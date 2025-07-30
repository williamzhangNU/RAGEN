
from transformers import AutoTokenizer
import hydra
import os
from pathlib import Path
from typing import List, Dict
import time
import json
import re
from vllm import LLM, SamplingParams

from verl import DataProto
from omegaconf import ListConfig, DictConfig, OmegaConf
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray.base import RayWorkerGroup

from ragen.env.spatial.env import SpatialGym
from ragen.utilities.visualization import visualize_json
from .es_manager import EnvStateManager
from .ctx_manager import ContextManager
from .base_llm import ConcurrentLLM

class VllmWrapperWg: # Thi is a developing class for eval and test
	def __init__(self, config, tokenizer):
		self.config = config
		self.tokenizer = tokenizer
		model_name = config.actor_rollout_ref.model.path
		ro_config = config.actor_rollout_ref.rollout
		self.llm = LLM(
			model_name,
			enable_sleep_mode=True,
			tensor_parallel_size=ro_config.tensor_model_parallel_size,
			dtype=ro_config.dtype,
			enforce_eager=ro_config.enforce_eager,
			gpu_memory_utilization=ro_config.gpu_memory_utilization,
			disable_custom_all_reduce=True,
			disable_mm_preprocessor_cache=True,
			skip_tokenizer_init=False,
			max_model_len=ro_config.max_model_len,
			disable_log_stats=ro_config.disable_log_stats,
			max_num_batched_tokens=ro_config.max_num_batched_tokens,
			enable_chunked_prefill=ro_config.enable_chunked_prefill,
			enable_prefix_caching=True,
		)
		print("LLM initialized")
		self.sampling_params = SamplingParams(
			max_tokens=ro_config.response_length,
			temperature=ro_config.val_kwargs.temperature,
			top_p=ro_config.val_kwargs.top_p,
			top_k=ro_config.val_kwargs.top_k,
			# min_p=0.1,
		)

	def generate_sequences(self, lm_inputs: DataProto):
		"""
		Convert the input ids to text, and then generate the sequences. Finally create a dataproto. 
		This aligns with the verl Worker Group interface.
		"""
		# NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
		# cache_action = lm_inputs.meta_info.get('cache_action', None)

		input_ids = lm_inputs.batch['input_ids']
		input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
		input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

		outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
		texts = [output.outputs[0].text for output in outputs] 
		lm_outputs = DataProto()
		lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
		lm_outputs.meta_info = lm_inputs.meta_info

		return lm_outputs
	
class ApiCallingWrapperWg:
	"""Wrapper class for API-based LLM calls that fits into the VERL framework"""
	
	def __init__(self, config, tokenizer):
		self.config = config
		self.tokenizer = tokenizer
		model_info = config.model_info[config.api_model_info.model_name]
		self.llm_kwargs = model_info.generation_kwargs
		
		
		self.llm = ConcurrentLLM(
			provider=model_info.provider_name,
			model_name=model_info.model_name,
			max_concurrency=config.api_model_info.max_concurrency
		)
		
		print(f'API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized')


	def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
		"""
		Convert the input ids to text, make API calls to generate responses, 
		and create a DataProto with the results.
		"""

		messages_list = lm_inputs.non_tensor_batch['messages_list'].tolist()
		results, failed_messages = self.llm.run_batch(
			messages_list=messages_list,
			**self.llm_kwargs
		)
		assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

		texts = [result["response"] for result in results]
		lm_outputs = DataProto()
		lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
		lm_outputs.meta_info = lm_inputs.meta_info
		
		return lm_outputs

class LLMAgentProxy:
	"""
	The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
	"""
	def __init__(self, config, actor_rollout_wg, tokenizer):
		self.config = config
		self.train_ctx_manager = ContextManager(config, tokenizer, mode="train")
		self.train_es_manager = EnvStateManager(config, mode="train")
		self.val_ctx_manager = ContextManager(config, tokenizer, mode="val")
		self.val_es_manager = EnvStateManager(config, mode="val")
		self.actor_wg = actor_rollout_wg
		self.tokenizer = tokenizer

	def generate_sequences(self, lm_inputs: DataProto):
		# TODO: add kv cache both for the vllm wrapper here and for verl vllm.
		if isinstance(self.actor_wg, RayWorkerGroup):
			padded_lm_inputs, pad_size = pad_dataproto_to_divisor(lm_inputs, self.actor_wg.world_size)
			padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
			lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
			lm_outputs.meta_info = lm_inputs.meta_info
			lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
		elif isinstance(self.actor_wg, VllmWrapperWg) or isinstance(self.actor_wg, ApiCallingWrapperWg):
			lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
		else:
			raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")

		return lm_outputs

	def rollout(self, dataproto: DataProto, val=False):
		es_manager = self.val_es_manager if val else self.train_es_manager
		ctx_manager = self.val_ctx_manager if val else self.train_ctx_manager
		env_outputs = es_manager.reset()

		for i in range(self.config.agent_proxy.max_turn):
			lm_inputs: DataProto = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
			lm_inputs.meta_info = dataproto.meta_info # TODO: setup vllm early stop when max length is reached. make sure this can be done
			lm_outputs: DataProto = self.generate_sequences(lm_inputs)
			env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)
			env_outputs: List[Dict] = es_manager.step(env_inputs)
			if len(env_outputs) == 0: # all finished
				break
		rollout_states = es_manager.get_rollout_states() 
		rollouts = ctx_manager.formulate_rollouts(rollout_states)
		# self.tokenizer.batch_decode(rollouts.batch['input_ids'], skip_special_tokens=False) # see all the trajectories
		return rollouts


def convert_omegaconf_to_python(obj):
	"""Recursively convert OmegaConf objects to standard Python types for JSON serialization."""
	if isinstance(obj, (DictConfig, ListConfig)):
		return OmegaConf.to_container(obj, resolve=True)
	elif isinstance(obj, dict):
		return {key: convert_omegaconf_to_python(value) for key, value in obj.items()}
	elif isinstance(obj, list):
		return [convert_omegaconf_to_python(item) for item in obj]
	else:
		return obj

def log_each_env_info(envs: Dict[int, "SpatialGym"], messages: List[Dict], env_ids: List[int], config, output_path: str):
	"""Logs detailed information for each environment and overall performance metrics."""
	aggregated_data = SpatialGym.aggregate_env_data(envs, messages, env_ids)

	saved_data = {
		'meta_info': {
			'model_name': config.model_path if config.eval_model_type == "vllm" else config.api_model_info.model_name,
			'n_envs': len(envs),
		},
		**aggregated_data,
	}

	# Convert any OmegaConf objects to standard Python types for JSON serialization
	saved_data = convert_omegaconf_to_python(saved_data)
	
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "w") as f:
		json.dump(saved_data, f, indent=2)

	conversation_output_path = config.output_path.replace('.json', '_conversations.txt')
	format_conversations(
		messages=messages,
		env_ids=env_ids,
		output_path=conversation_output_path
	)
	html_dir = os.path.dirname(output_path)
	base = Path(output_path).stem

	# Generate HTML dashboard
	html_name = f"{base}_dashboard.html"
	html_path = os.path.join(html_dir, html_name)
	dashboard_path = visualize_json(output_path, html_path, True)
	print(f"Environment data logged to {output_path}")
	print(f"Dashboard written to {dashboard_path}")
	return output_path



def format_conversations(messages: List[Dict], env_ids: List[int], output_path: str):
	"""Parse and format conversations with clear separation of introduction, turns, and think/answer sections."""
	formatted_conversations = []
	
	for env_id, message_list in zip(env_ids, messages):
		
		# Extract introduction (first part)
		introduction = message_list[0] if message_list else ""
		
		# Parse remaining parts into turns
		turns = []
		current_turn = {"user": "", "agent_think": "", "agent_answer": ""}
		
		for message in message_list[1:]:
			if message['role'] == "user":
				# Save previous turn if exists
				if current_turn["user"] or current_turn["agent_think"] or current_turn["agent_answer"]:
					turns.append(current_turn)
				# Start new turn
				current_turn = {"user": message['content'], "agent_think": "", "agent_answer": ""}
			elif message['role'] == "assistant":
				# Parse agent response for think/answer sections
				agent_text = message['content']
				
				# Extract think section
				think_match = re.search(r'<think>(.*?)</think>', agent_text, re.DOTALL)
				current_turn["agent_think"] = think_match.group(1).strip() if think_match else ""
				
				# Extract answer section
				answer_match = re.search(r'<answer>(.*?)</answer>', agent_text, re.DOTALL)
				current_turn["agent_answer"] = answer_match.group(1).strip() if answer_match else ""
				
				# If no think/answer tags, treat whole response as answer
				if not think_match and not answer_match:
					current_turn["agent_answer"] = agent_text.replace("Assistant:", "").strip()
		
		# Add final turn
		if current_turn["user"] or current_turn["agent_think"] or current_turn["agent_answer"]:
			turns.append(current_turn)
		
		formatted_conversations.append({
			"env_id": env_id,
			"introduction": introduction,
			"turns": turns
		})
	
	# Format for readable output
	formatted_text = ""
	for conv in formatted_conversations:
		formatted_text += f"=" * 80 + "\n"
		formatted_text += f"ENVIRONMENT {conv['env_id']}\n"
		formatted_text += f"=" * 80 + "\n\n"
		
		formatted_text += f"INTRODUCTION:\n{'-' * 40}\n{conv['introduction']}\n\n"
		
		for i, turn in enumerate(conv['turns'], 1):
			formatted_text += f"TURN {i}:\n{'-' * 40}\n"
			
			if turn["user"]:
				formatted_text += f"USER:\n{turn['user']}\n\n"
			
			if turn["agent_think"]:
				formatted_text += f"AGENT THINKING:\n{turn['agent_think']}\n\n"
			
			if turn["agent_answer"]:
				formatted_text += f"AGENT ANSWER:\n{turn['agent_answer']}\n\n"
		
		formatted_text += "\n"
	
	# Save formatted conversations
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		f.write(formatted_text)
	
	print(f"Formatted conversations saved to {output_path}")
	return output_path

@hydra.main(version_base=None, config_path="../../config", config_name="evaluate_spatial")
def main(config):
	"""
	Usage: python -m ragen.llm_agent.agent_proxy --config-name evaluate_spatial
	"""
	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
	if config.eval_model_type == "vllm":
		os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
		os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
		actor_wg = VllmWrapperWg(config, tokenizer)
	elif config.eval_model_type == "api":
		actor_wg = ApiCallingWrapperWg(config, tokenizer)
	else:
		raise ValueError(f"Unsupported eval model type: {config.eval_model_type}")
	proxy = LLMAgentProxy(config, actor_wg, tokenizer)

	start_time = time.time()
	rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample':config.actor_rollout_ref.rollout.do_sample, 'validate': True}), val=True)
	end_time = time.time()
	print(f'rollout time: {end_time - start_time} seconds')
	# print rollout rewards from the rm_scores
	rm_scores = rollouts.batch["rm_scores"]
	metrics = rollouts.meta_info["metrics"]
	avg_reward = rm_scores.sum(-1).mean().item()
	print(f'rollout rewards: {avg_reward}')
	print(f'metrics:')
	for k, v in metrics.items():
		print(f'{k}: {v}')


	# specific analysis for spatial env
	log_each_env_info(
		envs={env['env_id']: env['env'] for env in proxy.val_es_manager.envs},
		messages=rollouts.non_tensor_batch['messages_list'].tolist(),
		env_ids=rollouts.non_tensor_batch['env_ids'].tolist(),
		config=config,
		output_path=config.output_path
	)
	
	# format conversations for readable output
	

if __name__ == "__main__":
	main()