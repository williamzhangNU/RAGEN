
from transformers import AutoTokenizer
import hydra
import os
from typing import List, Dict
import time
import json
import datetime

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from .ctx_manager import ContextManager
from .es_manager import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
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
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs
        
        
        self.llm = ConcurrentLLM(
			provider=model_info.provider_name,
            model_name=model_info.model_name,
            max_concurrency=config.model_config.max_concurrency
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
        print(f'[DEBUG] texts: {texts}')
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


def log_each_env_info(envs: List[Dict], messages, env_ids, config, output_dir):
	saved_data = {
		'meta_info': {
			'model_name': config.actor_rollout_ref.model.path if config.eval_model_type == "api" else config.model_config.model_name,
			'n_envs': len(envs),
		},
		'overall_performance': {
			'exploration_efficiency': {
				'avg_coverage': 0.0,
				'avg_novelty': 0.0,
			},
			'evaluation_performance': {
				'avg_accuracy': 0.0,
			}
		},
		'env_data': [],
	}
	
	# Collect data from all environments
	total_envs = 0
	exp_eff_sum = {'coverage': 0.0, 'novelty': 0.0}
	eval_perf_sum = {'accuracy': 0.0}
	
	for _, (message, env_id) in enumerate(zip(messages, env_ids)):
		env = envs[env_id]
		env_info = env.get_env_info()
		exp_efficiency = env.get_exp_efficiency()
		eval_performance = env.get_eval_performance()
		
		saved_data['env_data'].append({
			"message": message,
			"env_info": env_info,
			"config": env_info.get("config", {}),
			"exploration_efficiency": exp_efficiency,
			"evaluation_performance": eval_performance
		})
		
		# Accumulate for overall statistics
		total_envs += 1
		for key in exp_eff_sum:
			exp_eff_sum[key] += exp_efficiency.get(key, 0)
		for key in eval_perf_sum:
			eval_perf_sum[key] += eval_performance.get(key, 0)
	
	# Calculate averages for overall performance
	if total_envs > 0:
		saved_data['overall_performance']['exploration_efficiency'] = {
			'avg_coverage': exp_eff_sum['coverage'] / total_envs,
			'avg_novelty': exp_eff_sum['novelty'] / total_envs,
		}
		saved_data['overall_performance']['evaluation_performance'] = {
			'avg_accuracy': eval_perf_sum['accuracy'] / total_envs,
		}
		
	
	

	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	os.makedirs(output_dir, exist_ok=True)
	log_filename = f"{output_dir}/env_log_{timestamp}.json"
	with open(log_filename, "w") as f:
		json.dump(saved_data, f, indent=2)
	
	print(f"Environment data logged to {log_filename}")
	return saved_data
	

@hydra.main(version_base=None, config_path="../../config", config_name="evaluate_spatial")
def main(config):
	"""
	Usage: python -m ragen.llm_agent.agent_proxy --config-name evaluate_spatial
	"""
	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
	print(config.system.CUDA_VISIBLE_DEVICES)
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
	
	
	# specific analysis for spatial env
	saved_data = log_each_env_info(
		envs={env['env_id']: env['env'] for env in proxy.val_es_manager.envs},
		messages=rollouts.non_tensor_batch['messages_list'].tolist(),
		env_ids=rollouts.non_tensor_batch['env_ids'].tolist(),
		config=config,
		output_dir=config.output_dir
	)
	rollouts.meta_info["metrics"]["avg_accuracy"] = saved_data['overall_performance']['evaluation_performance']['avg_accuracy']
	rollouts.meta_info["metrics"]["avg_coverage"] = saved_data['overall_performance']['exploration_efficiency']['avg_coverage']
	rollouts.meta_info["metrics"]["avg_novelty"] = saved_data['overall_performance']['exploration_efficiency']['avg_novelty']


	metrics = rollouts.meta_info["metrics"]
	avg_reward = rm_scores.sum(-1).mean().item()
	print(f'rollout rewards: {avg_reward}')
	print(f'metrics:')
	for k, v in metrics.items():
		print(f'{k}: {v}')

	
	
	
	
	
	
	
	
	
	
	

if __name__ == "__main__":
	main()
