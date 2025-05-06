from ragen.llm_agent.agent_proxy import LLMAgentProxy, VllmWrapperWg
from transformers import AutoTokenizer
from verl import DataProto
import hydra
import os
import time

"""
Evaluate the SFT model on single-turn SpatialQA tasks
TODO: 
1. Use EvaluationData
"""

@hydra.main(version_base=None, config_path="../../config", config_name="evaluate_qa")
def main(config):
	# detect config name from python -m ragen.sft.evaluate --config-name evaluate_qa
	os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
	os.environ["CUDA_VISIBLE_DEVICES"] = config.system.CUDA_VISIBLE_DEVICES
	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
	actor_wg = VllmWrapperWg(config, tokenizer)
	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
	start_time = time.time()
	rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample':config.actor_rollout_ref.rollout.do_sample, 'validate': True}), val=True)
	end_time = time.time()
	print(f'rollout time: {end_time - start_time} seconds')
	print(f"rollout: {rollouts}")
	# print rollout rewards from the rm_scores
	rm_scores = rollouts.batch["rm_scores"]
	metrics = rollouts.meta_info["metrics"]
	avg_reward = rm_scores.sum(-1).mean().item()
	print(f'rollout rewards: {avg_reward}')
	print(f'metrics:')
	for k, v in metrics.items():
		print(f'{k}: {v}')
		
if __name__ == "__main__":
	main()