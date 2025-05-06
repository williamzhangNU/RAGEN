"""
The script is used to convert json spatial qa data to parquet format for sft.
Also convert the question-answer pair to the required data format identical to RAGEN
    - add <think> and <answer> tags
    - convert to chat message format as used in RAGEN
    - Refering to ctx_manager.py

TODO modify the function using the agent_proxy of single-turn qa
"""
import json
from typing import List, Dict, Union, Any
import argparse
from transformers import AutoTokenizer
from datasets import Dataset
import copy
import hydra
import os
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from ragen.llm_agent.ctx_manager import ContextManager
from ragen.llm_agent.es_manager import EnvStateManager

class SFTDataGenerator:
    """
    Generate SFT data using the same pipeline as LLMAgentProxy, keep consistent with RL
    """
    def __init__(self, config, tokenizer):
        self.config = config
        self.ctx_manager = ContextManager(config, tokenizer, mode=config.mode)
        self.es_manager = EnvStateManager(config, mode=config.mode)
        self.tokenizer = tokenizer
          
    def gen_sft_data(
            self,
            output_prompt_key: str = "question",
            output_response_key: str = "answer",
            enable_think: bool = True,
            output_path: str = "sft_dataset.json",
            seed: int = 42,
        ):
        """
        Parameters:
            enable_think: bool, whether to add <think> and <answer> tags
            output_path: str, path to save the sft data
        
        Format: (list of dict)
            - {prompt_key}: sharegpt-format messages, containing system and user messages
            - {response_key}: response string
        """
        env_outputs = self.es_manager.reset(seed=seed)
        lm_inputs: DataProto = self.ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
        messages_list = lm_inputs.non_tensor_batch['messages_list'].tolist()
        env_ids = lm_inputs.non_tensor_batch['env_ids'].tolist()
        envs = {env['env_id']: env['env'] for env in self.es_manager.envs}
        sft_data = []
        for messages, env_id in zip(messages_list, env_ids):
            env = envs[env_id]
            reasoning = f'<think>{env.eval_tasks[0].reasoning}</think> ' if enable_think else ''
            answer = f'<answer>{env.eval_tasks[0].answer}</answer>'
            messages = messages[:2]
            assert messages[0]['role'] == 'system' and messages[1]['role'] == 'user'
            response = f'{reasoning}{answer}'
            sft_data.append({
                output_prompt_key: messages,
                output_response_key: response
            })

        print(sft_data)
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # with open(output_path, 'w') as f:
        #     json.dump(sft_data, f, indent=4)
    

@hydra.main(version_base=None, config_path="../../config", config_name="gen_sft")
def main(config):
    """
    Usage: python -m ragen.utilities.gen_sft_data --config-name gen_sft
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    sft_data_generator = SFTDataGenerator(config, tokenizer)
    sft_data_generator.gen_sft_data(
        enable_think=config.enable_think,
        output_path=config.output_path,
        output_prompt_key=config.output_prompt_key,
        output_response_key=config.output_response_key,
        seed=config.seed,
    )

# Example usage
if __name__ == "__main__":
    main()