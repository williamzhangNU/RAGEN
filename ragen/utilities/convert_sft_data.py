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

def convert_sft_data(
    json_data: List[Dict[str, Any]], 
    output_path: str = "sft_dataset.json",
    output_prompt_key: str = "question",
    output_response_key: str = "answer",
    types: List[str] = None,
    prompt_kwargs: Dict[str, Any] = {}
) -> str:
    """
    Converts a list of QA JSON objects to a Parquet file compatible with SFTDataset.
    SFT data point: 
        - question:
        - answer:
    
    Args:
        json_data: List of dictionaries containing at least 'question' and 'answer' fields
        output_path: Path to save the output json file
        output_prompt_key: Key in the output parquet for the prompt (default: 'question')
        output_response_key: Key in the output parquet for the response (default: 'answer')
        types: List of question types to filter by
        prompt_kwargs: Additional keyword arguments for prompt formatting
    
    Returns:
        Path to the saved JSON file
    """
    def _convert_to_chat_message(prompt: str) -> List[Dict[str, str]]:
        THINK_PROMPT = "first wrapping your thoughts in <think>...</think>, then " if prompt_kwargs.get("enable_think", True) else ""
        return [
            {"role": "system", "content": f"You're a helpful assistant. You always respond by {THINK_PROMPT}giving your answer in <answer>...</answer>. Max response length: {prompt_kwargs.get('max_tokens', 400)} words (tokens)."}, 
            {"role": "user", "content": prompt}
        ]
    
    def _convert_to_answer_format(answer: str) -> str:
        THINK_TOKEN = "<think>...</think>" if prompt_kwargs.get("enable_think", True) else ""
        return f"{THINK_TOKEN}<answer>{answer}</answer>"
    
    # Filter the json data by question_type if types is provided
    if types is not None:
        json_data = [item for item in json_data if item["meta_data"]["type"] in types]
    
    # Process the data
    processed_data = [{output_prompt_key: _convert_to_chat_message(item['question']), output_response_key: _convert_to_answer_format(item['answer'])} for item in json_data]
    
    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=4)

    

# Example usage
if __name__ == "__main__":
    
    # # unit test
    # json_data = [
    #     {
    #         "question": "What is machine learning?",
    #         "answer": "Machine learning is a subfield of artificial intelligence...",
    #         "dataset_name": "ml_qa",
    #         "meta_data": {"question_type": "direction", "type": "EgoDirectionSS"},
    #         "extra": {"difficulty": "beginner"}
    #     },
    #     {
    #         "question": "Explain neural networks.",
    #         "answer": "Neural networks are computing systems inspired by biological neural networks...",
    #         "dataset_name": "ml_qa",
    #         "meta_data": {"question_type": "direction", "type": "EgoDirectionSS"},
    #         "extra": {"difficulty": "intermediate"}
    #     }
    # ]
    
    # # Convert to Parquet
    # convert_sft_data(json_data, prompt_kwargs={"enable_think": True, "max_tokens": 400})


    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--types", nargs='+', type=str, default=None)
    parser.add_argument("--enable_think", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=400)
    args = parser.parse_args()

    with open(args.json_file, "r") as f:
        json_data = json.load(f)

    convert_sft_data(
        json_data,
        output_path=args.output_path,
        types=args.types,
        prompt_kwargs={"enable_think": args.enable_think, "max_tokens": args.max_tokens}
    )