import yaml
import os
import argparse
import json
from typing import Dict, Any
import time

def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_config(env_name: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Load configuration from base.yaml and environment-specific yaml file."""
    with open("config/base.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment-specific configuration
    env_config_path = f"config/env/{env_name}.yaml"
    if os.path.exists(env_config_path):
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
            config = deep_update(config, env_config)
    else:
        raise ValueError(f"Environment config not found: {env_config_path}")
    
    # Apply command line overrides if provided
    if overrides:
        config = deep_update(config, overrides)
    
    # Override with environment variables (CONFIG_XXX__YYY format)
    for key, value in os.environ.items():
        if key.startswith('CONFIG_'):
            # Remove CONFIG_ prefix and split by double underscore for nested keys
            key_path = key[7:].lower().split('__')
            
            # Convert value to appropriate type
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.replace('.', '').isdigit():
                value = float(value) if '.' in value else int(value)
                
            # Navigate to the correct nested dictionary
            current = config
            for k in key_path[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[key_path[-1]] = value
    
    return config

def parse_override_args(args_list):
    """Parse override arguments in the format key=value."""
    overrides = {}
    for arg in args_list:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Convert value to appropriate type
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.lower() == 'null':
                value = None
            elif value.replace('.', '').isdigit():
                value = float(value) if '.' in value else int(value)
            
            # Handle nested keys
            current = overrides
            key_parts = key.split('.')
            for part in key_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[key_parts[-1]] = value
    return overrides

def main():
    parser = argparse.ArgumentParser(description='Run training with config')
    parser.add_argument('env_name', nargs='?', default='sokoban', help='Environment name')
    parser.add_argument('overrides', nargs='*', help='Config overrides in the format key=value')
    parser.add_argument('--dry-run', action='store_true', help='Print the configuration and exit')
    
    args = parser.parse_args()
    
    # Parse command line overrides
    overrides = parse_override_args(args.overrides) if args.overrides else {}
    
    # Load configuration
    config = load_config(args.env_name, overrides)
    
    # Calculate max_prompt_length (could also be done in the YAML directly)
    config['data']['max_prompt_length'] = (
        config['data']['max_start_length'] +
        config['data']['max_response_length'] * (config['training']['max_turns'] - 1) +
        config['data']['max_obs_length'] * config['training']['max_turns']
    )
    config['critic']['model']['path'] = config['model']['path']
    
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['system']['cuda_visible_devices'])
    os.environ['VLLM_ATTENTION_BACKEND'] = config['system']['vllm_attention_backend']
    
    # Import the main module and run training
    import importlib
    main_module = importlib.import_module('main')
    main_module.main(config)

if __name__ == "__main__":
    main()