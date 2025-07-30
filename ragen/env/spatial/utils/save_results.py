import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf

from ragen.env.spatial.utils.visualization import visualize_json
from ragen.env.spatial.Base.tos_base.core.room import Room
from ragen.env.spatial.Base.tos_base.utils.room_utils import set_initial_pos_as_origin

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

def plot_room(room_dict: Dict, out_dir: str, config_name: str, sample_idx: int, turn_idx: int) -> Optional[str]:
	"""Plot room from state and return image filename"""
	img_folder = os.path.join(out_dir, "images", config_name, f"sample_{sample_idx+1}")
	os.makedirs(img_folder, exist_ok=True)
	
	room = Room.from_dict(room_dict)
	transformed_room = set_initial_pos_as_origin(room)
	
	img_name = f"turn_{turn_idx+1}.png" if turn_idx > 0 else "initial_room.png"
	img_path = os.path.join(img_folder, img_name)
	transformed_room.plot(render_mode='img', save_path=img_path)
	
	return os.path.join("images", config_name, f"sample_{sample_idx+1}", img_name)

def log_each_env_info(envs, messages: List[Dict], env_ids: List[int], config, output_path: str):
	"""Logs detailed information for each environment and overall performance metrics."""
	aggregated_data = list(envs.values())[0].__class__.aggregate_env_data(envs, messages, env_ids)
	out_dir = os.path.dirname(output_path)

	# Plot rooms and add image paths to data
	for config_name, group in aggregated_data["config_groups"].items():
		for sample_idx, env_data in enumerate(group["env_data"]):
			# Plot initial room
			initial_room_dict = env_data["env_info"]["initial_room"]
			initial_img_path = plot_room(initial_room_dict, out_dir, config_name, sample_idx, 0)
			env_data["initial_room_image"] = initial_img_path
			
			# Plot room for each turn
			for turn_log in env_data["env_turn_logs"]:
				if turn_log["room_state"]:
					turn_idx = turn_log["turn_number"]
					img_path = plot_room(turn_log["room_state"], out_dir, config_name, sample_idx, turn_idx)
					turn_log["room_image"] = img_path

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

	html_dir = os.path.dirname(output_path)
	base = Path(output_path).stem

	# Generate HTML dashboard
	html_name = f"{base}_dashboard.html"
	html_path = os.path.join(html_dir, html_name)
	dashboard_path = visualize_json(output_path, html_path, True)
	print(f"Environment data logged to {output_path}")
	print(f"Dashboard written to {dashboard_path}")
	return output_path
