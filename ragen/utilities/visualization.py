from html import escape
from itertools import zip_longest
import re
import os
from typing import List, Dict
from pathlib import Path
from ragen.env.spatial.env import SpatialGym

THINK_RE  = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
ANS_RE    = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

def _extract(tag_re, text):
	m = tag_re.search(text)
	return m.group(1).strip() if m else ""

def split_into_turns(messages):
	"""
	messages: List[{'role': 'user'/'assistant'/..., 'content': str}]
	Returns: List[{'user':..., 'think':..., 'answer':..., 'raw':..., 'reward':..., 'state':...}]
	"""
	turns = []
	cur = {"user": "", "think": "", "answer": "", "reward": "", "state": "", "raw": ""}

	for m in messages:
		role = m.get("role", "")
		content = m.get("content", "")
		cur["raw"] += content + "\n"

		if role == "user":
			# split reward/state if you want
			# example: reward block at top of content
			# simple heuristic:
			parts = content.split("Turn")
			# keep whole thing as user text anyway
			cur["user"] = content
			# optional: parse reward/state if your format is stable
		elif role == "assistant":
			cur["think"]  = _extract(THINK_RE, content)
			cur["answer"] = _extract(ANS_RE, content)

			# finalize a turn when we see an assistant
			turns.append(cur)
			cur = {"user": "", "think": "", "answer": "", "reward": "", "state": "", "raw": ""}

	# flush if dangling
	if any(cur.values()):
		turns.append(cur)

	return turns

def dict_to_html(d):
	return "".join(f"<div><strong>{escape(str(k))}:</strong> {escape(str(v))}</div>" for k,v in d.items())

def squash_exp_logs(exp_log: list[dict]) -> list[dict]:
    """
    Merge every two consecutive entries into one dict.
    If the length is odd, keep the last one as-is.
    Heuristic: even idx = cumulative snapshot, odd idx = action info.
    """
    merged = []
    it = iter(exp_log)
    for first in it:
        try:
            second = next(it)
            combined = {**first, **second}
        except StopIteration:
            combined = first
        merged.append(combined)
    return merged

def visualize(envs: Dict[int, "SpatialGym"], messages: List[Dict], env_ids: List[int], config, output_path: str):
	aggregated = SpatialGym.aggregate_env_data(envs, messages, env_ids)
	meta = {
		'model_name': config.model_path if config.eval_model_type == "vllm" else config.api_model_info.model_name,
		'n_envs': len(envs),
	}

	# Save initial room plot
	html_dir = os.path.dirname(output_path)
	base = Path(output_path).stem

	# Generate HTML dashboard
	html_name = f"{base}_dashboard.html"
	html_path = os.path.join(html_dir, html_name)
	with open(html_path, 'w') as f:
		f.write("""<!DOCTYPE html>
		<html>
		<head>
		<meta charset="utf-8">
		<title>SpatialGym Dashboard</title>
		<style>
		body{font-family:Arial,Helvetica,sans-serif;margin:0;padding:0;background:#fafafa;}
		#nav{position:fixed;top:0;width:100%;background:#333;color:#fff;padding:10px 0;text-align:center;z-index:999;}
		#nav button{margin:0 8px;padding:6px 12px;border:none;background:#555;color:#fff;cursor:pointer;border-radius:4px;}
		#nav button:hover{background:#777;}
		#counter{margin-left:12px;font-size:0.9em;opacity:0.8;}

		.sample-page{display:none;padding:80px 24px 24px 24px; max-width:1000px;margin:auto;}
		.sample-page.active{display:block;}

		.turn{background:#fff;border:1px solid #ddd;border-radius:6px;margin:14px 0;padding:12px;}
		.turn h3{margin:0 0 6px 0;font-size:16px;color:#333;}
		.block{padding:8px 10px;border-radius:4px;margin:6px 0;font-size:14px;line-height:1.5;}
		.block.user{background:#e8f4ff;border-left:4px solid #4299e1;}
		.block.think{background:#fff7e6;border-left:4px solid #ed8936;font-style:italic;}
		.block.answer{background:#e6ffed;border-left:4px solid #38a169;}

		.metrics{margin-top:8px;font-size:13px;color:#444;}
		.metrics div{margin:2px 0;}
		img.room{max-width:220px;height:auto;border:1px solid #ccc;margin-top:6px;}

		h1{margin-top:40px;color:#222;}
		h2{color:#444;margin-top:20px;}
		</style>

		<script>
		let currentPage = 0;
		function showPage(n,total){
  		currentPage = Math.max(0, Math.min(total-1, n));
  		const pages = document.querySelectorAll('.sample-page');
  		pages.forEach((p,i)=>{ p.classList.toggle('active', i===currentPage); });
  		document.getElementById('counter').innerText = (currentPage+1)+' / '+total;
		}
		function nextPage(total){ showPage(currentPage+1,total); }
		function prevPage(total){ showPage(currentPage-1,total); }
		</script>
		</head>
		<body>
	""")
	# model info header strip
		f.write(f"<div id='nav'><button onclick=\"prevPage({len(aggregated['env_data'])})\">Prev</button>"
			f"<button onclick=\"nextPage({len(aggregated['env_data'])})\">Next</button>"
			f"<span id='counter'></span></div>\n")

		f.write(f"<h1 style='padding-top:60px;text-align:center;'>Model: {escape(meta['model_name'])}</h1>\n")

		total_pages = len(aggregated['env_data'])

		for idx, entry in enumerate(aggregated['env_data']):

		# metrics: static + per-turn
			em_static = entry.get('exploration_efficiency', {})
			ev_static = entry.get('evaluation_performance', {})
			metrics   = {**em_static, **ev_static}

		# write room config and plot room
			env_id  = env_ids[idx]
			room    = envs[env_id].initial_room
			env_info = envs[env_id].get_env_info()
			env_config = {**env_info['config']}
			turn_img_name = f"{base}_turn{idx+1}.png"
			turn_img_path = os.path.join(html_dir, turn_img_name)
			room.plot(render_mode='img', save_path=turn_img_path)

		# write section
			f.write(f"<section class='sample-page' id='page{idx}'>\n")
			f.write(f"<h2>Sample {idx+1}</h2>\n")
			f.write(f"<img src='{turn_img_name}' alt='room' class='room'>\n")
			f.write("<div class='metrics'><strong>Env Info</strong>\n")
			for k,v in env_config.items():
				f.write(f"<div>{escape(str(k))}: {escape(str(v))}</div>")
			f.write("</div>\n")  # metrics
			turns = split_into_turns(entry.get("message", []))
			exp_log_raw = entry.get("exploration_metrics_log", [])
			exp_log = squash_exp_logs(exp_log_raw)
			eval_log = entry.get("evaluation_metrics_log", [])

			for t_idx, (turn, em) in enumerate(zip_longest(turns, exp_log, fillvalue={})):
			  
				per_turn_metrics = {}
				if isinstance(em, dict): per_turn_metrics.update(em)
				met_html = dict_to_html(per_turn_metrics) or "<div>(none)</div>"

				f.write("<div class='turn'>\n")
				f.write(f"<h3>Turn {t_idx+1}</h3>\n")

				if turn.get("user"):
					f.write(f"<div class='block user'><strong>User</strong><br>{turn['user']}</div>\n")
				if turn.get("think"):
					f.write(f"<div class='block think'><strong>Agent Think</strong><br>{turn['think']}</div>\n")
				if turn.get("answer"):
					f.write(f"<div class='block answer'><strong>Agent Answer</strong><br>{turn['answer']}</div>\n")

				f.write("<div class='metrics'><strong>Per Turn Metrics</strong>")
				f.write(met_html)
				f.write("</div>\n")  # metrics

				f.write("</div>\n")  # .turn

			f.write("<div class='metrics'><strong>Final Metrics</strong>\n")
			for k,v in metrics.items():
				f.write(f"<div>{escape(str(k))}: {escape(str(v))}</div>")
			f.write("</div>\n")  # metrics

			f.write("</div>\n")  # turn
			f.write("</section>\n")

		# finish body
		f.write(f"<script>showPage(0,{total_pages});</script>\n")
		f.write("</body></html>")

	print(f"Dashboard written to {html_path}")
	return html_path