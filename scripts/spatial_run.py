#!/usr/bin/env python3
import argparse
import subprocess
import sys
import shlex
import os
from omegaconf import OmegaConf
from hydra import initialize, compose
import time
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run agent_proxy sequentially for multiple tasks: "
            "python -m ragen.llm_agent.agent_proxy tags=[task] num=<num> "
            "[override=true] [output_dir=...] [eval_model_type=...] [api_model_info.model_name=.../model_path=...]"
        )
    )
    parser.add_argument("--num_per_task", type=int, default=1, help="num of each task. Default: 1")
    parser.add_argument(
        "--task",
        dest="tasks",
        nargs="+",
        required=False,
        default=["ActiveRot"],
        help="Task tags. Space-separated, or a single comma-separated arg, e.g. --task ActiveRot,ActiveDir. Defaults to [ActiveRot] if omitted.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="If set, will override the active exploration history",
    )
    parser.add_argument(
        "--cogmap",
        action="store_true",
        help="If set, will enable cognitive map evaluation",
    )
    parser.add_argument(
        "--override-cogmap",
        action="store_true",
        help="If set, will override the cognitive map cache",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="For each task, set output_dir=<base>. Default base=results. If model_name contains slashes, the last segment is used for the directory name.",
    )
    parser.add_argument(
        "--eval_model_type",
        type=str,
        choices=["api", "vllm"],
        default="api",
        help="Override eval_model_type. Choices: api or vllm. Default: api",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4.1-mini",
        help="If eval_model_type=api, overrides api_model_info.model_name; if vllm, overrides model_path. Default: gpt-5-mini. If contains '/', the last segment is used for the output directory.",
    )
    parser.add_argument(
        "--no_think",
        action="store_true",
        help="If set, will disable think",
    )
    return parser.parse_args()


def normalize_tasks(tasks):
    # Support both: --task A B and --task A,B
    if len(tasks) == 1 and ("," in tasks[0]):
        return [t.strip() for t in tasks[0].split(",") if t.strip()]
    return tasks


def run_for_task(
    task: str,
    cwd: str,
    output_dir: str,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "ragen.llm_agent.agent_proxy",
        "--config-path",
        f"{os.path.abspath(output_dir)}",
        f"tags=[{task}]",
    ]
    effective_output_dir = os.path.abspath(os.path.join(output_dir, task))
    cmd.append(f"output_dir={effective_output_dir}")

    print("Running:", " ".join(shlex.quote(c) for c in cmd), f"(cwd={cwd})", flush=True)
    completed = subprocess.run(cmd, cwd=cwd)
    return completed.returncode

def save_config(args, output_dir):
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="evaluate_spatial",overrides=[
            f"tags={args.tasks}",
            f"num_per_task={args.num_per_task}",
            f"output_dir={output_dir}",
            f"eval_model_type={args.eval_model_type}",
            f"api_model_info.model_name={args.model_name}",
            f"model_path={args.model_name}",
            f"override={args.override}",
            f"evaluate_cogmap={args.cogmap}",
            f"agent_proxy.enable_think={not args.no_think}"
            f"override_cogmap={args.override_cogmap}",
        ])
    OmegaConf.save(cfg, os.path.join(output_dir, "evaluate_spatial.yaml"))

def main():
    args = parse_args()
    output_dir = os.path.join(args.output_dir, args.model_name.replace("\\", "/").rstrip("/").split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)
    tasks = normalize_tasks(args.tasks)
    save_config(args, output_dir)
    # Project root dir so that -m can find the ragen package
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for t in tasks:
        rc = run_for_task(
            t,
            cwd=root_dir,
            output_dir=output_dir,
        )
        if rc != 0:
            sys.exit(rc)
    sys.exit(0)


if __name__ == "__main__":
    main()
