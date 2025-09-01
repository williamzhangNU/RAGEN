#!/usr/bin/env python3
import argparse
import subprocess
import sys
import shlex
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run agent_proxy sequentially for multiple tasks: "
            "python -m ragen.llm_agent.agent_proxy tags=[task] num=<num> "
            "[override=true] [output_dir=...] [eval_model_type=...] [api_model_info.model_name=.../model_path=...]"
        )
    )
    parser.add_argument("--num", type=int, default=1, help="num of each task. Default: 1")
    parser.add_argument(
        "--task",
        dest="tasks",
        nargs="+",
        required=False,
        default=None,
        help="Task tags. Space-separated, or a single comma-separated arg, e.g. --task ActiveRot,ActiveDir. Defaults to [ActiveRot] if omitted.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="If set, will override the active exploration history",
    )
    parser.add_argument(
        "--output_dir",
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
        default="gpt-5-mini",
        help="If eval_model_type=api, overrides api_model_info.model_name; if vllm, overrides model_path. Default: gpt-5-mini. If contains '/', the last segment is used for the output directory.",
    )
    return parser.parse_args()


def normalize_tasks(tasks):
    # If --task not provided, use [ActiveRot]
    if tasks is None:
        return ["ActiveRot"]
    # Support both: --task A B and --task A,B
    if len(tasks) == 1 and ("," in tasks[0]):
        return [t.strip() for t in tasks[0].split(",") if t.strip()]
    return tasks


def run_for_task(
    task: str,
    num: int,
    cwd: str,
    override: bool,
    output_dir: str,
    eval_model_type: str,
    model_name: str,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "ragen.llm_agent.agent_proxy",
        f"tags=[{task}]",
        f"num={num}",
    ]
    if override:
        cmd.append("override=true")

    # Directory segment: if model_name contains slashes (or backslashes), use the last segment
    model_seg = model_name.replace("\\", "/").rstrip("/").split("/")[-1]
    effective_output_dir = os.path.join(output_dir, model_seg, task)
    cmd.append(f"output_dir={effective_output_dir}")

    # Backend type override
    cmd.append(f"eval_model_type={eval_model_type}")

    # Model override: api uses api_model_info.model_name; vllm uses model_path
    if eval_model_type == "vllm":
        cmd.append(f"model_path={model_name}")
    else:
        cmd.append(f"api_model_info.model_name={model_name}")

    print("Running:", " ".join(shlex.quote(c) for c in cmd), f"(cwd={cwd})", flush=True)
    completed = subprocess.run(cmd, cwd=cwd)
    return completed.returncode


def main():
    args = parse_args()
    tasks = normalize_tasks(args.tasks)
    # Project root dir so that -m can find the ragen package
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for t in tasks:
        rc = run_for_task(
            t,
            args.num,
            cwd=root_dir,
            override=args.override,
            output_dir=args.output_dir,
            eval_model_type=args.eval_model_type,
            model_name=args.model_name,
        )
        if rc != 0:
            sys.exit(rc)
    sys.exit(0)


if __name__ == "__main__":
    main()
