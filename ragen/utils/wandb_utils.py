import wandb


def download_wandb(run_id, out_dir="./log", team="ragen", project="RAGEN"):
    api = wandb.Api()
    run = api.run(f"{team}/{project}/{run_id}")
    files = run.files()
    for file in files:
        file.download(out_dir + "/" + run_id, exist_ok=True)


# def download_wandb_logs(run_id, out_dir="./log", team="zihanwang-ai-northwestern-university", project="RAGEN"):
#     import wandb
#     from pathlib import Path
    
#     # Ensure output directory exists
#     Path(out_dir).mkdir(parents=True, exist_ok=True)
    
#     # Get the run
#     api = wandb.Api()
#     run = api.run(f"{team}/{project}/{run_id}")
    
#     # Use wandb.restore to download output.log
#     log_file = wandb.restore("logs", run_path="/".join(run.path))
    
#     if log_file:
#         # Copy the file to our desired output location
#         output_path = Path(out_dir) / f"{run_id}_output.log"
#         with open(log_file.name, 'r') as src, open(output_path, 'w') as dst:
#             dst.write(src.read())
#         print(f"Successfully downloaded logs to {output_path}")
#     else:
#         print("Failed to download logs")
        
# usage: 
# from ragen.utils.wandb import download_wandb
# download_wandb("9o465jqj")
def download_wandb_metrics(run_id, out_dir="./log", team="zihanwang-ai-northwestern-university", project="RAGEN", save_format="csv"):
    """
    Download all metrics from a W&B run and save them in a folder named after the experiment.
    
    Args:
        run_id (str): The run ID to download metrics from
        out_dir (str): Directory to save the metrics file
        team (str): W&B team name
        project (str): W&B project name
        save_format (str): Format to save the metrics ('csv' or 'json')
    
    Returns:
        tuple: Paths to the saved metrics and summary files
    """
    import wandb
    import pandas as pd
    from pathlib import Path
    
    # Initialize W&B API
    #wandb.login()
    api = wandb.Api()
    
    # Get the run
    run = api.run(f"{team}/{project}/{run_id}")
    
    # Get the experiment name from the run
    exp_name = run.name if run.name else "unnamed_exp"
    
    # Get the history metrics
    history_df = pd.DataFrame(run.history())
    
    # Get summary metrics (these are usually final values)
    summary_dict = {k: v for k, v in run.summary.items() 
                   if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
    
    # Create output directory with just the experiment name
    out_path = Path(out_dir) / exp_name
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save the metrics
    if save_format.lower() == 'csv':
        metrics_path = out_path / 'metrics.csv'
        history_df.to_csv(metrics_path, index=False)
        
        summary_path = out_path / 'summary.csv'
        pd.DataFrame([summary_dict]).to_csv(summary_path, index=False)
    else:  # json
        metrics_path = out_path / 'metrics.json'
        history_df.to_json(metrics_path, orient='records')
        
        summary_path = out_path / 'summary.json'
        pd.DataFrame([summary_dict]).to_json(summary_path, orient='records')
    
    return str(metrics_path), str(summary_path)