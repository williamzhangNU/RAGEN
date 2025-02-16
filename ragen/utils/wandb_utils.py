
import wandb
import pandas as pd
from pathlib import Path

def download_wandb(run_id, out_dir="./log", team="ragen", project="RAGEN"):
    api = wandb.Api()
    run = api.run(f"{team}/{project}/{run_id}")
    files = run.files()
    for file in files:
        file.download(out_dir + "/" + run_id, exist_ok=True)


def download_wandb_metrics(run_id, out_dir="./log", team="ragen", project="RAGEN", save_format="csv"):
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