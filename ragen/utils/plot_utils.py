import matplotlib.pyplot as plt
from typing import Dict, Any
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
import seaborn as sns
import os
import pandas as pd
from matplotlib.ticker import MaxNLocator
import numpy as np
def plot_line_chart(processed_data: Dict[str, Dict[str, Any]], 
                font_config: Dict[str, int] = None,
                colors: Dict[str, str] = None,
                y_data_label: str = "High Arm Rate",
                x_data_label: str = "Steps",
                save_path: str = './figure/test.pdf',
                figure_size: Tuple[int, int] = (10, 6),
                use_log_scale: bool = False,
                log_base: float = 2.0,
                smooth: bool = False,
                smoothing_factor: int = None) -> plt.Figure:
    """
    Create a line plot from processed metrics data with optional logarithmic x-axis and smoothing.
    Integer ticks are enforced on the x-axis.
    
    Args:
        processed_data (dict): Dictionary containing processed metrics data
        font_config (dict, optional): Configuration for font sizes
        colors (dict, optional): Dictionary mapping plot names to colors
        y_data_label (str): Label for y-axis
        x_data_label (str): Label for x-axis
        save_path (str): Path to save the figure (must end with .pdf)
        figure_size (tuple): Figure size in inches (width, height)
        use_log_scale (bool): Whether to use logarithmic scale for x-axis
        log_base (float): Base for logarithmic scale (default: 2.0)
        smooth (bool): Whether to apply curve smoothing
        smoothing_factor (int, optional): Number of points to use in smoothing interpolation.
                                        If None, automatically calculated based on data size.
    
    Returns:
        matplotlib.figure.Figure: The generated plot figure
    """
    from scipy.interpolate import make_interp_spline
    import numpy as np
    
    # Initialize colors dictionary if None
    if colors is None:
        colors = {}
    
    # Set default font configuration if not provided
    default_sizes = {
        'xlabel_size': 30,
        'ylabel_size': 30,
        'title_size': 24,
        'legend_size': 18,
        'xtick_size': 24,
        'ytick_size': 24
    }
    if font_config is None:
        font_config = {}
    for key, value in default_sizes.items():
        if key not in font_config:
            font_config[key] = value
    
    # Create figure and axis with specified size
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Set the style
    sns.set_style("whitegrid")
    
    # Set x-axis to logarithmic scale if specified
    if use_log_scale:
        ax.set_xscale('log', base=log_base)
        
        # For log scale, we need to handle the tick locations carefully
        all_x_values = []
        for data in processed_data.values():
            all_x_values.extend(data[x_data_label])
        min_x = min(all_x_values)
        max_x = max(all_x_values)
        
        # Generate integer tick positions that cover the data range
        tick_positions = []
        current = int(np.ceil(min_x))
        while current <= max_x:
            if current >= min_x:
                tick_positions.append(current)
            current = int(current * log_base)
        
        # Set the custom tick positions and labels
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(int(x)) for x in tick_positions])
    else:
        # For linear scale, use MaxNLocator to force integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Plot each line
    for plot_name, data in processed_data.items():
        x = np.array(data[x_data_label])
        y = np.array(data[y_data_label])
        color = colors.get(plot_name)  # Get color from dictionary, will be None if not found
        
        if smooth and len(x) > 3:  # Need at least 4 points for cubic spline
            # Remove duplicate x values by averaging corresponding y values
            unique_x, unique_indices = np.unique(x, return_inverse=True)
            unique_y = np.array([np.mean(y[unique_indices == i]) for i in range(len(unique_x))])
            
            # Sort the unique values
            sort_idx = np.argsort(unique_x)
            x_sorted = unique_x[sort_idx]
            y_sorted = unique_y[sort_idx]
            
            # Calculate automatic smoothing factor if not provided
            if smoothing_factor is None:
                n_points = len(x_sorted)
                if n_points < 50:
                    # For small datasets, use 3x the number of points
                    smoothing_factor = n_points * 3
                else:
                    # For larger datasets, use a more aggressive smoothing
                    smoothing_factor = n_points * 2
            
            try:
                # Create smooth curve using spline interpolation
                x_smooth = np.linspace(min(x_sorted), max(x_sorted), smoothing_factor)
                
                # Adjust spline degree based on number of points
                k = min(3, len(x_sorted) - 1)  # cubic spline if possible, lower order if not
                spl = make_interp_spline(x_sorted, y_sorted, k=k)
                y_smooth = spl(x_smooth)
                
                # Plot the smoothed line
                ax.plot(x_smooth, y_smooth, label=plot_name, color=color)
                
                # Plot original points with reduced opacity
                ax.scatter(x, y, color=color, alpha=0.3, s=20)
            except ValueError:
                # Fallback to original plotting if smoothing fails
                print(f"Warning: Smoothing failed for {plot_name}, falling back to original data")
                sns.lineplot(x=x, y=y, label=plot_name, color=color, ax=ax)
        else:
            # Plot original data without smoothing
            sns.lineplot(x=x, y=y, label=plot_name, color=color, ax=ax)
    
    # Customize the plot
    ax.set_xlabel(x_data_label, fontsize=font_config['xlabel_size'])
    ax.set_ylabel(y_data_label, fontsize=font_config['ylabel_size'])
    #ax.set_title(f"{y_data_label} vs {x_data_label}", fontsize=font_config['title_size'])
    ax.legend(fontsize=font_config['legend_size'],   # (x, y) coordinates relative to the axes
         loc='upper left') 
    
    # Set tick label sizes
    ax.tick_params(axis='x', labelsize=font_config['xtick_size'])
    ax.tick_params(axis='y', labelsize=font_config['ytick_size'])
    
    # Format x-axis ticks to show actual values without scientific notation
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure in PDF format with high DPI
    plt.savefig(save_path, format='pdf', dpi=600, bbox_inches='tight')
    
    # Close the figure
    plt.close()
    
def extract_metrics(file_plot_dict,metric_names_mapping,hard_cut_off=1000):
    processed_data = {}
    
    for file_name, plot_name in file_plot_dict.items():
        # Read the CSV file
        df = pd.read_csv(file_name)
        
        dp={}
        dp["Steps"] = df["_step"].values
        for k,v in metric_names_mapping.items():
            dp[v] = df[k].values
        
        processed_data[plot_name] = dp
    # make sure the length is the same for all, check the longest common length and perform truncation
    for k,v in processed_data.items():
        print(k,len(v['Steps']))
    min_len = min([len(v['Steps']) for v in processed_data.values()])
    min_len = min(min_len,hard_cut_off)
    for k,v in processed_data.items():
        for kk,vv in v.items():
            processed_data[k][kk] = vv[:min_len]
    return processed_data


