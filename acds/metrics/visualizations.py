import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional


def scatter_metrics_vs_performance(df, metric_names: List[str], performance_metric: str, title: str, save_path: Optional[str] = None):
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metric_names):
        # scatter plot with  x-axis: metric y-axis: performance_metric color: n_modules marker: connection type

        sns.scatterplot(data=df, x=metric, y=performance_metric, hue='n_modules', style='connection_type', ax=ax, palette='inferno')
        ax.set_title(f'{metric} vs {performance_metric}')
        ax.set_xlabel(metric)
        ax.set_ylabel(performance_metric)
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()



def spider_plot(df, metric_names: List[str], title: str, save_path: Optional[str] = None):
    categories = metric_names
    N = len(categories)
    
    # Sort n_modules in increasing order
    n_modules_unique = np.sort(df['n_modules'].unique())
    nm = len(n_modules_unique)
    
    # Heuristic to decide number of columns: aim for roughly square grid
    n_cols = int(np.ceil(np.sqrt(nm)))
    n_rows = int(np.ceil(nm / n_cols))
    
    # Create subplots with appropriate grid layout
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows), subplot_kw=dict(polar=True))
    
    # Handle case of single or reshaped subplot array
    if nm == 1:
        axs = np.array([axs])
    else:
        axs = axs.flatten()
    
    # Calculate angles for spider plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Plot for each n_modules value
    for idx, n_modules in enumerate(n_modules_unique):
        ax = axs[idx]
        
        # Get data for this n_modules value
        group = df[df['n_modules'] == n_modules]
        values = group[metric_names].mean().values.flatten().tolist()
        values += values[:1]
        
        # Configure subplot
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Plot data
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'n_modules={n_modules}', color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
        ax.set_title(f'n_modules = {n_modules}', size=12, pad=20)
    
    # Hide unused subplots
    for idx in range(nm, len(axs)):
        axs[idx].set_visible(False)
    
    plt.suptitle(title, size=16, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Create a sample dataframe
    data = {
        'correlation_dimension': np.random.rand(100),
        'participation_ratio': np.random.rand(100),
        'n_modules': np.random.choice([2**n for n in range(1, 8)], size=100),
        'connection_type': np.random.choice([f'type{n}' for n in range(1, 5)], size=100)
    }
    data['performance_metric'] = data['correlation_dimension'] * 0.5 + data['participation_ratio'] * 0.5 + np.random.rand(100) * 0.1
    df = pd.DataFrame(data)

    # scatter_metrics_vs_performance(
    #     df,
    #     metric_names=['correlation_dimension', 'participation_ratio'],
    #     performance_metric='performance_metric',
    #     title='Attractor Dimension Metrics vs Performance',
    #     save_path=None
    # )

    spider_plot(
        df,
        metric_names=['correlation_dimension', 'participation_ratio', 'performance_metric'],
        title='Average Metrics Spider Plot',
        save_path=None  
    )
