"""
Visualization utilities for attractor dimension metrics.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import numpy as np
from typing import List, Optional

__all__ = [
    "scatter_metrics_vs_performance",
    "spider_plot",
    "lineplot_by_n_modules",
    "heatmap_metrics",
]


def scatter_metrics_vs_performance(
    df,
    metric_names: List[str],
    performance_metric: str,
    title: str,
    save_path: Optional[str] = None,
    semilogy: bool = False,
):
    n_metrics = len(metric_names)

    fig, axes = plt.subplots(
        1,
        n_metrics,
        figsize=(5 * n_metrics, 4),
        constrained_layout=True
    )

    axes = np.atleast_1d(axes)
    df['log_n_modules'] = df['n_modules'].apply(np.log)
    for i, (ax, metric) in enumerate(zip(axes, metric_names)):
        sns.scatterplot(
            data=df,
            x=metric,
            y=performance_metric,
            hue="n_modules",
            style="connection_matrix",
            palette="inferno",
            ax=ax,
            hue_norm=LogNorm(vmax=20),
            legend=(i==0),  # create legend on first axis only
            size=1,
        )

        ax.set_title(f"{metric} vs {performance_metric}")
        ax.set_xlabel(metric)
        ax.set_ylabel(performance_metric)

        if semilogy:
            ax.set_yscale("log")

    # Extract legend handles from the first axis
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()
    # Place a single legend to the right of all subplots
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    return fig, axes


def spider_plot(df, metrics: List[str], metrics_names, title: str, save_path: Optional[str] = None):
    categories = metrics
    N = len(categories)
    # normalize metrics to [0, 1]
    df_norm = df.copy()
    for metric in metrics:
        min_val = df[metric].min()
        max_val = df[metric].max()
        df_norm[metric] = (df[metric] - min_val) / (max_val - min_val)
    # Filter for n_modules > 1
    df_filtered = df_norm[df_norm['n_modules'] > 1]
    df_one_module = df_norm[df_norm['n_modules'] == 1]
    # Get unique connection types

    connection_types = df_filtered['connection_matrix'].unique()
    
    # Calculate angles for spider plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Create single polar plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Configure polar axes
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([metrics_names[metric] for metric in categories], size=20, position=(0, -0.1))
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Define color palette for different connection types
    colors = sns.color_palette("husl", len(connection_types))
    
    # Plot for each connection type
    for idx, conn_type in enumerate(connection_types):
        df_conn = df_filtered[df_filtered['connection_matrix'] == conn_type]
        
        # Get the maximum value for each metric across all parameter combinations
        values = df_conn[metrics].max().values.flatten().tolist()
        values += values[:1]
        c = colors[idx]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'{conn_type.capitalize()}', color=c)
        ax.fill(angles, values, alpha=0.15, color=c)
    
    # Optionally plot the one module case for reference
    values_one_module = df_one_module[metrics].mean().values.flatten().tolist()
    values_one_module += values_one_module[:1]
    ax.plot(angles, values_one_module, linewidth=2, linestyle='dashed',
            label='Monolithic ESN', color='black')
    ax.fill(angles, values_one_module, alpha=0.1, color='black')

    
    # Hide y tick labels
    

    ax.set_yticklabels([])  # hide y tick labels
    
    # legend in separate figure
    
    fig.suptitle(title, fontsize=24, y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    fig_legend, ax_legend = plt.subplots(figsize=(2, 2))
    ax_legend.legend(*ax.get_legend_handles_labels(), loc='center', bbox_to_anchor=(0.5, 0.5), frameon=False, fontsize=12)
    ax_legend.axis('off')
    fig_legend.tight_layout(pad=0)
    if save_path:
        fig_legend.savefig(save_path.replace('.pdf', '_legend.pdf'), dpi=150, bbox_inches='tight', pad_inches=0)
    return fig, ax

def lineplot_by_n_modules(df, att_dim_metrics, att_dim_metrics_renamed, n_datasets, mode='all', save_path: Optional[str] = None):
    """Line plot of n_modules vs metrics for each dataset and connection matrix.

    Args:
        save_path: Optional path template with a ``{}`` placeholder for the
            hidden dimension value, e.g. ``'figures/lineplot_hdim_{}.pdf'``.
            If *None*, the figure is not saved.

    Returns:
        list[tuple[matplotlib.figure.Figure, np.ndarray]]: (fig, axes) pairs,
        one per unique hidden dimension.
    """
    figures = []
    for hdim in df['n_hid'].unique():
        fig, axes = plt.subplots(n_datasets, len(att_dim_metrics), figsize=(5*len(att_dim_metrics), 4*n_datasets), squeeze=False)
        df_hdim = df[df['n_hid'] == hdim]
            
        for j, dataset in enumerate(df_hdim['dataset'].unique()):
            df_dataset = df_hdim[df_hdim['dataset'] == dataset]
            if mode == 'max':
                df_dataset = df_dataset.groupby(['n_modules', 'connection_matrix'])[att_dim_metrics].max().reset_index()
            for i, metric in enumerate(att_dim_metrics):
                ax = axes[j, i]
                sns.lineplot(data=df_dataset, x='n_modules', y=metric, style='connection_matrix', markers=True, ax=ax, hue='connection_matrix')
                ax.set_title( f'Metric: {att_dim_metrics_renamed[metric]}')
                ax.set_xlabel('n_modules')
                ax.set_ylabel(att_dim_metrics_renamed[metric])
                ax.legend(title=f' Dataset: {dataset} \n Connection Matrix', bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.suptitle(f'Attractor Dimension Metrics vs Number of Modules (Hidden Dimension = {hdim})', fontsize=16)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.format(hdim), bbox_inches='tight')
        plt.show()
        figures.append((fig, axes))
    return figures


def heatmap_metrics(
    df,
    row_var,
    col_var,
    metrics_renamed,
    n_datasets,
    n_cms,
    att_dim_metrics,
    labelsize=16,
    ticksize=16,
    titlesize=18,
    annot_kws={"size": 14},
    save_path: Optional[str] = None,
):


    # aggregate
    df_grouped = (
        df[df["n_modules"] > 1]
        .groupby([row_var, col_var, "dataset", "connection_matrix"])
        .mean(numeric_only=True)
        .reset_index()
    )

    n_metrics = len(att_dim_metrics)
    assert n_metrics == 4, "This layout assumes exactly 4 metrics."

    # outer figure (2x2 metrics)
    fig, axs = plt.subplots(2, 2, figsize=(10 * n_cms, 10 * n_datasets))
    #reduce spacing between subplots
    fig.subplots_adjust(wspace=0.05, hspace=0.2)
    axs = axs.flatten()

    for idx, metric in enumerate(att_dim_metrics):
        # outer grid logic (2x2)
        show_y_ticks = idx in [0, 2]   # colonna sinistra
        show_x_ticks = idx in [2, 3]   # riga inferiore

        # inner grid: datasets x connection matrices
        inner_fig, inner_axs = plt.subplots(
            n_datasets,
            n_cms,
            figsize=(5 * n_cms, 5 * n_datasets),
            squeeze=False,
        )

        for i, (dataset, df_dataset) in enumerate(df_grouped.groupby("dataset")):
            for j, (cm, df_cm) in enumerate(df_dataset.groupby("connection_matrix")):

                pivot = df_cm.pivot(
                    index=row_var, columns=col_var, values=metric
                )

                ax = inner_axs[i, j]

                sns.heatmap(
                    pivot,
                    ax=ax,
                    cmap="viridis",
                    annot=True,
                    fmt=".2f",
                    cbar=False,
                    annot_kws=annot_kws,
                )

                ax.set_title(
                    f"{dataset.upper()} - {cm.capitalize()}",
                    fontsize=titlesize,
                )

                # ---- axis visibility logic ----

                # y-axis: only first column
                # ----- Y axis -----
                if show_y_ticks and j == 0:
                    ax.set_ylabel(r"$N_h$", fontsize=labelsize)
                    ax.tick_params(axis="y", labelsize=ticksize)
                else:
                    ax.set_ylabel("")
                    ax.set_yticks([])

                # ----- X axis -----
                if show_x_ticks and i == n_datasets - 1:
                    ax.set_xlabel(r"$M$", fontsize=labelsize)
                    ax.tick_params(axis="x", labelsize=ticksize)
                else:
                    ax.set_xlabel("")
                    ax.set_xticks([])

        inner_fig.tight_layout()

        # draw inner fig to image
        inner_fig.canvas.draw()
        w, h = inner_fig.canvas.get_width_height()

        try:
            img = np.frombuffer(
                inner_fig.canvas.tostring_rgb(), dtype=np.uint8
            ).reshape(h, w, 3)
        except AttributeError:
            img = np.asarray(inner_fig.canvas.buffer_rgba())[:, :, :3]

        axs[idx].imshow(img)
        axs[idx].axis("off")
        axs[idx].set_title(metrics_renamed[metric], fontsize=titlesize + 2)

        plt.close(inner_fig)

    # global labels
    # fig.supxlabel(r"$M$", fontsize=labelsize + 4, y=0.04)
    # fig.supylabel(r"$N_h$", fontsize=labelsize + 4, x=0.04)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    return fig, axs