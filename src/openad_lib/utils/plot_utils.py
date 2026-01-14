"""
Plotting utilities for openad_lib examples.

Provides consistent styling across all example scripts.
"""

import os
import matplotlib.pyplot as plt

def setup_plot_style():
    """Apply consistent plot styling for all examples."""
    plt.style.use('bmh')

def save_plot(fig_or_path, filename, project_root=None):
    """
    Save plot to images directory with consistent settings.
    
    Args:
        fig_or_path: Either a matplotlib figure or path to project root
        filename: Name of the file to save
        project_root: Path to project root (if fig_or_path is a figure)
    """
    if project_root is None:
        # fig_or_path is actually the project_root
        project_root = fig_or_path
    
    images_dir = os.path.join(project_root, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    save_path = os.path.join(images_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    return save_path

def format_plot(ax=None, title=None, xlabel=None, ylabel=None, legend=True):
    """
    Apply consistent formatting to a plot.
    
    Args:
        ax: Matplotlib axis object (if None, uses current axis)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        legend: Whether to show legend
    """
    if ax is None:
        ax = plt.gca()
    
    if title:
        ax.set_title(title, fontsize=16, pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    
    if legend:
        ax.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
