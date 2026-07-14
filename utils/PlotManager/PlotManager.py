import os
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.LogManager.LogManager import LogManager
from utils.common_variables import *

file_name = os.path.splitext(os.path.basename(__file__))[0]


class PlotManager:
    def __init__(self) -> None:
        """
        Create a plot manager using the framework logger.

        :return: None.
        """
        self.lm = LogManager('main')

    def _save_current_figure(self, path: str, filename: str, dpi_value: int = 800) -> None:
        """
        Save the active matplotlib figure.

        :param path: [str] Output directory.
        :param filename: [str] Output filename.
        :param dpi_value: [int] Figure resolution.
        :return: None.
        """
        plt.savefig(f"{path}{filename}", dpi=dpi_value)

    def plot_line(
        self,
        path: str,
        type_ca: str,
        x_values: Any,
        y_values: Any,
        x_label: str,
        y_label: str,
        title: str,
        filename: str,
        marker: str = 'o',
        markersize: int = 3,
    ) -> None:
        """
        Plot and save one line chart.

        :param path: [str] Output directory.
        :param type_ca: [str] Co-action or layer name used for color and label.
        :param x_values: [Any] X values.
        :param y_values: [Any] Y values.
        :param x_label: [str] X-axis label.
        :param y_label: [str] Y-axis label.
        :param title: [str] Plot title.
        :param filename: [str] Output filename.
        :param marker: [str] Marker style.
        :param markersize: [int] Marker size.
        :return: None.
        """
        plt.figure()
        plt.plot(x_values, y_values, color=color_dict[type_ca], linestyle='--', label=type_ca,
                 marker=marker, markersize=markersize)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.title(title)
        plt.grid(True)
        plt.show()
        self._save_current_figure(path, filename)
        self.lm.printl(f"{file_name}. plot_line finished. {filename} saved.")

    def plot_grid_line(
        self,
        path_analysis: str,
        filename: str,
        df: pd.DataFrame,
        subset_column: str,
        x_column: str,
        y_column: str,
        x_label: str,
        y_label: str,
        title: str,
    ) -> None:
        """
        Plot a grid of line charts, one subplot per subset value.

        :param path_analysis: [str] Output directory.
        :param filename: [str] Output filename.
        :param df: [pd.DataFrame] Input dataframe.
        :param subset_column: [str] Column defining subplots.
        :param x_column: [str] X-value column.
        :param y_column: [str] Y-value column.
        :param x_label: [str] X-axis label.
        :param y_label: [str] Y-axis label.
        :param title: [str] Plot title prefix.
        :return: None.
        """
        layer_list = list(df[subset_column].unique())
        n_layers = len(layer_list)
        if n_layers == 0:
            self.lm.printl(f"{file_name}: plot_grid_line skipped because no subsets are available.")
            return

        ncols = math.ceil(math.sqrt(n_layers))  # Number of columns
        nrows = math.ceil(n_layers / ncols)  # Number of rows

        # Create subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
        # Matplotlib returns a scalar Axes for a 1x1 grid; normalize it
        # so the same iteration code works for one or many co-action layers.
        axes = np.atleast_1d(axes).flatten()

        # subset: for instance the type of co-action/layer
        for i, subset in enumerate(layer_list):
            ax = axes[i]
            subset_df = df[df[subset_column] == subset]
            ax.plot(subset_df[x_column], subset_df[y_column], marker='o', label=f'{subset}', markersize=2)
            ax.set_title(f'{title}: {subset}')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.grid(True)
            ax.legend()

        # Remove any empty subplots
        for j in range(n_layers, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        self._save_current_figure(path_analysis, filename)
        plt.show()
        self.lm.printl(f"{file_name}. plot_grid_line finished. {filename} saved.")

    def plot_histogram(
        self,
        path_analysis: str,
        type_ca: str,
        values: Any,
        x_label: str,
        y_label: str,
        title: str,
        filename: str,
    ) -> None:
        """
        Plot and save one histogram.

        :param path_analysis: [str] Output directory.
        :param type_ca: [str] Co-action or layer name used for color and label.
        :param values: [Any] Values to plot.
        :param x_label: [str] X-axis label.
        :param y_label: [str] Y-axis label.
        :param title: [str] Plot title.
        :param filename: [str] Output filename.
        :return: None.
        """
        color = color_dict.get(type_ca, 'blue')
        # Plotting the distribution
        plt.figure()
        plt.hist(values, bins=30, edgecolor='black', color=color, label=type_ca, alpha=0.2)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.legend()
        # Save the figure with high resolution (300 dpi)
        self._save_current_figure(path_analysis, filename)
        plt.show()
        self.lm.printl(f"{file_name}. plot_histogram finished. {filename} saved.")

    def plot_grid_combinations(
        self,
        df: pd.DataFrame,
        path: str,
        filename: str,
        column1: str,
        column2: str,
        x_column: str,
        y_column: str,
        x_label: str,
        y_label: str,
        step: float,
    ) -> None:
        """
        Plot pairwise grid combinations between two categorical columns.

        :param df: [pd.DataFrame] Input dataframe.
        :param path: [str] Output directory.
        :param filename: [str] Output filename.
        :param column1: [str] Row category column.
        :param column2: [str] Column category column.
        :param x_column: [str] X-value column.
        :param y_column: [str] Y-value column.
        :param x_label: [str] X-axis label.
        :param y_label: [str] Y-axis label.
        :param step: [float] X tick step when threshold values are present.
        :return: None.
        """
        if df.empty:
            self.lm.printl(f"{file_name}: plot_grid_combinations skipped because dataframe is empty.")
            return

        # Get unique actions
        unique_actions1 = df[column1].unique()
        unique_actions2 = df[column2].unique()
        if len(unique_actions1) == 0 or len(unique_actions2) == 0:
            self.lm.printl(f"{file_name}: plot_grid_combinations skipped because no category pairs are available.")
            return

        # Create subplots
        fig, axes = plt.subplots(len(unique_actions1), len(unique_actions2), figsize=(15, 15))
        axes = np.atleast_2d(axes)

        # Plot each pair
        for i, coAction1 in enumerate(unique_actions1):
            for j, coAction2 in enumerate(unique_actions2):
                if i <= j:  # Upper diagonal condition
                    ax = axes[i, j]
                    subset = df[(df[column1] == coAction1) & (df[column2] == coAction2)]
                    if not subset.empty:
                        ax.plot(subset[x_column], subset[y_column], marker='o', markersize=2,
                                label=f'{coAction1} \n {coAction2}')
                        #                 ax.set_title(f'{coAction1} vs {coAction2}')
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        if 'threshold' in subset.columns:
                            ax.set_xticks(np.arange(min(subset['threshold']), max(subset['threshold'])+step, step))
                        # ax.set_xlim(0, 0.105)
                        ax.tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels
                        ax.legend()
                        ax.grid(True)

                    else:
                        ax.set_visible(False)  # Hide the subplot if no data
                else:
                    axes[i, j].axis('off')  # Hide the lower triangle plots

        # Add column labels
        for ax, col in zip(axes[0], unique_actions2):
            ax.annotate(f'{col}', xy=(0.5, 1), xytext=(0, 10),
                        xycoords='axes fraction', textcoords='offset points',
                        size='x-large', ha='center', va='baseline')

        # Add row labels
        for ax, row in zip(axes[:, 0], unique_actions1):
            ax.annotate(f'{row}', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 50, 0),
                        xycoords='axes fraction', textcoords='offset points',
                        size='x-large', ha='right', va='center', rotation=90)

        # Adjust layout to fit annotations
        plt.tight_layout()
        plt.show()
        self._save_current_figure(path, filename, dpi)

        self.lm.printl(f"{file_name}: plot_grid_combinations finished.")
