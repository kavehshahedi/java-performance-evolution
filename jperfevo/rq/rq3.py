import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import os

class RQ3:
    def __init__(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Plotting class for RQ3 visualizations.
        
        :param df: DataFrame containing the method performance data
        :type df: pd.DataFrame
        :param output_dir: Directory to save the plots
        :type output_dir: str
        """
        self.df = df.copy()
        self._preprocess_data()
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._add_derived_metrics()

        self._set_style()

    def _preprocess_data(self):
        """Preprocess the data for analysis."""
        # Drop invalid rows (i.e., duplicates)
        self.df['commit_date'] = pd.to_datetime(df['commit_date'], unit='s')
        self.df = self.df.dropna(subset=['effect_size'])
        self.df = self.df.reindex(self.df['effect_size'].abs().sort_values().index).drop_duplicates(['project_id', 'commit_id', 'method_name'], keep='first')

    def save_plot(self, plt, filename: str) -> None:
        """
        Helper method to save plots with consistent settings.
        
        :param plt: Matplotlib plot object
        :type plt: plt
        :param filename: Name of the file to save the plot
        :type filename: str
        """
        filepath = os.path.join(self.output_dir, f"{filename}.pdf")
        plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        # plt.show()
        
    def _set_style(self):
        """Set consistent style for all visualizations."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        # Set font size
        plt.rc('font', size=16)
        plt.rc('axes', titlesize=18)
        plt.rc('axes', labelsize=16)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        plt.rc('legend', fontsize=14)
        plt.rc('figure', titlesize=20)

    def _set_standard_legend_style(self, ax, padding_factor: float = 0.02, title: Optional[str] = None) -> None:
        """
        Helper method to standardize legend appearance across all plots with adaptive positioning.
        
        :param ax: Matplotlib axes object
        :type ax: plt.Axes
        :param padding_factor: Padding factor for legend position adjustment
        :type padding_factor: float
        :param title: Title for the legend
        :type title: Optional[str]
        """
        # Get the figure and axes dimensions
        fig = ax.get_figure()
        fig_width_inches = fig.get_figwidth()
        fig_height_inches = fig.get_figheight()
        
        # Get the axes position in figure coordinates
        bbox = ax.get_position()
        plot_width = bbox.width
        
        # Calculate the right position for legend
        # Move legend outside plot area with some padding
        legend_x = 1 + (padding_factor * plot_width)
        
        legend = ax.legend(
            bbox_to_anchor=(legend_x, 1),
            title=title,
            fontsize=12,
            title_fontsize=12,
            frameon=True,
            borderaxespad=0,
            loc='upper left'
        )
        
        # Remove the default title padding
        legend._legend_box.align = "left"
        
        # Get the legend width in figure coordinates
        legend_width = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).width
        
        # If legend goes beyond figure bounds, adjust figure size to accommodate it
        total_width_needed = bbox.x1 + (legend_width * 1.1)  # 1.1 adds a small right margin
        if total_width_needed > 1:
            # Calculate new figure width
            new_fig_width = fig_width_inches / bbox.x1
            fig.set_figwidth(new_fig_width)
            
            # Update tight_layout with new dimensions
            plt.tight_layout()
            
            # Reposition legend after tight_layout adjustment
            bbox = ax.get_position()
            legend_x = 1 + (padding_factor * bbox.width)
            legend.set_bbox_to_anchor((legend_x, 1))

    def _add_derived_metrics(self):
        """Add derived metrics for analysis."""
        # Convert size to categorical
        self.df['size_category'] = pd.qcut(
            self.df['size'],
            q=3,
            labels=['Small', 'Medium', 'Large']
        )
        
        # Add volatility metric (rolling std of effect sizes)
        self.df['performance_volatility'] = self.df.groupby('project_id')['effect_size']\
            .transform(lambda x: x.rolling(window=5, min_periods=1).std())
        
        # Calculate relative change frequency
        self.df['change_frequency'] = self.df.groupby('project_id')['commit_id']\
            .transform('count') / self.df.groupby('project_id')['size'].transform('first')
    
    def plot_domain_patterns(self):
        """
        Visualization of performance distribution across different domains.
        """
        plt.figure(figsize=(10, 8), dpi=300)
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4) # type: ignore
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        
        # Set custom colors for each change type
        improvement_color = '#27ae60'
        regression_color = '#e74c3c'
        unchanged_color = '#3498db'
        colors = [regression_color, improvement_color, unchanged_color]
        
        # Split data by change type
        improvements = self.df[self.df['change_type'] == 'Improvement']
        regressions = self.df[self.df['change_type'] == 'Regression']
        improvements = improvements[improvements['effect_size'] < improvements['effect_size'].quantile(0.99)]
        regressions = regressions[regressions['effect_size'] > regressions['effect_size'].quantile(0.01)]

        # Sort based on domain name order
        improvements['domain'] = pd.Categorical(improvements['domain'], categories=sorted(improvements['domain'].unique()), ordered=True)
        regressions['domain'] = pd.Categorical(regressions['domain'], categories=sorted(regressions['domain'].unique()), ordered=True)
        
        # Plot improvements
        sns.boxplot(data=improvements, x='domain', y='effect_size', ax=ax1,
                color=colors[1], flierprops={'alpha': 0.5})
        
        # Plot regressions
        sns.boxplot(data=regressions, x='domain', y='effect_size', ax=ax2,
                color=colors[0], flierprops={'alpha': 0.5})
        
        thresholds = [0.474, 0.33, 0.147]
        labels = ['Large effect', 'Medium effect', 'Small effect']
        for threshold, color, label in zip(thresholds, colors, labels):
            ax1.axhline(y=threshold, color=color, linestyle='--', label=label, alpha=0.5, zorder=0)
            ax2.axhline(y=-threshold, color=color, linestyle='--', label=label, alpha=0.5, zorder=0)

        # Customize improvement plot
        ax1.set_title('Performance Improvements')
        ax1.set_xlabel('')
        ax1.set_ylabel('Effect Size')
        # ax1.legend(title='Effect Size Thresholds', loc='upper right')
        self._set_standard_legend_style(ax1, title='Effect Size Thresholds')
        
        # Customize regression plot
        ax2.set_title('Performance Regressions')
        ax2.set_xlabel('Domain')
        ax2.set_ylabel('Effect Size')
        # ax2.legend(title='Effect Size Thresholds', loc='lower right')
        self._set_standard_legend_style(ax2, title='Effect Size Thresholds')
        
        plt.tight_layout()

        # Save the plot
        self.save_plot(plt, 'domain_performance_distribution')
    
    def plot_size_patterns(self):
        """Visualize size-related patterns."""
        plt.figure(figsize=(10, 8), dpi=300)
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4) # type: ignore
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        # Set custom colors for each change type
        improvement_color = '#27ae60'
        regression_color = '#e74c3c'
        unchanged_color = '#3498db'
        colors = [regression_color, improvement_color, unchanged_color]
        
        # Define consistent domain colors using a color palette
        unique_domains = sorted(self.df['domain'].unique())  # Sort to ensure consistency
        domain_colors = dict(zip(unique_domains, sns.husl_palette(n_colors=len(unique_domains))))
        
        # Split data by change type
        improvements = self.df[self.df['change_type'] == 'Improvement']
        regressions = self.df[self.df['change_type'] == 'Regression']  
        
        # Remove outliers
        improvements = improvements[improvements['effect_size'] < improvements['effect_size'].quantile(0.99)]
        regressions = regressions[regressions['effect_size'] > regressions['effect_size'].quantile(0.01)]  
        
        # Improvements
        sns.boxplot(data=improvements, x='size_category', y='effect_size', 
                    hue='domain', ax=ax1, flierprops={'alpha': 0.5},
                    hue_order=unique_domains, palette=domain_colors)
        
        # Regressions
        sns.boxplot(data=regressions, x='size_category', y='effect_size', 
                    hue='domain', ax=ax2, flierprops={'alpha': 0.5},
                    hue_order=unique_domains, palette=domain_colors)
        
        # Add threshold lines
        thresholds = [0.474, 0.33, 0.147]
        labels = ['Large effect', 'Medium effect', 'Small effect']
        for threshold, color, label in zip(thresholds, colors, labels):
            ax1.axhline(y=threshold, color=color, linestyle='--', label=label, alpha=0.5, zorder=0)
            ax2.axhline(y=-threshold, color=color, linestyle='--', label=label, alpha=0.5, zorder=0)
        
        # Customize improvement plot
        ax1.set_title('Performance Improvements')
        ax1.set_xlabel('')
        ax1.set_ylabel('Effect Size')
        self._set_standard_legend_style(ax1, title='Domains')
        
        # Customize regression plot
        ax2.set_title('Performance Regressions')
        ax2.set_xlabel('Project Size Category')
        ax2.set_ylabel('Effect Size')
        self._set_standard_legend_style(ax2, title='Domains')

        # Separate plots into three vertical groups
        # Add axvline to separate the plots
        for ax in [ax1, ax2]:
            ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.25)
            ax.axvline(x=1.5, color='black', linestyle='--', alpha=0.25)
        
        # Adjust layout to accommodate the legend
        plt.subplots_adjust(right=0.85)

        # Save the plot
        self.save_plot(plt, 'size_vs_effect_boxplot')

if __name__ == "__main__":
    dataset = os.path.join('dataset', 'dataset.csv')
    output_dir = os.path.join('plots', 'RQ3')
    df = pd.read_csv(dataset)
    visualizer = RQ3(df, output_dir)

    visualizer.plot_domain_patterns()
    visualizer.plot_size_patterns()