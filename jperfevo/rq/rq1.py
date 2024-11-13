from git import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class RQ1:
    def __init__(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Plotting class for RQ1 visualizations.

        :param df: DataFrame containing the method performance data
        :type df: pd.DataFrame
        :param output_dir: Directory to save the plots
        :type output_dir: str
        """
        self.df = df

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._preprocess_data()
                
        self._set_style()

        self.change_type_order = ['Unchanged', 'Regression', 'Improvement']

    def _set_style(self) -> None:
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

    def save_plot(self, plt, filename: str) -> None:
        """
        Helper method to save plots with consistent settings.
        
        :param plt: Matplotlib plot object
        :type plt: plt
        :param filename: Filename to save the plot
        :type filename: str
        """
        filepath = os.path.join(self.output_dir, f"{filename}.pdf")
        plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        # plt.show()

    def _preprocess_data(self) -> None:
        """Preprocess the data for visualization."""

        df['commit_date'] = pd.to_datetime(df['commit_date'], unit='s')
        self.df = self.df.dropna(subset=['effect_size'])
        self.df = self.df.reindex(self.df['effect_size'].abs().sort_values().index).drop_duplicates(['project_id', 'commit_id', 'method_name'], keep='first')
        self.df = self.df[self.df['commit_date'] >= '2016-01-01']

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

    def plot_effect_size_evolution(self):
        """
        Plot the evolution of effect sizes over time with trend analysis and stability metrics.
        Shows rolling statistics and separate analyses for improvements and regressions.
        """
        plt.figure(figsize=(15, 10), dpi=300)
        
        # Create two subplots - one for improvements, one for regressions
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3) # type: ignore
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        
        # Calculate rolling statistics (4-month windows)
        window = '120D'
        
        # Process improvements
        improvements = self.df[self.df['change_type'] == 'Improvement'].copy()
        improvements = improvements.groupby('commit_date')['effect_size'].mean().reset_index()
        improvements.sort_values('commit_date', inplace=True)
        improvements.set_index('commit_date', inplace=True)
        improvements = improvements.ffill()
        improvements = improvements.bfill()
        improvements = improvements[improvements['effect_size'] > improvements['effect_size'].quantile(0.01)]
        
        # Calculate rolling statistics for improvements
        imp_rolling_mean = improvements['effect_size'].rolling(window, min_periods=1).mean()
        
        # Process regressions
        regressions = self.df[self.df['change_type'] == 'Regression'].copy()
        regressions = regressions.groupby('commit_date')['effect_size'].mean().reset_index()
        regressions.sort_values('commit_date', inplace=True)
        regressions.set_index('commit_date', inplace=True)
        regressions = regressions.ffill()
        regressions = regressions.bfill()
        regressions = regressions[regressions['effect_size'] < regressions['effect_size'].quantile(0.99)]
        regressions['effect_size'] = -regressions['effect_size']
        
        # Calculate rolling statistics for regressions
        reg_rolling_mean = regressions['effect_size'].rolling(window, min_periods=1).mean()

        # Smooth out the rolling statistics
        imp_rolling_mean = imp_rolling_mean.ewm(span=5).mean()
        reg_rolling_mean = reg_rolling_mean.ewm(span=5).mean()
        
        # Plot improvements
        ax1.scatter(improvements.index, improvements['effect_size'], 
                alpha=0.5, color='#2ecc71', s=30)
        ax1.plot(imp_rolling_mean.index, imp_rolling_mean, 
                color='#27ae60', linewidth=3.5, label='4-month rolling mean')
        
        # Add threshold lines for improvements
        ax1.axhline(y=0.474, color='#e74c3c', linestyle='--', 
                    label='Large effect threshold')
        ax1.axhline(y=0.33, color='#f39c12', linestyle='--', 
                    label='Medium effect threshold')
        ax1.axhline(y=0.147, color='#3498db', linestyle='--', 
                    label='Small effect threshold')
        
        # Plot regressions
        ax2.scatter(regressions.index, -regressions['effect_size'], 
                alpha=0.5, color='#e74c3c', s=30)
        ax2.plot(reg_rolling_mean.index, -reg_rolling_mean, 
                color='#c0392b', linewidth=3.5, label='4-month rolling mean')
        
        # Add threshold lines for regressions
        ax2.axhline(y=-0.474, color='#e74c3c', linestyle='--', 
                    label='Large effect threshold')
        ax2.axhline(y=-0.33, color='#f39c12', linestyle='--', 
                    label='Medium effect threshold')
        ax2.axhline(y=-0.147, color='#3498db', linestyle='--', 
                    label='Small effect threshold')
        
        # Add axis labels
        ax1.set_ylabel('Effect Size')
        ax2.set_ylabel('Effect Size')
        ax2.set_xlabel('Commit Date')

        ax1.set_title('Performance Improvements')
        ax2.set_title('Performance Regressions')
        
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        self._set_standard_legend_style(ax1, title='Effect Size Thresholds')
        self._set_standard_legend_style(ax2, title='Effect Size Thresholds')
        
        plt.tight_layout()

        # Save the plot
        self.save_plot(plt, 'effect_size_evolution_analysis')
        
    def plot_effect_size_distribution(self):
        """Create a distribution plot showing effect sizes with box plots, points, and density curves."""
        fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
        
        # Define colors
        improvement_color = '#27ae60'
        improvement_color_dark = '#1e8a4b'
        regression_color = '#e74c3c'
        regression_color_dark = '#ae382b'
        unchanged_color = '#3498db'
        unchanged_color_dark = '#2a80b9'
        colors = [improvement_color, regression_color, unchanged_color]
        
        # Create box plots
        box_props = dict(linewidth=0.5)
        whisker_props = dict(linewidth=2)
        flier_props = dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5)
        
        # For improvements
        improvements = self.df[self.df['change_type'] == 'Improvement']['effect_size']
        improvements = improvements[improvements < improvements.quantile(0.99)]
        ax.boxplot(improvements, positions=[0], widths=0.5,
                        patch_artist=True,
                        boxprops=dict(facecolor=improvement_color, **box_props),
                        whiskerprops=whisker_props,
                        medianprops=dict(linewidth=3, color=improvement_color_dark),
                        flierprops=flier_props)
        
        # For regressions (negate values for visual clarity)
        regressions = -self.df[self.df['change_type'] == 'Regression']['effect_size']
        regressions = regressions[regressions < regressions.quantile(0.99)]
        ax.boxplot(regressions, positions=[1], widths=0.5,
                        patch_artist=True,
                        boxprops=dict(facecolor=regression_color, **box_props),
                        whiskerprops=whisker_props,
                        medianprops=dict(linewidth=3, color=regression_color_dark),
                        flierprops=flier_props)
        
        # For unchanged methods
        unchanged = self.df[self.df['change_type'] == 'Unchanged']['effect_size']
        ax.boxplot(unchanged, positions=[2], widths=0.5,
                        patch_artist=True,
                        boxprops=dict(facecolor=unchanged_color, **box_props),
                        whiskerprops=whisker_props,
                        medianprops=dict(linewidth=3, color=unchanged_color_dark),
                        flierprops=flier_props)
        
        # Add individual points with jitter
        for idx, data in enumerate([improvements, regressions, unchanged]):
            x = np.random.normal(idx, 0.01, size=len(data))
            ax.scatter(x, data, alpha=0.25, s=20, 
                    color=[improvement_color, regression_color, unchanged_color][idx])
        
        # Add threshold lines
        thresholds = [0.474, 0.33, 0.147]
        labels = ['Large effect', 'Medium effect', 'Small effect']
        
        for threshold, color, label in zip(thresholds, colors, labels):
            ax.axhline(y=threshold, color=color, linestyle='--', label=label, alpha=0.5, zorder=0)
        
        # Customize the plot
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Improvements', 'Regressions', 'Unchanged'])
        ax.set_ylabel('Absolute Effect Size')
        ax.set_xlabel('Performance Change Type')
        ax.grid(True, alpha=0.3)
        
        self._set_standard_legend_style(ax, title='Effect Size Thresholds')
        
        plt.tight_layout()
        
        # Save the plot
        self.save_plot(plt, 'effect_size_distribution')
        
    def plot_effect_size_categories(self):
        """
        Plot the distribution of effect size categories with a split visualization:
        """
        plt.figure(figsize=(15, 6), dpi=300)
        
        # Create grid for two subplots with more space between them
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4) # type: ignore
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        
        # Filter out negligible changes and recalculate distributions
        df_significant = self.df[self.df['effect_size_interpretation'].isin(['Small', 'Medium', 'Large'])]
        
        # Calculate both absolute counts and percentages
        abs_counts = pd.crosstab(
            df_significant['change_type'],
            df_significant['effect_size_interpretation']
        )
        
        pct_dist = pd.crosstab(
            df_significant['change_type'],
            df_significant['effect_size_interpretation'],
            normalize='index'
        ) * 100
        
        # Color scheme for effect sizes
        colors = {
            'Small': '#2ecc71',    # green
            'Medium': '#f1c40f',   # yellow
            'Large': '#e74c3c'     # red
        }
        
        # Sort effect size categories in order of magnitude
        effect_size_order = ['Small', 'Medium', 'Large']
        
        # Process improvements
        imp_data = pct_dist.loc['Improvement'][effect_size_order]
        imp_counts = abs_counts.loc['Improvement'][effect_size_order]
        
        # Process regressions
        reg_data = pct_dist.loc['Regression'][effect_size_order]
        reg_counts = abs_counts.loc['Regression'][effect_size_order]
        
        # Plot improvements (top)
        bars = ax1.bar(range(len(effect_size_order)), imp_data, 
                    color=[colors[cat] for cat in effect_size_order])
        
        # Add percentage and count labels on bars (moved up slightly)
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'\n{height:.1f}%',
                    ha='center', va='bottom')
        
        # Plot regressions (bottom)
        bars = ax2.bar(range(len(effect_size_order)), reg_data,
                    color=[colors[cat] for cat in effect_size_order])
        
        # Add percentage and count labels on bars (moved up slightly)
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'\n{height:.1f}%',
                    ha='center', va='bottom')
        
        # Customize improvement plot
        ax1.set_title('Performance Improvements')
        ax1.set_ylabel('Percentage of Changes')
        ax1.set_xticks(range(len(effect_size_order)))
        ax1.set_xticklabels(['Small', 'Medium', 'Large'])
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, ax1.get_ylim()[1] + 10)
        
        # Customize regression plot
        ax2.set_title('Performance Regressions')
        ax2.set_ylabel('Percentage of Changes')
        ax2.set_xlabel('Effect Size Category')
        ax2.set_xticks(range(len(effect_size_order)))
        ax2.set_xticklabels(['Small', 'Medium', 'Large'])
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, ax2.get_ylim()[1] + 10)
        
        plt.tight_layout()
        self.save_plot(plt, 'effect_size_categories')

    def analyze_project_lifecycles(self):
        """
        Analyze performance changes across different project lifecycle stages.
        """
        # Calculate project age for each commit
        self.df['project_age'] = (self.df['commit_date'] - self.df['commit_date'].min()) / np.timedelta64(365, 'D')
        total_lifespan = self.df['project_age'].max()
        
        # Define stage boundaries
        stage_duration = total_lifespan / 3
        
        def get_stage(age):
            if age <= stage_duration:
                return 'Early Stage'
            elif age <= 2 * stage_duration:
                return 'Middle Stage'
            else:
                return 'Late Stage'
        
        # Add stage column
        self.df['lifecycle_stage'] = self.df['project_age'].apply(get_stage)
        
        # Calculate distributions for each stage and change type
        stage_distribution = pd.crosstab(
            self.df['lifecycle_stage'],
            self.df['change_type'],
            normalize='index'
        ) * 100  # Convert to percentages
        
        # Reorder columns to match the standard order
        stage_distribution = stage_distribution[self.change_type_order]
        
        # Round to 2 decimal places
        stage_distribution = stage_distribution.round(2)
        
        # Sort by stage order
        stage_order = ['Early Stage', 'Middle Stage', 'Late Stage']
        stage_distribution = stage_distribution.reindex(stage_order)
        
        # Print information about the analysis
        print(f"\nProject Lifecycle Analysis")
        print(f"Total project lifespan: {total_lifespan:.2f} years")
        print(f"Stage duration: {stage_duration:.2f} years each")
        
        # Print the distribution table
        print("\nDistribution of Performance Changes by Lifecycle Stage")
        print("-" * 80)
        print(f"{'Lifecycle Stage':<25} {'Unchanged %':>15} {'Regression %':>15} {'Improvement %':>15}")
        print("-" * 80)
        
        for stage in stage_order:
            if stage in stage_distribution.index:
                row = stage_distribution.loc[stage]
                print(f"{stage:<25} {row['Unchanged']:>15.2f} {row['Regression']:>15.2f} {row['Improvement']:>15.2f}")
        
        print("-" * 80)
        
if __name__ == "__main__":
    dataset = os.path.join('dataset', 'dataset.csv')
    output_dir = os.path.join('plots', 'RQ1')
    df = pd.read_csv(dataset)
    visualizer = RQ1(df, output_dir)
    
    visualizer.plot_effect_size_evolution()
    visualizer.plot_effect_size_distribution()
    visualizer.plot_effect_size_categories()
    visualizer.analyze_project_lifecycles()