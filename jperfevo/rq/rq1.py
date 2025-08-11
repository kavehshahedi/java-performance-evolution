from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from scipy.stats import chi2_contingency
import warnings

warnings.filterwarnings("ignore")


class RQ1:
    def __init__(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        RQ1 class for analyzing performance change patterns across project lifecycle stages.
        """
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._preprocess_data()
        self._set_style()
        self.change_type_order = ['Unchanged', 'Regression', 'Improvement']

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
            fontsize=24,
            title_fontsize=24,
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

    def _preprocess_data(self) -> None:
        """Preprocess data with formal definitions of improvements/regressions."""
        # Convert commit date and filter
        self.df['commit_date'] = pd.to_datetime(self.df['commit_date'], unit='s')
        self.df = self.df.dropna(subset=['effect_size'])
        self.df = self.df[self.df['commit_date'] >= '2016-01-01']

        # Formally define significant performance changes
        # Based on Mann-Whitney U test (p < 0.05) and non-negligible effect size (|effect_size| >= 0.147)
        self.df['is_significant_change'] = (
            (self.df['change_type'].isin(['Improvement', 'Regression'])) &
            (abs(self.df['effect_size']) >= 0.147)
        )

        # Normalize time by project lifespan (0 to 1)
        project_time_ranges = self.df.groupby('project_id')['commit_date'].agg(['min', 'max'])

        def normalize_time(row):
            project_id = row['project_id']
            commit_date = row['commit_date']
            min_date = project_time_ranges.loc[project_id, 'min']
            max_date = project_time_ranges.loc[project_id, 'max']

            if max_date == min_date:
                return 0.0
            return (commit_date - min_date) / (max_date - min_date)  # type: ignore

        self.df['normalized_time'] = self.df.apply(normalize_time, axis=1)

        # Add time bins for analysis
        self.df['time_bin'] = pd.cut(self.df['normalized_time'],
                                     bins=10,
                                     labels=False,
                                     include_lowest=True)

    def save_plot(self, plt, filename: str) -> None:
        """
        Helper method to save plots with consistent settings.

        :param plt: Matplotlib plot object
        :type plt: plt
        :param filename: Filename to save the plot
        :type filename: str
        """
        pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
        png_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(png_path, format='png', dpi=72, bbox_inches='tight')
        plt.close()

    def plot_performance_change_proportions_over_time(self):
        """
        Plot proportions of commits with performance improvements/regressions over normalized time.
        """
        fig, ax = plt.subplots(figsize=(26, 8), dpi=300)

        # Calculate proportions for each time bin
        time_analysis = []
        for time_bin in range(10):
            bin_data = self.df[self.df['time_bin'] == time_bin]
            if len(bin_data) > 0:
                total_commits = len(bin_data)
                improvements = len(bin_data[bin_data['change_type'] == 'Improvement'])
                regressions = len(bin_data[bin_data['change_type'] == 'Regression'])
                unchanged = len(bin_data[bin_data['change_type'] == 'Unchanged'])

                time_analysis.append({
                    'time_bin': time_bin,
                    'normalized_time': (time_bin + 0.5) / 10,  # Mid-point of bin
                    'improvement_prop': improvements / total_commits * 100,
                    'regression_prop': regressions / total_commits * 100,
                    'unchanged_prop': unchanged / total_commits * 100,
                    'instability_rate': (improvements + regressions) / total_commits * 100,
                    'total_commits': total_commits
                })

        time_df = pd.DataFrame(time_analysis)

        # Plot proportions with emphasis on instability
        ax.plot(time_df['normalized_time'], time_df['improvement_prop'],
                'o-', color='#27ae60', linewidth=6, markersize=8,
                label='Performance Improvements', alpha=0.8)
        ax.plot(time_df['normalized_time'], time_df['regression_prop'],
                'o-', color='#e74c3c', linewidth=6, markersize=8,
                label='Performance Regressions', alpha=0.8)
        ax.plot(time_df['normalized_time'], time_df['instability_rate'],
                's--', color='#9b59b6', linewidth=6, markersize=8,
                label='Performance Instability Rate', alpha=0.9)

        ax.set_xlabel('Normalized Project Lifetime (0=Start, 1=End)', fontsize=26, labelpad=20)
        ax.set_ylabel('Percentage of Commits (%)', fontsize=26, labelpad=20)
        ax.set_title('Performance Change Patterns Across Project Lifecycle', fontsize=30, pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

        # Set x-ticks and y-ticks font size
        x_ticks = np.arange(0, 1.1, 0.2)
        y_ticks = np.arange(0, 101, 10)
        ax.set_xticks(x_ticks[1:])
        ax.set_xticklabels([f'{t:0.2}' for t in x_ticks][1:], fontsize=24)
        ax.set_yticks(y_ticks[1:])
        ax.set_yticklabels([f'{int(t)}%' for t in y_ticks][1:], fontsize=24)

        ax.set_ylim(0, max(time_df['regression_prop'].max(), time_df['improvement_prop'].max(), time_df['instability_rate'].max()) + 5)

        # Add lifecycle stage boundaries
        ax.axvline(x=0.33, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0.67, color='gray', linestyle='--', alpha=0.3)
        ax.text(0.165, ax.get_ylim()[1] * 0.95, 'Early Stage', ha='center', fontweight='bold', fontsize=20)
        ax.text(0.5, ax.get_ylim()[1] * 0.95, 'Middle Stage', ha='center', fontweight='bold', fontsize=20)
        ax.text(0.835, ax.get_ylim()[1] * 0.95, 'Late Stage', ha='center', fontweight='bold', fontsize=20)

        plt.tight_layout()
        self._set_standard_legend_style(plt.gca())
        self.save_plot(plt, 'performance_proportions_normalized_timeline')

        return time_df

    def plot_lifecycle_stage_comparison(self):
        """
        Create a focused comparison of performance changes across lifecycle stages.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 6), dpi=300)

        # Get lifecycle stage data
        lifecycle_data = self.analyze_lifecycle_stages_detailed()

        stages = lifecycle_data['stage'].tolist()
        improvement_pcts = lifecycle_data['improvement_pct'].tolist()
        regression_pcts = lifecycle_data['regression_pct'].tolist()
        instability_rates = lifecycle_data['significant_change_rate'].tolist()

        x = np.arange(len(stages))
        width = 0.35

        # Bar chart comparing improvements vs regressions
        bars1 = ax1.bar(x - width/2, improvement_pcts, width, label='Improvements',
                        color='#27ae60', alpha=0.8)
        bars2 = ax1.bar(x + width/2, regression_pcts, width, label='Regressions',
                        color='#e74c3c', alpha=0.8)

        ax1.set_xlabel('Project Lifecycle Stage', fontsize=26, labelpad=20)
        ax1.set_ylabel('Percentage of Changes (%)', fontsize=26, labelpad=20)
        ax1.set_title('Performance Change Distribution by Lifecycle Stage', fontsize=30, pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(stages, fontsize=24)
        ax1.set_yticks(np.arange(0, 101, 10))
        ax1.set_yticklabels([f'{int(t)}%' for t in np.arange(0, 101, 10)], fontsize=24)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, max(max(improvement_pcts), max(regression_pcts)) + 3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=20)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=20)

        # Instability rate trend
        ax2.plot(stages, instability_rates, 'o-', color='#9b59b6', linewidth=6,
                 markersize=10, label='Performance Instability Rate')
        ax2.fill_between(stages, instability_rates, alpha=0.3, color='#9b59b6')

        ax2.set_xlabel('Project Lifecycle Stage', fontsize=26, labelpad=20)
        ax2.set_ylabel('Performance Instability Rate (%)', fontsize=26, labelpad=20)
        ax2.set_title('Performance Instability Across Project Lifecycle', fontsize=30, pad=20)
        ax1.set_xticks(x)
        ax2.set_xticklabels(stages, fontsize=24)
        ax2.set_yticks(np.arange(0, 101, 2.5)[1:])
        ax2.set_yticklabels([f'{int(t)}%' for t in np.arange(0, 101, 2.5)[1:]], fontsize=24)
        ax2.set_ylim(min(instability_rates) * 0.9, max(instability_rates) * 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add hlines for early/middle/late stage boundaries
        for ir in instability_rates:
            ax2.axhline(y=ir, color='#9b59b6', linestyle='--', alpha=0.3)

        plt.tight_layout()
        self._set_standard_legend_style(ax1, padding_factor=0.02, title='Change Type')
        self._set_standard_legend_style(ax2, padding_factor=0.02, title='Instability Rate')
        self.save_plot(plt, 'lifecycle_stage_comparison')

        return lifecycle_data

    def generate_lifecycle_stage_statistics(self):
        """
        Generate detailed statistics for lifecycle stage comparison analysis.
        """
        print("="*80)
        print("LIFECYCLE STAGE COMPARISON - STATISTICAL SUMMARY")
        print("="*80)

        # Calculate stage-wise statistics
        stage_stats = {}

        for stage in ['Early Stage', 'Middle Stage', 'Late Stage']:
            stage_data = self.df[self.df['lifecycle_stage'] == stage]
            total = len(stage_data)

            improvements = len(stage_data[stage_data['change_type'] == 'Improvement'])
            regressions = len(stage_data[stage_data['change_type'] == 'Regression'])
            neutral = len(stage_data[stage_data['change_type'] == 'Unchanged'])

            stage_stats[stage] = {
                'total_changes': total,
                'improvements': improvements,
                'regressions': regressions,
                'neutral': neutral,
                'improvement_rate': improvements / total * 100,
                'regression_rate': regressions / total * 100,
                'instability_rate': (improvements + regressions) / total * 100,
                'neutral_rate': neutral / total * 100
            }

        # Print detailed statistics
        print(f"\nSTAGE-WISE BREAKDOWN:")
        for stage, stats in stage_stats.items():
            print(f"\n{stage.upper()} STAGE:")
            print(f"  Total changes: {stats['total_changes']:,}")
            print(f"  Improvements: {stats['improvements']:,} ({stats['improvement_rate']:.1f}%)")
            print(f"  Regressions: {stats['regressions']:,} ({stats['regression_rate']:.1f}%)")
            print(f"  Neutral: {stats['neutral']:,} ({stats['neutral_rate']:.1f}%)")
            print(f"  Instability Rate: {stats['instability_rate']:.1f}%")

        # Comparative analysis
        print(f"\nCOMPARATIVE ANALYSIS:")

        # Instability pattern
        early_inst = stage_stats['Early Stage']['instability_rate']
        middle_inst = stage_stats['Middle Stage']['instability_rate']
        late_inst = stage_stats['Late Stage']['instability_rate']

        print(f"Instability progression: Early={early_inst:.1f}% → Middle={middle_inst:.1f}% → Late={late_inst:.1f}%")
        print(f"U-shaped pattern: {'Yes' if middle_inst < early_inst and middle_inst < late_inst else 'No'}")
        print(f"Early vs Middle difference: {early_inst - middle_inst:.1f} percentage points")
        print(f"Late vs Middle difference: {late_inst - middle_inst:.1f} percentage points")

        # Improvement vs regression ratios
        early_ratio = stage_stats['Early Stage']['improvement_rate'] / stage_stats['Early Stage']['regression_rate']
        middle_ratio = stage_stats['Middle Stage']['improvement_rate'] / stage_stats['Middle Stage']['regression_rate']
        late_ratio = stage_stats['Late Stage']['improvement_rate'] / stage_stats['Late Stage']['regression_rate']

        print(f"\nIMPROVEMENT-TO-REGRESSION RATIOS:")
        print(f"Early: {early_ratio:.2f} | Middle: {middle_ratio:.2f} | Late: {late_ratio:.2f}")
        print(f"Most improvement-favorable stage: {max(stage_stats.keys(), key=lambda x: stage_stats[x]['improvement_rate'] / stage_stats[x]['regression_rate'])}")

        # Test for significant differences across stages
        contingency_table = [
            [stage_stats[stage]['improvements'], stage_stats[stage]['regressions'], stage_stats[stage]['neutral']]
            for stage in ['Early Stage', 'Middle Stage', 'Late Stage']
        ]

        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        print(f"\nSTATISTICAL SIGNIFICANCE:")
        print(f"Chi-square test for stage differences:")
        print(f"  Chi-square statistic: {chi2:.3f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant difference across stages: {'Yes' if p_value < 0.05 else 'No'}")  # type: ignore

        # Effect size (Cramér's V)
        n = sum(stage_stats[stage]['total_changes'] for stage in stage_stats.keys())
        cramers_v = np.sqrt(chi2 / (n * (min(len(contingency_table), len(contingency_table[0])) - 1)))  # type: ignore
        print(f"  Effect size (Cramér's V): {cramers_v:.3f} ({'Small' if cramers_v < 0.1 else 'Medium' if cramers_v < 0.3 else 'Large'})")

        return stage_stats

    def analyze_lifecycle_stages_detailed(self):
        """
        Detailed analysis of performance changes across project lifecycle stages.
        """
        # Define lifecycle stages
        def get_lifecycle_stage(normalized_time):
            if normalized_time <= 0.33:
                return 'Early Stage'
            elif normalized_time <= 0.67:
                return 'Middle Stage'
            else:
                return 'Late Stage'

        self.df['lifecycle_stage'] = self.df['normalized_time'].apply(get_lifecycle_stage)

        # Calculate detailed statistics
        stage_stats = []
        for stage in ['Early Stage', 'Middle Stage', 'Late Stage']:
            stage_data = self.df[self.df['lifecycle_stage'] == stage]
            total = len(stage_data)

            if total > 0:
                improvements = len(stage_data[stage_data['change_type'] == 'Improvement'])
                regressions = len(stage_data[stage_data['change_type'] == 'Regression'])
                unchanged = len(stage_data[stage_data['change_type'] == 'Unchanged'])

                # Significant changes
                sig_improvements = len(stage_data[
                    (stage_data['change_type'] == 'Improvement') &
                    (stage_data['is_significant_change'])
                ])
                sig_regressions = len(stage_data[
                    (stage_data['change_type'] == 'Regression') &
                    (stage_data['is_significant_change'])
                ])

                stage_stats.append({
                    'stage': stage,
                    'total_commits': total,
                    'improvement_pct': improvements / total * 100,
                    'regression_pct': regressions / total * 100,
                    'unchanged_pct': unchanged / total * 100,
                    'sig_improvement_pct': sig_improvements / total * 100,
                    'sig_regression_pct': sig_regressions / total * 100,
                    'significant_change_rate': (sig_improvements + sig_regressions) / total * 100
                })

        return pd.DataFrame(stage_stats)

    def calculate_effect_size_statistics(self):
        """
        Calculate concrete statistics for effect sizes by change type.
        """
        stats_by_type = {}

        for change_type in ['Improvement', 'Regression', 'Unchanged']:
            data = self.df[self.df['change_type'] == change_type]['effect_size']
            if change_type == 'Regression':
                data = abs(data)  # Use absolute values for comparison

            stats_by_type[change_type] = {
                'count': len(data),
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'q25': data.quantile(0.25),
                'q75': data.quantile(0.75),
                'small_effect_pct': len(data[(data >= 0.147) & (data < 0.33)]) / len(data) * 100,
                'medium_effect_pct': len(data[(data >= 0.33) & (data < 0.474)]) / len(data) * 100,
                'large_effect_pct': len(data[data >= 0.474]) / len(data) * 100,
            }

        return stats_by_type

    def statistical_comparison_improvements_vs_regressions(self):
        """
        Statistical comparison between improvements and regressions.
        """
        improvements = self.df[self.df['change_type'] == 'Improvement']['effect_size']
        regressions = abs(self.df[self.df['change_type'] == 'Regression']['effect_size'])

        # Ensure numeric and drop NaNs
        improvements = pd.to_numeric(improvements, errors='coerce').dropna()
        regressions = pd.to_numeric(regressions, errors='coerce').dropna()

        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(improvements, regressions,
                                                alternative='two-sided')

        # Effect size comparison (Cohen's d)
        imp_arr = improvements.to_numpy(dtype=float)
        reg_arr = regressions.to_numpy(dtype=float)
        pooled_std = np.sqrt(
            ((len(imp_arr) - 1) * np.var(imp_arr, ddof=1) +
             (len(reg_arr) - 1) * np.var(reg_arr, ddof=1)) /
            (len(imp_arr) + len(reg_arr) - 2)
        )
        cohens_d = (np.mean(imp_arr) - np.mean(reg_arr)) / pooled_std if pooled_std != 0 else np.nan

        return {
            'mann_whitney_statistic': statistic,
            'mann_whitney_p_value': p_value,
            'cohens_d': cohens_d,
            'improvements_mean': improvements.mean(),
            'regressions_mean': regressions.mean(),
            'improvements_median': improvements.median(),
            'regressions_median': regressions.median()
        }

    def analyze_individual_projects_top5(self):
        """
        Analyze performance change patterns for top 5 projects by data points.
        """
        # Identify top 5 projects by data points
        project_counts = self.df.groupby('project_id').size().sort_values(ascending=False)
        top5_projects = project_counts.head(5).index.tolist()

        # Create visualization - 5 plots vertically aligned
        fig, axes = plt.subplots(5, 1, figsize=(26, 24))

        project_stats = {}

        for i, project in enumerate(top5_projects):
            project_data = self.df[self.df['project_id'] == project].copy()

            # Normalize timeline for this specific project (0 = start, 1 = end)
            min_time = project_data['normalized_time'].min()
            max_time = project_data['normalized_time'].max()
            project_data['project_timeline'] = (project_data['normalized_time'] - min_time) / (max_time - min_time)

            window_size = 0.2
            step_size = 0.1
            window_centers = np.arange(window_size/2, 1 - window_size/2 + step_size, step_size)

            improvements_prop = []
            regressions_prop = []
            instability_prop = []

            for center in window_centers:
                window_start = max(0, center - window_size/2)
                window_end = min(1, center + window_size/2)

                window_data = project_data[
                    (project_data['project_timeline'] >= window_start) &
                    (project_data['project_timeline'] <= window_end)
                ]

                if len(window_data) >= 5:
                    total = len(window_data)
                    imp_count = len(window_data[window_data['change_type'] == 'Improvement'])
                    reg_count = len(window_data[window_data['change_type'] == 'Regression'])

                    improvements_prop.append(imp_count / total * 100)
                    regressions_prop.append(reg_count / total * 100)
                    instability_prop.append((imp_count + reg_count) / total * 100)
                else:
                    improvements_prop.append(np.nan)
                    regressions_prop.append(np.nan)
                    instability_prop.append(np.nan)

            ax = axes[i]

            valid_indices = ~np.isnan(instability_prop)
            valid_centers = window_centers[valid_indices]
            valid_imp = np.array(improvements_prop)[valid_indices]
            valid_reg = np.array(regressions_prop)[valid_indices]
            valid_inst = np.array(instability_prop)[valid_indices]

            ax.plot(valid_centers, valid_imp, 'o-', color='#27ae60', linewidth=6, markersize=6,
                    label='Performance Improvements', alpha=0.8)
            ax.plot(valid_centers, valid_reg, 'o-', color='#e74c3c', linewidth=6, markersize=6,
                    label='Performance Regressions', alpha=0.8)
            ax.plot(valid_centers, valid_inst, 's--', color='#9b59b6', linewidth=6, markersize=6,
                    label='Performance Instability Rate', alpha=0.9)

            ax.set_title(f'{project}\n({project_counts[project]} method-level changes)', fontsize=28, pad=20)

            xticks = np.arange(-0.2, 1.3, 0.2)
            ax.set_xticks(xticks[1:])
            ax.set_xticklabels([f'{t:0.2}' for t in xticks][1:], fontsize=24)
            yticks = np.arange(0, 101, 20)
            ax.set_yticks(yticks[1:])
            ax.set_yticklabels([f'{int(t)}%' for t in yticks][1:], fontsize=24)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(valid_imp.max(), valid_reg.max(), valid_inst.max()) + 5)
            ax.set_xlim(0, 1)

            if i == len(top5_projects) - 1:
                ax.set_xlabel('Normalized Project Timeline (0=Start, 1=End)', fontsize=26, labelpad=20)

            if i == 2:
                ax.set_ylabel('Percentage of Changes (%)', fontsize=26, labelpad=20)

            total_changes = len(project_data)
            improvements_count = len(project_data[project_data['change_type'] == 'Improvement'])
            regressions_count = len(project_data[project_data['change_type'] == 'Regression'])

            # Lifecycle stage analysis
            early_data = project_data[project_data['project_timeline'] <= 0.33]
            middle_data = project_data[(project_data['project_timeline'] > 0.33) &
                                       (project_data['project_timeline'] <= 0.67)]
            late_data = project_data[project_data['project_timeline'] > 0.67]

            # Calculate volatility (variance in instability across timeline)
            volatility = np.nanvar(valid_inst) if len(valid_inst) > 0 else 0

            project_stats[project] = {
                'total_changes': total_changes,
                'improvement_rate': improvements_count / total_changes * 100,
                'regression_rate': regressions_count / total_changes * 100,
                'instability_rate': (improvements_count + regressions_count) / total_changes * 100,
                'early_instability': len(early_data[early_data['change_type'].isin(['Improvement', 'Regression'])]) / len(early_data) * 100 if len(early_data) > 0 else 0,
                'middle_instability': len(middle_data[middle_data['change_type'].isin(['Improvement', 'Regression'])]) / len(middle_data) * 100 if len(middle_data) > 0 else 0,
                'late_instability': len(late_data[late_data['change_type'].isin(['Improvement', 'Regression'])]) / len(late_data) * 100 if len(late_data) > 0 else 0,
                'volatility': volatility,
                'peak_instability': np.nanmax(valid_inst) if len(valid_inst) > 0 else 0,
                'timeline_coverage': len(valid_inst) / len(window_centers) * 100
            }

        plt.tight_layout()
        self._set_standard_legend_style(axes[0], padding_factor=0.02, title='Change Type')
        self.save_plot(plt, 'individual_project_performance_patterns_top5')

        return project_stats

    def generate_individual_project_statistics(self):
        """
        Generate detailed statistics for individual project analysis.
        """
        project_stats = self.analyze_individual_projects_top5()

        print("\n" + "="*80)
        print("INDIVIDUAL PROJECT ANALYSIS - STATISTICAL SUMMARY")
        print("="*80)

        instability_rates = [stats['instability_rate'] for stats in project_stats.values()]
        volatilities = [stats['volatility'] for stats in project_stats.values()]
        peak_instabilities = [stats['peak_instability'] for stats in project_stats.values()]

        print(f"\nCROSS-PROJECT VARIATION ANALYSIS:")
        print(f"Instability rate range: {min(instability_rates):.1f}% to {max(instability_rates):.1f}%")
        print(f"Instability rate standard deviation: {np.std(instability_rates):.1f}%")
        print(f"Volatility range: {min(volatilities):.1f} to {max(volatilities):.1f}")
        print(f"Peak instability range: {min(peak_instabilities):.1f}% to {max(peak_instabilities):.1f}%")

        mean_instability = np.mean(instability_rates)
        mean_volatility = np.mean(volatilities)

        high_instability_projects = [p for p, s in project_stats.items() if s['instability_rate'] > mean_instability]
        low_instability_projects = [p for p, s in project_stats.items() if s['instability_rate'] <= mean_instability]

        high_volatility_projects = [p for p, s in project_stats.items() if s['volatility'] > mean_volatility]
        low_volatility_projects = [p for p, s in project_stats.items() if s['volatility'] <= mean_volatility]

        print(f"\nSTABILITY CATEGORIZATION:")
        print(f"High instability (>{mean_instability:.1f}%): {', '.join(high_instability_projects)}")
        print(f"Low instability (≤{mean_instability:.1f}%): {', '.join(low_instability_projects)}")
        print(f"High volatility (>{mean_volatility:.1f}): {', '.join(high_volatility_projects)}")
        print(f"Low volatility (≤{mean_volatility:.1f}): {', '.join(low_volatility_projects)}")

        print(f"\nDETAILED PROJECT PROFILES:")
        for project, stats in project_stats.items():
            lifecycle_pattern = "Early-heavy" if stats['early_instability'] > stats['late_instability'] else "Late-heavy"
            stability_level = "High" if stats['instability_rate'] > mean_instability else "Low"
            volatility_level = "High" if stats['volatility'] > mean_volatility else "Low"

            print(f"\n{project.upper()}:")
            print(f"  Changes: {stats['total_changes']} | Instability: {stats['instability_rate']:.1f}% | Volatility: {stats['volatility']:.1f}")
            print(f"  Pattern: {lifecycle_pattern} | Stability: {stability_level} | Volatility: {volatility_level}")
            print(f"  Lifecycle: Early={stats['early_instability']:.1f}%, Middle={stats['middle_instability']:.1f}%, Late={stats['late_instability']:.1f}%")

        instability_rates = [stats['instability_rate'] for stats in project_stats.values()]
        cv = np.std(instability_rates) / np.mean(instability_rates) * 100  # Coefficient of variation

        print(f"\nVARIATION ANALYSIS:")
        print(f"Instability rate range: {min(instability_rates):.1f}% to {max(instability_rates):.1f}%")
        print(f"Standard deviation: {np.std(instability_rates):.1f}%")
        print(f"Coefficient of variation: {cv:.1f}%")
        print(f"High variation across projects: {'Yes' if cv > 25 else 'No'}")

        return project_stats


if __name__ == "__main__":
    dataset = os.path.join('dataset', 'dataset.csv')
    output_dir = os.path.join('plots', 'RQ1')
    df = pd.read_csv(dataset)
    analyzer = RQ1(df, output_dir)

    time_proportions = analyzer.plot_performance_change_proportions_over_time()
    lifecycle_comparison = analyzer.plot_lifecycle_stage_comparison()
    lifecycle_stats = analyzer.analyze_lifecycle_stages_detailed()
    effect_size_stats = analyzer.calculate_effect_size_statistics()
    comparison_stats = analyzer.statistical_comparison_improvements_vs_regressions()
    analyzer.generate_lifecycle_stage_statistics()
    analyzer.analyze_individual_projects_top5()
    analyzer.generate_individual_project_statistics()

    print("=== LIFECYCLE STAGE ANALYSIS ===")
    print(lifecycle_stats.round(2))

    print("\n=== EFFECT SIZE STATISTICS ===")
    for change_type, stats in effect_size_stats.items():
        print(f"\n{change_type}:")
        for key, value in stats.items():
            print(f"  {key}: {value:.3f}")

    print("\n=== STATISTICAL COMPARISON ===")
    for key, value in comparison_stats.items():
        print(f"{key}: {value:.4f}")
