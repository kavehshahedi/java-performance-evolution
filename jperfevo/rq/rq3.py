import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr, kruskal
import numpy as np
from tabulate import tabulate
from typing import Optional


class RQ3:
    def __init__(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        RQ3 class for analyzing the impact of developer experience and code change complexity on performance.
        """
        self.df = df.copy()
        self._preprocess_data()

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._set_style()

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

    def _preprocess_data(self):
        """Preprocess the data for analysis."""
        self.df = self.df.dropna(subset=['effect_size'])
        self.df = self.df.reindex(self.df['effect_size'].abs().sort_values().index).drop_duplicates(['project_id', 'commit_id', 'method_name'], keep='first')

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

    def show_experience_change_distribution(self):
        """Create and display experience vs change type distribution table."""
        # Calculate percentages for each experience category and change type
        pivot_table = pd.crosstab(
            self.df['experience_category'],
            self.df['change_type'],
            normalize='index',
        ) * 100

        # Add total column (number of data points for each experience category)
        pivot_table['Total Data Points'] = self.df['experience_category'].value_counts()

        # Format the table using tabulate
        table = tabulate(
            pivot_table.reset_index(),  # type: ignore
            headers=['Experience Level', 'Improvement', 'Regression', 'Unchanged', 'Total Data Points'],
            tablefmt='grid',
            floatfmt='.2f'
        )

        print("\nExperience Level vs Change Type Distribution (%):")
        print(table)

    def analyze_experience_differences(self) -> dict:
        """
        Analyze if there are statistically significant differences between experience groups.
        """
        # Convert experience categories to numeric for analysis
        experience_map = {'Junior': 0, 'Mid': 1, 'Senior': 2}
        experience_values = self.df['experience_category'].map(experience_map)

        # Get effect sizes for each group
        junior_effects = self.df[self.df['experience_category'] == 'Junior']['effect_size']
        mid_effects = self.df[self.df['experience_category'] == 'Mid']['effect_size']
        senior_effects = self.df[self.df['experience_category'] == 'Senior']['effect_size']

        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(junior_effects, mid_effects, senior_effects)

        means = {
            'Junior': f'{(junior_effects.mean()*100):.2f}',
            'Mid': f'{(mid_effects.mean()*100):.2f}',
            'Senior': f'{(senior_effects.mean()*100):.2f}'
        }

        return {
            'significant_difference': p_value < 0.05,
            'p_value': p_value,
            'f_statistic': f_stat,
            'group_means': means
        }

    def comprehensive_experience_analysis(self):
        """
        Provide statistical analysis of experience vs performance.
        """
        print("="*80)
        print("DEVELOPER EXPERIENCE VS PERFORMANCE - COMPREHENSIVE ANALYSIS")
        print("="*80)

        for experience in ['Junior', 'Mid', 'Senior']:
            subset = self.df[self.df['experience_category'] == experience]
            total = len(subset)

            improvements = len(subset[subset['change_type'] == 'Improvement'])
            regressions = len(subset[subset['change_type'] == 'Regression'])
            unchanged = len(subset[subset['change_type'] == 'Unchanged'])

            print(f"\n{experience.upper()} DEVELOPERS (n={total}):")
            print(f"  Improvements: {improvements} ({improvements/total*100:.1f}%)")
            print(f"  Regressions: {regressions} ({regressions/total*100:.1f}%)")
            print(f"  Unchanged: {unchanged} ({unchanged/total*100:.1f}%)")

            # Effect size statistics
            effect_sizes = subset['effect_size'].abs()
            print(f"  Mean |Effect Size|: {effect_sizes.mean():.3f}")
            print(f"  Median |Effect Size|: {effect_sizes.median():.3f}")
            print(f"  Effect Size Std: {effect_sizes.std():.3f}")

        junior_effects = self.df[self.df['experience_category'] == 'Junior']['effect_size'].abs()
        mid_effects = self.df[self.df['experience_category'] == 'Mid']['effect_size'].abs()
        senior_effects = self.df[self.df['experience_category'] == 'Senior']['effect_size'].abs()

        f_stat, p_anova = f_oneway(junior_effects, mid_effects, senior_effects)

        ss_between = f_stat * (len(junior_effects) + len(mid_effects) + len(senior_effects) - 3)
        ss_total = ss_between + sum([(x - x.mean())**2 for x in [junior_effects, mid_effects, senior_effects]]).sum()  # type: ignore
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        print(f"\nANOVA RESULTS:")
        print(f"  F-statistic: {f_stat:.3f}")
        print(f"  p-value: {p_anova:.6f}")
        print(f"  Effect size (η²): {eta_squared:.6f}")
        print(f"  Variance explained: {eta_squared*100:.1f}%")

        contingency_data = []
        categories = ['Junior', 'Mid', 'Senior']

        for category in categories:
            subset = self.df[self.df['experience_category'] == category]
            improvements = len(subset[subset['change_type'] == 'Improvement'])
            regressions = len(subset[subset['change_type'] == 'Regression'])
            unchanged = len(subset[subset['change_type'] == 'Unchanged'])
            contingency_data.append([improvements, regressions, unchanged])

        chi2, p_chi2, dof, expected = chi2_contingency(contingency_data)

        n = sum(sum(row) for row in contingency_data)
        cramers_v = np.sqrt(chi2 / (n * (min(len(contingency_data), len(contingency_data[0])) - 1)))  # type: ignore

        print(f"\nCHI-SQUARE TEST:")
        print(f"  Chi-square: {chi2:.3f}")
        print(f"  p-value: {p_chi2:.6f}")
        print(f"  Cramér's V: {cramers_v:.3f}")

        print(f"\n95% CONFIDENCE INTERVALS FOR IMPROVEMENT RATES:")

        for category in categories:
            subset = self.df[self.df['experience_category'] == category]
            improvements = len(subset[subset['change_type'] == 'Improvement'])
            total = len(subset)
            prop = improvements / total
            se = np.sqrt(prop * (1 - prop) / total)
            ci_lower = prop - 1.96 * se
            ci_upper = prop + 1.96 * se
            print(f"  {category}: {prop*100:.1f}% [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")

    def analyze_complexity_performance_correlation(self):
        """
        Analyze correlation between complexity scores and performance impact.
        Test if complexity is actually predictive of performance.
        """
        print("\n" + "="*80)
        print("CODE CHANGE COMPLEXITY VS PERFORMANCE CORRELATION")
        print("="*80)

        df_valid = self.df[self.df['method_change_complexity'] != -1].copy()

        complexity_scores = df_valid['method_change_complexity']
        absolute_effect_sizes = df_valid['effect_size'].abs()

        pearson_r, pearson_p = pearsonr(complexity_scores, absolute_effect_sizes)
        spearman_r, spearman_p = spearmanr(complexity_scores, absolute_effect_sizes)

        print(f"CORRELATION ANALYSIS:")
        print(f"  Pearson correlation: r = {pearson_r:.3f}, p = {pearson_p:.6f}")
        print(f"  Spearman correlation: ρ = {spearman_r:.3f}, p = {spearman_p:.6f}")
        print(f"  Variance explained: R² = {pearson_r**2:.3f} ({pearson_r**2*100:.1f}%)")  # type: ignore

        df_valid['complexity_category'] = pd.qcut(
            df_valid['method_change_complexity'],
            q=4,
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        print(f"\nCOMPLEXITY CATEGORIES ANALYSIS:")
        for category in ['Low', 'Medium', 'High', 'Very High']:
            subset = df_valid[df_valid['complexity_category'] == category]
            total = len(subset)

            improvements = len(subset[subset['change_type'] == 'Improvement'])
            regressions = len(subset[subset['change_type'] == 'Regression'])
            unchanged = len(subset[subset['change_type'] == 'Unchanged'])

            print(f"\n{category.upper()} COMPLEXITY (n={total}):")
            print(f"  Improvements: {improvements} ({improvements/total*100:.1f}%)")
            print(f"  Regressions: {regressions} ({regressions/total*100:.1f}%)")
            print(f"  Unchanged: {unchanged} ({unchanged/total*100:.1f}%)")
            print(f"  Mean |Effect Size|: {subset['effect_size'].abs().mean():.3f}")

        complexity_groups = [
            df_valid[df_valid['complexity_category'] == cat]['effect_size'].abs().values
            for cat in ['Low', 'Medium', 'High', 'Very High']
        ]

        h_stat, p_kruskal = kruskal(*complexity_groups)

        print(f"\nKRUSKAL-WALLIS TEST:")
        print(f"  H-statistic: {h_stat:.3f}")
        print(f"  p-value: {p_kruskal:.6f}")

        contingency_complex = []
        for category in ['Low', 'Medium', 'High', 'Very High']:
            subset = df_valid[df_valid['complexity_category'] == category]
            improvements = len(subset[subset['change_type'] == 'Improvement'])
            regressions = len(subset[subset['change_type'] == 'Regression'])
            unchanged = len(subset[subset['change_type'] == 'Unchanged'])
            contingency_complex.append([improvements, regressions, unchanged])

        chi2_complex, p_chi2_complex, _, _ = chi2_contingency(contingency_complex)
        n_complex = sum(sum(row) for row in contingency_complex)
        cramers_v_complex = np.sqrt(chi2_complex / (n_complex * (min(len(contingency_complex), len(contingency_complex[0])) - 1)))  # type: ignore

        print(f"\nCHI-SQUARE TEST (COMPLEXITY):")
        print(f"  Chi-square: {chi2_complex:.3f}")
        print(f"  p-value: {p_chi2_complex:.6f}")
        print(f"  Cramér's V: {cramers_v_complex:.3f}")

    def plot_experience_and_complexity_impact_analysis(self):
        """
        Create a visualization showing proportions of performance impacts by experience and complexity.
        """
        df_valid = self.df[self.df['method_change_complexity'] != -1].copy()

        df_valid['complexity_category'] = pd.qcut(
            df_valid['method_change_complexity'],
            q=4,
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(23, 10), width_ratios=[1, 1.3])
        fig.subplots_adjust(wspace=0.18)

        colors = {
            'Improvement': '#27ae60',
            'Regression': '#e74c3c',
            'Unchanged': '#3498db'
        }

        # 1. Experience Level Analysis (Top Left)
        experience_data = []
        for exp_level in ['Junior', 'Mid', 'Senior']:
            subset = self.df[self.df['experience_category'] == exp_level]
            total = len(subset)

            improvements = len(subset[subset['change_type'] == 'Improvement'])
            regressions = len(subset[subset['change_type'] == 'Regression'])
            unchanged = len(subset[subset['change_type'] == 'Unchanged'])

            experience_data.append({
                'Experience': exp_level,
                'Improvement_pct': improvements / total * 100,
                'Regression_pct': regressions / total * 100,
                'Unchanged_pct': unchanged / total * 100,
                'Total': total,
                'Improvement_count': improvements,
                'Regression_count': regressions
            })

        exp_df = pd.DataFrame(experience_data)

        x_pos = np.arange(len(exp_df))
        width = 0.6

        bars1 = ax1.bar(x_pos, exp_df['Improvement_pct'], width,
                        label='Improvements', color=colors['Improvement'], alpha=0.8)
        bars2 = ax1.bar(x_pos, exp_df['Regression_pct'], width,
                        bottom=exp_df['Improvement_pct'],
                        label='Regressions', color=colors['Regression'], alpha=0.8)
        bars3 = ax1.bar(x_pos, exp_df['Unchanged_pct'], width,
                        bottom=exp_df['Improvement_pct'] + exp_df['Regression_pct'],
                        label='Unchanged', color=colors['Unchanged'], alpha=0.8)

        for i, (exp, row) in enumerate(exp_df.iterrows()):
            if row['Improvement_pct'] > 8:
                ax1.text(i, row['Improvement_pct']/2, f'{row["Improvement_pct"]:.1f}%',
                         ha='center', va='center', fontweight='bold', color='white', fontsize=22)

            if row['Regression_pct'] > 8:
                ax1.text(i, row['Improvement_pct'] + row['Regression_pct']/2,
                         f'{row["Regression_pct"]:.1f}%',
                         ha='center', va='center', fontweight='bold', color='white', fontsize=22)

            if row['Unchanged_pct'] > 8:
                ax1.text(i, row['Improvement_pct'] + row['Regression_pct'] + row['Unchanged_pct']/2,
                         f'{row["Unchanged_pct"]:.1f}%',
                         ha='center', va='center', fontweight='bold', color='white', fontsize=22)

        ax1.set_xlabel('Developer Experience Level', fontsize=26, labelpad=20)
        ax1.set_ylabel('Percentage of Commits (%)', fontsize=26, labelpad=20)
        ax1.set_title('Performance Impact Distribution\nby Developer Experience', fontsize=30, pad=20)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(exp_df['Experience'], fontsize=24)
        ax1.set_yticks(np.arange(0, 101, 20))
        ax1.set_yticklabels([f"{tick}%" for tick in np.arange(0, 101, 20)], fontsize=24)
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Complexity Level Analysis (Top Right)
        complexity_data = []
        for comp_level in ['Low', 'Medium', 'High', 'Very High']:
            subset = df_valid[df_valid['complexity_category'] == comp_level]
            total = len(subset)

            improvements = len(subset[subset['change_type'] == 'Improvement'])
            regressions = len(subset[subset['change_type'] == 'Regression'])
            unchanged = len(subset[subset['change_type'] == 'Unchanged'])

            complexity_data.append({
                'Complexity': comp_level,
                'Improvement_pct': improvements / total * 100,
                'Regression_pct': regressions / total * 100,
                'Unchanged_pct': unchanged / total * 100,
                'Total': total,
                'Improvement_count': improvements,
                'Regression_count': regressions
            })

        comp_df = pd.DataFrame(complexity_data)

        x_pos_comp = np.arange(len(comp_df))
        bars1 = ax2.bar(x_pos_comp, comp_df['Improvement_pct'], width,
                        label='Improvements', color=colors['Improvement'], alpha=0.8)
        bars2 = ax2.bar(x_pos_comp, comp_df['Regression_pct'], width,
                        bottom=comp_df['Improvement_pct'],
                        label='Regressions', color=colors['Regression'], alpha=0.8)
        bars3 = ax2.bar(x_pos_comp, comp_df['Unchanged_pct'], width,
                        bottom=comp_df['Improvement_pct'] + comp_df['Regression_pct'],
                        label='Unchanged', color=colors['Unchanged'], alpha=0.8)

        for i, (comp, row) in enumerate(comp_df.iterrows()):
            if row['Improvement_pct'] > 8:
                ax2.text(i, row['Improvement_pct']/2, f'{row["Improvement_pct"]:.1f}%',
                         ha='center', va='center', fontweight='bold', color='white', fontsize=22)

            if row['Regression_pct'] > 8:
                ax2.text(i, row['Improvement_pct'] + row['Regression_pct']/2,
                         f'{row["Regression_pct"]:.1f}%',
                         ha='center', va='center', fontweight='bold', color='white', fontsize=22)

            if row['Unchanged_pct'] > 8:
                ax2.text(i, row['Improvement_pct'] + row['Regression_pct'] + row['Unchanged_pct']/2,
                         f'{row["Unchanged_pct"]:.1f}%',
                         ha='center', va='center', fontweight='bold', color='white', fontsize=22)

        ax2.set_xlabel('Code Change Complexity Level', fontsize=26, labelpad=20)
        ax2.set_title('Performance Impact Distribution\nby Change Complexity', fontsize=30, pad=20)
        ax2.set_xticks(x_pos_comp)
        ax2.set_xticklabels(comp_df['Complexity'], fontsize=24)
        ax2.set_yticks(np.arange(0, 101, 20))
        ax2.set_yticklabels([f"{tick}%" for tick in np.arange(0, 101, 20)], fontsize=24)
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis='y')
        self._set_standard_legend_style(ax2, padding_factor=0.02, title='Change Type')

        plt.tight_layout()
        self.save_plot(plt, 'experience_complexity_comprehensive_analysis')

        return exp_df, comp_df


if __name__ == "__main__":
    dataset = os.path.join('dataset', 'dataset.csv')
    output_dir = os.path.join('plots', 'RQ3')
    df = pd.read_csv(dataset)

    visualizer = RQ3(df, output_dir)
    visualizer.show_experience_change_distribution()
    visualizer.comprehensive_experience_analysis()
    visualizer.analyze_complexity_performance_correlation()
    visualizer.plot_experience_and_complexity_impact_analysis()
