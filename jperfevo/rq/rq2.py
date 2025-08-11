from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
import itertools
import seaborn as sns
import os


class RQ2:
    def __init__(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        RQ2 class for analyzing the impact of code change types on performance.
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

        # Split code change labels into separate rows
        self.df_expanded = self.df.copy()
        self.df_expanded['code_change_label'] = self.df_expanded['code_change_label'].fillna('Unknown')
        self.df_expanded = self.df_expanded.assign(
            code_change_label=self.df_expanded['code_change_label'].str.split('+')
        ).explode('code_change_label')

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

    def generate_code_change_statistics(self):
        """
        Generate comprehensive statistics for code change impact analysis.
        """
        print("="*80)
        print("CODE CHANGE TYPE IMPACT - DETAILED STATISTICS")
        print("="*80)

        df_valid = self.df_expanded[self.df_expanded['code_change_label'] != 'Unknown']
        results = {}
        for change_type in df_valid['code_change_label'].unique():
            subset = df_valid[df_valid['code_change_label'] == change_type]
            total = len(subset)

            improvements = len(subset[subset['change_type'] == 'Improvement'])
            regressions = len(subset[subset['change_type'] == 'Regression'])
            unchanged = len(subset[subset['change_type'] == 'Unchanged'])

            significant_changes = subset[subset['change_type'].isin(['Improvement', 'Regression'])]
            effect_sizes = significant_changes['effect_size'].abs()

            results[change_type] = {
                'total_commits': total,
                'improvements': improvements,
                'regressions': regressions,
                'Unchanged': unchanged,
                'improvement_rate': improvements / total * 100,
                'regression_rate': regressions / total * 100,
                'unchanged_rate': unchanged / total * 100,
                'mean_effect_size': effect_sizes.mean(),
                'median_effect_size': effect_sizes.median(),
                'max_effect_size': effect_sizes.max(),
                'risk_ratio': regressions / improvements if improvements > 0 else float('inf')
            }

        # Print detailed table
        print(f"\n{'Category':<25} {'Total':<8} {'Impr%':<8} {'Regr%':<8} {'Unch%':<8} {'Risk':<8} {'Mean ES':<8}")
        print("-" * 80)

        for category, stats in sorted(results.items()):
            risk_str = f"{stats['risk_ratio']:.2f}" if stats['risk_ratio'] != float('inf') else "∞"
            impr_str = f"{stats['improvements']} ({stats['improvement_rate']:.1f}%)"
            regr_str = f"{stats['regressions']} ({stats['regression_rate']:.1f}%)"
            unch_str = f"{stats['Unchanged']} ({stats['unchanged_rate']:.1f}%)"
            print(f"{category:<25} {stats['total_commits']:<8} "
                  f"{impr_str:<12} {regr_str:<12} {unch_str:<12} "
                  f"{risk_str:<8} {stats['mean_effect_size']:<8.3f}")

        print(f"\n{'STATISTICAL SIGNIFICANCE ANALYSIS:'}")

        categories = list(results.keys())
        contingency_data = []

        for category in categories:
            stats = results[category]
            contingency_data.append([stats['improvements'], stats['regressions'], stats['Unchanged']])

        chi2, p_value, dof, expected = chi2_contingency(contingency_data)

        print(f"Chi-square test for differences across categories:")
        print(f"  Chi-square statistic: {chi2:.3f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")  # type: ignore

        # Effect size (Cramér's V)
        n = sum(stats['total_commits'] for stats in results.values())
        cramers_v = np.sqrt(chi2 / (n * (min(len(contingency_data), len(contingency_data[0])) - 1)))  # type: ignore
        print(f"  Effect size (Cramér's V): {cramers_v:.3f}")

        print(f"\n{'CATEGORY RANKINGS:'}")

        # Sort by regression rate (risk)
        high_risk = sorted(results.items(), key=lambda x: x[1]['regression_rate'], reverse=True)[:3]
        print(f"Highest regression risk:")
        for i, (cat, stats) in enumerate(high_risk, 1):
            print(f"  {i}. {cat}: {stats['regression_rate']:.1f}% ({stats['regressions']}/{stats['total_commits']} commits)")

        # Sort by improvement rate (benefit)
        high_benefit = sorted(results.items(), key=lambda x: x[1]['improvement_rate'], reverse=True)[:3]
        print(f"\nHighest improvement potential:")
        for i, (cat, stats) in enumerate(high_benefit, 1):
            print(f"  {i}. {cat}: {stats['improvement_rate']:.1f}% ({stats['improvements']}/{stats['total_commits']} commits)")

        return results

    def generate_effect_size_distribution_by_category(self):
        """
        Analyze effect size distributions by category with statistical rigor.
        """
        print("\n" + "="*80)
        print("EFFECT SIZE DISTRIBUTION BY CODE CHANGE CATEGORY")
        print("="*80)

        df_valid = self.df_expanded[self.df_expanded['code_change_label'] != 'Unknown']

        effect_categories = {
            'Small': (0.147, 0.33),
            'Medium': (0.33, 0.474),
            'Large': (0.474, float('inf'))
        }

        for change_type in df_valid['code_change_label'].unique():
            subset = df_valid[df_valid['code_change_label'] == change_type]
            improvements = subset[subset['change_type'] == 'Improvement']
            regressions = subset[subset['change_type'] == 'Regression']

            print(f"\n{change_type.upper()}:")
            print(f"  Total significant changes: {len(improvements) + len(regressions)}")

            if len(improvements) > 0:
                print(f"  Improvements ({len(improvements)} commits):")
                for size_label, (min_val, max_val) in effect_categories.items():
                    if max_val == float('inf'):
                        count = len(improvements[improvements['effect_size'] >= min_val])
                    else:
                        count = len(improvements[(improvements['effect_size'] >= min_val) &
                                                 (improvements['effect_size'] < max_val)])
                    percentage = count / len(improvements) * 100
                    print(f"    {size_label}: {count} ({percentage:.1f}%)")

            if len(regressions) > 0:
                print(f"  Regressions ({len(regressions)} commits):")
                for size_label, (min_val, max_val) in effect_categories.items():
                    if max_val == float('inf'):
                        count = len(regressions[regressions['effect_size'].abs() >= min_val])
                    else:
                        count = len(regressions[(regressions['effect_size'].abs() >= min_val) &
                                                (regressions['effect_size'].abs() < max_val)])
                    percentage = count / len(regressions) * 100
                    print(f"    {size_label}: {count} ({percentage:.1f}%)")

    def plot_proportional_impact_by_category(self):
        """
        Create a visualization showing proportions instead of just boxplots.
        """
        df_valid = self.df_expanded[self.df_expanded['code_change_label'] != 'Unknown']
        proportions_data = []

        label_map = {
            'Algorithmic Change': 'ALG',
            'Control Flow/Loop Changes': 'CF',
            'Data Structure & Variable Changes': 'DS',
            'Refactoring & Code Cleanup': 'REF',
            'Exception & Input/Output Handling': 'ER',
            'Concurrency/Parallelism': 'CON',
            'API/Library Call Changes': 'API'
        }

        for category in label_map.keys():
            subset = df_valid[df_valid['code_change_label'] == category]
            total = len(subset)

            improvements = len(subset[subset['change_type'] == 'Improvement'])
            regressions = len(subset[subset['change_type'] == 'Regression'])
            unchanged = len(subset[subset['change_type'] == 'Unchanged'])

            proportions_data.append({
                'Category': label_map.get(category, category),
                'Improvements': improvements / total * 100,
                'Regressions': regressions / total * 100,
                'Unchanged': unchanged / total * 100,
                'Total': total
            })

        proportions_df = pd.DataFrame(proportions_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 10))

        categories = proportions_df['Category']
        improvements = proportions_df['Improvements']
        regressions = proportions_df['Regressions']
        unchanged = proportions_df['Unchanged']

        width = 0.75
        ax1.bar(categories, improvements, width, label='Improvements', color='#27ae60')
        ax1.bar(categories, regressions, width, bottom=improvements, label='Regressions', color='#e74c3c')
        ax1.bar(categories, unchanged, width, bottom=improvements + regressions, label='Unchanged', color='#3498db')

        # Annotate percentages inside each bar segment
        for i, (imp, reg, unc) in enumerate(zip(improvements, regressions, unchanged)):
            # Improvements
            if imp > 0:
                ax1.text(i, imp / 2, f"{imp:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=18)
            # Regressions
            if reg > 0:
                ax1.text(i, imp + reg / 2, f"{reg:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=18)
            # Unchanged
            if unc > 0:
                ax1.text(i, imp + reg + unc / 2, f"{unc:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=18)

        ax1.set_ylabel('Percentage of Commits', fontsize=26, labelpad=20)
        ax1.set_xlabel('Code Change Category', fontsize=26, labelpad=20)
        ax1.set_title('Performance Impact Distribution by Category', fontsize=30, pad=20)
        xticks = label_map.values()
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(xticks, fontsize=22)
        yticks = np.arange(0, 101, 20)
        ax1.set_yticks(yticks)
        ax1.set_yticklabels([f"{tick}%" for tick in yticks], fontsize=22)
        self._set_standard_legend_style(ax1, padding_factor=0.02, title='Change Type')

        # Right plot: Sample sizes
        ax2.bar(categories, proportions_df['Total'], color='#95a5a6', alpha=0.7)
        ax2.set_ylabel('Number of Commits', fontsize=26, labelpad=20)
        ax2.set_xlabel('Code Change Category', fontsize=26, labelpad=20)
        ax2.set_title('Sample Size by Category', fontsize=30, pad=20)
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(xticks, fontsize=22)
        ax2.set_yticks(np.arange(0, max(proportions_df['Total']) + 1, 50))
        ax2.set_yticklabels([f"{tick}" for tick in np.arange(0, max(proportions_df['Total']) + 1, 50)], fontsize=22)
        ax2.set_ylim(0, max(proportions_df['Total']) * 1.1)  # Add some space above bars

        # Add sample size annotations
        for i, (cat, total) in enumerate(zip(categories, proportions_df['Total'])):
            ax2.text(i, total + max(proportions_df['Total']) * 0.01, str(total),
                     ha='center', va='bottom', fontweight='bold', fontsize=18, color='black')

        plt.tight_layout()
        self.save_plot(plt, 'proportional_impact_by_category')

    def analyze_statistical_significance_between_categories(self):
        """
        Perform pairwise statistical tests between categories.
        """
        print("\n" + "="*60)
        print("PAIRWISE STATISTICAL COMPARISONS")
        print("="*60)

        df_valid = self.df_expanded[self.df_expanded['code_change_label'] != 'Unknown']
        categories = df_valid['code_change_label'].unique()

        # Pairwise comparisons for improvement vs regression rates
        significant_pairs = []

        for cat1, cat2 in itertools.combinations(categories, 2):
            subset1 = df_valid[df_valid['code_change_label'] == cat1]
            subset2 = df_valid[df_valid['code_change_label'] == cat2]

            impr1 = len(subset1[subset1['change_type'] == 'Improvement'])
            regr1 = len(subset1[subset1['change_type'] == 'Regression'])
            impr2 = len(subset2[subset2['change_type'] == 'Improvement'])
            regr2 = len(subset2[subset2['change_type'] == 'Regression'])

            if impr1 + regr1 > 0 and impr2 + regr2 > 0:  # Ensure we have data
                contingency_table = [[impr1, regr1], [impr2, regr2]]

                try:
                    odds_ratio, p_value = fisher_exact(contingency_table)

                    if p_value < 0.05:  # type: ignore
                        significant_pairs.append({
                            'category1': cat1,
                            'category2': cat2,
                            'p_value': p_value,
                            'odds_ratio': odds_ratio,
                            'cat1_impr_rate': impr1 / (impr1 + regr1) * 100,
                            'cat2_impr_rate': impr2 / (impr2 + regr2) * 100
                        })
                except:
                    continue

        if significant_pairs:
            print("Statistically significant differences found:")
            for pair in significant_pairs:
                print(f"\n{pair['category1']} vs {pair['category2']}:")
                print(f"  p-value: {pair['p_value']:.4f}")
                print(f"  {pair['category1']} improvement rate: {pair['cat1_impr_rate']:.1f}%")
                print(f"  {pair['category2']} improvement rate: {pair['cat2_impr_rate']:.1f}%")
                print(f"  Odds ratio: {pair['odds_ratio']:.2f}")
        else:
            print("No statistically significant differences found between categories.")

        return significant_pairs


if __name__ == "__main__":
    dataset = os.path.join('dataset', 'dataset.csv')
    output_dir = os.path.join('plots', 'RQ2')
    df = pd.read_csv(dataset)

    visualizer = RQ2(df, output_dir)
    visualizer.generate_code_change_statistics()
    visualizer.generate_effect_size_distribution_by_category()
    visualizer.plot_proportional_impact_by_category()
    visualizer.analyze_statistical_significance_between_categories()
