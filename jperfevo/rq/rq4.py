import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, chi2_contingency, f_oneway
import os
from typing import Optional, Dict
import warnings

warnings.filterwarnings("ignore")


class RQ4:
    def __init__(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        RQ4 class for analyzing the impact of domain and size on performance.
        """
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._preprocess_data()
        self._set_style()
        self.change_type_order = ['Unchanged', 'Regression', 'Improvement']

    def _preprocess_data(self):
        """Preprocess data with domain and size categorization."""
        self.df['commit_date'] = pd.to_datetime(self.df['commit_date'], unit='s')
        self.df = self.df.dropna(subset=['effect_size'])
        self.df = self.df[self.df['commit_date'] >= '2016-01-01']

        self.df['size_kloc'] = self.df['size'] / 1000  # Convert to KLOC

        size_thresholds = self.df['size_kloc'].quantile([0.33, 0.67]).values

        def categorize_size(kloc):
            if kloc <= size_thresholds[0]:
                return 'Small'
            elif kloc <= size_thresholds[1]:
                return 'Medium'
            else:
                return 'Large'

        self.df['size_category'] = self.df['size_kloc'].apply(categorize_size)

        if 'domain' not in self.df.columns:
            domain_mapping = {}
            self.df['domain'] = self.df['project_id'].map(domain_mapping).fillna('Unknown')

        for change_type in ['Improvement', 'Regression']:
            mask = self.df['change_type'] == change_type
            if change_type == 'Improvement':
                threshold = self.df[mask]['effect_size'].quantile(0.99)
                self.df = self.df[~(mask & (self.df['effect_size'] > threshold))]
            else:
                threshold = self.df[mask]['effect_size'].quantile(0.01)
                self.df = self.df[~(mask & (self.df['effect_size'] < threshold))]

    def _set_style(self):
        """Set consistent style for all visualizations."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rc('font', size=16)
        plt.rc('axes', titlesize=18)
        plt.rc('axes', labelsize=16)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        plt.rc('legend', fontsize=14)
        plt.rc('figure', titlesize=20)

    def _set_standard_legend_style(self, ax, padding_factor: float = 0.02, title: Optional[str] = None) -> None:
        """Standardize legend appearance across all plots."""
        fig = ax.get_figure()
        bbox = ax.get_position()
        legend_x = 1 + (padding_factor * bbox.width)

        legend = ax.legend(
            bbox_to_anchor=(legend_x, 1),
            title=title,
            fontsize=24,
            title_fontsize=24,
            frameon=True,
            borderaxespad=0,
            loc='upper left'
        )

        legend._legend_box.align = "left"
        legend_width = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).width

        total_width_needed = bbox.x1 + (legend_width * 1.1)
        if total_width_needed > 1:
            new_fig_width = fig.get_figwidth() / bbox.x1
            fig.set_figwidth(new_fig_width)
            plt.tight_layout()
            bbox = ax.get_position()
            legend_x = 1 + (padding_factor * bbox.width)
            legend.set_bbox_to_anchor((legend_x, 1))

    def save_plot(self, plt, filename: str) -> None:
        """Save plots with consistent settings."""
        pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
        png_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(png_path, format='png', dpi=72, bbox_inches='tight')
        plt.close()

    def analyze_domain_patterns_comprehensive(self) -> Dict:
        """
        Statistical analysis of domain-specific patterns.
        """
        print("="*80)
        print("DOMAIN-SPECIFIC PERFORMANCE PATTERNS - STATISTICAL ANALYSIS")
        print("="*80)

        domain_stats = {}

        for domain in self.df['domain'].unique():
            if domain == 'Unknown':
                continue

            domain_data = self.df[self.df['domain'] == domain]
            total = len(domain_data)

            improvements = len(domain_data[domain_data['change_type'] == 'Improvement'])
            regressions = len(domain_data[domain_data['change_type'] == 'Regression'])
            unchanged = len(domain_data[domain_data['change_type'] == 'Unchanged'])

            imp_effects = domain_data[domain_data['change_type'] == 'Improvement']['effect_size']
            reg_effects = domain_data[domain_data['change_type'] == 'Regression']['effect_size'].abs()

            domain_stats[domain] = {
                'total_changes': total,
                'improvement_count': improvements,
                'regression_count': regressions,
                'unchanged_count': unchanged,
                'improvement_rate': improvements / total * 100,
                'regression_rate': regressions / total * 100,
                'instability_rate': (improvements + regressions) / total * 100,
                'improvement_mean_effect': imp_effects.mean() if len(imp_effects) > 0 else 0,
                'improvement_std_effect': imp_effects.std() if len(imp_effects) > 0 else 0,
                'regression_mean_effect': reg_effects.mean() if len(reg_effects) > 0 else 0,
                'regression_std_effect': reg_effects.std() if len(reg_effects) > 0 else 0,
                'effect_size_range_imp': imp_effects.max() - imp_effects.min() if len(imp_effects) > 0 else 0,
                'effect_size_range_reg': reg_effects.max() - reg_effects.min() if len(reg_effects) > 0 else 0
            }

        print(f"\nDOMAIN BREAKDOWN:")
        for domain, stats in domain_stats.items():
            print(f"\n{domain.upper()} (n={stats['total_changes']}):")
            print(f"  Improvements: {stats['improvement_count']} ({stats['improvement_rate']:.1f}%)")
            print(f"  Regressions: {stats['regression_count']} ({stats['regression_rate']:.1f}%)")
            print(f"  Instability: {stats['instability_rate']:.1f}%")
            print(f"  Improvement effect: μ={stats['improvement_mean_effect']:.3f}, σ={stats['improvement_std_effect']:.3f}")
            print(f"  Regression effect: μ={stats['regression_mean_effect']:.3f}, σ={stats['regression_std_effect']:.3f}")

        # Statistical significance testing across domains
        contingency_data = []
        domain_names = []

        for domain, stats in domain_stats.items():
            contingency_data.append([stats['improvement_count'], stats['regression_count'], stats['unchanged_count']])
            domain_names.append(domain)

        chi2, p_value, dof, expected = chi2_contingency(contingency_data)
        n = sum(stats['total_changes'] for stats in domain_stats.values())
        cramers_v = np.sqrt(chi2 / (n * (min(len(contingency_data), len(contingency_data[0])) - 1)))  # type: ignore

        print(f"\nSTATISTICAL SIGNIFICANCE ACROSS DOMAINS:")
        print(f"  Chi-square statistic: {chi2:.3f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Cramér's V: {cramers_v:.3f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")  # type: ignore

        # Kruskal-Wallis test for effect size differences
        effect_groups = []
        for domain in domain_names:
            domain_effects = self.df[self.df['domain'] == domain]['effect_size'].abs()
            effect_groups.append(domain_effects.values)

        h_stat, p_kruskal = kruskal(*effect_groups)

        print(f"\nKRUSKAL-WALLIS TEST (Effect Sizes):")
        print(f"  H-statistic: {h_stat:.3f}")
        print(f"  p-value: {p_kruskal:.6f}")

        volatility_scores = {domain: stats['improvement_std_effect'] + stats['regression_std_effect']
                             for domain, stats in domain_stats.items()}

        stable_domains = sorted(volatility_scores.items(), key=lambda x: x[1])[:2]
        volatile_domains = sorted(volatility_scores.items(), key=lambda x: x[1])[-2:]

        print(f"\nSTABILITY CLASSIFICATION:")
        print(f"  Stable domains: {', '.join([d[0] for d in stable_domains])}")
        print(f"  Volatile domains: {', '.join([d[0] for d in volatile_domains])}")

        stable_effects = []
        volatile_effects = []

        for domain, _ in stable_domains:
            stable_effects.extend(self.df[self.df['domain'] == domain]['effect_size'].abs().values)
        for domain, _ in volatile_domains:
            volatile_effects.extend(self.df[self.df['domain'] == domain]['effect_size'].abs().values)

        stable_effects = np.array(stable_effects)
        volatile_effects = np.array(volatile_effects)

        pooled_std = np.sqrt(((len(stable_effects) - 1) * np.var(stable_effects, ddof=1) +
                             (len(volatile_effects) - 1) * np.var(volatile_effects, ddof=1)) /
                             (len(stable_effects) + len(volatile_effects) - 2))

        cohens_d = (np.mean(volatile_effects) - np.mean(stable_effects)) / pooled_std if pooled_std != 0 else 0

        print(f"  Cohen's d (Volatile vs Stable): {cohens_d:.3f}")
        print(f"  Effect size interpretation: {'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'}")

        return domain_stats

    def analyze_size_patterns_comprehensive(self) -> Dict:
        """
        Statistical analysis of size-specific patterns.
        """
        print("\n" + "="*80)
        print("SIZE-SPECIFIC PERFORMANCE PATTERNS - STATISTICAL ANALYSIS")
        print("="*80)

        size_stats = {}

        for size_cat in ['Small', 'Medium', 'Large']:
            size_data = self.df[self.df['size_category'] == size_cat]
            total = len(size_data)

            improvements = len(size_data[size_data['change_type'] == 'Improvement'])
            regressions = len(size_data[size_data['change_type'] == 'Regression'])
            unchanged = len(size_data[size_data['change_type'] == 'Unchanged'])

            imp_effects = size_data[size_data['change_type'] == 'Improvement']['effect_size']
            reg_effects = size_data[size_data['change_type'] == 'Regression']['effect_size'].abs()

            size_stats[size_cat] = {
                'total_changes': total,
                'improvement_count': improvements,
                'regression_count': regressions,
                'unchanged_count': unchanged,
                'improvement_rate': improvements / total * 100,
                'regression_rate': regressions / total * 100,
                'instability_rate': (improvements + regressions) / total * 100,
                'improvement_mean_effect': imp_effects.mean() if len(imp_effects) > 0 else 0,
                'improvement_std_effect': imp_effects.std() if len(imp_effects) > 0 else 0,
                'regression_mean_effect': reg_effects.mean() if len(reg_effects) > 0 else 0,
                'regression_std_effect': reg_effects.std() if len(reg_effects) > 0 else 0,
                'mean_kloc': size_data['size_kloc'].mean(),
                'median_kloc': size_data['size_kloc'].median()
            }

        print(f"\nSIZE CATEGORY BREAKDOWN:")
        for size_cat, stats in size_stats.items():
            print(f"\n{size_cat.upper()} PROJECTS (n={stats['total_changes']}):")
            print(f"  Mean size: {stats['mean_kloc']:.1f} KLOC | Median: {stats['median_kloc']:.1f} KLOC")
            print(f"  Improvements: {stats['improvement_count']} ({stats['improvement_rate']:.1f}%)")
            print(f"  Regressions: {stats['regression_count']} ({stats['regression_rate']:.1f}%)")
            print(f"  Instability: {stats['instability_rate']:.1f}%")
            print(f"  Improvement effect: μ={stats['improvement_mean_effect']:.3f}, σ={stats['improvement_std_effect']:.3f}")
            print(f"  Regression effect: μ={stats['regression_mean_effect']:.3f}, σ={stats['regression_std_effect']:.3f}")

        contingency_data_size = []
        size_names = ['Small', 'Medium', 'Large']

        for size_cat in size_names:
            stats = size_stats[size_cat]
            contingency_data_size.append([stats['improvement_count'], stats['regression_count'], stats['unchanged_count']])

        chi2_size, p_size, _, _ = chi2_contingency(contingency_data_size)
        n_size = sum(stats['total_changes'] for stats in size_stats.values())
        cramers_v_size = np.sqrt(chi2_size / (n_size * (min(len(contingency_data_size), len(contingency_data_size[0])) - 1)))  # type: ignore

        print(f"\nSTATISTICAL SIGNIFICANCE ACROSS SIZES:")
        print(f"  Chi-square statistic: {chi2_size:.3f}")
        print(f"  p-value: {p_size:.6f}")
        print(f"  Cramér's V: {cramers_v_size:.3f}")
        print(f"  Significant difference: {'Yes' if p_size < 0.05 else 'No'}")  # type: ignore

        return size_stats

    def analyze_domain_size_interaction(self) -> Dict:
        """
        Analyze interaction effects between domain and size.
        """
        print("\n" + "="*80)
        print("DOMAIN-SIZE INTERACTION ANALYSIS")
        print("="*80)

        interaction_stats = {}

        unique_domains = self.df['domain'].unique()
        unique_sizes = ['Small', 'Medium', 'Large']

        groups = []
        group_labels = []

        for domain in unique_domains:
            if domain == 'Unknown':
                continue
            for size in unique_sizes:
                group_data = self.df[(self.df['domain'] == domain) & (self.df['size_category'] == size)]
                if len(group_data) >= 5:  # Minimum sample size
                    groups.append(group_data['effect_size'].abs().values)
                    group_labels.append(f"{domain}-{size}")

        if len(groups) >= 3:
            f_stat, p_anova = f_oneway(*groups)

            print(f"INTERACTION EFFECTS ANALYSIS:")
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  p-value: {p_anova:.6f}")
            print(f"  Significant interaction: {'Yes' if p_anova < 0.05 else 'No'}")

        print(f"\nDETAILED INTERACTION PATTERNS:")

        for domain in unique_domains:
            if domain == 'Unknown':
                continue

            print(f"\n{domain.upper()}:")
            domain_interaction = {}

            for size in unique_sizes:
                combo_data = self.df[(self.df['domain'] == domain) & (self.df['size_category'] == size)]
                if len(combo_data) > 0:
                    total = len(combo_data)
                    improvements = len(combo_data[combo_data['change_type'] == 'Improvement'])
                    regressions = len(combo_data[combo_data['change_type'] == 'Regression'])

                    imp_effects = combo_data[combo_data['change_type'] == 'Improvement']['effect_size']
                    reg_effects = combo_data[combo_data['change_type'] == 'Regression']['effect_size'].abs()

                    domain_interaction[size] = {
                        'total': total,
                        'improvement_rate': improvements / total * 100,
                        'regression_rate': regressions / total * 100,
                        'instability_rate': (improvements + regressions) / total * 100,
                        'imp_mean_effect': imp_effects.mean() if len(imp_effects) > 0 else 0,
                        'reg_mean_effect': reg_effects.mean() if len(reg_effects) > 0 else 0
                    }

                    print(f"  {size}: n={total}, Instability={domain_interaction[size]['instability_rate']:.1f}%, "
                          f"Imp={domain_interaction[size]['imp_mean_effect']:.3f}, "
                          f"Reg={domain_interaction[size]['reg_mean_effect']:.3f}")

            interaction_stats[domain] = domain_interaction

        return interaction_stats

    def plot_domain_proportional_analysis(self):
        """
        Create proportional analysis visualization for domains.
        """
        proportions_data = []

        for domain in self.df['domain'].unique():
            if domain == 'Unknown':
                continue

            domain_data = self.df[self.df['domain'] == domain]
            total = len(domain_data)

            improvements = len(domain_data[domain_data['change_type'] == 'Improvement'])
            regressions = len(domain_data[domain_data['change_type'] == 'Regression'])
            unchanged = len(domain_data[domain_data['change_type'] == 'Unchanged'])

            proportions_data.append({
                'Domain': domain,
                'Improvements': improvements / total * 100,
                'Regressions': regressions / total * 100,
                'Unchanged': unchanged / total * 100,
                'Total': total,
                'Instability': (improvements + regressions) / total * 100
            })

        proportions_df = pd.DataFrame(proportions_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 10))

        # Left plot: Proportional stacked bars
        domains = proportions_df['Domain']
        improvements = proportions_df['Improvements']
        regressions = proportions_df['Regressions']
        unchanged = proportions_df['Unchanged']

        width = 0.75
        ax1.bar(domains, improvements, width, label='Improvements', color='#27ae60')
        ax1.bar(domains, regressions, width, bottom=improvements, label='Regressions', color='#e74c3c')
        ax1.bar(domains, unchanged, width, bottom=improvements + regressions, label='Unchanged', color='#3498db')

        for i, (imp, reg, unc) in enumerate(zip(improvements, regressions, unchanged)):
            if imp > 8:
                ax1.text(i, imp / 2, f"{imp:.1f}%", ha='center', va='center',
                         color='white', fontweight='bold', fontsize=24)
            if reg > 2:
                ax1.text(i, imp + reg / 2, f"{reg:.1f}%", ha='center', va='center',
                         color='white', fontweight='bold', fontsize=24)
            if unc > 8:
                ax1.text(i, imp + reg + unc / 2, f"{unc:.1f}%", ha='center', va='center',
                         color='white', fontweight='bold', fontsize=24)

        ax1.set_ylabel('Percentage of Changes (%)', fontsize=26, labelpad=20)
        ax1.set_xlabel('Project Domain', fontsize=26, labelpad=20)
        ax1.set_title('Performance Impact Distribution by Domain', fontsize=30, pad=20)
        ax1.set_xticks(range(len(domains)))
        ax1.set_xticklabels(domains, fontsize=26, rotation=45, ha='right')
        ax1.set_yticks(np.arange(0, 101, 20))
        ax1.set_yticklabels([f"{tick}%" for tick in np.arange(0, 101, 20)], fontsize=24)
        self._set_standard_legend_style(ax1, title='Change Type')

        # Right plot: Instability rates
        instability_bars = ax2.bar(domains, proportions_df['Instability'], color='#9b59b6', alpha=0.7)
        ax2.set_ylabel('Performance Instability Rate (%)', fontsize=26, labelpad=20)
        ax2.set_xlabel('Project Domain', fontsize=26, labelpad=20)
        ax2.set_title('Performance Instability by Domain', fontsize=30, pad=20)
        ax2.set_xticks(range(len(domains)))
        ax2.set_xticklabels(domains, fontsize=26, rotation=45, ha='right')
        ax2.set_yticks(np.arange(0, max(proportions_df['Instability']) + 10, 10))
        ax2.set_yticklabels([f"{tick:.0f}%" for tick in np.arange(0, max(proportions_df['Instability']) + 10, 10)], fontsize=24)

        for i, (domain, total, instability) in enumerate(zip(domains, proportions_df['Total'], proportions_df['Instability'])):
            ax2.text(i, instability + 1, f"n={total}", ha='center', va='bottom',
                     fontweight='bold', fontsize=22, color='black')

        plt.tight_layout()
        self.save_plot(plt, 'domain_proportional_analysis')

        return proportions_df

    def plot_size_proportional_analysis(self):
        """
        Create proportional analysis visualization for size categories.
        """
        proportions_data = []

        for size_cat in ['Small', 'Medium', 'Large']:
            size_data = self.df[self.df['size_category'] == size_cat]
            total = len(size_data)

            improvements = len(size_data[size_data['change_type'] == 'Improvement'])
            regressions = len(size_data[size_data['change_type'] == 'Regression'])
            unchanged = len(size_data[size_data['change_type'] == 'Unchanged'])

            proportions_data.append({
                'Size': size_cat,
                'Improvements': improvements / total * 100,
                'Regressions': regressions / total * 100,
                'Unchanged': unchanged / total * 100,
                'Total': total,
                'Instability': (improvements + regressions) / total * 100,
                'Mean_KLOC': size_data['size_kloc'].mean()
            })

        proportions_df = pd.DataFrame(proportions_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

        # Left plot: Proportional stacked bars
        sizes = proportions_df['Size']
        improvements = proportions_df['Improvements']
        regressions = proportions_df['Regressions']
        unchanged = proportions_df['Unchanged']

        width = 0.6
        ax1.bar(sizes, improvements, width, label='Improvements', color='#27ae60')
        ax1.bar(sizes, regressions, width, bottom=improvements, label='Regressions', color='#e74c3c')
        ax1.bar(sizes, unchanged, width, bottom=improvements + regressions, label='Unchanged', color='#3498db')

        for i, (imp, reg, unc) in enumerate(zip(improvements, regressions, unchanged)):
            if imp > 5:
                ax1.text(i, imp / 2, f"{imp:.1f}%", ha='center', va='center',
                         color='white', fontweight='bold', fontsize=24)
            if reg > 5:
                ax1.text(i, imp + reg / 2, f"{reg:.1f}%", ha='center', va='center',
                         color='white', fontweight='bold', fontsize=24)
            if unc > 5:
                ax1.text(i, imp + reg + unc / 2, f"{unc:.1f}%", ha='center', va='center',
                         color='white', fontweight='bold', fontsize=24)

        ax1.set_ylabel('Percentage of Changes (%)', fontsize=26, labelpad=20)
        ax1.set_xlabel('Project Size Category', fontsize=26, labelpad=20)
        ax1.set_title('Performance Impact Distribution by Project Size', fontsize=30, pad=20)
        ax1.set_xticks(range(len(sizes)))
        ax1.set_xticklabels(sizes, fontsize=24)
        ax1.set_yticks(np.arange(0, 101, 20))
        ax1.set_yticklabels([f"{tick}%" for tick in np.arange(0, 101, 20)], fontsize=24)
        self._set_standard_legend_style(ax1, title='Change Type')

        # Right plot: Size characteristics
        instability_bars = ax2.bar(sizes, proportions_df['Instability'], width=0.8,
                                   color='#9b59b6', alpha=0.7, label='Instability Rate')

        for i, (size, total, instability) in enumerate(zip(sizes, proportions_df['Total'], proportions_df['Instability'])):
            ax2.text(i, instability + 1, f"n={total}", ha='center', va='bottom',
                     fontweight='bold', fontsize=22, color='black')

        ax2.set_ylabel('Performance Instability Rate (%)', fontsize=26, labelpad=20)
        ax2.set_xlabel('Project Size Category', fontsize=26, labelpad=20)
        ax2.set_title('Size Characteristics and Performance Instability', fontsize=30, pad=20)
        ax2.set_xticks(range(len(sizes)))
        ax2.set_xticklabels(sizes, fontsize=24)
        ax2.set_ylim(0, max(proportions_df['Instability']) * 1.1)
        ax2.set_yticks(np.arange(0, max(proportions_df['Instability']) + 5, 5))
        ax2.set_yticklabels([f"{tick:.0f}%" for tick in np.arange(0, max(proportions_df['Instability']) + 5, 5)], fontsize=24)

        self._set_standard_legend_style(ax2, title='Size Characteristics')

        plt.tight_layout()
        self.save_plot(plt, 'size_proportional_analysis')

        return proportions_df

    def plot_domain_size_interaction_heatmap(self):
        """
        Create interaction heatmap showing domain-size combinations.
        """
        domains = [d for d in self.df['domain'].unique() if d != 'Unknown']
        sizes = ['Small', 'Medium', 'Large']

        labels_mapping = {
            'Monitoring': 'MON',
            'System Programming': 'SYS',
            'Data Processing': 'DP',
            'Web Server': 'WS',
            'Networking': 'NET',
            'Testing': 'TEST'
        }

        instability_matrix = np.zeros((len(domains), len(sizes)))
        sample_size_matrix = np.zeros((len(domains), len(sizes)))
        improvement_matrix = np.zeros((len(domains), len(sizes)))
        regression_matrix = np.zeros((len(domains), len(sizes)))

        for i, domain in enumerate(domains):
            for j, size in enumerate(sizes):
                combo_data = self.df[(self.df['domain'] == domain) & (self.df['size_category'] == size)]
                if len(combo_data) > 0:
                    total = len(combo_data)
                    improvements = len(combo_data[combo_data['change_type'] == 'Improvement'])
                    regressions = len(combo_data[combo_data['change_type'] == 'Regression'])

                    instability_matrix[i, j] = (improvements + regressions) / total * 100
                    sample_size_matrix[i, j] = total
                    improvement_matrix[i, j] = improvements / total * 100
                    regression_matrix[i, j] = regressions / total * 100
                else:
                    instability_matrix[i, j] = np.nan
                    sample_size_matrix[i, j] = 0
                    improvement_matrix[i, j] = np.nan
                    regression_matrix[i, j] = np.nan

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(28, 24))

        # Instability heatmap
        sns.heatmap(instability_matrix, xticklabels=sizes, yticklabels=domains,
                    annot=True, fmt='.1f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'Instability Rate (%)'})
        ax1.set_title('Performance Instability Rate\nby Domain-Size', fontsize=30, pad=20)
        ax1.set_xlabel('Project Size Category', fontsize=26, labelpad=20)
        ax1.set_ylabel('Project Domain', fontsize=26, labelpad=20)
        ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=24)
        new_yticks = [labels_mapping.get(d, d) for d in domains]
        ax1.set_yticklabels(new_yticks, fontsize=24)

        # Sample size heatmap
        sns.heatmap(sample_size_matrix, xticklabels=sizes, yticklabels=domains,
                    annot=True, fmt='.0f', cmap='Blues', ax=ax2, cbar_kws={'label': 'Sample Size'})
        ax2.set_title('Sample Size\nby Domain-Size', fontsize=30, pad=20)
        ax2.set_xlabel('Project Size Category', fontsize=26, labelpad=20)
        ax2.set_ylabel('Project Domain', fontsize=26, labelpad=20)
        ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=24)
        ax2.set_yticklabels(new_yticks, fontsize=24)

        # Improvement heatmap
        sns.heatmap(improvement_matrix, xticklabels=sizes, yticklabels=domains,
                    annot=True, fmt='.1f', cmap='Greens', ax=ax3, cbar_kws={'label': 'Improvement Rate (%)'})
        ax3.set_title('Performance Improvement Rate\nby Domain-Size', fontsize=30, pad=20)
        ax3.set_xlabel('Project Size Category', fontsize=26, labelpad=20)
        ax3.set_ylabel('Project Domain', fontsize=26, labelpad=20)
        ax3.set_xticklabels(ax3.get_xticklabels(), fontsize=24)
        ax3.set_yticklabels(new_yticks, fontsize=24)

        # Regression heatmap
        sns.heatmap(regression_matrix, xticklabels=sizes, yticklabels=domains,
                    annot=True, fmt='.1f', cmap='Reds', ax=ax4, cbar_kws={'label': 'Regression Rate (%)'})
        ax4.set_title('Performance Regression Rate\nby Domain-Size', fontsize=30, pad=20)
        ax4.set_xlabel('Project Size Category', fontsize=26, labelpad=20)
        ax4.set_ylabel('Project Domain', fontsize=26, labelpad=20)
        ax4.set_xticklabels(ax4.get_xticklabels(), fontsize=24)
        ax4.set_yticklabels(new_yticks, fontsize=24)

        for ax in [ax1, ax2, ax3, ax4]:
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=24)
            cbar.set_label(cbar.ax.get_ylabel(), fontsize=24, labelpad=20)

            for text in ax.texts:
                text.set_fontsize(28)
                text.set_fontweight('bold')

        plt.subplots_adjust(wspace=0.25, hspace=0.35)

        self.save_plot(plt, 'domain_size_interaction_heatmap')

        return {
            'instability_matrix': instability_matrix,
            'sample_size_matrix': sample_size_matrix,
            'improvement_matrix': improvement_matrix,
            'regression_matrix': regression_matrix,
            'domains': domains,
            'sizes': sizes
        }

    def generate_comprehensive_rq4_analysis(self):
        """
        Run RQ4 analysis and generate all statistics.
        """
        print("="*100)
        print("RQ4: COMPREHENSIVE DOMAIN AND SIZE PERFORMANCE PATTERN ANALYSIS")
        print("="*100)

        # Domain analysis
        domain_stats = self.analyze_domain_patterns_comprehensive()

        # Size analysis
        size_stats = self.analyze_size_patterns_comprehensive()

        # Interaction analysis
        interaction_stats = self.analyze_domain_size_interaction()

        print(f"\nGENERATING VISUALIZATIONS...")
        domain_props = self.plot_domain_proportional_analysis()
        size_props = self.plot_size_proportional_analysis()
        interaction_data = self.plot_domain_size_interaction_heatmap()

        print(f"\n" + "="*80)
        print("SUMMARY STATISTICS FOR PAPER WRITING")
        print("="*80)

        total_changes = len(self.df)
        total_domains = len([d for d in self.df['domain'].unique() if d != 'Unknown'])

        print(f"Dataset overview:")
        print(f"  Total method changes analyzed: {total_changes:,}")
        print(f"  Number of domains: {total_domains}")
        print(f"  Size categories: 3 (Small, Medium, Large)")

        domain_volatility = {domain: stats['improvement_std_effect'] + stats['regression_std_effect']
                             for domain, stats in domain_stats.items()}
        most_stable = min(domain_volatility.items(), key=lambda x: x[1])
        most_volatile = max(domain_volatility.items(), key=lambda x: x[1])

        print(f"\nKey findings:")
        print(f"  Most stable domain: {most_stable[0]} (volatility: {most_stable[1]:.3f})")
        print(f"  Most volatile domain: {most_volatile[0]} (volatility: {most_volatile[1]:.3f})")
        print(f"  Volatility difference: {(most_volatile[1] / most_stable[1] - 1) * 100:.1f}% higher")

        # Size effects
        size_instability = {size: stats['instability_rate'] for size, stats in size_stats.items()}
        print(f"  Size instability rates: Small={size_instability['Small']:.1f}%, "
              f"Medium={size_instability['Medium']:.1f}%, Large={size_instability['Large']:.1f}%")

        return {
            'domain_stats': domain_stats,
            'size_stats': size_stats,
            'interaction_stats': interaction_stats,
            'domain_props': domain_props,
            'size_props': size_props,
            'interaction_data': interaction_data
        }


if __name__ == "__main__":
    dataset = os.path.join('dataset', 'dataset.csv')
    output_dir = os.path.join('plots', 'RQ4')
    df = pd.read_csv(dataset)

    visualizer = RQ4(df, output_dir)
    visualizer.generate_comprehensive_rq4_analysis()
