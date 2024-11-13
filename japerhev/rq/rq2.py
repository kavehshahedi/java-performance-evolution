from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os

class RQ2:
    def __init__(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Plotting class for RQ2 visualizations.

        :param df: DataFrame containing the method performance data
        :type df: pd.DataFrame
        :param output_dir: Directory to save the plots
        :type output_dir: str
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
        :param filename: Name of the file to save the plot
        :type filename: str
        """
        filepath = os.path.join(self.output_dir, f"{filename}.pdf")
        plt.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        # plt.show()
        
    def _preprocess_data(self):
        """Preprocess the data for analysis."""
        # Drop invalid rows (i.e., duplicates)
        self.df = self.df.dropna(subset=['effect_size'])
        self.df = self.df.reindex(self.df['effect_size'].abs().sort_values().index).drop_duplicates(['project_id', 'commit_id', 'method_name'], keep='first')

        # Create experience categories
        self.df['experience_category'] = pd.qcut(
            self.df['experience'],
            q=3,
            labels=['Junior', 'Mid', 'Senior']
        )
        
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
            pivot_table.reset_index(), # type: ignore
            headers=['Experience Level', 'Improvement', 'Regression', 'Unchanged', 'Total Data Points'],
            tablefmt='grid',
            floatfmt='.2f'
        )
        
        print("\nExperience Level vs Change Type Distribution (%):")
        print(table)
        
    def plot_code_change_impact(self):
        """Create boxplot for code change label impact on effect size."""
        # Filter out rows with null code_change_label
        df_valid = self.df_expanded[self.df_expanded['code_change_label'] != 'Unknown']

        # Map the x-axis labels to more readable format
        label_map = {
            'Algorithmic Change': 'ALG',
            'Control Flow/Loop Changes': 'CF',
            'Data Structure & Variable Changes': 'DS',
            'Refactoring & Code Cleanup': 'REF',
            'Exception & Input/Output Handling': 'IO',
            'Concurrency/Parallelism': 'CON',
            'API/Library Call Changes': 'API'
        }
        
        plt.figure(figsize=(21, 5), dpi=300)

        # Set custom colors for each change type
        improvement_color = '#27ae60'
        regression_color = '#e74c3c'
        unchanged_color = '#3498db'
        colors = [unchanged_color, regression_color, improvement_color]
        sns.boxplot(
            data=df_valid,
            x='code_change_label',
            y='effect_size',
            hue='change_type',
            palette=colors,
        )

        # Customize appearance
        plt.xticks(
            ticks=plt.xticks()[0], # type: ignore
            labels=[label_map[label.get_text()] for label in plt.gca().get_xticklabels()],
        )

        plt.xlabel('Code Change Type')
        plt.ylabel('Effect Size')
        # plt.title('Effect Size Distribution by Code Change Type')
        self._set_standard_legend_style(plt.gca())
        plt.tight_layout()
        
        # Save the plot
        self.save_plot(plt, 'code_change_impact')
        
    def plot_complexity_impact(self):
        """Create improved visualization for method change complexity impact."""
        # Filter valid complexity values
        df_valid = df[df['method_change_complexity'] != -1].copy()
        
        # Create complexity categories
        df_valid['complexity_category'] = pd.qcut(
            df_valid['method_change_complexity'],
            q=4,
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        plt.figure(figsize=(12, 5), dpi=300)
        
        # Set custom colors for each change type
        improvement_color = '#27ae60'
        regression_color = '#e74c3c'
        unchanged_color = '#3498db'
        colors = [unchanged_color, improvement_color, regression_color]
        sns.boxplot(
            data=df_valid,
            x='complexity_category',
            y='effect_size',
            hue='change_type',
            palette=colors,
        )
        
        # Customize appearance
        plt.xlabel('Method Change Complexity')
        plt.ylabel('Effect Size')
        # plt.title('Effect Size Distribution by Complexity Level', fontsize=14, pad=20)
        self._set_standard_legend_style(plt.gca())
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        self.save_plot(plt, 'complexity_impact')

if __name__ == "__main__":
    dataset = os.path.join('dataset', 'dataset.csv')
    output_dir = os.path.join('plots', 'RQ2')
    df = pd.read_csv(dataset)
    visualizer = RQ2(df, output_dir)
    
    visualizer.show_experience_change_distribution()
    visualizer.plot_code_change_impact()
    visualizer.plot_complexity_impact()