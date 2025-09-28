import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

logger = logging.getLogger(__name__)


class EvaluationPlotter:
    """Creates visualizations for RAG evaluation results."""

    def __init__(self, results_dir: str = "qdrant_ms_marco_small"):
        """Initialize the plotter with a results directory.
        
        Args:
            results_dir: Directory containing evaluation result JSON files
        """
        self.results_dir = Path(results_dir)
        # Set matplotlib style for clean plots
        plt.style.use('default')
        sns.set_palette("husl")

    def load_results(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Load evaluation results from JSON file.
        
        Args:
            filename: Specific file to load, or None for most recent
            
        Returns:
            Dictionary containing evaluation results
        """
        if filename is None:
            # Find most recent evaluation file
            json_files = list(self.results_dir.glob("evaluation_results_*.json"))
            if not json_files:
                raise FileNotFoundError(f"No evaluation results found in {self.results_dir}")
            filename = max(json_files, key=lambda x: x.stat().st_mtime)
        else:
            filename = self.results_dir / filename

        logger.info(f"Loading evaluation results from {filename}")
        with open(filename, 'r') as f:
            return json.load(f)

    def plot_retrieval_metrics(self, results: Dict[str, Any], figsize: tuple = (15, 10)) -> plt.Figure:
        """Plot comprehensive retrieval metrics overview.
        
        Args:
            results: Evaluation results dictionary
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        metrics = results.get('retrieval_metrics', {})
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Retrieval Metrics Overview', fontsize=16, fontweight='bold')
        
        # Precision@K, Recall@K, NDCG@K
        k_metrics = ['precision', 'recall', 'ndcg']
        metric_colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        for i, (metric, color) in enumerate(zip(k_metrics, metric_colors)):
            # Find all K values for this metric
            k_values = []
            metric_values = []
            
            for key, value in metrics.items():
                if key.startswith(f"{metric}@"):
                    try:
                        k = int(key.split("@")[1])
                        k_values.append(k)
                        metric_values.append(value)
                    except (ValueError, IndexError):
                        continue
            
            if k_values and metric_values:
                # Sort by K value
                sorted_data = sorted(zip(k_values, metric_values))
                k_values, metric_values = zip(*sorted_data)
                
                axes[0, i].bar(range(len(k_values)), metric_values, color=color, alpha=0.7)
                axes[0, i].set_title(f'{metric.title()}@K')
                axes[0, i].set_xlabel('K')
                axes[0, i].set_ylabel(metric.title())
                axes[0, i].set_xticks(range(len(k_values)))
                axes[0, i].set_xticklabels(k_values)
                
                # Add value labels on bars
                for j, v in enumerate(metric_values):
                    axes[0, i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Summary metrics
        summary_metrics = ['mrr', 'map', 'f1']
        summary_values = [metrics.get(metric, 0) for metric in summary_metrics]
        summary_labels = ['MRR', 'MAP', 'F1']
        
        bars = axes[1, 0].bar(summary_labels, summary_values, color='gold', alpha=0.7)
        axes[1, 0].set_title('Summary Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, summary_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Response time vs Performance
        avg_response_time = results.get('avg_response_time', 0)
        mrr_score = metrics.get('mrr', 0)
        
        axes[1, 1].scatter([avg_response_time], [mrr_score], s=100, color='red', alpha=0.7)
        axes[1, 1].set_xlabel('Avg Response Time (s)')
        axes[1, 1].set_ylabel('MRR Score')
        axes[1, 1].set_title('Performance vs Speed')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Coverage metrics
        coverage_metrics = {
            'Retrieved': metrics.get('retrieved_count', 0),
            'Relevant': metrics.get('relevant_count', 0),
            'True Positives': metrics.get('true_positives', 0)
        }
        
        bars = axes[1, 2].bar(coverage_metrics.keys(), coverage_metrics.values(), 
                             color='purple', alpha=0.7)
        axes[1, 2].set_title('Retrieval Coverage')
        axes[1, 2].set_ylabel('Average Count')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig

    def plot_generation_metrics(self, results: Dict[str, Any], figsize: tuple = (12, 8)) -> plt.Figure:
        """Plot generation quality metrics.
        
        Args:
            results: Evaluation results dictionary
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        metrics = results.get('generation_metrics', {})
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Generation Quality Metrics', fontsize=16, fontweight='bold')
        
        # Quality metrics
        quality_metrics = {
            'Answer Relevance': metrics.get('answer_relevance', 0),
            'Support Coverage': metrics.get('support_coverage', 0),
            'Support Density': metrics.get('support_density', 0)
        }
        
        bars = axes[0, 0].bar(quality_metrics.keys(), quality_metrics.values(), 
                             color='lightblue', alpha=0.7)
        axes[0, 0].set_title('Answer Quality Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, quality_metrics.values()):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Hallucination and accuracy
        hallucination_rate = metrics.get('hallucination_rate', 0)
        accuracy_rate = 1 - hallucination_rate
        
        labels = ['Accurate', 'Hallucinated']
        sizes = [accuracy_rate, hallucination_rate]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = axes[0, 1].pie(sizes, labels=labels, colors=colors, 
                                                 autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Answer Accuracy vs Hallucination')
        
        # Text metrics (ROUGE, F1, etc.)
        text_metrics = {k: v for k, v in metrics.items() 
                       if k in ['exact_match', 'f1', 'rouge_l_f1']}
        
        if text_metrics:
            bars = axes[1, 0].bar(text_metrics.keys(), text_metrics.values(),
                                 color='orange', alpha=0.7)
            axes[1, 0].set_title('Text Similarity Metrics')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, text_metrics.values()):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        # Answer length distribution
        avg_length = metrics.get('answer_len_tokens', 0)
        axes[1, 1].bar(['Average Answer Length'], [avg_length], color='gold', alpha=0.7)
        axes[1, 1].set_title('Answer Length (Tokens)')
        axes[1, 1].set_ylabel('Token Count')
        axes[1, 1].text(0, avg_length + 0.5, f'{avg_length:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig

    def plot_response_time_analysis(self, results: Dict[str, Any], figsize: tuple = (12, 6)) -> plt.Figure:
        """Plot response time analysis.
        
        Args:
            results: Evaluation results dictionary
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        response_times = results.get('response_times', [])
        
        if not response_times:
            logger.warning("No response times found in results")
            return plt.figure(figsize=figsize)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Response Time Analysis', fontsize=16, fontweight='bold')
        
        # Response time distribution
        axes[0].hist(response_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(np.mean(response_times), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(response_times):.2f}s')
        axes[0].axvline(np.median(response_times), color='orange', linestyle='--', 
                       label=f'Median: {np.median(response_times):.2f}s')
        axes[0].set_title('Response Time Distribution')
        axes[0].set_xlabel('Response Time (seconds)')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Response time trend
        axes[1].plot(range(len(response_times)), response_times, marker='o', alpha=0.7)
        axes[1].axhline(np.mean(response_times), color='red', linestyle='--', alpha=0.7)
        axes[1].set_title('Response Time Trend')
        axes[1].set_xlabel('Query Number')
        axes[1].set_ylabel('Response Time (seconds)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def create_evaluation_report(self, results: Optional[Dict[str, Any]] = None, 
                               output_dir: str = "evaluation_plots", 
                               filename: Optional[str] = None) -> Path:
        """Generate complete evaluation report with all visualizations.
        
        Args:
            results: Evaluation results dictionary (loaded if None)
            output_dir: Directory to save plots
            filename: Specific results file to load
            
        Returns:
            Path to output directory
        """
        if results is None:
            results = self.load_results(filename)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Generating evaluation report in {output_path}")
        
        # Generate all plots
        try:
            fig1 = self.plot_retrieval_metrics(results)
            fig1.savefig(output_path / "retrieval_metrics.png", dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            fig2 = self.plot_generation_metrics(results)
            fig2.savefig(output_path / "generation_metrics.png", dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            fig3 = self.plot_response_time_analysis(results)
            fig3.savefig(output_path / "response_time_analysis.png", dpi=300, bbox_inches='tight')
            plt.close(fig3)
            
            logger.info(f"Evaluation plots saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            raise
        
        return output_path

    def compare_results(self, result_files: List[str], figsize: tuple = (12, 8)) -> plt.Figure:
        """Compare metrics across multiple evaluation runs.
        
        Args:
            result_files: List of result filenames to compare
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        comparison_data = []
        
        for i, file in enumerate(result_files):
            try:
                results = self.load_results(file)
                row = {'run': f'Run {i+1} ({Path(file).stem.split("_")[-1]})'}
                
                # Retrieval metrics
                retrieval = results.get('retrieval_metrics', {})
                row.update({
                    'MRR': retrieval.get('mrr', 0),
                    'MAP': retrieval.get('map', 0),
                    'Precision@1': retrieval.get('precision@1', 0),
                    'F1': retrieval.get('f1', 0)
                })
                
                # Generation metrics
                generation = results.get('generation_metrics', {})
                row.update({
                    'Answer Relevance': generation.get('answer_relevance', 0),
                    'Hallucination Rate': generation.get('hallucination_rate', 0),
                    'Support Coverage': generation.get('support_coverage', 0)
                })
                
                # Performance
                row['Avg Response Time'] = results.get('avg_response_time', 0)
                
                comparison_data.append(row)
                
            except Exception as e:
                logger.warning(f"Could not load {file}: {e}")
                continue
        
        if not comparison_data:
            raise ValueError("No valid result files could be loaded")
        
        df = pd.DataFrame(comparison_data)
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Evaluation Results Comparison', fontsize=16, fontweight='bold')
        
        # Retrieval metrics comparison
        retrieval_cols = ['MRR', 'MAP', 'Precision@1', 'F1']
        df[retrieval_cols].plot(kind='bar', ax=axes[0, 0], alpha=0.7)
        axes[0, 0].set_title('Retrieval Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Generation metrics comparison
        generation_cols = ['Answer Relevance', 'Support Coverage']
        df[generation_cols].plot(kind='bar', ax=axes[0, 1], alpha=0.7)
        axes[0, 1].set_title('Generation Quality')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Hallucination comparison
        df['Hallucination Rate'].plot(kind='bar', ax=axes[1, 0], color='red', alpha=0.7)
        axes[1, 0].set_title('Hallucination Rate')
        axes[1, 0].set_ylabel('Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Response time comparison
        df['Avg Response Time'].plot(kind='bar', ax=axes[1, 1], color='orange', alpha=0.7)
        axes[1, 1].set_title('Average Response Time')
        axes[1, 1].set_ylabel('Seconds')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig


def plot_latest_results(results_dir: str = "qdrant_ms_marco_small") -> Dict[str, Any]:
    """Convenience function to plot the latest evaluation results.
    
    Args:
        results_dir: Directory containing result files
        
    Returns:
        The loaded results dictionary
    """
    plotter = EvaluationPlotter(results_dir)
    results = plotter.load_results()
    
    # Display plots
    fig1 = plotter.plot_retrieval_metrics(results)
    plt.show()
    
    fig2 = plotter.plot_generation_metrics(results)
    plt.show()
    
    fig3 = plotter.plot_response_time_analysis(results)
    plt.show()
    
    return results