import argparse
import logging
import sys
from pathlib import Path

from dataset.visualization import EvaluationPlotter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main plotting script entry point."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots from RAG evaluation results"
    )
    parser.add_argument(
        "--results-dir",
        default="qdrant_ms_marco_small",
        help="Directory containing evaluation result files",
    )
    parser.add_argument(
        "--results-file", help="Specific evaluation results file to plot"
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_plots",
        help="Output directory for generated plots",
    )
    parser.add_argument(
        "--compare", nargs="+", help="Compare multiple result files (provide filenames)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (in addition to saving)",
    )

    args = parser.parse_args()

    try:
        plotter = EvaluationPlotter(args.results_dir)

        if args.compare:
            logger.info(f"Comparing results from {len(args.compare)} files")
            fig = plotter.compare_results(args.compare)

            output_path = Path(args.output_dir)
            output_path.mkdir(exist_ok=True)
            fig.savefig(output_path / "comparison.png", dpi=300, bbox_inches="tight")

            if args.show:
                import matplotlib.pyplot as plt

                plt.show()
            else:
                plt.close(fig)

            logger.info(f"Comparison plot saved to {output_path / 'comparison.png'}")

        else:
            # Generate full evaluation report
            results = plotter.load_results(args.results_file)
            output_path = plotter.create_evaluation_report(
                results, args.output_dir, args.results_file
            )

            if args.show:
                # Load and display plots
                import matplotlib.pyplot as plt
                from matplotlib.image import imread

                plot_files = list(output_path.glob("*.png"))
                if plot_files:
                    fig, axes = plt.subplots(
                        len(plot_files), 1, figsize=(12, 8 * len(plot_files))
                    )
                    if len(plot_files) == 1:
                        axes = [axes]

                    for ax, plot_file in zip(axes, plot_files, strict=False):
                        img = imread(plot_file)
                        ax.imshow(img)
                        ax.set_title(plot_file.stem.replace("_", " ").title())
                        ax.axis("off")

                    plt.tight_layout()
                    plt.show()

            # Print summary
            retrieval_metrics = results.get("retrieval_metrics", {})
            generation_metrics = results.get("generation_metrics", {})

            print(f"\n{'=' * 50}")
            print("EVALUATION SUMMARY")
            print(f"{'=' * 50}")
            print(f"Results file: {args.results_file or 'latest'}")
            print(f"Plots saved to: {output_path}")
            print("\nKey Metrics:")
            print(f"  MRR: {retrieval_metrics.get('mrr', 0):.3f}")
            print(f"  MAP: {retrieval_metrics.get('map', 0):.3f}")
            print(f"  Precision@1: {retrieval_metrics.get('precision@1', 0):.3f}")
            print(
                f"  Answer Relevance: "
                f"{generation_metrics.get('answer_relevance', 0):.3f}"
            )
            print(
                f"  Hallucination Rate: "
                f"{generation_metrics.get('hallucination_rate', 0):.3f}"
            )
            print(f"  Avg Response Time: {results.get('avg_response_time', 0):.2f}s")
            print(f"{'=' * 50}")

    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
