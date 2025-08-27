"""
MS MARCO dataset evaluation package.

This package provides tools for loading, processing, and evaluating
MS MARCO datasets with comprehensive retrieval and generation metrics.
"""

from .dataframe_dataset import DataFrameDataset
from .prepare_ms_marco import MSMarcoDataset
from .run_marco_eval import run_marco_eval

__all__ = ['DataFrameDataset', 'MSMarcoDataset', 'run_marco_eval']
