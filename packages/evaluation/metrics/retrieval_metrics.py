"""
Evaluation Metrics Calculator

Standard information retrieval evaluation metrics implementation.
"""

import math


class RetrievalMetrics:
    """Metrics for evaluating retrieval performance in RAG systems."""

    @staticmethod
    def precision_at_k(
        retrieved_docs: list[str],
        relevant_docs: list[str],
        k: int,
    ) -> float:
        """
        Calculate precision@k metric.

        Parameters:
            retrieved_docs (list[str]): List of retrieved document IDs
                in rank order
            relevant_docs (list[str]): List of relevant document IDs
            k (int): Rank position to evaluate precision at

        Returns:
            float: Precision@k score (0.0-1.0)
        """
        if not retrieved_docs or k <= 0:
            return 0.0

        # Convert relevant_docs to set for faster lookup
        relevant_set = set(relevant_docs)

        # Only consider the top k results
        top_k = retrieved_docs[:k]

        # Count relevant docs in top k
        relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_set)

        return relevant_count / min(k, len(retrieved_docs))

    @staticmethod
    def recall_at_k(
        retrieved_docs: list[str],
        relevant_docs: list[str],
        k: int,
    ) -> float:
        """
        Calculate recall@k metric.

        Parameters:
            retrieved_docs (list[str]): List of retrieved document IDs
                in rank order
            relevant_docs (list[str]): List of relevant document IDs
            k (int): Rank position to evaluate recall at

        Returns:
            float: Recall@k score (0.0-1.0)
        """
        if not retrieved_docs or not relevant_docs or k <= 0:
            return 0.0

        # Convert relevant_docs to set for faster lookup
        relevant_set = set(relevant_docs)

        # Only consider the top k results
        top_k = retrieved_docs[:k]

        # Count relevant docs in top k
        relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_set)

        return relevant_count / len(relevant_set)

    @staticmethod
    def mean_reciprocal_rank(
        retrieved_docs: list[str],
        relevant_docs: list[str],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR) metric.

        Parameters:
            retrieved_docs (list[str]): List of retrieved document IDs
                in rank order
            relevant_docs (list[str]): List of relevant document IDs

        Returns:
            float: MRR score (0.0-1.0)
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0

        # Convert relevant_docs to set for faster lookup
        relevant_set = set(relevant_docs)

        # Find the rank of the first relevant document
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_set:
                return 1.0 / (i + 1)  # +1 because ranks start at 1

        return 0.0  # No relevant document found

    @staticmethod
    def ndcg_at_k(
        retrieved_docs: list[str],
        relevance_scores: dict[str, float],
        k: int,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG) at rank k.

        Parameters:
            retrieved_docs (list[str]): List of retrieved document IDs
                in rank order
            relevance_scores (dict[str, float]): Dictionary mapping doc_ids
                to relevance scores
            k (int): Rank position to evaluate NDCG at

        Returns:
            float: NDCG@k score (0.0-1.0)
        """
        if not retrieved_docs or not relevance_scores or k <= 0:
            return 0.0

        # Only consider the top k results
        top_k = retrieved_docs[: min(k, len(retrieved_docs))]

        # Calculate DCG - sum of (relevance / log2(rank+1))
        dcg = 0.0
        for i, doc_id in enumerate(top_k):
            rel = relevance_scores.get(doc_id, 0.0)
            # Add 2 to i because:
            #  - 1 for converting 0-index to 1-index
            #  - 1 for the log base conversion helper
            dcg += rel / math.log2(i + 2)

        # Calculate ideal DCG (IDCG)
        # Sort relevance scores in descending order and take top k
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances):
            idcg += rel / math.log2(i + 2)

        # Avoid division by zero
        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def aqwv(
        retrieved_docs: list[str],
        relevant_docs: list[str],
        beta: float = 40.0,
        collection_size: int = -1,
    ) -> float:
        """
        Calculate Average Query Weighted Value (AQWV) metric.

        Formula: AQWV = 1 - pMiss - β * pFA

        Parameters:
            retrieved_docs (list[str]): List of retrieved document IDs
                in rank order
            relevant_docs (list[str]): List of relevant document IDs
            beta (float): Weight factor for false alarms
                (default: 40.0 from MATERIAL)
            collection_size (int): Total size of document collection

        Returns:
            float: AQWV score

        Raises:
            ValueError: If collection_size is not positive
        """

        if collection_size <= 0:
            raise ValueError(f"collection_size must be positive, got {collection_size}")

        if not relevant_docs:
            # If no relevant docs exist for this query, pMiss = 0
            # Only false alarms matter
            num_false_alarms = len(retrieved_docs)
            p_fa = num_false_alarms / collection_size if collection_size > 0 else 0.0
            return 1.0 - beta * p_fa

        # Convert to sets for efficient operations
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)

        # Calculate misses and false alarms
        true_positives = len(retrieved_set.intersection(relevant_set))
        num_misses = len(relevant_set) - true_positives
        num_false_alarms = len(retrieved_set) - true_positives

        # Calculate rates
        p_miss = num_misses / len(relevant_set)
        p_fa = (
            num_false_alarms / (collection_size - len(relevant_set))
            if (collection_size - len(relevant_set)) > 0
            else 0.0
        )

        # Calculate AQWV
        aqwv = 1.0 - p_miss - beta * p_fa

        return aqwv

    @staticmethod
    def average_precision(
        retrieved_docs: list[str],
        relevant_docs: list[str],
    ) -> float:
        """Calculate Average Precision (AP) for a single query.

        Args:
            retrieved_docs: List of retrieved document IDs in rank order
            relevant_docs: List of relevant document IDs

        Returns:
            Average Precision score
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0

        relevant_set = set(relevant_docs)
        precision_sum = 0.0
        relevant_retrieved = 0

        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_set:
                relevant_retrieved += 1
                precision_at_i = relevant_retrieved / (i + 1)
                precision_sum += precision_at_i

        return precision_sum / len(relevant_docs)

    @staticmethod
    def mean_average_precision(
        retrieved_docs_per_query: list[list[str]],
        relevant_docs_per_query: list[list[str]],
    ) -> float:
        """Calculate Mean Average Precision (MAP) across multiple queries.

        Args:
            retrieved_docs_per_query: List of retrieved doc lists for each query
            relevant_docs_per_query: List of relevant doc lists for each query

        Returns:
            MAP score
        """
        if not retrieved_docs_per_query or not relevant_docs_per_query:
            return 0.0

        if len(retrieved_docs_per_query) != len(relevant_docs_per_query):
            raise ValueError("Retrieved and relevant docs lists must have same length")

        ap_scores = []
        for retrieved_docs, relevant_docs in zip(
            retrieved_docs_per_query, relevant_docs_per_query, strict=False
        ):
            ap = RetrievalMetrics.average_precision(retrieved_docs, relevant_docs)
            ap_scores.append(ap)

        return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
