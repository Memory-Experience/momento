import re
import string
from collections import Counter

import numpy as np

from api.models.embedding.embedding_model_interface import EmbeddingModel
from .cross_encoder_scorer import CrossEncoderScorer


# Generation metrics
class GenerationMetrics:
    """
    Collection of metrics for evaluating text generation quality.

    Provides both answer correctness metrics (exact match, F1, ROUGE-L)
    and generation quality metrics (relevance, faithfulness, similarity).
    Includes methods for normalizing text and detecting hallucinations.
    """

    _ARTICLES = {"a", "an", "the"}
    _PUNCT_TABLE = str.maketrans("", "", string.punctuation)
    _THINKING_TAG_PATTERN = r"<think>.*?</think>"
    _SOURCE_TAG_PATTERN = r"<source>.*?</source>"

    @staticmethod
    def empty_gold_answer_guard(gold_answers: list[str]) -> bool:
        return len(gold_answers) == 0 or all(
            (g is None or g.strip() == "" or g.strip() == "()") for g in gold_answers
        )

    @staticmethod
    def _normalize(text: str) -> str:
        if text is None:
            return ""
        text = re.sub(
            GenerationMetrics._THINKING_TAG_PATTERN, "", text, flags=re.DOTALL
        )
        text = re.sub(GenerationMetrics._SOURCE_TAG_PATTERN, "", text, flags=re.DOTALL)
        text = text.lower()
        text = text.translate(GenerationMetrics._PUNCT_TABLE)
        text = re.sub(r"\s+", " ", text).strip()
        # remove articles (SQuAD style)
        tokens = [t for t in text.split() if t not in GenerationMetrics._ARTICLES]
        return " ".join(tokens)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return GenerationMetrics._normalize(text).split()

    # Correctness vs gold answers
    @staticmethod
    def exact_match(pred: str, gold_answers: list[str]) -> float:
        """
        Compute exact match score between prediction and gold answers.

        Parameters:
            pred (str): Predicted answer text
            gold_answers (list[str]): List of reference answer strings

        Returns:
            float: 1.0 if prediction matches any gold answer, else 0.0
        """
        pred_norm = GenerationMetrics._normalize(pred)
        return (
            1.0
            if any(pred_norm == GenerationMetrics._normalize(g) for g in gold_answers)
            else 0.0
        )

    @staticmethod
    def f1(pred: str, gold_answers: list[str]) -> float:
        """
        Compute token-level F1 score between prediction and gold answers.

        Parameters:
            pred (str): Predicted answer text
            gold_answers (list[str]): List of reference answer strings

        Returns:
            float: Maximum F1 score across all gold answers (0.0-1.0)
        """
        if GenerationMetrics.empty_gold_answer_guard(gold_answers):
            return 0.0

        def f1_pair(p: str, g: str) -> float:
            pt, gt = GenerationMetrics._tokens(p), GenerationMetrics._tokens(g)
            if not pt and not gt:
                return 1.0
            if not pt or not gt:
                return 0.0
            common = sum((Counter(pt) & Counter(gt)).values())
            if common == 0:
                return 0.0
            precision = common / len(pt)
            recall = common / len(gt)
            return 2 * precision * recall / (precision + recall)

        return max(f1_pair(pred, g) for g in gold_answers)

    @staticmethod
    def rouge_l_f1(pred: str, gold_answers: list[str]) -> float:
        """
        Compute ROUGE-L F1 score using longest common subsequence.

        Parameters:
            pred (str): Predicted answer text
            gold_answers (list[str]): List of reference answer strings

        Returns:
            float: Maximum ROUGE-L F1 score across all gold answers
        """
        if GenerationMetrics.empty_gold_answer_guard(gold_answers):
            return 0.0

        # LCS-based ROUGE-L F1 (single-ref max)
        def lcs(a: list[str], b: list[str]) -> int:
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                ai = a[i - 1]
                for j in range(1, n + 1):
                    if ai == b[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]

        pt = GenerationMetrics._tokens(pred)
        best = 0.0
        for g in gold_answers:
            gt = GenerationMetrics._tokens(g)
            if not pt or not gt:
                continue
            lcs_len = lcs(pt, gt)
            if lcs_len == 0:
                continue
            prec = lcs_len / len(pt)
            rec = lcs_len / len(gt)
            f1 = 2 * prec * rec / (prec + rec)
            best = max(best, f1)
        return best

    # Answer relevance to query (proxy when no judge available)
    @staticmethod
    def answer_relevance_to_query(answer: str, query: str) -> float:
        """
        Compute answer relevance to query using token F1.

        Provides a cheap topicality proxy by computing token-level F1
        between the answer and query text.

        Parameters:
            answer (str): Generated answer text
            query (str): Original query text

        Returns:
            float: Token F1 score between answer and query (0.0-1.0)
        """
        # token F1 between query and answer (cheap topicality proxy)
        return GenerationMetrics.f1(answer, [query])

    # Faithfulness to retrieved docs
    @staticmethod
    def _content_words(tokens: list[str]) -> list[str]:
        # keep non-stopword-ish tokens (very light filter)
        stopish = GenerationMetrics._ARTICLES | {
            "of",
            "in",
            "on",
            "for",
            "to",
            "and",
            "or",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "with",
            "by",
            "as",
            "that",
            "this",
            "it",
            "its",
            "at",
            "from",
            "into",
            "about",
            "over",
            "after",
            "before",
            "between",
            "than",
            "then",
            "so",
            "if",
            "but",
            "because",
            "while",
            "during",
            "per",
        }
        return [t for t in tokens if t not in stopish and not t.isdigit()]

    @staticmethod
    def faithfulness_signals(
        answer: str,
        retrieved_docs_ordered: list[str],
        retrieved_docs_map: dict[str, str],
        top_k_docs: int = 5,
    ) -> dict[str, float]:
        """
        Compute faithfulness signals for generated answer.

        Measures how well the answer is grounded in retrieved documents
        by computing coverage and density metrics.

        Parameters:
            answer (str): Generated answer text
            retrieved_docs_ordered (list[str]): Ordered list of
                retrieved doc IDs
            retrieved_docs_map (dict[str, str]): Mapping from doc ID
                to doc text
            top_k_docs (int): Number of top documents to consider

        Returns:
            dict[str, float]: Dictionary with keys: support_coverage
                (percent of unique content words in answer that appear
                in retrieved docs), support_density (percent of all
                answer tokens appearing in docs), hallucination_rate
                (1 - support_density)
        """
        ans_toks = GenerationMetrics._tokens(answer)
        ans_content = GenerationMetrics._content_words(ans_toks)

        if not ans_toks:
            return {
                "support_coverage": 1.0,
                "support_density": 1.0,
                "hallucination_rate": 0.0,
            }

        # Build evidence bag from top-K docs
        top_ids = retrieved_docs_ordered[:top_k_docs]
        evidence_text = " ".join(
            retrieved_docs_map.get(doc_id, "") for doc_id in top_ids
        )
        ev_toks = set(GenerationMetrics._tokens(evidence_text))

        if not ev_toks:
            return {
                "support_coverage": 0.0,
                "support_density": 0.0,
                "hallucination_rate": 1.0,
            }

        # Coverage over unique content words
        unique_content = set(ans_content) if ans_content else set()
        covered_content = sum(1 for t in unique_content if t in ev_toks)
        support_coverage = (
            covered_content / max(1, len(unique_content)) if unique_content else 0.0
        )

        # Density over all answer tokens (precision-like)
        covered_tokens = sum(1 for t in ans_toks if t in ev_toks)
        support_density = covered_tokens / len(ans_toks)

        hallucination_rate = 1.0 - support_density
        return {
            "support_coverage": support_coverage,
            "support_density": support_density,
            "hallucination_rate": hallucination_rate,
        }

    @staticmethod
    async def sbert_similarity(
        answer: str,
        gold_answers: list[str],
        embedder: EmbeddingModel,
        reduction: str = "max",  # "max" or "mean"
    ) -> float:
        """
        Compute SBERT cosine similarity between answer and gold answers.

        Parameters:
            answer (str): Generated answer string
            gold_answers (list[str]): List of reference answers
            embedder (EmbeddingModel): Embedding model instance
                (e.g., SbertEmbeddingModel)
            reduction (str): Reduction method for multiple similarities:
                "max" (default) or "mean"

        Returns:
            float: Cosine similarity in [-1.0, 1.0]. If embeddings were
                normalized, typically in [0, 1] for natural language.
                Returns 0.0 if gold_answers is empty or contains only
                empty strings.
        """
        if GenerationMetrics.empty_gold_answer_guard(gold_answers):
            return 0.0

        # Get embeddings concurrently
        ans_vec_task = await embedder.embed_text(
            GenerationMetrics._normalize(answer) or ""
        )
        gold_vec_tasks = [
            await embedder.embed_text(GenerationMetrics._normalize(g) or "")
            for g in gold_answers
        ]

        ans_vec = np.array(ans_vec_task, dtype=float)
        gold_vecs = [np.array(t, dtype=float) for t in gold_vec_tasks]

        # Cosine similarity. If vectors are normalized, this is a dot product.
        def _cos(u: np.ndarray, v: np.ndarray) -> float:
            # Stable cosine even if not normalized in the embedder
            u_norm = np.linalg.norm(u)
            v_norm = np.linalg.norm(v)
            if u_norm == 0.0 or v_norm == 0.0:
                return 0.0
            return float(np.dot(u, v) / (u_norm * v_norm))

        sims = [_cos(ans_vec, gv) for gv in gold_vecs]
        if reduction == "mean":
            return float(np.mean(sims))
        return float(np.max(sims))

    @staticmethod
    async def cross_encoder_similarity(
        answer: str,
        gold_answers: list[str],
        scorer: CrossEncoderScorer,
        reduction: str = "max",  # "max" or "mean"
    ) -> float:
        """
        Score semantic similarity using cross-encoder model.

        Uses a cross-encoder (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        to score relevance between answer and gold answers.

        Parameters:
            answer (str): Generated answer text
            gold_answers (list[str]): List of reference answer strings
            scorer (CrossEncoderScorer): Cross-encoder scorer instance
            reduction (str): Reduction method: "max" (default) or "mean"

        Returns:
            float: Single relevance score (max or mean across gold answers)
        """
        if GenerationMetrics.empty_gold_answer_guard(gold_answers):
            return 0.0
        # Synchronous call (wrapper handles async under the hood)
        return await scorer.best_of(
            GenerationMetrics._normalize(answer) or "",
            [GenerationMetrics._normalize(g) or "" for g in gold_answers],
            reduction=reduction,
        )
