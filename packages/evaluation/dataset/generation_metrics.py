import re
import string
from collections import Counter


# Generation metrics
class GenerationMetrics:
    _ARTICLES = {"a", "an", "the"}
    _PUNCT_TABLE = str.maketrans("", "", string.punctuation)

    @staticmethod
    def _normalize(text: str) -> str:
        if text is None:
            return ""
        end_idx = text.find("</think>")
        if end_idx != -1:
            text = text[end_idx + len("</think>") :]
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
        pred_norm = GenerationMetrics._normalize(pred)
        return (
            1.0
            if any(pred_norm == GenerationMetrics._normalize(g) for g in gold_answers)
            else 0.0
        )

    @staticmethod
    def f1(pred: str, gold_answers: list[str]) -> float:
        if not gold_answers:
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
        support_coverage: percent of unique content words in the answer
            that appear in retrieved docs (top-K)
        support_density: percent of all answer tokens that appear in
            retrieved docs (precision-like)
        hallucination_rate: 1 - support_density
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
