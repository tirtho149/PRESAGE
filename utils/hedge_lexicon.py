"""
utils/hedge_lexicon.py
======================
Hedge lexicon bootstrap (Appendix E / §6 RQ5).

Appendix E:
    "Seed hedge terms (possibly, unclear, hard to tell, uncertain, approximately,
     may, might, could) are used to extract co-occurring n-grams scored by PMI.
     Top-200 PMI candidates are manually reviewed. Final lexicon validated against
     BioScope (Vincze et al., 2008) and CoNLL 2010 Shared Task hedging data."

Reference: Vincze et al. (2008) "The BioScope corpus: Biomedical texts annotated
           for uncertainty, negation and their scopes." BioNLP 2008.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from math import log
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# Appendix E: seed hedge terms
SEED_HEDGE_TERMS: List[str] = [
    "possibly", "unclear", "hard to tell", "uncertain",
    "approximately", "may", "might", "could",
    # PathogenAgent-style hedge markers (plant diagnosis uncertainty)
    "possibly", "unclear", "hard to tell",
    # Additional uncertainty markers common in VLM outputs
    "appears to be", "seems like", "likely", "probably",
    "not sure", "difficult to determine", "cannot confirm",
    "ambiguous", "unclear", "hard to say",
]

# Deduplicate
SEED_HEDGE_TERMS = list(dict.fromkeys(SEED_HEDGE_TERMS))


class HedgeLexiconBuilder:
    """
    PMI-based hedge lexicon bootstrap (Appendix E).

    Usage
    -----
    builder = HedgeLexiconBuilder()
    builder.fit(corpus_texts)
    lexicon = builder.get_lexicon(top_k=200)
    """

    def __init__(self, ngram_range: Tuple[int, int] = (1, 3)):
        self.ngram_range = ngram_range
        self._ngram_counts: Counter = Counter()
        self._context_counts: Counter = Counter()  # n-grams co-occurring with seeds
        self._total_ngrams: int = 0
        self.lexicon_: Optional[Set[str]] = None

    def _tokenize(self, text: str) -> List[str]:
        return re.sub(r"[^\w\s']", " ", text.lower()).split()

    def _extract_ngrams(self, tokens: List[str]) -> List[str]:
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i:i + n]))
        return ngrams

    def fit(self, texts: List[str], seed_terms: Optional[List[str]] = None) -> "HedgeLexiconBuilder":
        """
        Fit on corpus texts; compute PMI scores for n-grams co-occurring with seeds.
        """
        seeds = set(seed_terms or SEED_HEDGE_TERMS)
        context_window = 5  # words around seed

        all_ngrams: List[str] = []
        context_ngrams: Counter = Counter()

        for text in texts:
            tokens = self._tokenize(text)
            ngrams = self._extract_ngrams(tokens)
            all_ngrams.extend(ngrams)

            # Find seed positions and collect context n-grams
            for i, token in enumerate(tokens):
                for seed in seeds:
                    seed_tokens = seed.split()
                    end = i + len(seed_tokens)
                    if tokens[i:end] == seed_tokens:
                        # Collect n-grams in window [i-context_window, end+context_window]
                        lo = max(0, i - context_window)
                        hi = min(len(tokens), end + context_window)
                        ctx_tokens = tokens[lo:hi]
                        for ng in self._extract_ngrams(ctx_tokens):
                            if ng not in seeds:
                                context_ngrams[ng] += 1

        self._ngram_counts = Counter(all_ngrams)
        self._context_counts = context_ngrams
        self._total_ngrams = len(all_ngrams)
        return self

    def pmi_scores(self) -> Dict[str, float]:
        """Compute PMI score for each n-gram relative to hedge context."""
        total = self._total_ngrams
        if total == 0:
            return {}

        total_context = sum(self._context_counts.values()) or 1
        scores = {}
        for ng, ctx_count in self._context_counts.items():
            ng_count = self._ngram_counts.get(ng, 0)
            if ng_count == 0:
                continue
            p_ng = ng_count / total
            p_ctx = ctx_count / total_context
            p_joint = ctx_count / total  # approximation
            pmi = log(p_joint / (p_ng * p_ctx) + 1e-10)
            scores[ng] = pmi
        return scores

    def get_lexicon(self, top_k: int = 200) -> Set[str]:
        """Return top-k PMI n-grams as the hedge lexicon (Appendix E)."""
        scores = self.pmi_scores()
        sorted_terms = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        lexicon = set(SEED_HEDGE_TERMS)
        lexicon.update(t for t, _ in sorted_terms)
        self.lexicon_ = lexicon
        return lexicon


# ---------------------------------------------------------------------------
# Hedge score computation (for routing trace analysis)
# ---------------------------------------------------------------------------

class HedgeScorer:
    """
    Compute hedge score for an agent message using the lexicon.
    Used in §6 RQ5: hedge propagation between PathogenAgent and SeverityAgent.
    """

    def __init__(self, lexicon: Optional[Set[str]] = None):
        self.lexicon = lexicon or set(SEED_HEDGE_TERMS)

    def score(self, text: str) -> float:
        """
        Hedge score = fraction of sentences containing at least one hedge term.
        Normalised to [0, 1].
        """
        text_lower = text.lower()
        sentences = re.split(r"[.!?]\s+", text_lower)
        if not sentences:
            return 0.0
        hedged = sum(
            1 for sent in sentences
            if any(term in sent for term in self.lexicon)
        )
        return hedged / len(sentences)

    def score_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.score(t) for t in texts])

    @staticmethod
    def default() -> "HedgeScorer":
        """Return scorer with seed lexicon (no PMI fitting required)."""
        return HedgeScorer(lexicon=set(SEED_HEDGE_TERMS))
