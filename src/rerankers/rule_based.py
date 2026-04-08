"""Rule-based reranker implementation."""

import numpy as np
from .base import BaseReranker


class RuleBasedReranker(BaseReranker):
    """Rule-based reranker combining multiple signals.

    Formula:
    final_score = w1 * recall_score + w2 * text_similarity + w3 * relation_overlap
    """

    def __init__(self, config, text_embeddings=None, item_relations=None):
        """Initialize rule-based reranker.

        Args:
            config: Configuration dict
            text_embeddings: Dict of item_id -> text embedding
            item_relations: Dict of item_id -> set of related item IDs
        """
        super().__init__(config)

        # Get weights
        rerank_config = config.get('reranking', {}).get('rule_based', {})
        self.w1 = rerank_config.get('w1_recall', 0.5)
        self.w2 = rerank_config.get('w2_text_sim', 0.3)
        self.w3 = rerank_config.get('w3_relation', 0.2)

        # Features
        self.text_embeddings = text_embeddings or {}
        self.item_relations = item_relations or {}

    def compute_text_similarity(self, item_id, user_history):
        """Compute text similarity between item and user history.

        Args:
            item_id: Candidate item ID
            user_history: List of item IDs from user history

        Returns:
            Similarity score
        """
        if not self.text_embeddings or not user_history:
            return 0.0

        # Get candidate embedding
        cand_emb = self.text_embeddings.get(item_id)
        if cand_emb is None:
            return 0.0

        # Get history embeddings
        hist_embs = []
        for hist_item in user_history:
            if hist_item in self.text_embeddings:
                hist_embs.append(self.text_embeddings[hist_item])

        if not hist_embs:
            return 0.0

        # Compute average similarity
        hist_embs = np.array(hist_embs)

        if isinstance(cand_emb, np.ndarray):
            cand_emb = cand_emb.reshape(1, -1)
        else:
            cand_emb = np.array(cand_emb).reshape(1, -1)

        # Cosine similarity
        similarities = np.dot(hist_embs, cand_emb.T).flatten()

        # Normalize
        cand_norm = np.linalg.norm(cand_emb)
        hist_norms = np.linalg.norm(hist_embs, axis=1)

        if cand_norm > 0 and np.any(hist_norms > 0):
            normalized = similarities / (cand_norm * hist_norms + 1e-10)
            return np.max(normalized)

        return 0.0

    def compute_relation_overlap(self, item_id, user_history):
        """Compute relation overlap between item and user history.

        Args:
            item_id: Candidate item ID
            user_history: List of item IDs from user history

        Returns:
            Relation overlap score
        """
        if not self.item_relations or not user_history:
            return 0.0

        # Get related items for candidate
        cand_related = self.item_relations.get(item_id, set())

        # Get related items for history
        hist_related = set()
        for hist_item in user_history:
            if hist_item in self.item_relations:
                hist_related.update(self.item_relations[hist_item])

        # Compute overlap
        if len(cand_related) == 0:
            return 0.0

        overlap = len(cand_related & hist_related) / len(cand_related)
        return overlap

    def rerank(self, user_id, candidates, user_history, base_scores):
        """Rerank candidates.

        Args:
            user_id: User ID
            candidates: List of candidate item IDs
            user_history: List of user's historical item IDs
            base_scores: Dict of item_id -> base score

        Returns:
            List of (item_id, new_score) tuples, reranked
        """
        results = []

        for item_id in candidates:
            # Base recall score (normalized)
            base_score = base_scores.get(item_id, 0.0)
            max_score = max(base_scores.values()) if base_scores else 1.0
            recall_score = base_score / max_score if max_score > 0 else 0.0

            # Text similarity to history
            text_sim = self.compute_text_similarity(item_id, user_history)

            # Relation overlap
            rel_overlap = self.compute_relation_overlap(item_id, user_history)

            # Combined score
            final_score = (
                self.w1 * recall_score +
                self.w2 * text_sim +
                self.w3 * rel_overlap
            )

            results.append((item_id, final_score))

        # Sort by final score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results
