"""Base reranker interface."""

from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """Base class for rerankers."""

    def __init__(self, config):
        """Initialize reranker.

        Args:
            config: Configuration dict
        """
        self.config = config

    @abstractmethod
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
        pass

    def prepare_user_history_features(self, user_history, item_features):
        """Prepare features from user history for reranking.

        Args:
            user_history: List of item IDs from user history
            item_features: Dict mapping item_id to feature dict

        Returns:
            Feature dict
        """
        pass
