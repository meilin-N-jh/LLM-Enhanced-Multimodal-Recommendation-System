"""LLM-enhanced profile-based reranker."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.llm_profiles import LocalLLMUserProfiler, TemplateUserProfiler
from src.text_encoder import TextEncoder

from .base import BaseReranker


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom <= 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a matrix."""
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return matrix / norms


class LLMProfileReranker(BaseReranker):
    """Reranker that augments candidate ranking with an LLM preference profile."""

    def __init__(
        self,
        config: dict,
        text_embeddings: Dict[str, np.ndarray],
        image_embeddings: Optional[Dict[str, np.ndarray]],
        item_relations: Dict[str, set],
        item_metadata: Dict[str, dict],
        user_feedback_lookup: Optional[Dict[str, List[dict]]] = None,
        device: str = "cpu",
    ):
        super().__init__(config)

        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings or {}
        self.item_relations = item_relations
        self.item_metadata = item_metadata
        self.user_feedback_lookup = user_feedback_lookup or {}
        self.device = device

        rerank_cfg = config.get("reranking", {})
        weight_cfg = rerank_cfg.get("profile_based", {})
        self.candidate_pool_size = rerank_cfg.get("candidate_pool_size", 50)
        self.w_base = weight_cfg.get("w_base", 0.45)
        self.w_history = weight_cfg.get("w_history_text", 0.2)
        self.w_image = weight_cfg.get("w_history_image", 0.1)
        self.w_profile = weight_cfg.get("w_llm_profile", 0.2)
        self.w_relation = weight_cfg.get("w_relation", 0.1)
        self.w_metadata = weight_cfg.get("w_metadata", 0.05)

        llm_cfg = rerank_cfg.get("llm_profile", {})
        max_history_items = llm_cfg.get("max_history_items", 5)
        model_path = llm_cfg.get("model_path", "models/Qwen2.5-14B-Instruct")
        cache_path = llm_cfg.get(
            "cache_path",
            "outputs/llm_profiles/default_user_profiles.json",
        )

        if llm_cfg.get("enabled", False) and Path(model_path).exists():
            self.profiler = LocalLLMUserProfiler(
                item_metadata=item_metadata,
                model_path=model_path,
                cache_path=cache_path,
                max_history_items=max_history_items,
                max_new_tokens=llm_cfg.get("max_new_tokens", 96),
                device=device,
            )
        else:
            self.profiler = TemplateUserProfiler(
                item_metadata=item_metadata,
                max_history_items=max_history_items,
            )

        text_model_path = "models/all-MiniLM-L6-v2"
        text_model_name = (
            text_model_path
            if Path(text_model_path).exists()
            else config["model"]["text_encoder_model"]
        )
        self.profile_encoder = TextEncoder(
            model_name=text_model_name,
            device=device,
        )
        self.profile_embedding_cache: Dict[str, np.ndarray] = {}

    def _normalize_base_scores(self, candidates: List[str], base_scores: Dict[str, float]) -> Dict[str, float]:
        scores = np.array([base_scores.get(item_id, 0.0) for item_id in candidates], dtype=np.float32)
        if len(scores) == 0:
            return {}
        min_score = float(scores.min())
        max_score = float(scores.max())
        if max_score - min_score < 1e-8:
            return {item_id: 0.0 for item_id in candidates}
        return {
            item_id: float((base_scores.get(item_id, 0.0) - min_score) / (max_score - min_score))
            for item_id in candidates
        }

    def _get_profile_embedding(self, user_id: str, user_history: List[str]) -> np.ndarray:
        if user_id in self.profile_embedding_cache:
            return self.profile_embedding_cache[user_id]

        feedback_rows = self.user_feedback_lookup.get(user_id, [])
        profile_text = self.profiler.build_profile(user_id, user_history, feedback_rows)
        profile_emb = self.profile_encoder.encode_single(profile_text)
        self.profile_embedding_cache[user_id] = profile_emb
        return profile_emb

    def compute_history_similarity(self, item_id: str, user_history: List[str]) -> float:
        candidate_emb = self.text_embeddings.get(item_id)
        if candidate_emb is None or not user_history:
            return 0.0

        similarities = []
        for hist_item in user_history[-10:]:
            hist_emb = self.text_embeddings.get(hist_item)
            if hist_emb is not None:
                similarities.append(_cosine_similarity(candidate_emb, hist_emb))

        if not similarities:
            return 0.0
        return float(np.mean(similarities))

    def compute_profile_similarity(self, user_id: str, item_id: str, user_history: List[str]) -> float:
        item_emb = self.text_embeddings.get(item_id)
        if item_emb is None:
            return 0.0
        profile_emb = self._get_profile_embedding(user_id, user_history)
        return _cosine_similarity(profile_emb, item_emb)

    def compute_history_image_similarity(self, item_id: str, user_history: List[str]) -> float:
        candidate_emb = self.image_embeddings.get(item_id)
        if candidate_emb is None or not user_history:
            return 0.0

        similarities = []
        for hist_item in user_history[-10:]:
            hist_emb = self.image_embeddings.get(hist_item)
            if hist_emb is not None:
                similarities.append(_cosine_similarity(candidate_emb, hist_emb))

        if not similarities:
            return 0.0
        return float(np.mean(similarities))

    def compute_relation_overlap(self, item_id: str, user_history: List[str]) -> float:
        related_items = self.item_relations.get(item_id, set())
        if not related_items or not user_history:
            return 0.0
        history_set = set(user_history[-20:])
        return len(related_items & history_set) / max(len(related_items), 1)

    def compute_metadata_overlap(self, item_id: str, user_history: List[str]) -> float:
        candidate = self.item_metadata.get(item_id, {})
        if not candidate or not user_history:
            return 0.0

        categories = Counter()
        brands = Counter()
        for hist_item in user_history[-10:]:
            meta = self.item_metadata.get(hist_item, {})
            if meta.get("category"):
                categories[meta["category"]] += 1
            if meta.get("brand"):
                brands[meta["brand"]] += 1

        category_score = float(categories[candidate.get("category", "")] > 0)
        brand_score = float(brands[candidate.get("brand", "")] > 0)
        return 0.5 * category_score + 0.5 * brand_score

    def rerank(self, user_id, candidates, user_history, base_scores):
        normalized_base = self._normalize_base_scores(candidates, base_scores)
        candidate_count = len(candidates)
        reranked = []
        if candidate_count == 0:
            return reranked

        base_arr = np.array([normalized_base.get(item_id, 0.0) for item_id in candidates], dtype=np.float32)
        history_arr = np.zeros(candidate_count, dtype=np.float32)
        image_arr = np.zeros(candidate_count, dtype=np.float32)
        profile_arr = np.zeros(candidate_count, dtype=np.float32)
        relation_arr = np.zeros(candidate_count, dtype=np.float32)
        metadata_arr = np.zeros(candidate_count, dtype=np.float32)

        text_candidate_mask = np.array([item_id in self.text_embeddings for item_id in candidates], dtype=bool)
        if text_candidate_mask.any():
            candidate_text = np.stack(
                [self.text_embeddings[item_id] for item_id, keep in zip(candidates, text_candidate_mask) if keep]
            ).astype(np.float32)
            candidate_text = _normalize_rows(candidate_text)

            history_text_ids = [item_id for item_id in user_history[-10:] if item_id in self.text_embeddings]
            if history_text_ids:
                history_text = np.stack([self.text_embeddings[item_id] for item_id in history_text_ids]).astype(np.float32)
                history_text = _normalize_rows(history_text)
                history_arr[text_candidate_mask] = (candidate_text @ history_text.T).mean(axis=1)

            profile_emb = _normalize_rows(self._get_profile_embedding(user_id, user_history).reshape(1, -1))[0]
            profile_arr[text_candidate_mask] = candidate_text @ profile_emb

        image_candidate_mask = np.array([item_id in self.image_embeddings for item_id in candidates], dtype=bool)
        if image_candidate_mask.any():
            candidate_image = np.stack(
                [self.image_embeddings[item_id] for item_id, keep in zip(candidates, image_candidate_mask) if keep]
            ).astype(np.float32)
            candidate_image = _normalize_rows(candidate_image)

            history_image_ids = [item_id for item_id in user_history[-10:] if item_id in self.image_embeddings]
            if history_image_ids:
                history_image = np.stack([self.image_embeddings[item_id] for item_id in history_image_ids]).astype(np.float32)
                history_image = _normalize_rows(history_image)
                image_arr[image_candidate_mask] = (candidate_image @ history_image.T).mean(axis=1)

        history_tail = user_history[-20:]
        history_set = set(history_tail)
        categories = Counter()
        brands = Counter()
        for hist_item in user_history[-10:]:
            meta = self.item_metadata.get(hist_item, {})
            if meta.get("category"):
                categories[meta["category"]] += 1
            if meta.get("brand"):
                brands[meta["brand"]] += 1

        for idx, item_id in enumerate(candidates):
            related_items = self.item_relations.get(item_id, set())
            if related_items and history_tail:
                relation_arr[idx] = len(related_items & history_set) / max(len(related_items), 1)

            candidate_meta = self.item_metadata.get(item_id, {})
            if candidate_meta and user_history:
                category_score = float(categories[candidate_meta.get("category", "")] > 0)
                brand_score = float(brands[candidate_meta.get("brand", "")] > 0)
                metadata_arr[idx] = 0.5 * category_score + 0.5 * brand_score

        final_scores = (
            self.w_base * base_arr
            + self.w_history * history_arr
            + self.w_image * image_arr
            + self.w_profile * profile_arr
            + self.w_relation * relation_arr
            + self.w_metadata * metadata_arr
        )

        for idx, item_id in enumerate(candidates):
            reranked.append(
                (
                    item_id,
                    float(final_scores[idx]),
                    {
                        "base": round(float(base_arr[idx]), 4),
                        "history": round(float(history_arr[idx]), 4),
                        "image": round(float(image_arr[idx]), 4),
                        "profile": round(float(profile_arr[idx]), 4),
                        "relation": round(float(relation_arr[idx]), 4),
                        "metadata": round(float(metadata_arr[idx]), 4),
                    },
                )
            )

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
