"""User preference profiling for LLM-enhanced recommendation."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import torch


STOPWORDS = {
    "with",
    "that",
    "this",
    "from",
    "have",
    "your",
    "they",
    "will",
    "about",
    "their",
    "into",
    "were",
    "been",
    "also",
    "very",
    "more",
    "than",
    "just",
    "such",
    "when",
    "what",
    "where",
    "which",
    "while",
    "make",
    "made",
    "using",
    "used",
    "like",
    "love",
    "good",
    "great",
    "really",
    "product",
    "products",
}


class TemplateUserProfiler:
    """Build a lightweight preference summary from user history."""

    def __init__(self, item_metadata: Dict[str, dict], max_history_items: int = 5):
        self.item_metadata = item_metadata
        self.max_history_items = max_history_items

    def _extract_keywords(self, texts: List[str], top_k: int = 8) -> List[str]:
        words = Counter()
        for text in texts:
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
            for token in tokens:
                if token not in STOPWORDS:
                    words[token] += 1
        return [word for word, _ in words.most_common(top_k)]

    def build_profile(
        self,
        user_id: str,
        history_items: List[str],
        feedback_rows: Optional[List[dict]] = None,
    ) -> str:
        """Build a deterministic text profile from recent interactions."""
        feedback_rows = feedback_rows or []
        recent_items = history_items[-self.max_history_items :]

        categories = Counter()
        brands = Counter()
        titles = []
        free_text = []

        for item_id in recent_items:
            meta = self.item_metadata.get(item_id, {})
            if meta.get("category"):
                categories[meta["category"]] += 1
            if meta.get("brand"):
                brands[meta["brand"]] += 1
            if meta.get("title"):
                titles.append(meta["title"])
            if meta.get("description"):
                free_text.append(meta["description"])

        for row in feedback_rows[-self.max_history_items :]:
            if row.get("review_title"):
                free_text.append(row["review_title"])
            if row.get("review_text"):
                free_text.append(row["review_text"])

        top_categories = ", ".join(cat for cat, _ in categories.most_common(3)) or "unknown categories"
        top_brands = ", ".join(brand for brand, _ in brands.most_common(3)) or "mixed brands"
        key_titles = "; ".join(titles[:3]) or "few recorded items"
        keywords = ", ".join(self._extract_keywords(titles + free_text))
        keywords = keywords or "general beauty and personal care"

        return (
            f"User {user_id} prefers categories: {top_categories}. "
            f"Frequently engaged brands: {top_brands}. "
            f"Representative purchased items: {key_titles}. "
            f"Preference keywords: {keywords}."
        )


class LocalLLMUserProfiler(TemplateUserProfiler):
    """Generate user preference summaries with a local instruct model."""

    def __init__(
        self,
        item_metadata: Dict[str, dict],
        model_path: str,
        cache_path: Optional[str] = None,
        max_history_items: int = 5,
        max_new_tokens: int = 96,
        device: str = "cuda",
    ):
        super().__init__(item_metadata=item_metadata, max_history_items=max_history_items)
        self.model_path = model_path
        self.cache_path = Path(cache_path) if cache_path else None
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._cache = self._load_cache()
        self._tokenizer = None
        self._model = None

    def _load_cache(self) -> Dict[str, str]:
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as handle:
            json.dump(self._cache, handle, indent=2, ensure_ascii=False)

    def _lazy_load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto" if self.device.startswith("cuda") else None,
        )

    def _build_prompt(
        self,
        user_id: str,
        history_items: List[str],
        feedback_rows: Optional[List[dict]] = None,
    ) -> str:
        feedback_rows = feedback_rows or []
        recent_items = history_items[-self.max_history_items :]
        template_summary = super().build_profile(user_id, history_items, feedback_rows)

        lines = []
        for item_id in recent_items:
            meta = self.item_metadata.get(item_id, {})
            parts = [meta.get("title") or item_id]
            if meta.get("brand"):
                parts.append(f"brand={meta['brand']}")
            if meta.get("category"):
                parts.append(f"category={meta['category']}")
            if meta.get("description"):
                parts.append(f"description={meta['description'][:200]}")
            lines.append(" | ".join(parts))

        for row in feedback_rows[-self.max_history_items :]:
            review_bits = []
            if row.get("review_title"):
                review_bits.append(f"title={row['review_title'][:100]}")
            if row.get("review_text"):
                review_bits.append(f"text={row['review_text'][:220]}")
            if review_bits:
                lines.append("review: " + " | ".join(review_bits))

        lines_text = "\n".join(f"- {line}" for line in lines) or "- no detailed history available"

        return (
            "You are helping an e-commerce recommender system.\n"
            "Given a user's recent purchase history and review snippets, write a concise "
            "preference profile that helps rank future beauty-product candidates.\n"
            "Return 3 short bullet lines only:\n"
            "1. likely product interests\n"
            "2. favored brands, attributes, or use cases\n"
            "3. likely next-item intent\n\n"
            f"Seed summary:\n{template_summary}\n\n"
            f"Recent history for user {user_id}:\n{lines_text}\n"
        )

    def build_profile(
        self,
        user_id: str,
        history_items: List[str],
        feedback_rows: Optional[List[dict]] = None,
    ) -> str:
        if user_id in self._cache:
            return self._cache[user_id]

        prompt = self._build_prompt(user_id, history_items, feedback_rows)

        try:
            self._lazy_load()
            messages = [
                {"role": "system", "content": "You are an expert product preference profiler."},
                {"role": "user", "content": prompt},
            ]
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(rendered, return_tensors="pt").to(self._model.device)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            generated = outputs[0][inputs.input_ids.shape[1] :]
            profile = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
            if not profile:
                raise ValueError("empty profile")
        except Exception:
            profile = super().build_profile(user_id, history_items, feedback_rows)

        self._cache[user_id] = profile
        self._save_cache()
        return profile
