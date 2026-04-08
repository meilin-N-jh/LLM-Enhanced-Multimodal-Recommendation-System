"""Inference for the LLM-enhanced multimodal recommender."""

from __future__ import annotations

import argparse

import numpy as np

from src.runtime import build_item_lookup, create_reranker, load_checkpoint_bundle
from src.utils import get_device, load_config


def main():
    parser = argparse.ArgumentParser(description="Run recommendation inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--user_id", type=str, required=True)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--no_rerank", action="store_true")
    parser.add_argument("--enable_llm_profile", action="store_true")
    parser.add_argument("--disable_llm_profile", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_rerank:
        config["ablation"]["no_rerank"] = True
    if args.enable_llm_profile:
        config.setdefault("reranking", {}).setdefault("llm_profile", {})["enabled"] = True
    if args.disable_llm_profile:
        config.setdefault("reranking", {}).setdefault("llm_profile", {})["enabled"] = False

    device = get_device(config.get("device", "cuda"))
    _, bundle, model = load_checkpoint_bundle(args.model_path, config=config, device=device)

    user2idx = bundle["user2idx"]
    if args.user_id not in user2idx:
        raise KeyError(f"User {args.user_id} not found in training/test splits")

    item_lookup = build_item_lookup(bundle["items"])
    train_df = bundle["train_df"]
    user_history = train_df.sort_values(["user_id", "timestamp"]).groupby("user_id")["item_id"].apply(list).to_dict()
    exclude_items = set(user_history.get(args.user_id, []))
    reranker = None if config["ablation"].get("no_rerank", False) else create_reranker(config, bundle, device)

    scores = model.predict(user2idx[args.user_id], exclude_items=exclude_items)
    candidate_pool = max(args.top_n, config.get("reranking", {}).get("candidate_pool_size", args.top_n))
    top_indices = np.argsort(-scores)[:candidate_pool]
    candidates = [bundle["idx2item"][i] for i in top_indices]
    base_scores = {item_id: float(scores[bundle["item2idx"][item_id]]) for item_id in candidates}

    if reranker is not None:
        reranked = reranker.rerank(args.user_id, candidates, user_history.get(args.user_id, []), base_scores)
        recommendations = reranked[: args.top_n]
    else:
        recommendations = [(item_id, base_scores[item_id], {}) for item_id in candidates[: args.top_n]]

    print(f"Recommendations for user {args.user_id}")
    for rank, row in enumerate(recommendations, start=1):
        item_id = row[0]
        score = row[1]
        details = row[2] if len(row) > 2 else {}
        meta = item_lookup.get(item_id, {})
        title = meta.get("title", item_id)
        brand = meta.get("brand", "")
        category = meta.get("category", "")
        print(
            f"{rank:02d}. {item_id} | score={score:.4f} | title={title[:100]} | "
            f"brand={brand[:30]} | category={category[:60]} | signals={details}"
        )


if __name__ == "__main__":
    main()
