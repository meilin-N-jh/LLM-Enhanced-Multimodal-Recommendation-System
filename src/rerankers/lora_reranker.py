"""LoRA reranker interface for small local models."""

import os
import torch
from .base import BaseReranker
from .prompt_utils import build_rerank_prompt, parse_rerank_response


class LoRAReranker(BaseReranker):
    """LoRA reranker using small local models.

    This is an interface for local LLM-based reranking.
    Requires local LLM setup (e.g., Qwen2.5-1.5B-Instruct, Phi-3-mini).
    Falls back to mock mode if no LLM available.
    """

    def __init__(self, config, model_name=None, lora_path=None):
        """Initialize LoRA reranker.

        Args:
            config: Configuration dict
            model_name: Name/path of local model
            lora_path: Path to LoRA weights
        """
        super().__init__(config)

        self.model_name = model_name or "microsoft/Phi-3-mini-4k-instruct"
        self.lora_path = lora_path

        # Try to load model, fall back to mock
        self.use_mock = True
        self.model = None
        self.tokenizer = None

        self._try_load_model()

    def _try_load_model(self):
        """Try to load local LLM with LoRA."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            print(f"Loading model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            # Load LoRA if provided
            if self.lora_path and os.path.exists(self.lora_path):
                print(f"Loading LoRA weights from {self.lora_path}")
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)

            self.use_mock = False
            print("LoRA reranker loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load LLM: {e}")
            print("Falling back to mock mode")
            self.use_mock = True

    def rerank(self, user_id, candidates, user_history, base_scores):
        """Rerank candidates using LLM.

        Args:
            user_id: User ID
            candidates: List of candidate item IDs
            user_history: List of user's historical item IDs
            base_scores: Dict of item_id -> base score

        Returns:
            List of (item_id, new_score) tuples, reranked
        """
        if self.use_mock:
            # Fall back to base scores
            return [(item_id, base_scores.get(item_id, 0.0)) for item_id in candidates]

        # Use LLM for reranking
        results = []

        for item_id in candidates:
            # Build prompt
            prompt = build_rerank_prompt(
                user_id=user_id,
                candidate_item=item_id,
                user_history=user_history,
                base_score=base_scores.get(item_id, 0.0)
            )

            # Generate
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False
                    )

                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Parse score
                score = parse_rerank_response(response)
                results.append((item_id, score))

            except Exception as e:
                print(f"Error reranking {item_id}: {e}")
                results.append((item_id, base_scores.get(item_id, 0.0)))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        return results


def train_lora_reranker(config, train_data, model_path, output_dir, epochs=3):
    """Train LoRA reranker.

    This is a template for training LoRA on reranking data.
    Requires local LLM setup.

    Args:
        config: Configuration dict
        train_data: Training data
        model_path: Path to base model
        output_dir: Output directory for LoRA weights
        epochs: Number of epochs
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("Error: transformers and peft required for LoRA training")
        print("pip install transformers peft")
        return

    print("LoRA training template")
    print("This would fine-tune a small model for reranking")
    print(f"Base model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}")
    print("\nNote: Full implementation requires training data preparation")
