# app/utils/token_metrics.py

from typing import Dict, Optional

class TokenMetrics:
    """
    A helper class to manage and aggregate token usage metrics by category.
    This class is used to track token consumption for policy, artifact, 
    and standards processing during batch operations.
    """
    def __init__(self):
        """Initializes the metric categories with zero values."""
        self.categories = {
            "policy": {"llm_input_tokens": 0, "llm_output_tokens": 0, "rerank_tokens": 0, "embedding_tokens": 0},
            "artifact": {"llm_input_tokens": 0, "llm_output_tokens": 0, "rerank_tokens": 0, "embedding_tokens": 0},
            "standards": {"llm_input_tokens": 0, "llm_output_tokens": 0, "rerank_tokens": 0, "embedding_tokens": 0},
        }

    def update(self, category: str, usage: Optional[Dict[str, int]]):
        """
        Updates the token count for a specific category (e.g., 'policy').

        Args:
            category: The category to update ('policy', 'artifact', 'standards').
            usage: A dictionary containing token counts for the operation.
        """
        if not usage or category not in self.categories:
            return
        
        target = self.categories[category]
        target["llm_input_tokens"] += usage.get("llm_input_tokens", 0)
        target["llm_output_tokens"] += usage.get("llm_output_tokens", 0)
        target["rerank_tokens"] += usage.get("rerank_tokens", 0)
        target["embedding_tokens"] += usage.get("embedding_tokens", 0)

    def add(self, other_metrics: Dict[str, Dict[str, int]]):
        """
        Adds metrics from another structured dictionary to this one, used for
        aggregating cumulative totals.

        Args:
            other_metrics: A dictionary with the same structure as self.categories.
        """
        for category, usage in other_metrics.items():
            self.update(category, usage)

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        """Returns the metrics as a dictionary."""
        return self.categories
