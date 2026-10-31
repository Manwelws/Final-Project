"""Domain value objects for business cost and threshold calculation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostMatrix:
    """Financial cost matrix for customer churn classification.

    Attributes:
        cost_fn: Business loss of a missed churner (False Negative), e.g. $200 customer lifetime value.
        cost_fp: Cost of retention outreach to a retained customer (False Positive), e.g. $20 incentive.
    """

    cost_fn: float = 200.0
    cost_fp: float = 20.0

    def compute_loss(self, fn: int, fp: int) -> float:
        """Calculate total business loss given false negatives and false positives."""
        return float(self.cost_fn * fn + self.cost_fp * fp)

    def total_cost(self, fn: int, fp: int) -> float:
        """Alias for compute_loss."""
        return self.compute_loss(fn, fp)

