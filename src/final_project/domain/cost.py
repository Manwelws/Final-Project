"""Domain value objects for business cost and threshold calculation."""

from __future__ import annotations

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

    def compute_net_savings(self, default_loss: float, optimal_loss: float) -> float:
        """Calculate dollar savings achieved by cost-optimal threshold over default 0.50 cutoff."""
        return float(default_loss - optimal_loss)

    def compute_loss_reduction_pct(self, default_loss: float, optimal_loss: float) -> float:
        """Calculate percentage loss reduction relative to baseline default cutoff."""
        if default_loss <= 0.0:
            return 0.0
        return float((default_loss - optimal_loss) / default_loss * 100.0)

    def compute_campaign_roi(self, gross_revenue_saved: float, campaign_cost: float) -> float:
        """Calculate campaign return-on-investment multiple."""
        if campaign_cost <= 0.0:
            return 0.0
        return float(gross_revenue_saved / campaign_cost)
