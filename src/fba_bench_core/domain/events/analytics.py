"""Analytics and system events and commands."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field

from ..models import DemandProfile
from .base import BaseEvent, Command


class ForecastUpdated(BaseEvent):
    """Event emitted when a product forecast/demand profile is updated."""

    event_type: Literal["forecast_updated"] = "forecast_updated"

    product_id: str
    demand_profile: DemandProfile


class AnomalyDetected(BaseEvent):
    """Generic anomaly detection event used by monitoring/analytics.

    - summary: short text describing the anomaly type.
    - metrics: optional structured metrics that explain the anomaly (small dict).
    """

    event_type: Literal["anomaly_detected"] = "anomaly_detected"

    summary: str
    metrics: dict[str, Any] | None = None
    severity: Literal["low", "medium", "high", "critical"] | None = "low"


class ReforecastDemandCommand(Command):
    """Request to recompute demand forecasts for a product over a timeframe."""

    command_type: Literal["reforecast_demand"] = "reforecast_demand"

    product_id: str
    timeframe_days: int = Field(..., gt=0)
    reason: str | None = None


class AdjustFulfillmentLatencyCommand(Command):
    """Command to set or adjust fulfillment latency targets (in days)."""

    command_type: Literal["adjust_fulfillment_latency"] = "adjust_fulfillment_latency"

    product_id: str
    new_latency_days: int = Field(..., ge=0)

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class NegotiateSupplyCommand(Command):
    """Command to negotiate supply terms with a supplier.

    - negotiation_terms: structured terms being negotiated (e.g., price, lead_time).
    """

    command_type: Literal["negotiate_supply"] = "negotiate_supply"

    supplier_id: str
    product_id: str
    negotiation_terms: dict[str, Any]

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
