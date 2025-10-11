"""Pricing and demand-related events and commands."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from ..models import CompetitorListing, DemandProfile
from .base import BaseEvent, Command


class SaleOccurred(BaseEvent):
    """Event emitted when a sale completes for a product.

    Notes / semantics:
    - `product_id` is the canonical identifier and is preferred; `product_sku` is
      optional and kept for compatibility with legacy callers.
    - `gross_margin` is represented as a Decimal fraction of revenue (e.g., 0.35
      for 35%). Negative values are allowed (loss-making sale) but are bounded
      by -1.0 for sanity.
    """

    event_type: Literal["sale_occurred"] = "sale_occurred"

    order_id: str
    product_id: str | None = None
    product_sku: str | None = None
    quantity: int = Field(..., ge=1)
    revenue: Decimal = Field(..., gt=Decimal("0"))
    currency: str = Field("USD", min_length=3, description="ISO currency code")
    channel: str | None = None
    customer_segment: str | None = None
    gross_margin: Decimal | None = Field(
        None,
        description="Gross margin expressed as Decimal fraction (0.0 == 0%, 1.0 == 100%)",
    )

    @field_validator("revenue", "gross_margin", mode="before")
    @classmethod
    def _coerce_decimal(cls, v):
        if v is None:
            return v
        if isinstance(v, Decimal):
            return v
        try:
            return Decimal(str(v))
        except Exception as exc:
            raise ValueError(f"Invalid monetary/ratio value: {v!r}") from exc

    @model_validator(mode="after")
    def _validate_gross_margin(self):
        if self.gross_margin is not None:
            if self.gross_margin < Decimal("-1") or self.gross_margin > Decimal("1"):
                raise ValueError("gross_margin must be between -1 and 1 (fractional)")
        return self


class PriceChangedExternally(BaseEvent):
    """Event representing an observed competitor price/listing change.

    Embeds a CompetitorListing to provide structured comparables for repricing logic.
    """

    event_type: Literal["price_changed_externally"] = "price_changed_externally"
    competitor_id: str
    listing: CompetitorListing


class DemandSpiked(BaseEvent):
    """Event signalling an abrupt increase in demand for a product.

    - delta: positive increase in expected demand (units or percentage depending on trigger)
    - trigger: short text describing why (e.g., 'seasonal', 'media_mention', 'stockout_competitor')
    - optional demand_profile allows attaching a refreshed DemandProfile for downstream forecasting.
    """

    event_type: Literal["demand_spiked"] = "demand_spiked"

    product_id: str
    delta: Decimal = Field(..., gt=Decimal("0"))
    trigger: str
    demand_profile: DemandProfile | None = None

    @field_validator("delta", mode="before")
    @classmethod
    def _coerce_delta(cls, v):
        if isinstance(v, Decimal):
            return v
        try:
            return Decimal(str(v))
        except Exception as exc:
            raise ValueError("delta must be numeric") from exc


class CompetitorAction(BaseEvent):
    """Event signalling a competitor's strategic action (e.g., price change, promotion launch).

    - action_type: categorized action (e.g., 'price_adjustment', 'promotion_launch', 'inventory_change')
    - details: optional structured details about the action.
    """

    event_type: Literal["competitor_action"] = "competitor_action"

    competitor_id: str
    action_type: str
    details: dict[str, Any] | None = None


class AdjustPriceCommand(Command):
    """Command to change the price for a product.

    Business rules:
    - proposed_price uses Decimal for monetary precision and must be >= 0.
    - `effective_from` indicates when price should take effect (None = immediate).
    - `channel` is optional to target marketplace/channel-level prices.
    """

    command_type: Literal["adjust_price"] = "adjust_price"

    product_id: str | None = None
    product_sku: str | None = None
    proposed_price: Decimal = Field(..., ge=Decimal("0"))
    effective_from: datetime | None = None
    channel: str | None = None

    @field_validator("proposed_price", mode="before")
    @classmethod
    def _coerce_price(cls, v):
        if isinstance(v, Decimal):
            return v
        try:
            return Decimal(str(v))
        except Exception as exc:
            raise ValueError("Invalid proposed_price") from exc


class LaunchPromotionCommand(Command):
    """Command instructing the system to launch a promotion.

    - discount_percent is Decimal fraction between 0 and 1.
    """

    command_type: Literal["launch_promotion"] = "launch_promotion"

    promotion_id: str
    product_ids: list[str] | None = None
    category: str | None = None
    discount_percent: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    start: datetime
    end: datetime | None = None
    channels: list[str] | None = None
    notes: str | None = None

    @field_validator("discount_percent", mode="before")
    @classmethod
    def _coerce_discount(cls, v):
        if isinstance(v, Decimal):
            return v
        try:
            return Decimal(str(v))
        except Exception as exc:
            raise ValueError("Invalid discount_percent") from exc


class MonitorCompetitorCommand(Command):
    """Command to initiate monitoring of a competitor's activities.

    - monitoring_focus: areas to monitor (e.g., 'pricing', 'inventory', 'promotions').
    """

    command_type: Literal["monitor_competitor"] = "monitor_competitor"

    competitor_id: str
    monitoring_focus: list[str]

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
