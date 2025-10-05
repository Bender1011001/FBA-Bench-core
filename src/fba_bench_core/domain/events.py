"""Expanded domain events and commands for FBA-Bench (Phase C).

Rationale:
- This module defines a richer, typed vocabulary of events and commands that
  capture operational verbs required by an FBA simulation (pricing, inventory,
  marketing, competitor intel, customer service, logistics, analytics).
- Contracts are implemented with Pydantic v2 and reuse hardened domain models
  from `.models` (Product, InventorySnapshot, CompetitorListing, DemandProfile).
- Monetary values use Decimal for precision. Numeric inputs are coerced to Decimal
  where appropriate. Validation enforces common business invariants (non-negative
  quantities, bounded percentages, etc.).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Literal, Optional, Tuple, Type, Union
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import (
    CompetitorListing,
    DemandProfile,
    InventorySnapshot,
    Product,
)

# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class BaseEvent(BaseModel):
    """Base contract for all domain events.

    Attributes:
    - event_type: Literal discriminator provided by subclasses.
    - timestamp: UTC timestamp when the event was recorded (defaults to now).
    - tick: non-negative simulation or system tick.
    - correlation_id: optional id to trace this event across services and workflows.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tick: int = Field(0, ge=0)
    correlation_id: Optional[str] = None


class Command(BaseModel):
    """Base contract for commands (intent to change system state).

    Commands are issued by agents or systems. Include optional metadata to
    enable observability and intent tracing:
    - issued_by: human or system identifier that created the command.
    - reason: free-text explanation for auditing.
    - correlation_id: align with events for traceability.
    - metadata: structured map for small typed values (avoid open-ended blobs).
      We deliberately use extra="forbid" at the model level to prevent accidental
      arbitrary attributes; metadata is the supported extensibility point.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    command_type: str
    issued_by: Optional[str] = None
    reason: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, v: Dict[str, Any]):
        # Keep metadata shallow and keyed by str to encourage typed schemas.
        if not isinstance(v, dict):
            raise TypeError("metadata must be a dict[str, Any]")
        for k in v.keys():
            if not isinstance(k, str):
                raise TypeError("metadata keys must be strings")
        return v


# ---------------------------------------------------------------------------
# Events: Pricing & Demand Signals
# ---------------------------------------------------------------------------


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
    product_id: Optional[str] = None
    product_sku: Optional[str] = None
    quantity: int = Field(..., ge=1)
    revenue: Decimal = Field(..., gt=Decimal("0"))
    currency: str = Field("USD", min_length=3, description="ISO currency code")
    channel: Optional[str] = None
    customer_segment: Optional[str] = None
    gross_margin: Optional[Decimal] = Field(
        None, description="Gross margin expressed as Decimal fraction (0.0 == 0%, 1.0 == 100%)"
    )

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

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

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


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
    demand_profile: Optional[DemandProfile] = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    @field_validator("delta", mode="before")
    @classmethod
    def _coerce_delta(cls, v):
        if isinstance(v, Decimal):
            return v
        try:
            return Decimal(str(v))
        except Exception as exc:
            raise ValueError("delta must be numeric") from exc


# ---------------------------------------------------------------------------
# Inventory & Fulfillment Events
# ---------------------------------------------------------------------------


class StockReplenished(BaseEvent):
    """Event emitted when stock is replenished at a warehouse.

    Either provide `snapshot_before` and `snapshot_after` (InventorySnapshot) or
    provide `quantity_added` and `warehouse_location`. `quantity_added` must be > 0.
    """

    event_type: Literal["stock_replenished"] = "stock_replenished"

    product_id: str
    snapshot_before: Optional[InventorySnapshot] = None
    snapshot_after: Optional[InventorySnapshot] = None
    warehouse_location: Optional[str] = None
    quantity_added: Optional[int] = Field(None, gt=0)

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_snapshots_or_quantity(self):
        if not (self.snapshot_before or self.snapshot_after or self.quantity_added):
            raise ValueError(
                "Provide snapshot_before/after or quantity_added to describe the replenishment"
            )
        return self


class StockDepleted(BaseEvent):
    """Event triggered when inventory reaches zero or falls below safety stock.

    - safety_stock: optional configured safety stock level (int)
    - current_snapshot: optional InventorySnapshot for reconciliation
    """

    event_type: Literal["stock_depleted"] = "stock_depleted"

    product_id: str
    safety_stock: Optional[int] = None
    current_snapshot: Optional[InventorySnapshot] = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class FulfillmentDelayed(BaseEvent):
    """Event emitted when an order fulfillment is delayed beyond SLA."""

    event_type: Literal["fulfillment_delayed"] = "fulfillment_delayed"

    order_id: str
    delay_hours: float = Field(..., ge=0.0)
    reason: Optional[str] = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


# ---------------------------------------------------------------------------
# Marketing & Customer Engagement Events
# ---------------------------------------------------------------------------


class PromotionLaunched(BaseEvent):
    """Event signalling that a promotion has been launched for products or categories.

    - discount_percent is expressed as Decimal fraction (0.0 - 1.0).
    """

    event_type: Literal["promotion_launched"] = "promotion_launched"

    promotion_id: str
    product_ids: Optional[list[str]] = None
    category: Optional[str] = None
    discount_percent: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    start: datetime
    end: Optional[datetime] = None
    channels: Optional[list[str]] = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    @field_validator("discount_percent", mode="before")
    @classmethod
    def _coerce_discount(cls, v):
        if isinstance(v, Decimal):
            return v
        try:
            return Decimal(str(v))
        except Exception as exc:
            raise ValueError("Invalid discount_percent") from exc


class CustomerComplaintLogged(BaseEvent):
    """Event representing a logged customer complaint tied to an order."""

    event_type: Literal["customer_complaint_logged"] = "customer_complaint_logged"

    complaint_id: str
    order_id: Optional[str] = None
    product_id: Optional[str] = None
    issue_type: str
    details: Optional[str] = None
    resolution_deadline: Optional[datetime] = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


# ---------------------------------------------------------------------------
# Analytics / System Events
# ---------------------------------------------------------------------------


class ForecastUpdated(BaseEvent):
    """Event emitted when a product forecast/demand profile is updated."""

    event_type: Literal["forecast_updated"] = "forecast_updated"

    product_id: str
    demand_profile: DemandProfile

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class AnomalyDetected(BaseEvent):
    """Generic anomaly detection event used by monitoring/analytics.

    - summary: short text describing the anomaly type.
    - metrics: optional structured metrics that explain the anomaly (small dict).
    """

    event_type: Literal["anomaly_detected"] = "anomaly_detected"

    summary: str
    metrics: Optional[Dict[str, Any]] = None
    severity: Optional[Literal["low", "medium", "high", "critical"]] = "low"

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


# ---------------------------------------------------------------------------
# Commands: Pricing
# ---------------------------------------------------------------------------


class AdjustPriceCommand(Command):
    """Command to change the price for a product.

    Business rules:
    - proposed_price uses Decimal for monetary precision and must be >= 0.
    - `effective_from` indicates when price should take effect (None = immediate).
    - `channel` is optional to target marketplace/channel-level prices.
    """

    command_type: Literal["adjust_price"] = "adjust_price"

    product_id: Optional[str] = None
    product_sku: Optional[str] = None
    proposed_price: Decimal = Field(..., ge=Decimal("0"))
    effective_from: Optional[datetime] = None
    channel: Optional[str] = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

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
    product_ids: Optional[list[str]] = None
    category: Optional[str] = None
    discount_percent: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    start: datetime
    end: Optional[datetime] = None
    channels: Optional[list[str]] = None
    notes: Optional[str] = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    @field_validator("discount_percent", mode="before")
    @classmethod
    def _coerce_discount(cls, v):
        if isinstance(v, Decimal):
            return v
        try:
            return Decimal(str(v))
        except Exception as exc:
            raise ValueError("Invalid discount_percent") from exc


# ---------------------------------------------------------------------------
# Commands: Inventory / Operations
# ---------------------------------------------------------------------------


class PlaceReplenishmentOrderCommand(Command):
    """Command to place a replenishment order with a supplier."""

    command_type: Literal["place_replenishment_order"] = "place_replenishment_order"

    product_id: str
    quantity: int = Field(..., gt=0)
    supplier_id: str
    target_warehouse: Optional[str] = None
    priority: Optional[Literal["low", "normal", "high", "urgent"]] = "normal"

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class TransferInventoryCommand(Command):
    """Command to transfer inventory between warehouses."""

    command_type: Literal["transfer_inventory"] = "transfer_inventory"

    product_id: str
    from_warehouse: str
    to_warehouse: str
    quantity: int = Field(..., gt=0)

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class UpdateSafetyStockCommand(Command):
    """Command to update safety stock thresholds for a product."""

    command_type: Literal["update_safety_stock"] = "update_safety_stock"

    product_id: str
    new_safety_stock: int = Field(..., ge=0)

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


# ---------------------------------------------------------------------------
# Commands: Customer & Marketing
# ---------------------------------------------------------------------------


class ResolveCustomerIssueCommand(Command):
    """Command for customer service agents to resolve a logged complaint.

    - refund_amount uses Decimal for monetary values and must be >= 0.
    """

    command_type: Literal["resolve_customer_issue"] = "resolve_customer_issue"

    complaint_id: Optional[str] = None
    order_id: Optional[str] = None
    resolution_action: str
    refund_amount: Optional[Decimal] = Field(None, ge=Decimal("0"))

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    @field_validator("refund_amount", mode="before")
    @classmethod
    def _coerce_refund(cls, v):
        if v is None:
            return v
        if isinstance(v, Decimal):
            return v
        try:
            return Decimal(str(v))
        except Exception as exc:
            raise ValueError("Invalid refund_amount") from exc


class StartCustomerOutreachCommand(Command):
    """Command to start an outreach/campaign targeted at a customer segment."""

    command_type: Literal["start_customer_outreach"] = "start_customer_outreach"

    segment: str
    message_template: str
    goal_metrics: Optional[Dict[str, Any]] = None
    channels: Optional[list[str]] = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


# ---------------------------------------------------------------------------
# Commands: Analytical & Logistics
# ---------------------------------------------------------------------------


class ReforecastDemandCommand(Command):
    """Request to recompute demand forecasts for a product over a timeframe."""

    command_type: Literal["reforecast_demand"] = "reforecast_demand"

    product_id: str
    timeframe_days: int = Field(..., gt=0)
    reason: Optional[str] = None

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class AdjustFulfillmentLatencyCommand(Command):
    """Command to set or adjust fulfillment latency targets (in days)."""

    command_type: Literal["adjust_fulfillment_latency"] = "adjust_fulfillment_latency"

    product_id: str
    new_latency_days: int = Field(..., ge=0)

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


# ---------------------------------------------------------------------------
# Unions and registries
# ---------------------------------------------------------------------------

AnyEvent = Union[
    SaleOccurred,
    PriceChangedExternally,
    DemandSpiked,
    StockReplenished,
    StockDepleted,
    FulfillmentDelayed,
    PromotionLaunched,
    CustomerComplaintLogged,
    ForecastUpdated,
    AnomalyDetected,
]

AnyCommand = Union[
    AdjustPriceCommand,
    LaunchPromotionCommand,
    PlaceReplenishmentOrderCommand,
    TransferInventoryCommand,
    UpdateSafetyStockCommand,
    ResolveCustomerIssueCommand,
    StartCustomerOutreachCommand,
    ReforecastDemandCommand,
    AdjustFulfillmentLatencyCommand,
]

# Optional: runtime registries mapping event_type/command_type to class for lookups.

class EventRegistry(dict):
    """Lightweight compatibility wrapper around the event-type -> class mapping.

    This wrapper provides a small, well-defined surface used by older/auxiliary
    code paths and by the EventType enum initialization. It intentionally keeps
    introspection minimal and builds lightweight metadata on demand.
    """

    def __init__(self, mapping: Dict[str, Type[BaseEvent]]):
        super().__init__(mapping)

    def get_event_class(self, event_type: str) -> Optional[Type[BaseEvent]]:
        return self.get(event_type)

    @property
    def event_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return a mapping of event_type -> lightweight metadata dict.

        Each metadata dict contains:
        - event_class: the pydantic model class for the event
        - doc: class docstring (may be None)
        - event_type: canonical event_type string
        """
        md: Dict[str, Dict[str, Any]] = {}
        for k, cls in self.items():
            md[k] = {
                "event_class": cls,
                "doc": cls.__doc__,
                "event_type": getattr(cls, "event_type", k),
            }
        return md

# Instantiate the registry using a robust extraction of the event_type value
# to avoid triggering Pydantic's attribute machinery during module import.
def _camel_to_snake(name: str) -> str:
    # Convert CamelCase to snake_case reliably.
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace("__", "_").lower()


def _extract_event_type_from_class(cls: Type[BaseEvent]) -> str:
    # Prefer a simple literal default defined directly on the class (safe __dict__ access).
    val = cls.__dict__.get("event_type", None)
    if isinstance(val, str):
        return val
    # Fallback: derive snake_case from the class name (SaleOccurred -> sale_occurred)
    return _camel_to_snake(cls.__name__)

_event_registry: EventRegistry = EventRegistry(
    {
        _extract_event_type_from_class(cls): cls
        for cls in (
            SaleOccurred,
            PriceChangedExternally,
            DemandSpiked,
            StockReplenished,
            StockDepleted,
            FulfillmentDelayed,
            PromotionLaunched,
            CustomerComplaintLogged,
            ForecastUpdated,
            AnomalyDetected,
        )
    }
)

# Build the command registry robustly without invoking Pydantic attribute access.
def _extract_command_type_from_class(cls: Type[Command]) -> str:
    # Prefer a simple literal default defined directly on the class (safe __dict__ access).
    val = cls.__dict__.get("command_type", None)
    if isinstance(val, str):
        return val
    # Fallback: derive snake_case from the class name (AdjustPriceCommand -> adjust_price_command)
    return _camel_to_snake(cls.__name__)

# Build a robust command registry that exposes multiple lookup keys for compatibility:
# - the explicit class literal (if present)
# - snake_case derived from the class name
# - a shortened form that strips a trailing "_command" suffix
_command_registry: Dict[str, Type[Command]] = {}
for cls in (
    AdjustPriceCommand,
    LaunchPromotionCommand,
    PlaceReplenishmentOrderCommand,
    TransferInventoryCommand,
    UpdateSafetyStockCommand,
    ResolveCustomerIssueCommand,
    StartCustomerOutreachCommand,
    ReforecastDemandCommand,
    AdjustFulfillmentLatencyCommand,
):
    # canonical literal (safe __dict__ read)
    literal = cls.__dict__.get("command_type")
    if isinstance(literal, str):
        _command_registry[literal] = cls

    # derived snake_case
    derived = _camel_to_snake(cls.__name__)
    _command_registry.setdefault(derived, cls)

    # short form without trailing "_command"
    if derived.endswith("_command"):
        short = derived[: -len("_command")]
        _command_registry.setdefault(short, cls)


# -- EventType enum (dynamic, derived from the runtime registry) ----------------
from enum import Enum
import re

def _safe_member_name(s: str) -> str:
    """Create a valid enum member name from an arbitrary event_type string."""
    name = re.sub(r"\W+", "_", s).upper()
    if not name:
        name = "UNKNOWN"
    if name[0].isdigit():
        name = "_" + name
    return name

_event_type_members = {_safe_member_name(k): k for k in _event_registry.keys()}

# Create a string-valued Enum so members behave like their canonical event_type strings.
EventType = Enum("EventType", _event_type_members, type=str)  # type: ignore[call-arg]

# Attach minimal metadata helpers to each member for compatibility with legacy expectations.
for member in EventType:
    cls = _event_registry.get(member.value)
    # Provide direct reference to the model class and a small metadata dict.
    setattr(member, "event_class", cls)
    setattr(
        member,
        "metadata",
        {"event_class": cls, "doc": getattr(cls, "__doc__", None), "event_type": member.value},
    )

# Compatibility accessors (keep the previous function names and behaviors).
def get_event_class_for_type(event_type: str) -> Optional[Type[BaseEvent]]:
    """Return the event class for a given event_type or None if unknown."""
    return _event_registry.get_event_class(event_type)


def get_command_class_for_type(command_type: str) -> Optional[Type[Command]]:
    """Return the command class for a given command_type or None if unknown."""
    return _command_registry.get(command_type)


__all__ = [
    # Base types
    "BaseEvent",
    "Command",
    # Events
    "SaleOccurred",
    "PriceChangedExternally",
    "DemandSpiked",
    "StockReplenished",
    "StockDepleted",
    "FulfillmentDelayed",
    "PromotionLaunched",
    "CustomerComplaintLogged",
    "ForecastUpdated",
    "AnomalyDetected",
    "AnyEvent",
    # Commands
    "AdjustPriceCommand",
    "LaunchPromotionCommand",
    "PlaceReplenishmentOrderCommand",
    "TransferInventoryCommand",
    "UpdateSafetyStockCommand",
    "ResolveCustomerIssueCommand",
    "StartCustomerOutreachCommand",
    "ReforecastDemandCommand",
    "AdjustFulfillmentLatencyCommand",
    "AnyCommand",
    # Registries / helpers
    "get_event_class_for_type",
    "get_command_class_for_type",
    # EventType enum
    "EventType",
]