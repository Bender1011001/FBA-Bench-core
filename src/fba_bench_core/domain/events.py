"""Domain events and commands implemented as Pydantic contracts (Phase 4).

This module defines strict, validated event and command contracts used across
the system. Contracts are implemented using Pydantic v2 BaseModel with
Field validators and model_config applied where appropriate.
"""

from datetime import datetime
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------------------------
# Base event and concrete event definitions
# ---------------------------------------------------------------------------


class BaseEvent(BaseModel):
    """Base contract for domain events.

    Subclasses should override `event_type` with a Literal value to provide
    a strict discriminator. `timestamp` is recorded in UTC and defaults to the
    moment of model creation. `tick` represents a non-negative simulation or
    system tick and is validated to be >= 0.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tick: int = Field(0, ge=0)


class SaleOccurred(BaseEvent):
    """Event emitted when a sale completes for a given product SKU.

    Fields:
    - order_id: unique identifier for the order
    - product_sku: SKU of the sold product
    - quantity: number of units sold (>= 1)
    - revenue: total revenue for the sale (must be > 0)
    - currency: ISO currency code (restricted to 'USD' here)
    """

    event_type: Literal["sale_occurred"] = "sale_occurred"
    order_id: str
    product_sku: str
    quantity: int = Field(..., ge=1)
    revenue: float = Field(..., gt=0)
    currency: Literal["USD"] = "USD"


class StockUpdated(BaseEvent):
    """Event emitted when inventory levels for a SKU change.

    Fields:
    - product_sku: SKU whose levels were updated
    - previous_level: previous stock level (>= 0)
    - new_level: new stock level (>= 0)
    - reason: optional human-readable reason for the update
    """

    event_type: Literal["stock_updated"] = "stock_updated"
    product_sku: str
    previous_level: int = Field(..., ge=0)
    new_level: int = Field(..., ge=0)
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Command contracts
# ---------------------------------------------------------------------------


class Command(BaseModel):
    """Base contract for domain commands.

    Commands represent intentions to change system state. Subclasses should
    override `command_type` with a Literal discriminator.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    command_type: str


class AdjustPriceCommand(Command):
    """Command to change the price of a product SKU."""

    command_type: Literal["adjust_price"] = "adjust_price"
    product_sku: str
    new_price: float = Field(..., gt=0)
    reason: Optional[str] = None


class PlaceOrderCommand(Command):
    """Command representing a request to place an order for a SKU."""

    command_type: Literal["place_order"] = "place_order"
    product_sku: str
    quantity: int = Field(..., ge=1)
    notes: Optional[str] = None


# Union alias for typed command handling
AnyCommand = Union[AdjustPriceCommand, PlaceOrderCommand]

__all__ = [
    "BaseEvent",
    "SaleOccurred",
    "StockUpdated",
    "Command",
    "AdjustPriceCommand",
    "PlaceOrderCommand",
    "AnyCommand",
]