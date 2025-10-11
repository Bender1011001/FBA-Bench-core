"""Inventory and fulfillment-related events and commands."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..models import InventorySnapshot
from .base import BaseEvent, Command


class StockReplenished(BaseEvent):
    """Event emitted when stock is replenished at a warehouse.

    Either provide `snapshot_before` and `snapshot_after` (InventorySnapshot) or
    provide `quantity_added` and `warehouse_location`. `quantity_added` must be > 0.
    """

    event_type: Literal["stock_replenished"] = "stock_replenished"

    product_id: str
    snapshot_before: InventorySnapshot | None = None
    snapshot_after: InventorySnapshot | None = None
    warehouse_location: str | None = None
    quantity_added: int | None = Field(None, gt=0)

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
    safety_stock: int | None = None
    current_snapshot: InventorySnapshot | None = None


class FulfillmentDelayed(BaseEvent):
    """Event emitted when an order fulfillment is delayed beyond SLA."""

    event_type: Literal["fulfillment_delayed"] = "fulfillment_delayed"

    order_id: str
    delay_hours: float = Field(..., ge=0.0)
    reason: str | None = None


class SupplyDisruption(BaseEvent):
    """Event emitted when a supply chain disruption occurs (e.g., supplier delay, shortage).

    - disruption_type: type of disruption (e.g., 'supplier_delay', 'material_shortage', 'logistics_issue')
    - impact_description: optional details on the impact.
    """

    event_type: Literal["supply_disruption"] = "supply_disruption"

    product_id: str
    supplier_id: str
    disruption_type: str
    impact_description: str | None = None


class PlaceReplenishmentOrderCommand(Command):
    """Command to place a replenishment order with a supplier."""

    command_type: Literal["place_replenishment_order"] = "place_replenishment_order"

    product_id: str
    quantity: int = Field(..., gt=0)
    supplier_id: str
    target_warehouse: str | None = None
    priority: Literal["low", "normal", "high", "urgent"] | None = "normal"


class TransferInventoryCommand(Command):
    """Command to transfer inventory between warehouses."""

    command_type: Literal["transfer_inventory"] = "transfer_inventory"

    product_id: str
    from_warehouse: str
    to_warehouse: str
    quantity: int = Field(..., gt=0)


class UpdateSafetyStockCommand(Command):
    """Command to update safety stock thresholds for a product."""

    command_type: Literal["update_safety_stock"] = "update_safety_stock"

    product_id: str
    new_safety_stock: int = Field(..., ge=0)


class AdjustInventoryCommand(Command):
    """Command to adjust inventory levels for a product (e.g., manual correction, write-off).

    - adjustment_quantity: positive for increase, negative for decrease.
    - warehouse_location: optional specific warehouse.
    """

    command_type: Literal["adjust_inventory"] = "adjust_inventory"

    product_id: str
    adjustment_quantity: int
    warehouse_location: str | None = None
