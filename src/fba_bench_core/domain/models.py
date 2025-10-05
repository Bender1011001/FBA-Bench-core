"""
Domain models for FBA-Bench core domain layer (Phase B).

These models encode explicit business semantics used throughout the
simulation core: monetary precision, inventory invariants, marketplace
competitor listings, and demand profiles.

Design notes (high level):
- Monetary values use Decimal to avoid floating-point rounding issues in
  profitability calculations.
- The Product model enforces core invariants (non-negative stock, price >= cost).
  Exceptions (e.g., temporary loss-leaders) should be modeled by the simulation
  or application layers, not by the core contract.
- InventorySnapshot captures the momentary state of inventory for reconciliation,
  forecasting, and fulfillment logic.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Helper constants for documentation / defaults
DEFAULT_DECIMAL_PLACES = 2


class Product(BaseModel):
    """
    Product represents the fundamental trade unit in the simulation.

    Business invariants and rationale:
    - price and cost are Decimal to preserve financial precision.
    - price must be >= cost at the domain level to avoid modeling systemic losses.
      Allowing price == cost supports break-even scenarios; intentional loss-leading
      promotions (price < cost) are an application/simulation-level decision and
      should be represented via promotions/events rather than a contract-level rule.
    - stock is an available on-hand integer and must be >= 0.
    - max_inventory is optional; when defined it represents a physical or policy
      ceiling for stockage (used by replenishment logic).
    - fulfillment_latency (in days) expresses expected time-to-ship from the seller.
    """

    # Identifiers and classification
    product_id: str = Field(..., description="Canonical product identifier (internal).")
    sku: Optional[str] = Field(None, description="Stock keeping unit; may match marketplace SKU.")
    name: Optional[str] = Field(None, description="Human-readable product title.")
    category: Optional[str] = Field(None, description="Category or taxonomy for aggregation/segmentation.")

    # Monetary fields (Decimal for precision)
    cost: Decimal = Field(..., description="Per-unit landed cost (Decimal). Must be >= 0.")
    price: Decimal = Field(..., description="Per-unit listing price (Decimal). Must be >= cost.")

    # Inventory and fulfillment
    stock: int = Field(0, description="Available on-hand units (must be non-negative integer).")
    max_inventory: Optional[int] = Field(
        None,
        description="Optional maximum stock allowed; if set, stock must not exceed this ceiling.",
    )
    fulfillment_latency: Optional[int] = Field(
        None,
        description="Expected fulfillment latency in days (integer). Used by fulfillment & SLAs.",
        ge=0,
    )

    # Pydantic model configuration: validate assignments so runtime updates still enforce invariants.
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Field-level validators
    @field_validator("cost", "price", mode="before")
    @classmethod
    def _coerce_decimal(cls, v):
        """
        Ensure numeric inputs are converted to Decimal in a robust manner.
        Accept str/int/float but convert cautiously; floats may lose precision so
        passing strings or Decimal is preferred.
        """
        if isinstance(v, Decimal):
            return v
        if v is None:
            raise ValueError("Monetary fields cannot be None")
        try:
            return Decimal(str(v))
        except Exception as exc:
            raise ValueError(f"Invalid monetary value: {v!r}") from exc

    @field_validator("cost", "price")
    @classmethod
    def _non_negative_money(cls, v: Decimal):
        if v < Decimal("0"):
            raise ValueError("Monetary values must be non-negative")
        return v

    @field_validator("stock", "max_inventory", mode="before")
    @classmethod
    def _coerce_ints(cls, v):
        """Coerce numeric stock inputs to int and validate sign when possible."""
        if v is None:
            return v
        try:
            return int(v)
        except Exception as exc:
            raise ValueError("Stock and inventory limits must be integers") from exc

    @field_validator("stock")
    @classmethod
    def _stock_non_negative(cls, v: int):
        if v < 0:
            raise ValueError("stock must be >= 0")
        return v

    @model_validator(mode="after")
    def _price_vs_cost_and_inventory_limits(self):
        # price >= cost invariant
        if self.price < self.cost:
            # Domain-level rule: do not permit systemic losses at the model layer.
            # Applications that want to model discounts or strategic loss-leading
            # should apply those as transient events or promotion objects.
            raise ValueError("price must be greater than or equal to cost")

        # If max_inventory is specified, ensure stock is within [0, max_inventory]
        if self.max_inventory is not None:
            if self.max_inventory < 0:
                raise ValueError("max_inventory must be >= 0")
            if self.stock > self.max_inventory:
                raise ValueError("stock must not exceed max_inventory")

        return self


class InventorySnapshot(BaseModel):
    """
    Snapshot of inventory for a particular product at a specific time and location.

    Purpose:
    - Provide a normalized, auditable representation of available and reserved units
      used by forecasting, fulfillment routing, and reconciliation.
    """

    product_id: str = Field(..., description="Canonical product identifier this snapshot refers to.")
    available_units: int = Field(..., ge=0, description="Units available for sale/fulfillment.")
    reserved_units: int = Field(0, ge=0, description="Units reserved for pending orders.")
    warehouse_location: Optional[str] = Field(
        None,
        description="Identifier for warehouse/fulfillment center (optional).",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of snapshot capture.")

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_reserved_vs_available(self):
        if self.reserved_units > (self.available_units + self.reserved_units):
            # Defensive; reserved cannot exceed total physical units; this simple
            # check relies on the snapshot representing only on-hand + reserved.
            raise ValueError("reserved_units cannot exceed the total units represented")
        if self.available_units < 0 or self.reserved_units < 0:
            raise ValueError("available_units and reserved_units must be non-negative")
        return self


class CompetitorListing(BaseModel):
    """
    A single competitor listing describing a comparable SKU offered by a competitor.

    Rationale:
    - Simulations compare our product's price & fulfillment against competitor listings.
    - Fulfillment latency and marketplace are important signals for win-rate models.
    """

    sku: Optional[str] = Field(None, description="Competitor SKU, if available.")
    price: Decimal = Field(..., description="Competitor listing price (Decimal).")
    rating: Optional[float] = Field(None, ge=0.0, le=5.0, description="Optional customer rating (0-5).")
    fulfillment_latency: Optional[int] = Field(None, ge=0, description="Fulfillment latency in days.")
    marketplace: Optional[str] = Field(None, description="Marketplace or channel name (e.g., 'amazon.com').")

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @field_validator("price", mode="before")
    @classmethod
    def _coerce_price(cls, v):
        if isinstance(v, Decimal):
            return v
        try:
            return Decimal(str(v))
        except Exception as exc:
            raise ValueError("Invalid price for CompetitorListing") from exc

    @model_validator(mode="after")
    def _price_non_negative(self):
        if self.price < Decimal("0"):
            raise ValueError("Competitor listing price must be non-negative")
        return self


class Competitor(BaseModel):
    """
    Represents a competing seller in the marketplace.

    - listings: typed list of CompetitorListing rather than raw Product copies. This
      prevents accidental reuse of our Product model for external listings and keeps
      competitor metadata explicit.
    - operating_regions and primary_marketplace are optional metadata used by
      marketplace segmentation logic.
    """

    competitor_id: str = Field(..., description="Unique identifier for the competitor.")
    name: Optional[str] = Field(None, description="Display name for the competitor.")
    listings: List[CompetitorListing] = Field(default_factory=list, description="Listings offered by the competitor.")
    operating_regions: Optional[List[str]] = Field(
        None,
        description="ISO region codes or region identifiers where the competitor operates.",
    )
    primary_marketplace: Optional[str] = Field(None, description="Primary marketplace/channel name.")

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @model_validator(mode="after")
    def _unique_listing_skus(self):
        skus = [l.sku for l in self.listings if l.sku is not None]
        if len(skus) != len(set(skus)):
            raise ValueError("Competitor listings must have unique SKUs when provided")
        return self


class DemandProfile(BaseModel):
    """
    Simplified demand profile / customer segment demand model.

    Purpose:
    - Provide a compact representation of stochastic demand assumptions used by
      assortment and replenishment simulations. Fields are intentionally minimal;
      complex demand models belong in specialized modules.
    """

    product_id: str = Field(..., description="Product this demand profile pertains to.")
    daily_demand_mean: float = Field(..., ge=0.0, description="Mean expected daily demand (units/day).")
    daily_demand_std: float = Field(0.0, ge=0.0, description="Standard deviation for daily demand.")

    segment: Optional[str] = Field(None, description="Optional customer segment identifier (e.g., 'value', 'business').")

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @model_validator(mode="after")
    def _validate_stats(self):
        if self.daily_demand_std < 0:
            raise ValueError("daily_demand_std must be non-negative")
        return self


# Export list for explicit imports; keep stable names for external modules
__all__ = [
    "Product",
    "InventorySnapshot",
    "CompetitorListing",
    "Competitor",
    "DemandProfile",
]