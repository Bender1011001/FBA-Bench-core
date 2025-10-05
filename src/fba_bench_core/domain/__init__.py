"""Domain package exports for fba_bench_core.

Exposes the primary domain models and Pydantic contracts for external import.
"""

from .models import Product, Competitor
from .events import (
    BaseEvent,
    SaleOccurred,
    StockUpdated,
    Command,
    AdjustPriceCommand,
    PlaceOrderCommand,
    AnyCommand,
)

__all__ = [
    "Product",
    "Competitor",
    "BaseEvent",
    "SaleOccurred",
    "StockUpdated",
    "Command",
    "AdjustPriceCommand",
    "PlaceOrderCommand",
    "AnyCommand",
]