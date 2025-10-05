"""Domain package exports for fba_bench_core.

This module exposes the core domain contracts and the expanded event/command
vocabulary introduced in Phase C.
"""

from .events import (
    # Base types
    BaseEvent,
    Command,
    # Events
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
    AnyEvent,
    # Commands
    AdjustPriceCommand,
    LaunchPromotionCommand,
    PlaceReplenishmentOrderCommand,
    TransferInventoryCommand,
    UpdateSafetyStockCommand,
    ResolveCustomerIssueCommand,
    StartCustomerOutreachCommand,
    ReforecastDemandCommand,
    AdjustFulfillmentLatencyCommand,
    AnyCommand,
    # helpers
    get_event_class_for_type,
    get_command_class_for_type,
)
from .models import (
    Product,
    InventorySnapshot,
    CompetitorListing,
    Competitor,
    DemandProfile,
)

__all__ = [
    # models
    "Product",
    "InventorySnapshot",
    "CompetitorListing",
    "Competitor",
    "DemandProfile",
    # events & commands
    "BaseEvent",
    "Command",
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
    "get_event_class_for_type",
    "get_command_class_for_type",
]