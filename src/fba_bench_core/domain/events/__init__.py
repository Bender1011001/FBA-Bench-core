"""Compatibility layer for fba_bench_core.domain.events.

This module re-exports all event classes, command classes, unions (AnyEvent, AnyCommand),
EventType enum, and helper functions from the modular domain/events submodules to
provide full backward compatibility. Existing code importing from fba_bench_core.domain.events
will continue to work unchanged.

Do not add new definitions here; maintain only re-exports and registry logic.
"""

from __future__ import annotations

import re
from enum import Enum

from .analytics import (
    AdjustFulfillmentLatencyCommand,
    AnomalyDetected,
    ForecastUpdated,
    NegotiateSupplyCommand,
    ReforecastDemandCommand,
)
from .base import BaseEvent, Command
from .inventory import (
    AdjustInventoryCommand,
    FulfillmentDelayed,
    PlaceReplenishmentOrderCommand,
    StockDepleted,
    StockReplenished,
    SupplyDisruption,
    TransferInventoryCommand,
    UpdateSafetyStockCommand,
)
from .marketing import (
    CustomerComplaintLogged,
    PromotionLaunched,
    ResolveCustomerIssueCommand,
    RespondToComplaintCommand,
    StartCustomerOutreachCommand,
)
from .pricing import (
    AdjustPriceCommand,
    CompetitorAction,
    DemandSpiked,
    LaunchPromotionCommand,
    MonitorCompetitorCommand,
    PriceChangedExternally,
    SaleOccurred,
)

# Unions for type checking
AnyEvent = (
    SaleOccurred
    | PriceChangedExternally
    | DemandSpiked
    | CompetitorAction
    | StockReplenished
    | StockDepleted
    | FulfillmentDelayed
    | SupplyDisruption
    | PromotionLaunched
    | CustomerComplaintLogged
    | ForecastUpdated
    | AnomalyDetected
)

AnyCommand = (
    AdjustPriceCommand
    | LaunchPromotionCommand
    | PlaceReplenishmentOrderCommand
    | TransferInventoryCommand
    | UpdateSafetyStockCommand
    | AdjustInventoryCommand
    | ResolveCustomerIssueCommand
    | StartCustomerOutreachCommand
    | RespondToComplaintCommand
    | ReforecastDemandCommand
    | AdjustFulfillmentLatencyCommand
    | NegotiateSupplyCommand
    | MonitorCompetitorCommand
)


# Registry setup (moved from original events.py)
def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace("__", "_").lower()


def _extract_event_type_from_class(cls: type[BaseEvent]) -> str:
    val = cls.__dict__.get("event_type", None)
    if isinstance(val, str):
        return val
    return _camel_to_snake(cls.__name__)


_event_registry: dict[str, type[BaseEvent]] = {
    _extract_event_type_from_class(cls): cls
    for cls in (
        SaleOccurred,
        PriceChangedExternally,
        DemandSpiked,
        CompetitorAction,
        StockReplenished,
        StockDepleted,
        FulfillmentDelayed,
        SupplyDisruption,
        PromotionLaunched,
        CustomerComplaintLogged,
        ForecastUpdated,
        AnomalyDetected,
    )
}


def _extract_command_type_from_class(cls: type[Command]) -> str:
    val = cls.__dict__.get("command_type", None)
    if isinstance(val, str):
        return val
    return _camel_to_snake(cls.__name__)


_command_registry: dict[str, type[Command]] = {}
for cls in (
    AdjustPriceCommand,
    LaunchPromotionCommand,
    PlaceReplenishmentOrderCommand,
    TransferInventoryCommand,
    UpdateSafetyStockCommand,
    AdjustInventoryCommand,
    ResolveCustomerIssueCommand,
    StartCustomerOutreachCommand,
    RespondToComplaintCommand,
    ReforecastDemandCommand,
    AdjustFulfillmentLatencyCommand,
    NegotiateSupplyCommand,
    MonitorCompetitorCommand,
):
    literal = cls.__dict__.get("command_type")
    if isinstance(literal, str):
        _command_registry[literal] = cls

    derived = _camel_to_snake(cls.__name__)
    _command_registry.setdefault(derived, cls)

    if derived.endswith("_command"):
        short = derived[: -len("_command")]
        _command_registry.setdefault(short, cls)


# EventType enum (dynamic, derived from the runtime registry)
def _safe_member_name(s: str) -> str:
    name = re.sub(r"\W+", "_", s).upper()
    if not name:
        name = "UNKNOWN"
    if name[0].isdigit():
        name = "_" + name
    return name


_event_type_members = {_safe_member_name(k): k for k in _event_registry.keys()}

EventType = Enum("EventType", _event_type_members, type=str)

# Attach metadata to enum members
for member in EventType:
    cls = _event_registry.get(member.value)
    setattr(member, "event_class", cls)
    setattr(
        member,
        "metadata",
        {
            "event_class": cls,
            "doc": getattr(cls, "__doc__", None),
            "event_type": member.value,
        },
    )


# Helper functions
def get_event_class_for_type(event_type: str) -> type[BaseEvent] | None:
    """Return the event class for a given event_type or None if unknown."""
    return _event_registry.get(event_type)


def get_command_class_for_type(command_type: str) -> type[Command] | None:
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
    "CompetitorAction",
    "StockReplenished",
    "StockDepleted",
    "FulfillmentDelayed",
    "SupplyDisruption",
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
    "AdjustInventoryCommand",
    "ResolveCustomerIssueCommand",
    "StartCustomerOutreachCommand",
    "RespondToComplaintCommand",
    "ReforecastDemandCommand",
    "AdjustFulfillmentLatencyCommand",
    "NegotiateSupplyCommand",
    "MonitorCompetitorCommand",
    "AnyCommand",
    # Registries / helpers
    "get_event_class_for_type",
    "get_command_class_for_type",
    # EventType enum
    "EventType",
]
