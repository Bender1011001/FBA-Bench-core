"""Tests for domain events and commands (happy paths, edge cases, validation)."""

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fba_bench_core.domain.events import (
    AdjustInventoryCommand,
    AdjustPriceCommand,
    CompetitorAction,
    DemandSpiked,
    MonitorCompetitorCommand,
    NegotiateSupplyCommand,
    PlaceReplenishmentOrderCommand,
    PriceChangedExternally,
    PromotionLaunched,
    ResolveCustomerIssueCommand,
    RespondToComplaintCommand,
    SaleOccurred,
    StockReplenished,
    SupplyDisruption,
    get_command_class_for_type,
    get_event_class_for_type,
)
from fba_bench_core.domain.models import CompetitorListing, DemandProfile


def test_sale_occurred_happy_path_and_defaults():
    evt = SaleOccurred(
        order_id="ord-1",
        product_id="prod-1",
        quantity=3,
        revenue=Decimal("29.97"),
        gross_margin=Decimal("0.25"),
    )
    assert evt.event_type == "sale_occurred"
    assert evt.tick == 0
    assert isinstance(evt.timestamp, datetime)
    assert evt.revenue == Decimal("29.97")
    assert evt.gross_margin == Decimal("0.25")


@pytest.mark.parametrize(
    "kwargs,errmsg",
    [
        ({"quantity": 0, "revenue": Decimal("10.00")}, "quantity"),
        ({"quantity": 1, "revenue": Decimal("-1.00")}, "revenue"),
        (
            {"quantity": 1, "revenue": Decimal("1.00"), "gross_margin": Decimal("2.0")},
            "gross_margin",
        ),
    ],
)
def test_sale_occurred_invalid_cases_raise(kwargs, errmsg):
    base = {"order_id": "o", "product_id": "p"}
    base.update(kwargs)
    with pytest.raises(ValidationError) as ei:
        SaleOccurred(
            order_id=base["order_id"],
            product_id=base["product_id"],
            **{k: v for k, v in base.items() if k not in ("order_id", "product_id")},
        )
    assert errmsg in str(ei.value)


def test_price_changed_externally_embeds_listing_and_invalid_price():
    listing = CompetitorListing(sku="ck-1", price="7.50", rating=4.0)
    evt = PriceChangedExternally(competitor_id="comp-1", listing=listing)
    assert evt.event_type == "price_changed_externally"
    assert evt.listing.price == Decimal("7.50")

    # invalid listing price should raise when constructing listing
    with pytest.raises(ValidationError):
        CompetitorListing(sku="ck-2", price="not-a-number")


def test_stock_replenished_requires_snapshot_or_quantity():
    # missing all descriptors should fail
    with pytest.raises(ValidationError):
        StockReplenished(product_id="p1")
    # valid with quantity_added
    evt = StockReplenished(product_id="p1", quantity_added=5)
    assert evt.quantity_added == 5
    # invalid negative quantity_added not allowed by gt=0
    with pytest.raises(ValidationError):
        StockReplenished(product_id="p2", quantity_added=0)


def test_promotion_launched_discount_coercion_and_bounds():
    start = datetime.utcnow()
    # valid as string -> Decimal
    promo = PromotionLaunched(
        promotion_id="promo1", discount_percent="0.50", start=start
    )
    assert promo.discount_percent == Decimal("0.50")
    assert promo.event_type == "promotion_launched"
    # out of bounds discount
    with pytest.raises(ValidationError):
        PromotionLaunched(
            promotion_id="promo2", discount_percent=Decimal("1.5"), start=start
        )


def test_demand_spiked_delta_and_optional_profile():
    # zero/negative delta invalid
    with pytest.raises(ValidationError):
        DemandSpiked(product_id="p1", delta="0", trigger="t")
    # valid with numeric string coercion
    dp_payload = DemandProfile(
        product_id="p1", daily_demand_mean=2.0, daily_demand_std=0.5
    )
    ds = DemandSpiked(
        product_id="p1", delta="1.25", trigger="seasonal", demand_profile=dp_payload
    )
    assert ds.delta == Decimal("1.25")
    assert ds.demand_profile.daily_demand_mean == 2.0


# -------------------------
# Commands
# -------------------------
def test_adjust_price_command_decimal_coercion_and_non_negative():
    cmd = AdjustPriceCommand(product_id="p1", proposed_price="3.99")
    assert isinstance(cmd.proposed_price, Decimal) and cmd.proposed_price == Decimal(
        "3.99"
    )
    with pytest.raises(ValidationError):
        AdjustPriceCommand(product_id="p1", proposed_price=Decimal("-0.01"))


def test_place_replenishment_order_command_valid_and_invalid():
    # invalid quantity (<=0)
    with pytest.raises(ValidationError):
        PlaceReplenishmentOrderCommand(product_id="p1", quantity=0, supplier_id="s1")
    # valid positive quantity and priority allowed
    cmd = PlaceReplenishmentOrderCommand(
        product_id="p1", quantity=10, supplier_id="s1", priority="urgent"
    )
    assert cmd.quantity == 10
    assert cmd.priority == "urgent"


def test_competitor_action_event():
    evt = CompetitorAction(
        competitor_id="comp-1",
        action_type="price_adjustment",
        details={"old_price": 10.0, "new_price": 9.5},
    )
    assert evt.event_type == "competitor_action"
    assert evt.competitor_id == "comp-1"
    assert evt.action_type == "price_adjustment"
    assert evt.details["new_price"] == 9.5


def test_supply_disruption_event():
    evt = SupplyDisruption(
        product_id="prod-1",
        supplier_id="sup-1",
        disruption_type="supplier_delay",
        impact_description="Delayed by 3 days",
    )
    assert evt.event_type == "supply_disruption"
    assert evt.product_id == "prod-1"
    assert evt.supplier_id == "sup-1"
    assert evt.disruption_type == "supplier_delay"


def test_adjust_inventory_command():
    cmd = AdjustInventoryCommand(
        product_id="prod-1", adjustment_quantity=10, warehouse_location="wh-1"
    )
    assert cmd.command_type == "adjust_inventory"
    assert cmd.product_id == "prod-1"
    assert cmd.adjustment_quantity == 10
    assert cmd.warehouse_location == "wh-1"


def test_respond_to_complaint_command():
    cmd = RespondToComplaintCommand(
        complaint_id="comp-1",
        response_action="apology",
        response_message="We apologize for the inconvenience.",
    )
    assert cmd.command_type == "respond_to_complaint"
    assert cmd.complaint_id == "comp-1"
    assert cmd.response_action == "apology"
    assert cmd.response_message == "We apologize for the inconvenience."


def test_negotiate_supply_command():
    cmd = NegotiateSupplyCommand(
        supplier_id="sup-1",
        product_id="prod-1",
        negotiation_terms={"price": 5.0, "lead_time": 7},
    )
    assert cmd.command_type == "negotiate_supply"
    assert cmd.supplier_id == "sup-1"
    assert cmd.product_id == "prod-1"
    assert cmd.negotiation_terms["price"] == 5.0


def test_monitor_competitor_command():
    cmd = MonitorCompetitorCommand(
        competitor_id="comp-1", monitoring_focus=["pricing", "inventory"]
    )
    assert cmd.command_type == "monitor_competitor"
    assert cmd.competitor_id == "comp-1"
    assert "pricing" in cmd.monitoring_focus


def test_resolve_customer_issue_command_refund_and_command_type_literal():
    with pytest.raises(ValidationError):
        ResolveCustomerIssueCommand(
            resolution_action="fix", refund_amount=Decimal("-1.00")
        )
    cmd_cls = get_command_class_for_type("adjust_inventory")
    assert cmd_cls is not None and cmd_cls is AdjustInventoryCommand
    cmd_cls = get_command_class_for_type("respond_to_complaint")
    assert cmd_cls is not None and cmd_cls is RespondToComplaintCommand
    cmd_cls = get_command_class_for_type("negotiate_supply")
    assert cmd_cls is not None and cmd_cls is NegotiateSupplyCommand
    cmd_cls = get_command_class_for_type("monitor_competitor")
    assert cmd_cls is not None and cmd_cls is MonitorCompetitorCommand

    evt_cls = get_event_class_for_type("competitor_action")
    assert evt_cls is not None and evt_cls is CompetitorAction
    evt_cls = get_event_class_for_type("supply_disruption")
    assert evt_cls is not None and evt_cls is SupplyDisruption

    cmd = ResolveCustomerIssueCommand(resolution_action="refund", refund_amount="2.50")
    assert cmd.refund_amount == Decimal("2.50")
    assert cmd.command_type == "resolve_customer_issue"


def test_registry_helpers_lookup_and_unknown_return_none():
    evt_cls = get_event_class_for_type("sale_occurred")
    assert evt_cls is not None and evt_cls is SaleOccurred
    assert get_event_class_for_type("no_such_event") is None

    cmd_cls = get_command_class_for_type("adjust_price")
    assert cmd_cls is not None and cmd_cls is AdjustPriceCommand
    assert get_command_class_for_type("no_such_command") is None


def test_eventtype_enum_and_metadata_exposed():
    """Ensure EventType enum initializes at import time and exposes metadata."""
    from fba_bench_core.domain.events import EventType, SaleOccurred

    # Member name generated from "sale_occurred" should be present
    assert hasattr(EventType, "SALE_OCCURRED")
    member = EventType.SALE_OCCURRED

    # The enum value is the canonical event_type string
    assert member.value == "sale_occurred"

    # Compatibility helpers attached during module import
    assert hasattr(member, "event_class")
    assert hasattr(member, "metadata")

    # Validate the metadata contents reference the expected class and event_type
    assert member.event_class is SaleOccurred
    assert member.metadata["event_type"] == "sale_occurred"
