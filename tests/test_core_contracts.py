"""
Comprehensive pytest suite for core contracts (domain models, events/commands, config).

This single-file suite replaces the earlier minimal tests and exercises:
- Domain models: Product, InventorySnapshot, CompetitorListing, Competitor, DemandProfile
- Events: SaleOccurred, PriceChangedExternally, StockReplenished, PromotionLaunched, etc.
- Commands: AdjustPriceCommand, PlaceReplenishmentOrderCommand, ResolveCustomerIssueCommand
- Configuration contracts: BaseAgentConfig, BaseServiceConfig

Notes:
- Tests assert Pydantic ValidationError for invalid inputs.
- Uses Decimal for precise monetary assertions.
- Tests reflect current model behavior in source files under src/fba_bench_core/.
"""

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fba_bench_core.config import BaseAgentConfig, BaseServiceConfig
from fba_bench_core.domain import (
    Competitor,
    CompetitorListing,
    DemandProfile,
    InventorySnapshot,
    Product,
)
from fba_bench_core.domain.events import (
    AdjustPriceCommand,
    DemandSpiked,
    PlaceReplenishmentOrderCommand,
    PriceChangedExternally,
    PromotionLaunched,
    ResolveCustomerIssueCommand,
    SaleOccurred,
    StockReplenished,
    get_command_class_for_type,
    get_event_class_for_type,
)


# -------------------------
# Domain model tests
# -------------------------
def test_product_happy_path_and_decimal_fields():
    """Product should accept Decimal/string numeric monetary values and preserve Decimal semantics."""
    p = Product(
        product_id="p1",
        sku="SKU-1",
        name="Widget",
        cost="2.50",
        price=Decimal("5.00"),
        stock=10,
        max_inventory=100,
        fulfillment_latency=2,
    )
    assert isinstance(p.cost, Decimal) and p.cost == Decimal("2.50")
    assert isinstance(p.price, Decimal) and p.price == Decimal("5.00")
    assert p.price >= p.cost
    assert 0 <= p.stock <= p.max_inventory


@pytest.mark.parametrize(
    "cost,price,exc_match",
    [
        ("5.00", "4.00", "price must be greater than or equal to cost"),  # price < cost
        ("-1.00", "1.00", "Monetary values must be non-negative"),  # negative cost
    ],
)
def test_product_invalid_price_and_negative_cost(cost, price, exc_match):
    with pytest.raises(ValidationError) as ei:
        Product(product_id="p2", cost=cost, price=price)
    errors = str(ei.value)
    assert exc_match in errors


def test_product_negative_stock_and_exceeds_max_inventory():
    # negative stock
    with pytest.raises(ValidationError):
        Product(product_id="p3", cost="1.0", price="2.0", stock=-5)
    # stock exceeding max_inventory
    with pytest.raises(ValidationError):
        Product(product_id="p4", cost="1.0", price="2.0", stock=101, max_inventory=100)


def test_product_invalid_decimal_string():
    with pytest.raises(ValidationError):
        Product(product_id="p5", cost="not-a-number", price="2.0")


def test_product_assignment_validation_and_model_copy():
    p = Product(product_id="p6", cost="1.00", price="2.00", stock=1)
    # Attempt invalid direct assignment should raise ValidationError because validate_assignment=True
    with pytest.raises(ValidationError):
        p.price = Decimal("-1.00")
    # Valid update via model_copy(update=...) should succeed and produce a new instance
    p2 = p.model_copy(update={"price": Decimal("3.00")})
    assert p2.price == Decimal("3.00")
    assert p.price == Decimal("2.00")  # original unchanged


def test_inventory_snapshot_negative_values_and_reserved_logic():
    # negative available_units rejected
    with pytest.raises(ValidationError):
        InventorySnapshot(product_id="p1", available_units=-1, reserved_units=0)
    # negative reserved_units rejected
    with pytest.raises(ValidationError):
        InventorySnapshot(product_id="p1", available_units=1, reserved_units=-2)
    # The current model validator uses a defensive check; create a normal snapshot
    s = InventorySnapshot(product_id="p1", available_units=5, reserved_units=2)
    assert s.available_units == 5 and s.reserved_units == 2


def test_competitor_listing_and_unique_skus():
    # valid listing
    cl = CompetitorListing(
        sku="csku1",
        price="3.00",
        rating=4.5,
        fulfillment_latency=1,
        marketplace="amazon.com",
    )
    assert isinstance(cl.price, Decimal) and cl.price == Decimal("3.00")
    # competitor with duplicate SKU should raise
    c = {
        "competitor_id": "comp1",
        "listings": [cl, CompetitorListing(sku="csku1", price="4.00")],
    }
    with pytest.raises(ValidationError):
        Competitor(**c)


def test_demand_profile_invalid_std_and_happy_path():
    with pytest.raises(ValidationError):
        DemandProfile(product_id="p1", daily_demand_mean=1.0, daily_demand_std=-0.1)
    dp = DemandProfile(product_id="p1", daily_demand_mean=2.0, daily_demand_std=0.5)
    assert dp.daily_demand_mean == 2.0 and dp.daily_demand_std == 0.5


# -------------------------
# Event tests
# -------------------------
def test_sale_occurred_happy_path_and_invalid_values():
    evt = SaleOccurred(
        order_id="o1",
        product_id="p1",
        quantity=2,
        revenue=Decimal("10.50"),
        gross_margin=Decimal("0.2"),
    )
    assert evt.event_type == "sale_occurred"
    assert isinstance(evt.timestamp, datetime)
    # invalid revenue (<=0) and invalid quantity (<1)
    with pytest.raises(ValidationError):
        SaleOccurred(order_id="o2", quantity=0, revenue=Decimal("1.00"))
    with pytest.raises(ValidationError):
        SaleOccurred(order_id="o3", quantity=1, revenue=Decimal("-1.00"))
    # gross_margin out of bounds
    with pytest.raises(ValidationError):
        SaleOccurred(
            order_id="o4",
            quantity=1,
            revenue=Decimal("1.00"),
            gross_margin=Decimal("2.0"),
        )


def test_price_changed_externally_embeds_competitor_listing():
    listing = CompetitorListing(sku="cs1", price="7.50", rating=4.0)
    evt = PriceChangedExternally(competitor_id="c1", listing=listing)
    assert evt.event_type == "price_changed_externally"
    assert evt.listing.sku == "cs1"


def test_stock_replenished_requires_snapshot_or_quantity():
    # missing snapshot_before/after and quantity_added should fail
    with pytest.raises(ValidationError):
        StockReplenished(product_id="p1")
    # valid with quantity_added
    evt = StockReplenished(product_id="p1", quantity_added=10)
    assert evt.quantity_added == 10


def test_promotion_launched_discount_bounds_and_coercion():
    start = datetime.utcnow()
    # valid discount as string/Decimal
    promo = PromotionLaunched(
        promotion_id="promo1", discount_percent="0.25", start=start, product_ids=["p1"]
    )
    assert isinstance(
        promo.discount_percent, Decimal
    ) and promo.discount_percent == Decimal("0.25")
    # discount out of range
    with pytest.raises(ValidationError):
        PromotionLaunched(
            promotion_id="promo2", discount_percent=Decimal("1.5"), start=start
        )


def test_demand_spiked_delta_coercion_and_validation():
    with pytest.raises(ValidationError):
        DemandSpiked(product_id="p1", delta="0", trigger="t")
    ds = DemandSpiked(product_id="p1", delta="1.5", trigger="seasonal")
    assert isinstance(ds.delta, Decimal) and ds.delta == Decimal("1.5")


# -------------------------
# Command tests
# -------------------------
def test_adjust_price_command_decimal_coercion_and_boundaries():
    cmd = AdjustPriceCommand(product_id="p1", proposed_price="3.50")
    assert isinstance(cmd.proposed_price, Decimal) and cmd.proposed_price == Decimal(
        "3.50"
    )
    # proposed_price negative invalid
    with pytest.raises(ValidationError):
        AdjustPriceCommand(product_id="p1", proposed_price=Decimal("-1.00"))


def test_place_replenishment_order_command_quantity_positive():
    with pytest.raises(ValidationError):
        PlaceReplenishmentOrderCommand(product_id="p1", quantity=0, supplier_id="s1")
    cmd = PlaceReplenishmentOrderCommand(
        product_id="p1", quantity=5, supplier_id="s1", priority="high"
    )
    assert cmd.quantity == 5 and cmd.priority == "high"


def test_resolve_customer_issue_command_refund_coercion_and_negative_invalid():
    with pytest.raises(ValidationError):
        ResolveCustomerIssueCommand(
            resolution_action="fix", refund_amount=Decimal("-1.0")
        )
    cmd = ResolveCustomerIssueCommand(resolution_action="refund", refund_amount="2.50")
    assert isinstance(cmd.refund_amount, Decimal) and cmd.refund_amount == Decimal(
        "2.50"
    )
    assert cmd.command_type == "resolve_customer_issue"


def test_registry_lookup_helpers_return_expected_and_none_for_unknown():
    evt_cls = get_event_class_for_type("sale_occurred")
    assert evt_cls is not None and evt_cls is SaleOccurred
    assert get_event_class_for_type("non_existent_event") is None

    cmd_cls = get_command_class_for_type("adjust_price")
    assert cmd_cls is not None and cmd_cls is AdjustPriceCommand
    assert get_command_class_for_type("non_existent_command") is None


# -------------------------
# Configuration tests
# -------------------------
def test_base_agent_config_happy_and_invalid_cases_and_immutability():
    cfg = BaseAgentConfig(
        agent_id="agent_1",
        poll_interval_seconds=30,
        metadata={"env": "test", "retries": 3},
    )
    assert cfg.agent_id == "agent_1"
    assert cfg.poll_interval_seconds == 30
    assert cfg.metadata["env"] == "test"
    # invalid agent_id (spaces)
    with pytest.raises(ValidationError):
        BaseAgentConfig(agent_id="invalid id")
    # negative poll_interval
    with pytest.raises(ValidationError):
        BaseAgentConfig(agent_id="agent2", poll_interval_seconds=-1)
    # nested metadata (non-primitive) should be rejected
    with pytest.raises(ValidationError):
        BaseAgentConfig(agent_id="agent3", metadata={"nested": {"a": 1}})

    # immutability: frozen models raise TypeError on assignment
    with pytest.raises(TypeError):
        cfg.agent_id = "new_id"
    # model_copy(update=...) should produce modified copy
    cfg2 = cfg.model_copy(update={"poll_interval_seconds": 60})
    assert cfg2.poll_interval_seconds == 60
    assert cfg.poll_interval_seconds == 30


def test_base_service_config_similar_semantics():
    s = BaseServiceConfig(service_id="svc1", metadata={"enabled": True})
    assert s.service_id == "svc1"
    with pytest.raises(ValidationError):
        BaseServiceConfig(service_id="bad id")
    with pytest.raises(TypeError):
        s.service_id = "svc2"
    s2 = s.model_copy(update={"default_region": "us-west-2"})
    assert s2.default_region == "us-west-2"
