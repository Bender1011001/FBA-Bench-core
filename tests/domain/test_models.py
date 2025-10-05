"""Tests for domain models: Product, InventorySnapshot, CompetitorListing, Competitor, DemandProfile."""
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fba_bench_core.domain.models import (
    Competitor,
    CompetitorListing,
    DemandProfile,
    InventorySnapshot,
    Product,
)


def test_product_happy_path_decimal_and_invariants():
    p = Product(
        product_id="prod-1",
        sku="SKU-1",
        cost="2.50",
        price=Decimal("5.00"),
        stock=10,
        max_inventory=100,
        fulfillment_latency=1,
    )
    assert isinstance(p.cost, Decimal) and p.cost == Decimal("2.50")
    assert isinstance(p.price, Decimal) and p.price == Decimal("5.00")
    assert p.price >= p.cost
    assert 0 <= p.stock <= p.max_inventory


@pytest.mark.parametrize(
    "cost,price,expected_msg",
    [
        ("5.00", "4.00", "price must be greater than or equal to cost"),
        ("-1.00", "1.00", "Monetary values must be non-negative"),
    ],
)
def test_product_price_vs_cost_and_negative_cost(cost, price, expected_msg):
    with pytest.raises(ValidationError) as exc:
        Product(product_id="prod-err", cost=cost, price=price)
    assert expected_msg in str(exc.value)


def test_product_negative_stock_and_exceeds_max_inventory():
    with pytest.raises(ValidationError):
        Product(product_id="p-neg-stock", cost="1.00", price="2.00", stock=-1)
    with pytest.raises(ValidationError):
        Product(product_id="p-too-many", cost="1.00", price="2.00", stock=11, max_inventory=10)


def test_product_invalid_decimal_string_for_money():
    with pytest.raises(ValidationError):
        Product(product_id="p-bad-money", cost="not-a-number", price="2.00")


def test_product_immutability_validate_assignment_and_model_copy():
    p = Product(product_id="p6", cost="1.00", price="2.00", stock=1)
    # invalid direct assignment should raise ValidationError due to validate_assignment
    with pytest.raises(ValidationError):
        p.price = Decimal("-1.00")
    # valid model_copy update creates a new instance and original remains unchanged
    p2 = p.model_copy(update={"price": Decimal("3.00")})
    assert p2.price == Decimal("3.00")
    assert p.price == Decimal("2.00")


def test_inventory_snapshot_validation_and_reserved_logic():
    # negative available/reserved should be rejected
    with pytest.raises(ValidationError):
        InventorySnapshot(product_id="s1", available_units=-1, reserved_units=0)
    with pytest.raises(ValidationError):
        InventorySnapshot(product_id="s1", available_units=1, reserved_units=-2)
    # valid snapshot
    s = InventorySnapshot(product_id="s2", available_units=5, reserved_units=2)
    assert s.available_units == 5 and s.reserved_units == 2


def test_competitor_listing_price_coercion_and_non_negative():
    cl = CompetitorListing(sku="c1", price="3.00", rating=4.0, fulfillment_latency=2, marketplace="amazon.com")
    assert isinstance(cl.price, Decimal) and cl.price == Decimal("3.00")
    # negative competitor price invalid
    with pytest.raises(ValidationError):
        CompetitorListing(sku="c2", price=Decimal("-1.00"))


def test_competitor_unique_listing_skus_enforced():
    l1 = CompetitorListing(sku="dup", price="1.00")
    l2 = CompetitorListing(sku="dup", price="2.00")
    with pytest.raises(ValidationError):
        Competitor(competitor_id="comp1", listings=[l1, l2])


def test_demand_profile_std_validation_and_happy_path():
    with pytest.raises(ValidationError):
        DemandProfile(product_id="d1", daily_demand_mean=1.0, daily_demand_std=-0.1)
    dp = DemandProfile(product_id="d2", daily_demand_mean=5.0, daily_demand_std=1.0)
    assert dp.daily_demand_mean == 5.0 and dp.daily_demand_std == 1.0