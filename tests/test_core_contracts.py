from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from fba_bench_core.domain.events import AdjustPriceCommand, SaleOccurred

# Try to import Product from domain.models, fall back to legacy models.product shim if needed.
try:
    from fba_bench_core.domain.models import Product  # type: ignore
except Exception:
    try:
        from models.product import Product  # type: ignore
    except Exception:
        Product = None  # type: ignore


def test_sale_occurred_defaults_and_validation():
    """SaleOccurred should default event_type to the domain literal and populate timestamp.

    Notes:
    - The domain implementation uses the Literal value 'sale_occurred' (lowercase).
    - Invalid explicit event_type is surfaced as a pydantic.ValidationError; assert the error is raised
      and that the ValidationError reports an error referencing the permitted literal to be robust
      against Pydantic v2 error shape differences.
    """
    evt = SaleOccurred(order_id="order-1", product_sku="SKU-1", quantity=2, revenue=19.99)
    assert evt.event_type == "sale_occurred"
    assert isinstance(evt.timestamp, datetime)
    # timestamp should be recent (within a minute)
    assert datetime.utcnow() - evt.timestamp < timedelta(minutes=1)
    assert evt.tick == 0

    # invalid explicit event_type should raise ValidationError and include the allowed literal in the error details
    with pytest.raises(ValidationError) as excinfo:
        SaleOccurred(
            event_type="competitor_sale",
            order_id="o2",
            product_sku="SKU-2",
            quantity=1,
            revenue=1.0,
        )
    # Ensure the ValidationError includes at least one error mentioning the allowed literal (robust check)
    errors = excinfo.value.errors()
    assert errors and any("sale_occurred" in str(e) or "sale_occurred" in str(e.get("ctx", {})) for e in errors)


def test_sale_occurred_accepts_explicit_timestamp():
    """SaleOccurred accepts an explicit timestamp and preserves it (should be naive UTC datetime)."""
    ts = datetime(2020, 1, 1, 12, 0, 0)
    evt = SaleOccurred(timestamp=ts, order_id="order-2", product_sku="SKU-2", quantity=1, revenue=5.0)
    assert evt.timestamp == ts


def test_adjust_price_command_enforces_positive_new_price_and_coercion():
    """AdjustPriceCommand requires new_price > 0.

    The command uses Pydantic Field(gt=0) and Pydantic will coerce numeric strings to float when possible.
    Tests cover:
    - valid float input
    - numeric-string coercion
    - invalid zero and negative values raise ValidationError
    """
    # valid numeric input
    cmd = AdjustPriceCommand(product_sku="SKU-1", new_price=9.99)
    assert cmd.new_price == 9.99

    # numeric string input should be coerced to float by Pydantic
    cmd_str = AdjustPriceCommand(product_sku="SKU-1", new_price="12.50")
    assert isinstance(cmd_str.new_price, float) and cmd_str.new_price == 12.5

    # invalid inputs
    with pytest.raises(ValidationError):
        AdjustPriceCommand(product_sku="SKU-1", new_price=0)
    with pytest.raises(ValidationError):
        AdjustPriceCommand(product_sku="SKU-1", new_price=-1.0)


def test_legacy_product_coercion_and_legacy_behavior():
    """Document the current legacy Product behavior (non-Pydantic).

    The Product implementation in src/models/product.py is a legacy, hand-rolled class that:
    - coercively normalizes price/cost via Product._to_money (accepts numerics, numeric-strings, and dicts with
      'amount' or 'cents'), returning either a Money-compatible object or a raw float when Money implementations
      are not available in the environment.
    - accepts negative inventory values (legacy behavior).
    - does not raise ValidationError for negative price/cost; tests should document this behavior rather than fail.
    - returns None from calculate_profit_margin when price <= 0 (documented behavior).
    """
    if Product is None:
        pytest.skip("Product class not available for testing")

    # Detect whether Product is implemented as Pydantic (unexpected for current legacy model).
    is_pydantic = any(
        hasattr(Product, attr) for attr in ("model_fields", "__pydantic_core__", "__fields__", "__pydantic_validator__")
    )
    if is_pydantic:
        pytest.skip("Domain Product is a Pydantic model in this environment; these legacy assertions are not applicable")

    # Construction with numeric price/cost and inventory; coercion should make price comparable to float
    p = Product(price=10.0, cost=5.0, inventory_units=3)
    assert p.price == 10.0
    assert p.cost == 5.0
    assert p.inventory_units == 3
    assert p.inventory == 3

    # Numeric-string inputs are accepted and coerced
    p_str = Product(price="12.34", cost="1.23", inventory_units="4")
    assert p_str.price == 12.34
    assert p_str.cost == 1.23
    assert p_str.inventory_units == 4

    # Dict inputs with 'cents' or 'amount' are accepted
    p_cents = Product(price={"cents": 123, "currency": "USD"})
    assert p_cents.price == 1.23

    p_amount = Product(price={"amount": "2.5", "currency": "USD"})
    assert p_amount.price == 2.5

    # Negative inventory is accepted in legacy implementation (documented behavior)
    p_neg_inv = Product(price=1.0, cost=0.5, inventory_units=-5)
    assert p_neg_inv.inventory_units == -5

    # Negative price/cost do NOT raise immediately in legacy model; they are accepted and stored.
    p_neg_price = Product(price=-1.0, cost=-0.5)
    assert p_neg_price.price == -1.0
    assert p_neg_price.cost == -0.5
    # Profit margin for price <= 0 returns None per implementation
    assert p_neg_price.calculate_profit_margin() is None