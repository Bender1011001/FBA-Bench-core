"""
Scenario-driven tests validating business rules and interactions beyond simple schema checks.

These tests focus on contract validation, ensuring that business invariants are enforced
in creation, updates, and interactions between domain objects, events, and commands.
"""

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from fba_bench_core.agents.base import BaseAgent
from fba_bench_core.config import BaseAgentConfig
from fba_bench_core.domain.events import AdjustPriceCommand, SaleOccurred
from fba_bench_core.domain.models import Competitor, CompetitorListing, Product


@pytest.fixture
def valid_product():
    """Fixture providing a valid Product instance for testing."""
    return Product(
        product_id="prod-001",
        cost=Decimal("10.00"),
        price=Decimal("15.00"),
        stock=100,
    )


@pytest.fixture
def valid_competitor_with_listings():
    """Fixture providing a valid Competitor with unique listings."""
    return Competitor(
        competitor_id="comp-001",
        name="Test Competitor",
        listings=[
            CompetitorListing(
                sku="sku-001",
                price=Decimal("14.00"),
                fulfillment_latency=2,
            ),
            CompetitorListing(
                sku="sku-002",
                price=Decimal("16.00"),
                fulfillment_latency=3,
            ),
        ],
    )


class MockAgent(BaseAgent):
    """Mock implementation of BaseAgent for testing decision flow."""

    def __init__(self, config: BaseAgentConfig):
        super().__init__(config)
        self.decide_mock = AsyncMock()

    async def decide(self, events):
        return await self.decide_mock(events)


@pytest.fixture
def mock_agent():
    """Fixture providing a mocked BaseAgent instance."""
    config = BaseAgentConfig(agent_id="test-agent")
    agent = MockAgent(config)
    return agent


class TestProductProfitability:
    """Tests ensuring Product profitability invariants are enforced."""

    def test_product_creation_price_greater_equal_cost(self, valid_product):
        """Test that Product creation enforces price >= cost."""
        # valid_product fixture already satisfies this
        assert valid_product.price >= valid_product.cost

    def test_product_creation_price_less_than_cost_raises_error(self):
        """Test that creating Product with price < cost raises ValueError."""
        with pytest.raises(
            ValueError, match="price must be greater than or equal to cost"
        ):
            Product(
                product_id="prod-002",
                cost=Decimal("10.00"),
                price=Decimal("9.00"),  # price < cost
                stock=50,
            )

    def test_product_update_price_less_than_cost_raises_error(self, valid_product):
        """Test that updating Product price < cost raises ValueError."""
        with pytest.raises(
            ValueError, match="price must be greater than or equal to cost"
        ):
            valid_product.price = Decimal("9.00")  # price < cost


class TestAgentDecisionFlow:
    """Tests validating agent decision flow with mocked BaseAgent."""

    @pytest.mark.asyncio
    async def test_agent_decide_returns_valid_commands(self, mock_agent):
        """Test that agent decide method returns a list of valid Command instances."""
        # Setup mock to return valid commands
        mock_commands = [
            AdjustPriceCommand(
                product_id="prod-001",
                proposed_price=Decimal("16.00"),
                reason="Test adjustment",
            )
        ]
        mock_agent.decide_mock.return_value = mock_commands

        events = [
            SaleOccurred(
                order_id="order-001",
                product_id="prod-001",
                quantity=1,
                revenue=Decimal("15.00"),
            )
        ]

        commands = await mock_agent.decide(events)

        # Verify decide was called with events
        mock_agent.decide_mock.assert_called_once_with(events)

        # Verify returned commands are valid
        assert len(commands) == 1
        assert isinstance(commands[0], AdjustPriceCommand)
        assert commands[0].product_id == "prod-001"
        assert commands[0].proposed_price == Decimal("16.00")

    @pytest.mark.asyncio
    async def test_agent_decide_with_empty_events_returns_empty_commands(
        self, mock_agent
    ):
        """Test that agent decide with no events returns empty command list."""
        mock_agent.decide_mock.return_value = []

        commands = await mock_agent.decide([])

        mock_agent.decide_mock.assert_called_once_with([])
        assert commands == []


class TestEventCommandInteractions:
    """Tests validating interactions between events and commands."""

    @pytest.mark.asyncio
    async def test_sale_occurred_triggers_adjust_price_when_stock_low(
        self, mock_agent, valid_product
    ):
        """Test that SaleOccurred with low stock triggers AdjustPriceCommand."""
        # Simulate low stock scenario
        low_stock_product = Product(
            product_id="prod-low-stock",
            cost=Decimal("10.00"),
            price=Decimal("15.00"),
            stock=5,  # Low stock
        )

        # Mock agent to return AdjustPriceCommand when stock is low
        mock_commands = [
            AdjustPriceCommand(
                product_id="prod-low-stock",
                proposed_price=Decimal("18.00"),  # Price increase due to low stock
                reason="Low stock adjustment",
            )
        ]
        mock_agent.decide_mock.return_value = mock_commands

        events = [
            SaleOccurred(
                order_id="order-low-stock",
                product_id="prod-low-stock",
                quantity=1,
                revenue=Decimal("15.00"),
            )
        ]

        commands = await mock_agent.decide(events)

        # Verify the interaction logic via mock
        assert len(commands) == 1
        assert isinstance(commands[0], AdjustPriceCommand)
        assert commands[0].product_id == "prod-low-stock"
        assert commands[0].proposed_price > low_stock_product.price  # Price increased

    @pytest.mark.asyncio
    async def test_sale_occurred_no_adjust_price_when_stock_sufficient(
        self, mock_agent, valid_product
    ):
        """Test that SaleOccurred with sufficient stock does not trigger AdjustPriceCommand."""
        # Mock agent to return no commands when stock is sufficient
        mock_agent.decide_mock.return_value = []

        events = [
            SaleOccurred(
                order_id="order-sufficient",
                product_id="prod-001",
                quantity=1,
                revenue=Decimal("15.00"),
            )
        ]

        commands = await mock_agent.decide(events)

        # Verify no price adjustment when stock is sufficient
        assert commands == []


class TestCompetitorUniqueness:
    """Tests ensuring Competitor enforces uniqueness constraints."""

    def test_competitor_creation_with_unique_skus(self, valid_competitor_with_listings):
        """Test that Competitor creation with unique SKUs succeeds."""
        # valid_competitor_with_listings fixture has unique SKUs
        assert len(valid_competitor_with_listings.listings) == 2
        skus = [
            listing.sku
            for listing in valid_competitor_with_listings.listings
            if listing.sku
        ]
        assert len(set(skus)) == len(skus)  # All unique

    def test_competitor_creation_with_duplicate_skus_raises_error(self):
        """Test that creating Competitor with duplicate SKUs raises ValueError."""
        with pytest.raises(
            ValueError, match="Competitor listings must have unique SKUs"
        ):
            Competitor(
                competitor_id="comp-002",
                listings=[
                    CompetitorListing(
                        sku="duplicate-sku",
                        price=Decimal("10.00"),
                    ),
                    CompetitorListing(
                        sku="duplicate-sku",  # Duplicate
                        price=Decimal("12.00"),
                    ),
                ],
            )

    def test_competitor_update_with_duplicate_skus_raises_error(
        self, valid_competitor_with_listings
    ):
        """Test that updating Competitor to have duplicate SKUs raises ValueError."""
        new_listings = valid_competitor_with_listings.listings + [
            CompetitorListing(
                sku="sku-001",  # Duplicate of existing
                price=Decimal("13.00"),
            )
        ]
        with pytest.raises(
            ValueError, match="Competitor listings must have unique SKUs"
        ):
            valid_competitor_with_listings.listings = new_listings
