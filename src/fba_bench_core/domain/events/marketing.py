"""Marketing and customer engagement-related events and commands."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import Field, field_validator

from .base import BaseEvent, Command


class PromotionLaunched(BaseEvent):
    """Event signalling that a promotion has been launched for products or categories.

    - discount_percent is expressed as Decimal fraction (0.0 - 1.0).
    """

    event_type: Literal["promotion_launched"] = "promotion_launched"

    promotion_id: str
    product_ids: list[str] | None = None
    category: str | None = None
    discount_percent: Decimal = Field(..., ge=Decimal("0"), le=Decimal("1"))
    start: datetime
    end: datetime | None = None
    channels: list[str] | None = None

    @field_validator("discount_percent", mode="before")
    @classmethod
    def _coerce_discount(cls, v):
        if isinstance(v, Decimal):
            return v
        try:
            return Decimal(str(v))
        except Exception as exc:
            raise ValueError("Invalid discount_percent") from exc


class CustomerComplaintLogged(BaseEvent):
    """Event representing a logged customer complaint tied to an order."""

    event_type: Literal["customer_complaint_logged"] = "customer_complaint_logged"

    complaint_id: str
    order_id: str | None = None
    product_id: str | None = None
    issue_type: str
    details: str | None = None
    resolution_deadline: datetime | None = None


class ResolveCustomerIssueCommand(Command):
    """Command for customer service agents to resolve a logged complaint.

    - refund_amount uses Decimal for monetary values and must be >= 0.
    """

    command_type: Literal["resolve_customer_issue"] = "resolve_customer_issue"

    complaint_id: str | None = None
    order_id: str | None = None
    resolution_action: str
    refund_amount: Decimal | None = Field(None, ge=Decimal("0"))

    @field_validator("refund_amount", mode="before")
    @classmethod
    def _coerce_refund(cls, v):
        if v is None:
            return v
        if isinstance(v, Decimal):
            return v
        try:
            return Decimal(str(v))
        except Exception as exc:
            raise ValueError("Invalid refund_amount") from exc


class StartCustomerOutreachCommand(Command):
    """Command to start an outreach/campaign targeted at a customer segment."""

    command_type: Literal["start_customer_outreach"] = "start_customer_outreach"

    segment: str
    message_template: str
    goal_metrics: dict[str, Any] | None = None
    channels: list[str] | None = None


class RespondToComplaintCommand(Command):
    """Command to respond to a customer complaint.

    - response_action: type of response (e.g., 'apology', 'refund', 'replacement')
    - response_message: optional detailed message to the customer.
    """

    command_type: Literal["respond_to_complaint"] = "respond_to_complaint"

    complaint_id: str
    response_action: str
    response_message: str | None = None
