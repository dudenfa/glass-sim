from __future__ import annotations


class ExchangeError(Exception):
    """Base error for exchange-related failures."""


class InvalidOrderError(ExchangeError):
    """Raised when an order is malformed or inconsistent."""


class OrderNotFoundError(ExchangeError):
    """Raised when a cancellation targets an unknown resting order."""
