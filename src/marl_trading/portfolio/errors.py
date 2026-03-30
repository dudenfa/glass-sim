from __future__ import annotations


class PortfolioError(RuntimeError):
    pass


class InsufficientCashError(PortfolioError):
    pass


class InsufficientInventoryError(PortfolioError):
    pass


class ReservationNotFoundError(PortfolioError):
    pass


class PortfolioInactiveError(PortfolioError):
    pass
