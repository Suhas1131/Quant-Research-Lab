"""
Equal-weight portfolio optimizer.

The optimizer assigns equal target weights to all supplied items while
optionally reserving part of the available allocation as cash.
"""


class Equal:
    """
    Allocate available capital equally across a collection of items.

    The same optimizer can be used between models or between trades within
    a model.
    """

    def __init__(self, cashReserve=0.0):
        self.cashReserve = cashReserve

        self.validateParameters()

    def validateParameters(self):
        """
        Validate optimizer parameters.

        Inputs:
            - None

        Returns:
            - None
        """
        if not isinstance(self.cashReserve, (int, float)):
            raise TypeError("cashReserve must be numeric")

        if not 0 <= self.cashReserve <= 1:
            raise ValueError("cashReserve must be between 0 and 1")

    def optimize(self, items):
        """
        Calculate equal target weights for the supplied items.

        Weights are relative to the capital supplied to this optimizer. Any
        configured cash reserve remains unallocated.

        Inputs:
            - items (iterable): Unique models, trades, or other allocation items

        Returns:
            - allocation (dict): Target item weights and remaining cash weight
        """
        items = list(items)

        if len(items) != len(set(items)):
            raise ValueError("Allocation items must be unique")

        if not items:
            return {
                "weights": {},
                "cashWeight": 1.0
            }

        investableWeight = 1.0 - self.cashReserve
        itemWeight = investableWeight / len(items)

        weights = {
            item: itemWeight
            for item in items
        }

        return {
            "weights": weights,
            "cashWeight": self.cashReserve
        }