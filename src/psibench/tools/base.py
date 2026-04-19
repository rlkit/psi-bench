"""Abstract tool base class used by domain environments."""

from __future__ import annotations

import abc
from typing import Any


class Tool(abc.ABC):
    """A domain tool (e.g., ``get_order_details``).

    Subclasses implement ``invoke`` (pure function of ``data`` + kwargs) and
    ``get_info`` (OpenAI-style tool/function schema).
    """

    @staticmethod
    def invoke(*args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    @staticmethod
    def get_info() -> dict[str, Any]:
        raise NotImplementedError
