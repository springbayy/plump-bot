"""Observation type re-export.

Keeping this module lets future encoders import observations without reaching
into the broader state module.
"""

from .state import Observation

__all__ = ["Observation"]
