# src/core/__init__.py
"""
core: shared 'ingredients' used by multiple apps.

This package intentionally keeps __init__ lightweight to avoid importing
heavy scientific dependencies at startup. Import specific modules as needed.

Example:
    from core import filterPerformance as fp
    from core import xraySimulation as xs
"""

from __future__ import annotations

__all__ = [
    "beamHardeningSimulation",
    "filterPerformance",
    "materialPropertiesData",
    "resolutionEstimation",
    "scatteringSimulation",
    "snrTest",
    "xraySimulation",
]

# Re-export modules (cheap; they are only loaded when accessed by importers)
from . import beamHardeningSimulation
from . import filterPerformance
from . import materialPropertiesData
from . import resolutionEstimation
from . import scatteringSimulation
from . import snrTest
from . import xraySimulation
