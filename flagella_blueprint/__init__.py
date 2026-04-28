"""Flagella class 2 gene network dynamic model as a process-bigraph Process.

Reproduces the quantitative blueprint of Kalir & Alon, "Using a Quantitative
Blueprint to Reprogram the Dynamics of the Flagella Gene Network",
Cell 117:713-720 (2004).
"""

from flagella_blueprint.processes import FlagellaProcess
from flagella_blueprint.composites import make_flagella_document

__all__ = ['FlagellaProcess', 'make_flagella_document']
