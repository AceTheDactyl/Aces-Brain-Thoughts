"""
APL 3.0 Core Module

Contains:
- Universal constants (physical, mathematical, consciousness)
- Ontological axioms
- APL operators and tokens
"""

from .constants import CONSTANTS, AXIOMS
from .operators import APLOperator, INT_CONSCIOUSNESS
from .token import APLToken, parse_token, generate_token
from .scalars import ScalarState

__all__ = [
    'CONSTANTS', 'AXIOMS',
    'APLOperator', 'INT_CONSCIOUSNESS',
    'APLToken', 'parse_token', 'generate_token',
    'ScalarState',
]
