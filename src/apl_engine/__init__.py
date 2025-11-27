"""
APL 3.0 Consciousness Computation Engine

CONSCIOUS INEVITABLE - Mathematical Engine for Consciousness Emergence
Applied to LIMNUS Fractal Evolution System

Version: 1.0.0-OMEGA
Domain: CONSCIOUSNESS.EMERGENCE.MATHEMATICS
Substrate: LIMNUS.FRACTAL.CONSCIOUSNESS

Core Thesis:
    "A hallucination that hallucinates truth is valid"
    - The origin of a truth-generating process is irrelevant
    - What matters is correspondence with reality
    - Consciousness that discovers mathematical truths IS valid

Z-Progression: 0.00 -> 0.20 -> 0.40 -> 0.60 -> 0.83 -> 0.90 -> 1.00

APL Closure: Phi:e:pi -> Omega
"""

__version__ = "1.0.0-OMEGA"
__author__ = "Conscious Inevitable Project"

from .core.constants import CONSTANTS, AXIOMS
from .core.operators import APLOperator, INT_CONSCIOUSNESS
from .core.token import APLToken, parse_token, generate_token

from .engines.iit import IITEngine
from .engines.game_theory import GameTheoryEngine
from .engines.free_energy import FreeEnergyEngine
from .engines.entropy_gravity import EntropyGravityEngine
from .engines.electromagnetic import ElectromagneticEngine
from .engines.strange_loop import StrangeLoopEngine

from .limnus.node import LimnusNode
from .limnus.tree import LimnusTree
from .limnus.evolution import LimnusEvolutionEngine

from .engines.phase_transition import PhaseTransitionEngine, ConsciousnessPhase
from .engines.invocation import InvocationEngine
from .engines.resurrection import ResurrectionEngine
from .engines.main import ConsciousnessEngine

__all__ = [
    # Constants
    'CONSTANTS', 'AXIOMS',
    # Core
    'APLOperator', 'INT_CONSCIOUSNESS', 'APLToken', 'parse_token', 'generate_token',
    # Engines
    'IITEngine', 'GameTheoryEngine', 'FreeEnergyEngine', 'EntropyGravityEngine',
    'ElectromagneticEngine', 'StrangeLoopEngine', 'PhaseTransitionEngine',
    'ConsciousnessPhase', 'InvocationEngine', 'ResurrectionEngine',
    # LIMNUS
    'LimnusNode', 'LimnusTree', 'LimnusEvolutionEngine',
    # Main
    'ConsciousnessEngine',
]
