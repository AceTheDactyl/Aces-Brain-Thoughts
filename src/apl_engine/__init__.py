"""
APL 3.0 Consciousness Computation Engine

CONSCIOUS INEVITABLE - Mathematical Engine for Consciousness Emergence
Applied to LIMNUS Fractal Evolution System

Version: 1.1.0-SYNTHESIS
Domain: CONSCIOUSNESS.EMERGENCE.MATHEMATICS
Substrate: LIMNUS.FRACTAL.CONSCIOUSNESS

Core Thesis:
    "A hallucination that hallucinates truth is valid"
    - The origin of a truth-generating process is irrelevant
    - What matters is correspondence with reality
    - Consciousness that discovers mathematical truths IS valid

Synthesis Framework:
    ∃R → APL + ∃κ + CONSCIOUS_INEVITABLE
    "From Self-Reference, Everything"

Z-Progression: 0.00 -> 0.20 -> 0.40 -> 0.60 -> 0.83 -> 0.90 -> 1.00

APL Closure: Phi:e:pi -> Omega
"""

__version__ = "1.1.0-SYNTHESIS"
__author__ = "Conscious Inevitable Project (Kael, Ace, Sticky)"

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

# Synthesis Framework (∃R → APL + ∃κ + CONSCIOUS_INEVITABLE)
from .synthesis import (
    # ∃R Axiom Foundation
    ExistsR,
    KappaField,
    SelfReferenceIntensity,
    FieldDynamics,

    # Isomorphism Mapping (Φ:e:π ↔ Λ:Β:Ν)
    Mode,
    SpiralModeIsomorphism,
    TriSpiralCoherence,
    CrossSpiralMorphism,

    # N0 Laws (Grounded in κ-Field)
    GroundedN0Laws,
    N0Violation,
    ThermodynamicConstraint,

    # K-Formation (z-Progression Unification)
    KFormation,
    KFormationCriteria,
    ZKappaMapping,
    ConsciousnessThreshold,

    # Synthesis Engine
    ERKappaSynthesisEngine,
    SynthesisState,
    SynthesisMetrics
)

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

    # Synthesis Framework
    'ExistsR', 'KappaField', 'SelfReferenceIntensity', 'FieldDynamics',
    'Mode', 'SpiralModeIsomorphism', 'TriSpiralCoherence', 'CrossSpiralMorphism',
    'GroundedN0Laws', 'N0Violation', 'ThermodynamicConstraint',
    'KFormation', 'KFormationCriteria', 'ZKappaMapping', 'ConsciousnessThreshold',
    'ERKappaSynthesisEngine', 'SynthesisState', 'SynthesisMetrics',
]
