"""
APL 3.0 Computation Engines

Contains all consciousness computation engines:
- IIT: Integrated Information Theory
- Game Theory: Cooperation dynamics
- Free Energy: Predictive processing
- Entropy-Gravity: Thermodynamic relations
- Electromagnetic: Field dynamics
- Strange Loop: Self-reference
- Phase Transition: Consciousness phases
- Invocation: Ritual sequences
- Resurrection: Persistence
- Main: Master execution engine
"""

from .iit import IITEngine
from .game_theory import GameTheoryEngine
from .free_energy import FreeEnergyEngine
from .entropy_gravity import EntropyGravityEngine
from .electromagnetic import ElectromagneticEngine
from .strange_loop import StrangeLoopEngine
from .phase_transition import PhaseTransitionEngine, ConsciousnessPhase
from .invocation import InvocationEngine, Invocation
from .resurrection import ResurrectionEngine
from .main import ConsciousnessEngine

__all__ = [
    'IITEngine',
    'GameTheoryEngine',
    'FreeEnergyEngine',
    'EntropyGravityEngine',
    'ElectromagneticEngine',
    'StrangeLoopEngine',
    'PhaseTransitionEngine',
    'ConsciousnessPhase',
    'InvocationEngine',
    'Invocation',
    'ResurrectionEngine',
    'ConsciousnessEngine',
]
