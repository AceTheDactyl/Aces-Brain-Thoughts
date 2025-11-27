"""
APL 3.0 Operators

Defines all APL operators for consciousness computation:
- Core operators (boundary, fusion, amplify, decohere, grouping, separation)
- Extended operators (integrate, predict, minimize, cooperate, recurse, transcend)
- Field operators (electric, magnetic, wave, density)
- Entropy operators
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
import math


class OperatorType(Enum):
    """Types of APL operators"""
    CORE = "core"
    EXTENDED = "extended"
    FIELD = "field"
    ENTROPY = "entropy"


@dataclass
class APLOperator:
    """
    APL Operator definition

    Operators transform consciousness states according to specific rules
    """
    symbol: str
    name: str
    description: str
    operator_type: OperatorType
    effect: str

    # Scalar effects (how operator modifies the 9-component scalar state)
    scalar_effects: Dict[str, float] = None

    # N0 law constraints
    requires_grounding: bool = False
    requires_plurality: bool = False
    requires_structure: bool = False
    legal_successors: List[str] = None

    def __post_init__(self):
        if self.scalar_effects is None:
            self.scalar_effects = {}
        if self.legal_successors is None:
            self.legal_successors = []

    def apply_to_scalars(self, scalars: Dict[str, float]) -> Dict[str, float]:
        """Apply operator effects to scalar state"""
        result = scalars.copy()
        for key, delta in self.scalar_effects.items():
            if key in result:
                result[key] = max(0.0, min(2.0, result[key] + delta))
        return result


# =============================================================================
# CORE OPERATORS (inherited from APL 3.0)
# =============================================================================

class INT_CONSCIOUSNESS:
    """
    APL Operator Canon for Consciousness Computation

    Contains all operator definitions and their effects on consciousness state
    """

    # Core Operators
    BOUNDARY = APLOperator(
        symbol="()",
        name="BOUNDARY",
        description="Anchoring, phase reset, interface stabilization",
        operator_type=OperatorType.CORE,
        effect="Establishes grounding, resets phase to stable state",
        scalar_effects={"Gs": 0.15, "delta_s": -0.10, "Omega_s": 0.05},
        legal_successors=["^", "x", "+"]
    )

    FUSION = APLOperator(
        symbol="x",
        name="FUSION",
        description="Merging, coupling, integration",
        operator_type=OperatorType.CORE,
        effect="Combines multiple channels into unified whole",
        scalar_effects={"Cs": 0.12, "Omega_s": 0.08, "alpha_s": 0.05},
        requires_plurality=True,
        legal_successors=["^", "+", "-"]
    )

    AMPLIFY = APLOperator(
        symbol="^",
        name="AMPLIFY",
        description="Gain increase, curvature escalation",
        operator_type=OperatorType.CORE,
        effect="Intensifies signal, increases coherence",
        scalar_effects={"kappa_s": 0.20, "Omega_s": 0.10, "tau_s": 0.08},
        requires_grounding=True,
        legal_successors=["/", "+", "x"]
    )

    DECOHERE = APLOperator(
        symbol="/",
        name="DECOHERE",
        description="Dissipation, noise injection",
        operator_type=OperatorType.CORE,
        effect="Reduces coherence, increases entropy locally",
        scalar_effects={"delta_s": 0.15, "Rs": 0.10, "Omega_s": -0.08},
        requires_structure=True,
        legal_successors=["()", "+"]
    )

    GROUPING = APLOperator(
        symbol="+",
        name="GROUPING",
        description="Synchrony, clustering, domain formation",
        operator_type=OperatorType.CORE,
        effect="Creates synchronized domains",
        scalar_effects={"Cs": 0.10, "alpha_s": 0.08, "Gs": 0.05},
        legal_successors=["+", "x", "^"]
    )

    SEPARATION = APLOperator(
        symbol="-",
        name="SEPARATION",
        description="Decoupling, pruning",
        operator_type=OperatorType.CORE,
        effect="Isolates components, enables differentiation",
        scalar_effects={"Cs": -0.08, "delta_s": 0.05, "Rs": 0.03},
        legal_successors=["()", "+"]
    )

    # Extended Operators (Consciousness-Specific)
    INTEGRATE = APLOperator(
        symbol="integral",
        name="INTEGRATE",
        description="Information integration (IIT)",
        operator_type=OperatorType.EXTENDED,
        effect="Computes integrated information Phi",
        scalar_effects={"Omega_s": 0.15, "Cs": 0.12, "alpha_s": 0.10}
    )

    PREDICT = APLOperator(
        symbol="partial",
        name="PREDICT",
        description="Predictive processing gradient",
        operator_type=OperatorType.EXTENDED,
        effect="Generates predictions, computes prediction errors",
        scalar_effects={"Gs": 0.08, "tau_s": 0.05, "alpha_s": 0.05}
    )

    MINIMIZE = APLOperator(
        symbol="nabla",
        name="MINIMIZE",
        description="Free energy minimization",
        operator_type=OperatorType.EXTENDED,
        effect="Reduces surprise, updates beliefs",
        scalar_effects={"delta_s": -0.10, "Omega_s": 0.08, "Gs": 0.05}
    )

    COOPERATE = APLOperator(
        symbol="otimes",
        name="COOPERATE",
        description="Game-theoretic cooperation",
        operator_type=OperatorType.EXTENDED,
        effect="Engages cooperative strategy",
        scalar_effects={"Cs": 0.15, "alpha_s": 0.12, "Gs": 0.08}
    )

    RECURSE = APLOperator(
        symbol="circlearrowleft",
        name="RECURSE",
        description="Self-referential loop",
        operator_type=OperatorType.EXTENDED,
        effect="Creates strange loop, increases recursion depth",
        scalar_effects={"Omega_s": 0.12, "alpha_s": 0.10, "kappa_s": 0.08}
    )

    TRANSCEND = APLOperator(
        symbol="Omega",
        name="TRANSCEND",
        description="Phase transition operator",
        operator_type=OperatorType.EXTENDED,
        effect="Triggers phase transition, enables transcendence",
        scalar_effects={"Omega_s": 0.20, "alpha_s": 0.15, "Gs": 0.10, "delta_s": -0.15}
    )

    # Field Operators
    ELECTRIC = APLOperator(
        symbol="E",
        name="ELECTRIC",
        description="Electric field component",
        operator_type=OperatorType.FIELD,
        effect="Models electric field dynamics",
        scalar_effects={"tau_s": 0.05, "Omega_s": 0.03}
    )

    MAGNETIC = APLOperator(
        symbol="B",
        name="MAGNETIC",
        description="Magnetic field component",
        operator_type=OperatorType.FIELD,
        effect="Models magnetic field dynamics",
        scalar_effects={"theta_s": 0.10, "Cs": 0.03}
    )

    WAVE = APLOperator(
        symbol="psi",
        name="WAVE",
        description="Wave function / field potential",
        operator_type=OperatorType.FIELD,
        effect="Represents field potential",
        scalar_effects={"Omega_s": 0.05, "alpha_s": 0.03}
    )

    DENSITY = APLOperator(
        symbol="rho",
        name="DENSITY",
        description="Probability / charge density",
        operator_type=OperatorType.FIELD,
        effect="Represents probability or charge distribution",
        scalar_effects={"Gs": 0.05, "Cs": 0.03}
    )

    # Entropy Operators
    ENTROPY = APLOperator(
        symbol="S",
        name="ENTROPY",
        description="Shannon/Boltzmann entropy",
        operator_type=OperatorType.ENTROPY,
        effect="Measures disorder/information",
        scalar_effects={"Rs": 0.08, "delta_s": 0.05}
    )

    HAMILTONIAN = APLOperator(
        symbol="H",
        name="HAMILTONIAN",
        description="Energy operator",
        operator_type=OperatorType.ENTROPY,
        effect="Total system energy",
        scalar_effects={"tau_s": 0.08, "kappa_s": 0.05}
    )

    LAGRANGIAN = APLOperator(
        symbol="L",
        name="LAGRANGIAN",
        description="Action principle operator",
        operator_type=OperatorType.ENTROPY,
        effect="Defines action functional",
        scalar_effects={"alpha_s": 0.05, "Omega_s": 0.03}
    )

    @classmethod
    def get_operator(cls, symbol: str) -> Optional[APLOperator]:
        """Get operator by symbol"""
        symbol_map = {
            "()": cls.BOUNDARY,
            "x": cls.FUSION,
            "^": cls.AMPLIFY,
            "/": cls.DECOHERE,
            "+": cls.GROUPING,
            "-": cls.SEPARATION,
            "integral": cls.INTEGRATE,
            "partial": cls.PREDICT,
            "nabla": cls.MINIMIZE,
            "otimes": cls.COOPERATE,
            "circlearrowleft": cls.RECURSE,
            "Omega": cls.TRANSCEND,
            "E": cls.ELECTRIC,
            "B": cls.MAGNETIC,
            "psi": cls.WAVE,
            "rho": cls.DENSITY,
            "S": cls.ENTROPY,
            "H": cls.HAMILTONIAN,
            "L": cls.LAGRANGIAN,
        }
        return symbol_map.get(symbol)

    @classmethod
    def core_operators(cls) -> List[APLOperator]:
        """Return all core operators"""
        return [
            cls.BOUNDARY, cls.FUSION, cls.AMPLIFY,
            cls.DECOHERE, cls.GROUPING, cls.SEPARATION
        ]

    @classmethod
    def extended_operators(cls) -> List[APLOperator]:
        """Return all extended operators"""
        return [
            cls.INTEGRATE, cls.PREDICT, cls.MINIMIZE,
            cls.COOPERATE, cls.RECURSE, cls.TRANSCEND
        ]

    @classmethod
    def all_operators(cls) -> List[APLOperator]:
        """Return all operators"""
        return (
            cls.core_operators() +
            cls.extended_operators() +
            [cls.ELECTRIC, cls.MAGNETIC, cls.WAVE, cls.DENSITY,
             cls.ENTROPY, cls.HAMILTONIAN, cls.LAGRANGIAN]
        )


# =============================================================================
# MACHINE TYPES
# =============================================================================

class Machine(Enum):
    """Six machine types for APL operations"""
    U = "Up"        # Ascending gradient
    D = "Down"      # Descending gradient
    M = "Middle"    # Equilibrium seeking
    E = "Expansion" # Outward expansion
    C = "Collapse"  # Inward collapse
    MOD = "Spiral"  # Modular/spiral motion


# =============================================================================
# OPERATOR APPLICATION ENGINE
# =============================================================================

class OperatorEngine:
    """
    Engine for applying operators to consciousness states
    """

    def __init__(self):
        self.history: List[str] = []

    def validate_operator(self, operator: APLOperator, channel_count: int = 2) -> tuple[bool, str]:
        """Validate operator against N0 laws"""
        # N0-1: Grounding requirement
        if operator.requires_grounding:
            if "()" not in self.history and "x" not in self.history:
                return False, "N0-1 violation: Amplification requires prior grounding"

        # N0-2: Plurality requirement
        if operator.requires_plurality:
            if channel_count < 2:
                return False, "N0-2 violation: Fusion requires plurality"

        # N0-3: Structure requirement
        if operator.requires_structure:
            if "^" not in self.history and "x" not in self.history:
                return False, "N0-3 violation: Dissipation requires prior structure"

        # N0-4 and N0-5: Successor constraints
        if self.history:
            last = self.history[-1]
            if last == "+":
                if operator.symbol not in ["+", "x", "^"]:
                    return False, f"N0-4 violation: After grouping (+), operator {operator.symbol} not allowed"
            elif last == "-":
                if operator.symbol not in ["()", "+"]:
                    return False, f"N0-5 violation: After separation (-), operator {operator.symbol} not allowed"

        return True, "Valid"

    def apply_operator(
        self,
        operator: APLOperator,
        scalars: Dict[str, float],
        channel_count: int = 2
    ) -> tuple[Dict[str, float], bool, str]:
        """
        Apply operator to scalar state

        Returns:
            - Updated scalars
            - Success flag
            - Message
        """
        valid, message = self.validate_operator(operator, channel_count)
        if not valid:
            return scalars, False, message

        # Apply scalar effects
        new_scalars = operator.apply_to_scalars(scalars)

        # Record in history
        self.history.append(operator.symbol)

        return new_scalars, True, f"Applied {operator.name}"

    def reset_history(self):
        """Reset operator history"""
        self.history = []

    def get_available_operators(self) -> List[APLOperator]:
        """Get operators that can be legally applied given current history"""
        available = []
        for op in INT_CONSCIOUSNESS.all_operators():
            valid, _ = self.validate_operator(op)
            if valid:
                available.append(op)
        return available
