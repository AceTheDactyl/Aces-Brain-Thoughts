"""
APL 3.0 LIMNUS Node

Individual node in the LIMNUS fractal tree
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum

from ..core.constants import CONSTANTS, Spiral, TruthState
from ..core.scalars import ScalarState
from ..core.token import APLToken, Machine


class NodeActivity(Enum):
    """Activity level of a node"""
    DORMANT = "dormant"
    AWAKENING = "awakening"
    BRANCHING = "branching"
    CLUSTERING = "clustering"
    INTEGRATING = "integrating"
    TRANSCENDING = "transcending"


@dataclass
class NodeState:
    """Complete state of a LIMNUS node"""
    z: float                    # Local consciousness level
    phi_local: float            # Local integrated information
    operator: str               # Current APL operator
    spiral: Spiral              # Dominant spiral
    truth: TruthState           # Truth state
    tier: int                   # Operation tier
    scalars: ScalarState        # 9-component scalar state
    phase: float                # Oscillation phase [0, 2*pi]
    frequency: float            # Oscillation frequency (Hz)
    strategy: str               # Game theory strategy
    F_local: float              # Local free energy
    activity: NodeActivity      # Activity level

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "z": self.z,
            "phi_local": self.phi_local,
            "operator": self.operator,
            "spiral": self.spiral.value,
            "truth": self.truth.value,
            "tier": self.tier,
            "scalars": self.scalars.to_dict(),
            "phase": self.phase,
            "frequency": self.frequency,
            "strategy": self.strategy,
            "F_local": self.F_local,
            "activity": self.activity.value
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'NodeState':
        """Create from dictionary"""
        return cls(
            z=data["z"],
            phi_local=data["phi_local"],
            operator=data["operator"],
            spiral=Spiral(data["spiral"]),
            truth=TruthState(data["truth"]),
            tier=data["tier"],
            scalars=ScalarState.from_dict(data["scalars"]),
            phase=data["phase"],
            frequency=data["frequency"],
            strategy=data["strategy"],
            F_local=data["F_local"],
            activity=NodeActivity(data["activity"])
        )


class LimnusNode:
    """
    A node in the LIMNUS fractal tree

    Each node represents a point of consciousness emergence
    with local state and connections to parent/children
    """

    # Depth-based mappings
    DEPTH_TO_OPERATOR = {
        6: "()",   # Root: Boundary
        5: "^",    # Trunk: Amplify
        4: "x",    # Branch: Fusion
        3: "+",    # Process: Grouping
        2: "^",    # Structure: Amplify
        1: "-",    # Terminal: Separation
    }

    DEPTH_TO_SPIRAL = {
        6: Spiral.PHI,
        5: Spiral.PHI,
        4: Spiral.E,
        3: Spiral.PI,
        2: Spiral.PHI,
        1: Spiral.E,
    }

    DEPTH_TO_TIER = {
        6: 1,
        5: 2,
        4: 2,
        3: 2,
        2: 3,
        1: 3,
    }

    DEPTH_TO_FREQUENCY = {
        6: 0.5,    # Delta
        5: 4.0,    # Theta
        4: 10.0,   # Alpha
        3: 20.0,   # Beta
        2: 40.0,   # Gamma
        1: 80.0,   # High gamma
    }

    DEPTH_TO_ACTIVITY = {
        6: NodeActivity.DORMANT,
        5: NodeActivity.AWAKENING,
        4: NodeActivity.BRANCHING,
        3: NodeActivity.CLUSTERING,
        2: NodeActivity.INTEGRATING,
        1: NodeActivity.TRANSCENDING,
    }

    def __init__(
        self,
        id: int,
        depth: int,
        parent: Optional['LimnusNode'] = None
    ):
        """
        Initialize a LIMNUS node

        Args:
            id: Unique node identifier
            depth: Depth in tree (6=root, 1=leaf)
            parent: Parent node (None for root)
        """
        self.id = id
        self.depth = depth
        self.parent = parent
        self.children: List['LimnusNode'] = []

        # Compute position
        self.position = self._compute_position()

        # Initialize state
        self.state = self._initialize_state()

        # Neighbors for game theory
        self.neighbors: List['LimnusNode'] = []

        # Prediction state for free energy
        self.prediction: Optional[Dict] = None

    def _compute_position(self) -> Tuple[float, float]:
        """Compute 2D position based on tree structure"""
        if self.parent is None:
            return (0.0, 0.0)

        # Position relative to parent using golden angle
        golden_angle = CONSTANTS.MATHEMATICAL.golden_angle
        parent_x, parent_y = self.parent.position

        # Distance from parent decreases with depth
        distance = CONSTANTS.PHI ** (self.depth - 1)

        # Angle based on child index among siblings
        if self.parent.children:
            child_idx = len(self.parent.children)
        else:
            child_idx = 0

        angle = child_idx * golden_angle

        x = parent_x + distance * math.cos(angle)
        y = parent_y + distance * math.sin(angle)

        return (x, y)

    def _initialize_state(self) -> NodeState:
        """Initialize node state based on depth"""
        # Z-value from depth (inverse relationship)
        # depth 6 -> z = 0.0, depth 1 -> z = 1.0
        z = 1.0 - (self.depth - 1) / 5.0

        # Transform with golden ratio
        z = z ** (1 / CONSTANTS.PHI)

        return NodeState(
            z=z,
            phi_local=0.1 * (7 - self.depth),
            operator=self.DEPTH_TO_OPERATOR[self.depth],
            spiral=self.DEPTH_TO_SPIRAL[self.depth],
            truth=TruthState.UNTRUE,
            tier=self.DEPTH_TO_TIER[self.depth],
            scalars=ScalarState.dormant() if self.depth > 4 else ScalarState.awakening(),
            phase=hash(self.id) % (2 * math.pi),  # Random initial phase
            frequency=self.DEPTH_TO_FREQUENCY[self.depth],
            strategy="tit_for_tat",
            F_local=1.0,
            activity=self.DEPTH_TO_ACTIVITY[self.depth]
        )

    def add_child(self, child: 'LimnusNode'):
        """Add a child node"""
        self.children.append(child)
        child.parent = self

    def get_siblings(self) -> List['LimnusNode']:
        """Get sibling nodes"""
        if self.parent is None:
            return []
        return [c for c in self.parent.children if c.id != self.id]

    def get_ancestors(self) -> List['LimnusNode']:
        """Get all ancestor nodes (path to root)"""
        ancestors = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors

    def get_descendants(self) -> List['LimnusNode']:
        """Get all descendant nodes"""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants

    def is_leaf(self) -> bool:
        """Check if node is a leaf"""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if node is the root"""
        return self.parent is None

    # =========================================================================
    # STATE UPDATES
    # =========================================================================

    def update_z(self, delta_z: float):
        """Update consciousness level"""
        self.state.z = max(0.0, min(1.0, self.state.z + delta_z))

    def update_phi(self, delta_phi: float):
        """Update integrated information"""
        self.state.phi_local = max(0.0, self.state.phi_local + delta_phi)

    def update_phase(self, dt: float):
        """Update oscillation phase"""
        self.state.phase = (
            self.state.phase + 2 * math.pi * self.state.frequency * dt
        ) % (2 * math.pi)

    def update_scalars(self, deltas: Dict[str, float]):
        """Update scalar state"""
        self.state.scalars = self.state.scalars.apply_deltas(deltas)

    def update_truth(self, new_truth: TruthState):
        """Update truth state"""
        self.state.truth = new_truth

    def update_operator(self, new_operator: str):
        """Update APL operator"""
        self.state.operator = new_operator

    def update_activity(self):
        """Update activity level based on z"""
        z = self.state.z
        if z < 0.2:
            self.state.activity = NodeActivity.DORMANT
        elif z < 0.4:
            self.state.activity = NodeActivity.AWAKENING
        elif z < 0.6:
            self.state.activity = NodeActivity.BRANCHING
        elif z < 0.8:
            self.state.activity = NodeActivity.CLUSTERING
        elif z < 0.95:
            self.state.activity = NodeActivity.INTEGRATING
        else:
            self.state.activity = NodeActivity.TRANSCENDING

    # =========================================================================
    # APL TOKEN GENERATION
    # =========================================================================

    def generate_token(self) -> APLToken:
        """Generate APL token from current state"""
        # Map operator to machine
        operator_to_machine = {
            "()": Machine.U,
            "^": Machine.E,
            "x": Machine.M,
            "+": Machine.D,
            "-": Machine.C,
            "/": Machine.MOD,
        }

        machine = operator_to_machine.get(self.state.operator, Machine.M)

        return APLToken(
            spiral=self.state.spiral,
            machine=machine,
            intent=f"node_{self.id}_d{self.depth}",
            truth=self.state.truth,
            tier=self.state.tier
        )

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict:
        """Convert node to dictionary"""
        return {
            "id": self.id,
            "depth": self.depth,
            "position": self.position,
            "state": self.state.to_dict(),
            "children_ids": [c.id for c in self.children],
            "parent_id": self.parent.id if self.parent else None,
            "neighbor_ids": [n.id for n in self.neighbors]
        }

    def __repr__(self) -> str:
        return f"LimnusNode(id={self.id}, depth={self.depth}, z={self.state.z:.3f})"
