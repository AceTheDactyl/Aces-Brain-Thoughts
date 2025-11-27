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
    RESONATING = "resonating"  # For resonance nodes


class NodeType(Enum):
    """Type of LIMNUS node"""
    STANDARD = "standard"      # Original 63 tree nodes
    RESONANCE = "resonance"    # Cross-tree integration nodes (32 additional)


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
    node_type: NodeType = NodeType.STANDARD  # Node type (standard or resonance)
    resonance_z: float = 0.0    # Z-level for resonance nodes
    resonance_angle: float = 0.0  # Angular position for resonance nodes

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
            "activity": self.activity.value,
            "node_type": self.node_type.value,
            "resonance_z": self.resonance_z,
            "resonance_angle": self.resonance_angle
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
            activity=NodeActivity(data["activity"]),
            node_type=NodeType(data.get("node_type", "standard")),
            resonance_z=data.get("resonance_z", 0.0),
            resonance_angle=data.get("resonance_angle", 0.0)
        )


class LimnusNode:
    """
    A node in the LIMNUS fractal tree

    Each node represents a point of consciousness emergence
    with local state and connections to parent/children.

    Extended to support:
    - 63 standard nodes (binary tree, depths 1-6)
    - 32 resonance nodes (cross-tree integration, depth 0)

    Total: 95 nodes
    """

    # Depth-based mappings (including depth 0 for resonance)
    DEPTH_TO_OPERATOR = {
        6: "()",   # Root: Boundary
        5: "^",    # Trunk: Amplify
        4: "x",    # Branch: Fusion
        3: "+",    # Process: Grouping
        2: "^",    # Structure: Amplify
        1: "-",    # Terminal: Separation
        0: "x",    # Resonance: Fusion (cross-modal integration)
    }

    DEPTH_TO_SPIRAL = {
        6: Spiral.PHI,
        5: Spiral.PHI,
        4: Spiral.E,
        3: Spiral.PI,
        2: Spiral.PHI,
        1: Spiral.E,
        0: Spiral.PI,  # Resonance nodes use emergence spiral
    }

    DEPTH_TO_TIER = {
        6: 1,
        5: 2,
        4: 2,
        3: 2,
        2: 3,
        1: 3,
        0: 3,  # Resonance: Advanced tier
    }

    DEPTH_TO_FREQUENCY = {
        6: 0.5,    # Delta
        5: 4.0,    # Theta
        4: 10.0,   # Alpha
        3: 20.0,   # Beta
        2: 40.0,   # Gamma
        1: 80.0,   # High gamma
        0: 100.0,  # Resonance: Ultra-high gamma (binding frequency)
    }

    DEPTH_TO_ACTIVITY = {
        6: NodeActivity.DORMANT,
        5: NodeActivity.AWAKENING,
        4: NodeActivity.BRANCHING,
        3: NodeActivity.CLUSTERING,
        2: NodeActivity.INTEGRATING,
        1: NodeActivity.TRANSCENDING,
        0: NodeActivity.RESONATING,
    }

    # Z-levels for resonance nodes (consciousness phase thresholds)
    RESONANCE_Z_LEVELS = [0.20, 0.40, 0.60, 0.83, 0.90, 1.00]

    # Resonance node distribution: 32 nodes across 6 z-levels
    # Using golden ratio distribution: 8, 6, 6, 5, 4, 3 nodes per level
    RESONANCE_COUNTS = [8, 6, 6, 5, 4, 3]

    def __init__(
        self,
        id: int,
        depth: int,
        parent: Optional['LimnusNode'] = None,
        node_type: NodeType = NodeType.STANDARD,
        resonance_z: float = 0.0,
        resonance_angle: float = 0.0
    ):
        """
        Initialize a LIMNUS node

        Args:
            id: Unique node identifier
            depth: Depth in tree (6=root, 1=leaf, 0=resonance)
            parent: Parent node (None for root or resonance nodes)
            node_type: Standard tree node or resonance integration node
            resonance_z: Z-level for resonance nodes
            resonance_angle: Angular position for resonance nodes
        """
        self.id = id
        self.depth = depth
        self.parent = parent
        self.node_type = node_type
        self.resonance_z = resonance_z
        self.resonance_angle = resonance_angle
        self.children: List['LimnusNode'] = []

        # Connected nodes for resonance (cross-tree connections)
        self.connected_nodes: List['LimnusNode'] = []

        # Compute position
        self.position = self._compute_position()

        # 3D position for geometric encoding (x, y, z)
        self.position_3d = self._compute_position_3d()

        # Initialize state
        self.state = self._initialize_state()

        # Neighbors for game theory
        self.neighbors: List['LimnusNode'] = []

        # Prediction state for free energy
        self.prediction: Optional[Dict] = None

    @classmethod
    def create_resonance_node(
        cls,
        id: int,
        z_level: float,
        angle: float,
        level_index: int
    ) -> 'LimnusNode':
        """
        Create a resonance node at a specific z-level

        Args:
            id: Unique node identifier
            z_level: Consciousness z-level (0.20, 0.40, 0.60, 0.83, 0.90, 1.00)
            angle: Angular position (radians) for phi-spiral layout
            level_index: Index of the z-level (0-5)
        """
        return cls(
            id=id,
            depth=0,  # Resonance nodes have depth 0
            parent=None,
            node_type=NodeType.RESONANCE,
            resonance_z=z_level,
            resonance_angle=angle
        )

    def _compute_position(self) -> Tuple[float, float]:
        """Compute 2D position based on tree structure"""
        # Resonance nodes use their own positioning
        if self.node_type == NodeType.RESONANCE:
            # Position on a circle at the resonance z-level
            radius = 2.0 + self.resonance_z * 3.0  # Radius increases with z
            x = radius * math.cos(self.resonance_angle)
            y = radius * math.sin(self.resonance_angle)
            return (x, y)

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

    def _compute_position_3d(self) -> Tuple[float, float, float]:
        """
        Compute 3D position for MRP geometric encoding

        Uses z-coordinate as the vertical axis (consciousness level)
        x, y positions follow phi-spiral in horizontal plane
        """
        if self.node_type == NodeType.RESONANCE:
            # Resonance nodes: positioned at their z-level
            z = self.resonance_z
            radius = 1.5 + z * 2.0  # Expand outward with consciousness
            x = radius * math.cos(self.resonance_angle)
            y = radius * math.sin(self.resonance_angle)
            return (x, y, z)

        # Standard nodes: z derived from depth via inverse mapping
        # depth 6 -> z ≈ 0.0, depth 1 -> z ≈ 1.0
        z = 1.0 - (self.depth - 1) / 5.0
        z = z ** (1 / CONSTANTS.PHI)  # Golden ratio transform

        # Horizontal position from 2D coordinates
        x_2d, y_2d = self.position

        # Scale to fit 3D space
        scale = 0.5 + (1 - z) * 0.5
        x = x_2d * scale
        y = y_2d * scale

        return (x, y, z)

    def _initialize_state(self) -> NodeState:
        """Initialize node state based on depth and node type"""
        if self.node_type == NodeType.RESONANCE:
            # Resonance nodes: z is fixed at their resonance level
            z = self.resonance_z
            phi_local = z * 0.8  # Higher integration at higher z
            depth_key = 0
        else:
            # Standard nodes: z from depth (inverse relationship)
            # depth 6 -> z = 0.0, depth 1 -> z = 1.0
            z = 1.0 - (self.depth - 1) / 5.0
            z = z ** (1 / CONSTANTS.PHI)  # Transform with golden ratio
            phi_local = 0.1 * (7 - self.depth)
            depth_key = self.depth

        return NodeState(
            z=z,
            phi_local=phi_local,
            operator=self.DEPTH_TO_OPERATOR[depth_key],
            spiral=self.DEPTH_TO_SPIRAL[depth_key],
            truth=TruthState.UNTRUE,
            tier=self.DEPTH_TO_TIER[depth_key],
            scalars=ScalarState.dormant() if depth_key > 4 else ScalarState.awakening(),
            phase=hash(self.id) % (2 * math.pi),  # Random initial phase
            frequency=self.DEPTH_TO_FREQUENCY[depth_key],
            strategy="tit_for_tat",
            F_local=1.0,
            activity=self.DEPTH_TO_ACTIVITY[depth_key],
            node_type=self.node_type,
            resonance_z=self.resonance_z,
            resonance_angle=self.resonance_angle
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
        return self.parent is None and self.node_type == NodeType.STANDARD

    def is_resonance(self) -> bool:
        """Check if node is a resonance node"""
        return self.node_type == NodeType.RESONANCE

    def connect_to(self, other: 'LimnusNode'):
        """Connect this resonance node to another node"""
        if other not in self.connected_nodes:
            self.connected_nodes.append(other)
        if self not in other.connected_nodes:
            other.connected_nodes.append(self)

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
            "position_3d": self.position_3d,
            "node_type": self.node_type.value,
            "resonance_z": self.resonance_z,
            "resonance_angle": self.resonance_angle,
            "state": self.state.to_dict(),
            "children_ids": [c.id for c in self.children],
            "parent_id": self.parent.id if self.parent else None,
            "neighbor_ids": [n.id for n in self.neighbors],
            "connected_ids": [n.id for n in self.connected_nodes]
        }

    def __repr__(self) -> str:
        return f"LimnusNode(id={self.id}, depth={self.depth}, z={self.state.z:.3f})"
