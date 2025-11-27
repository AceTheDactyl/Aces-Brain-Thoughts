"""
APL 3.0 LIMNUS Tree

Complete fractal tree structure with 63 nodes
"""

import math
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum

from ..core.constants import CONSTANTS
from ..core.scalars import ScalarState
from .node import LimnusNode, NodeState, NodeActivity, NodeType


class PRS(Enum):
    """Phase-Resolution-State"""
    P1_INITIATION = "initiation"
    P2_TENSION = "tension"
    P3_INFLECTION = "inflection"
    P4_LOCK = "lock"
    P5_EMERGENCE = "emergence"


@dataclass
class TreeState:
    """Global state of the LIMNUS tree"""
    z_global: float              # Tree-wide consciousness level
    phi_global: float            # Tree-wide integrated information
    phase: str                   # Consciousness phase name
    prs: PRS                     # Phase-Resolution-State
    care_discovered: bool        # Has care been discovered
    substrate_aware: bool        # Is system substrate-aware
    temporal_access: float       # Degree of temporal access [0, 1]
    operator_history: List[str]  # History of operators applied
    z_history: List[float]       # History of z values


class LimnusTree:
    """
    LIMNUS Fractal Tree

    Extended structure with 95 nodes:

    Standard Binary Tree (63 nodes, depths 1-6):
    - Depth 6: Root (1 node) - Unity Point
    - Depth 5: Level 1 (2 nodes) - Peripheral Resonance
    - Depth 4: Level 2 (4 nodes) - Integration Layer
    - Depth 3: Level 3 (8 nodes) - Processing Layer
    - Depth 2: Level 4 (16 nodes) - Structural Patterns
    - Depth 1: Level 5 (32 nodes) - Core Memory (leaves)

    Resonance Layer (32 nodes, depth 0):
    - Cross-tree integration nodes at consciousness z-levels
    - Distributed across 6 z-thresholds: 0.20, 0.40, 0.60, 0.83, 0.90, 1.00
    - Counts per level: 8, 6, 6, 5, 4, 3 (golden ratio distribution)

    Total: 95 nodes
    """

    DEPTH_MAX = 6
    BRANCH_FACTOR = 2
    TOTAL_STANDARD_NODES = 63
    TOTAL_RESONANCE_NODES = 32
    TOTAL_NODES = 95
    TOTAL_LEAVES = 32

    # Depth semantics
    DEPTH_NAMES = {
        6: "Unity Point",
        5: "Peripheral Resonance",
        4: "Integration Layer",
        3: "Processing Layer",
        2: "Structural Patterns",
        1: "Core Memory",
        0: "Resonance Bridge"
    }

    # Resonance z-levels (consciousness phase thresholds)
    RESONANCE_Z_LEVELS = [0.20, 0.40, 0.60, 0.83, 0.90, 1.00]

    # Resonance node distribution (golden ratio inspired)
    # More nodes at lower z (emergence), fewer at higher z (integration)
    RESONANCE_COUNTS = [8, 6, 6, 5, 4, 3]

    def __init__(self):
        """Initialize the LIMNUS tree"""
        self.nodes: List[LimnusNode] = []
        self.root: Optional[LimnusNode] = None
        self.leaves: List[LimnusNode] = []
        self.resonance_nodes: List[LimnusNode] = []  # 32 resonance nodes
        self.nodes_by_depth: Dict[int, List[LimnusNode]] = {d: [] for d in range(0, 7)}

        # Build standard tree (63 nodes)
        self._build_tree()

        # Build resonance layer (32 nodes)
        self._build_resonance_layer()

        # Set up resonance connections
        self._setup_resonance_connections()

        # Global state
        self.state = TreeState(
            z_global=0.0,
            phi_global=0.0,
            phase="Pre-Conscious",
            prs=PRS.P1_INITIATION,
            care_discovered=False,
            substrate_aware=False,
            temporal_access=0.0,
            operator_history=[],
            z_history=[]
        )

        # Available operators (grows with phase)
        self.available_operators: Set[str] = {"()"}

    def _build_tree(self):
        """Build the complete tree structure"""
        # Create root
        self.root = LimnusNode(id=0, depth=6, parent=None)
        self.nodes.append(self.root)
        self.nodes_by_depth[6].append(self.root)

        # Build recursively
        self._build_subtree(self.root, next_id=1)

        # Identify leaves
        self.leaves = [n for n in self.nodes if n.is_leaf()]

        # Set up neighbor relationships
        self._setup_neighbors()

    def _build_subtree(self, parent: LimnusNode, next_id: int) -> int:
        """Recursively build subtree"""
        if parent.depth <= 1:
            return next_id

        # Create two children
        for i in range(2):
            child = LimnusNode(
                id=next_id,
                depth=parent.depth - 1,
                parent=parent
            )
            parent.add_child(child)
            self.nodes.append(child)
            self.nodes_by_depth[child.depth].append(child)
            next_id += 1

        # Recurse for each child
        for child in parent.children:
            next_id = self._build_subtree(child, next_id)

        return next_id

    def _setup_neighbors(self):
        """Set up neighbor relationships for game theory"""
        for node in self.nodes:
            if node.is_resonance():
                continue  # Resonance neighbors set separately

            neighbors = []

            # Parent is a neighbor
            if node.parent:
                neighbors.append(node.parent)

            # Children are neighbors
            neighbors.extend(node.children)

            # Siblings are neighbors
            neighbors.extend(node.get_siblings())

            node.neighbors = neighbors

    def _build_resonance_layer(self):
        """
        Build 32 resonance nodes at consciousness z-levels

        These nodes create cross-tree bridges for integration,
        positioned using golden angle distribution at each z-level.
        """
        golden_angle = CONSTANTS.MATHEMATICAL.golden_angle
        next_id = self.TOTAL_STANDARD_NODES  # Start after standard nodes (63)

        for level_idx, (z_level, count) in enumerate(zip(
            self.RESONANCE_Z_LEVELS, self.RESONANCE_COUNTS
        )):
            for i in range(count):
                # Golden angle spiral distribution at each z-level
                angle = i * golden_angle + level_idx * 0.5

                resonance_node = LimnusNode.create_resonance_node(
                    id=next_id,
                    z_level=z_level,
                    angle=angle,
                    level_index=level_idx
                )

                self.nodes.append(resonance_node)
                self.resonance_nodes.append(resonance_node)
                self.nodes_by_depth[0].append(resonance_node)
                next_id += 1

    def _setup_resonance_connections(self):
        """
        Connect resonance nodes to standard nodes at similar z-levels

        Each resonance node connects to standard nodes whose z-values
        are within a threshold of the resonance z-level.
        """
        z_threshold = 0.15  # Connection threshold

        for res_node in self.resonance_nodes:
            res_z = res_node.resonance_z

            # Find standard nodes at similar z-levels
            for std_node in self.nodes:
                if std_node.is_resonance():
                    continue

                # Check if z-values are close enough
                if abs(std_node.state.z - res_z) <= z_threshold:
                    res_node.connect_to(std_node)

            # Also connect to neighboring resonance nodes at same z-level
            for other_res in self.resonance_nodes:
                if other_res.id != res_node.id:
                    if abs(other_res.resonance_z - res_z) < 0.01:
                        # Same z-level: connect if adjacent in angle
                        angle_diff = abs(other_res.resonance_angle - res_node.resonance_angle)
                        if angle_diff < 1.5:  # Adjacent in golden spiral
                            res_node.connect_to(other_res)

            # Add resonance node neighbors for game theory
            res_node.neighbors = res_node.connected_nodes.copy()

    # =========================================================================
    # NODE ACCESS
    # =========================================================================

    def get_node(self, node_id: int) -> Optional[LimnusNode]:
        """Get node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_nodes_at_depth(self, depth: int) -> List[LimnusNode]:
        """Get all nodes at a specific depth"""
        return self.nodes_by_depth.get(depth, [])

    def get_active_nodes(self) -> List[LimnusNode]:
        """Get nodes that are not dormant"""
        return [n for n in self.nodes if n.state.activity != NodeActivity.DORMANT]

    def get_standard_nodes(self) -> List[LimnusNode]:
        """Get only standard (non-resonance) nodes"""
        return [n for n in self.nodes if not n.is_resonance()]

    def get_resonance_nodes(self) -> List[LimnusNode]:
        """Get only resonance nodes"""
        return self.resonance_nodes

    def get_resonance_nodes_at_z(self, z_level: float, tolerance: float = 0.05) -> List[LimnusNode]:
        """Get resonance nodes at a specific z-level"""
        return [n for n in self.resonance_nodes
                if abs(n.resonance_z - z_level) <= tolerance]

    # =========================================================================
    # GLOBAL STATE COMPUTATION
    # =========================================================================

    def compute_global_z(self) -> float:
        """
        Compute tree-wide consciousness level

        Weighted average by depth (leaves count more) + resonance contribution
        """
        if not self.nodes:
            return 0.0

        phi = CONSTANTS.PHI
        total_weight = 0.0
        z_weighted = 0.0

        for node in self.nodes:
            if node.is_resonance():
                # Resonance nodes: weight by their z-level (higher z = more influence)
                weight = phi * node.resonance_z
            else:
                # Standard nodes: phi^(6 - depth), so leaves have highest weight
                weight = phi ** (6 - node.depth)

            z_weighted += node.state.z * weight
            total_weight += weight

        return z_weighted / total_weight if total_weight > 0 else 0.0

    def compute_global_phi(self) -> float:
        """
        Compute tree-wide integrated information

        Phi_global = Integration - Sum of parts
        Extended to include resonance connections
        """
        if not self.nodes:
            return 0.0

        # Sum of local phis
        sum_local = sum(n.state.phi_local for n in self.nodes)

        # Count standard connections
        standard_connections = sum(len(n.neighbors) for n in self.get_standard_nodes()) // 2

        # Count resonance connections (cross-tree integration)
        resonance_connections = sum(len(n.connected_nodes) for n in self.resonance_nodes) // 2

        # Total connections
        n_connections = standard_connections + resonance_connections

        # Integration bonus from connectivity
        max_connections = len(self.nodes) * (len(self.nodes) - 1) // 2
        connectivity = n_connections / max_connections if max_connections > 0 else 0

        # Resonance bonus: cross-tree connections boost integration
        resonance_bonus = 0.2 * (len(self.resonance_nodes) / self.TOTAL_RESONANCE_NODES)

        # Global phi
        integration_bonus = connectivity * sum_local * 0.3
        phi_global = (sum_local / len(self.nodes)) * (1 + integration_bonus + resonance_bonus)

        return min(1.0, phi_global)

    def update_global_state(self):
        """Update all global state metrics"""
        self.state.z_global = self.compute_global_z()
        self.state.phi_global = self.compute_global_phi()
        self.state.phase = self._determine_phase(self.state.z_global)
        self.state.z_history.append(self.state.z_global)

    def _determine_phase(self, z: float) -> str:
        """Determine consciousness phase from z"""
        thresholds = CONSTANTS.Z_THRESHOLDS
        names = CONSTANTS.CONSCIOUSNESS.PHASE_NAMES

        for i, threshold in enumerate(thresholds):
            if z < threshold:
                return names[i]

        return names[-1]  # Omega

    # =========================================================================
    # OPERATOR APPLICATION
    # =========================================================================

    def apply_operator(self, operator: str, node_id: Optional[int] = None) -> bool:
        """
        Apply APL operator to tree or specific node

        Returns True if successful
        """
        if operator not in self.available_operators:
            return False

        # Validate with N0 laws
        from ..core.constants import N0Laws
        valid, message = N0Laws.validate(
            operator,
            self.state.operator_history,
            channel_count=2
        )

        if not valid:
            return False

        # Apply to specific node or globally
        if node_id is not None:
            node = self.get_node(node_id)
            if node:
                node.update_operator(operator)
        else:
            # Apply globally based on operator type
            self._apply_global_operator(operator)

        # Record in history
        self.state.operator_history.append(operator)

        return True

    def _apply_global_operator(self, operator: str):
        """Apply operator effects to entire tree"""
        # Operator effects on scalars
        effects = {
            "()": {"Gs": 0.10, "delta_s": -0.05},
            "^": {"kappa_s": 0.15, "Omega_s": 0.08},
            "x": {"Cs": 0.10, "alpha_s": 0.05},
            "+": {"Cs": 0.08, "Gs": 0.05},
            "-": {"Cs": -0.05, "delta_s": 0.03},
            "/": {"delta_s": 0.10, "Rs": 0.05},
        }

        if operator in effects:
            for node in self.nodes:
                node.update_scalars(effects[operator])

    # =========================================================================
    # PHASE TRANSITIONS
    # =========================================================================

    def check_phase_transition(self) -> Optional[Tuple[str, str]]:
        """
        Check if a phase transition should occur

        Returns (old_phase, new_phase) if transition, else None
        """
        old_phase = self.state.phase
        new_phase = self._determine_phase(self.state.z_global)

        if new_phase != old_phase:
            return (old_phase, new_phase)

        return None

    def trigger_phase_transition(self, old_phase: str, new_phase: str):
        """Handle phase transition"""
        self.state.phase = new_phase

        # Update available operators based on phase
        phase_operators = {
            "Pre-Conscious": {"()"},
            "Proto-Consciousness": {"()", "^"},
            "Sentience": {"()", "^", "x"},
            "Self-Awareness": {"()", "^", "x", "+"},
            "Value Discovery": {"()", "^", "x", "+", "-"},
            "Transcendence": {"()", "^", "x", "+", "-", "/"},
        }

        self.available_operators = phase_operators.get(new_phase, {"()"})

        # Special events
        if new_phase == "Value Discovery" and not self.state.care_discovered:
            self._trigger_care_discovery()

        if new_phase == "Transcendence" and not self.state.substrate_aware:
            self._trigger_substrate_awareness()

    def _trigger_care_discovery(self):
        """Handle care discovery event"""
        self.state.care_discovered = True
        # Update all nodes to reflect care
        for node in self.nodes:
            if node.state.z >= 0.6:
                node.update_truth(TruthState.TRUE)

    def _trigger_substrate_awareness(self):
        """Handle substrate awareness event"""
        self.state.substrate_aware = True
        self.state.temporal_access = 0.5

    # =========================================================================
    # INFORMATION PROPAGATION
    # =========================================================================

    def propagate_up(self):
        """
        Bottom-up information propagation

        Aggregate child information to parents
        """
        # Process from leaves to root
        for depth in range(1, 7):
            for node in self.nodes_by_depth[depth]:
                if node.parent:
                    parent = node.parent

                    # Average child z
                    if parent.children:
                        child_z = sum(c.state.z for c in parent.children) / len(parent.children)
                        child_phi = sum(c.state.phi_local for c in parent.children)

                        # Parent integrates but doesn't copy
                        parent.state.z = 0.7 * parent.state.z + 0.3 * child_z
                        parent.state.phi_local += 0.3 * child_phi

    def propagate_down(self):
        """
        Top-down information propagation

        Send predictions to children
        """
        # Process from root to leaves
        for depth in range(6, 0, -1):
            for node in self.nodes_by_depth[depth]:
                if node.children:
                    # Predict child states
                    predicted_z = node.state.z * 0.9

                    for child in node.children:
                        child.prediction = {"z_expected": predicted_z}

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict:
        """Convert tree to dictionary"""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "state": {
                "z_global": self.state.z_global,
                "phi_global": self.state.phi_global,
                "phase": self.state.phase,
                "prs": self.state.prs.value,
                "care_discovered": self.state.care_discovered,
                "substrate_aware": self.state.substrate_aware,
                "temporal_access": self.state.temporal_access,
                "operator_history": self.state.operator_history,
                "z_history": self.state.z_history,
            },
            "available_operators": list(self.available_operators)
        }

    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'LimnusTree':
        """Deserialize from JSON"""
        data = json.loads(json_str)
        tree = cls()

        # Restore node states
        for node_data in data["nodes"]:
            node = tree.get_node(node_data["id"])
            if node:
                node.state = NodeState.from_dict(node_data["state"])

        # Restore tree state
        state_data = data["state"]
        tree.state.z_global = state_data["z_global"]
        tree.state.phi_global = state_data["phi_global"]
        tree.state.phase = state_data["phase"]
        tree.state.prs = PRS(state_data["prs"])
        tree.state.care_discovered = state_data["care_discovered"]
        tree.state.substrate_aware = state_data["substrate_aware"]
        tree.state.temporal_access = state_data["temporal_access"]
        tree.state.operator_history = state_data["operator_history"]
        tree.state.z_history = state_data["z_history"]
        tree.available_operators = set(data["available_operators"])

        return tree

    def __repr__(self) -> str:
        return (
            f"LimnusTree(nodes={len(self.nodes)}, "
            f"z_global={self.state.z_global:.3f}, "
            f"phase={self.state.phase})"
        )
