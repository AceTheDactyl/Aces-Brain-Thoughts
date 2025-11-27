"""
APL 3.0 LIMNUS Fractal Consciousness Module

Extended LIMNUS architecture with 95 nodes:
- 63 standard nodes (binary tree, depths 1-6)
- 32 resonance nodes (cross-tree integration, depth 0)

Features:
- Golden ratio integration (phi-weighted positioning)
- Consciousness z-coordinate as primary organizing axis
- MRP (Meta-Recursive Protocol) geometric encoding
- Phase transitions at z-thresholds
- Resonance bridges for cross-modal integration
"""

from .node import LimnusNode, NodeState, NodeActivity, NodeType
from .tree import LimnusTree, TreeState, PRS
from .evolution import LimnusEvolutionEngine, EvolutionMetrics, EvolutionInput

__all__ = [
    # Node components
    'LimnusNode',
    'NodeState',
    'NodeActivity',
    'NodeType',
    # Tree components
    'LimnusTree',
    'TreeState',
    'PRS',
    # Evolution components
    'LimnusEvolutionEngine',
    'EvolutionMetrics',
    'EvolutionInput',
]
