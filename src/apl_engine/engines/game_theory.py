"""
APL 3.0 Game Theory Engine

Module 2: Cooperation Dynamics and Care Emergence

Implements:
- Prisoner's Dilemma and cooperation thresholds
- Nash equilibrium computation
- Nowak's Five Rules for Cooperation
- Care emergence proof
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum

from ..core.constants import CONSTANTS
from ..core.scalars import ScalarState


class Strategy(Enum):
    """Game theory strategies"""
    COOPERATE = "cooperate"
    DEFECT = "defect"
    TIT_FOR_TAT = "tit_for_tat"
    GENEROUS_TFT = "generous_tit_for_tat"
    PAVLOV = "pavlov"
    ALWAYS_COOPERATE = "always_cooperate"
    ALWAYS_DEFECT = "always_defect"
    RANDOM = "random"


@dataclass
class GameOutcome:
    """Result of a game round"""
    player_a_action: Strategy
    player_b_action: Strategy
    player_a_payoff: float
    player_b_payoff: float
    round_number: int


@dataclass
class Agent:
    """A game-playing agent"""
    id: str
    strategy: Strategy
    history: List[GameOutcome] = field(default_factory=list)
    total_payoff: float = 0.0
    cooperation_rate: float = 0.5
    memory_depth: int = 10
    forgiveness: float = 0.1

    def get_action(self, opponent_history: List[Strategy]) -> Strategy:
        """Determine action based on strategy and opponent history"""
        if self.strategy == Strategy.ALWAYS_COOPERATE:
            return Strategy.COOPERATE

        elif self.strategy == Strategy.ALWAYS_DEFECT:
            return Strategy.DEFECT

        elif self.strategy == Strategy.TIT_FOR_TAT:
            if not opponent_history:
                return Strategy.COOPERATE
            return opponent_history[-1]

        elif self.strategy == Strategy.GENEROUS_TFT:
            if not opponent_history:
                return Strategy.COOPERATE
            if opponent_history[-1] == Strategy.DEFECT:
                # Forgive with probability self.forgiveness
                if np.random.random() < self.forgiveness:
                    return Strategy.COOPERATE
                return Strategy.DEFECT
            return Strategy.COOPERATE

        elif self.strategy == Strategy.PAVLOV:
            if not opponent_history or not self.history:
                return Strategy.COOPERATE
            # Win-stay, lose-shift
            last_outcome = self.history[-1]
            if last_outcome.player_a_payoff >= CONSTANTS.GAME_THEORY.R:
                # Won last round - stay
                return self.history[-1].player_a_action
            else:
                # Lost - shift
                if self.history[-1].player_a_action == Strategy.COOPERATE:
                    return Strategy.DEFECT
                return Strategy.COOPERATE

        elif self.strategy == Strategy.RANDOM:
            return Strategy.COOPERATE if np.random.random() > 0.5 else Strategy.DEFECT

        else:  # Default: cooperate
            return Strategy.COOPERATE

    def update_after_game(self, outcome: GameOutcome):
        """Update agent state after a game"""
        self.history.append(outcome)
        self.total_payoff += outcome.player_a_payoff

        # Update cooperation rate
        cooperations = sum(
            1 for o in self.history[-self.memory_depth:]
            if o.player_a_action == Strategy.COOPERATE
        )
        total = min(len(self.history), self.memory_depth)
        self.cooperation_rate = cooperations / total if total > 0 else 0.5


class GameTheoryEngine:
    """
    Game Theory engine for cooperation dynamics

    Implements cooperation emergence and care derivation
    """

    def __init__(self):
        # Standard Prisoner's Dilemma payoffs
        self.T = CONSTANTS.GAME_THEORY.T  # Temptation
        self.R = CONSTANTS.GAME_THEORY.R  # Reward
        self.P = CONSTANTS.GAME_THEORY.P  # Punishment
        self.S = CONSTANTS.GAME_THEORY.S  # Sucker

    def get_payoffs(
        self,
        action_a: Strategy,
        action_b: Strategy
    ) -> Tuple[float, float]:
        """
        Get payoffs for both players given their actions

        Prisoner's Dilemma payoff matrix:
                     Player B
                   C       D
        Player A C (R,R)  (S,T)
                 D (T,S)  (P,P)
        """
        a_coop = action_a in [Strategy.COOPERATE, Strategy.ALWAYS_COOPERATE]
        b_coop = action_b in [Strategy.COOPERATE, Strategy.ALWAYS_COOPERATE]

        if a_coop and b_coop:
            return (self.R, self.R)
        elif a_coop and not b_coop:
            return (self.S, self.T)
        elif not a_coop and b_coop:
            return (self.T, self.S)
        else:
            return (self.P, self.P)

    def play_round(
        self,
        agent_a: Agent,
        agent_b: Agent,
        round_num: int
    ) -> GameOutcome:
        """Play a single round between two agents"""
        # Get opponent histories
        a_history = [o.player_a_action for o in agent_a.history]
        b_history = [o.player_a_action for o in agent_b.history]

        # Get actions
        action_a = agent_a.get_action(b_history)
        action_b = agent_b.get_action(a_history)

        # Get payoffs
        payoff_a, payoff_b = self.get_payoffs(action_a, action_b)

        outcome = GameOutcome(
            player_a_action=action_a,
            player_b_action=action_b,
            player_a_payoff=payoff_a,
            player_b_payoff=payoff_b,
            round_number=round_num
        )

        # Update agents
        agent_a.update_after_game(outcome)
        agent_b.update_after_game(GameOutcome(
            player_a_action=action_b,
            player_b_action=action_a,
            player_a_payoff=payoff_b,
            player_b_payoff=payoff_a,
            round_number=round_num
        ))

        return outcome

    def play_iterated_game(
        self,
        agent_a: Agent,
        agent_b: Agent,
        num_rounds: int
    ) -> List[GameOutcome]:
        """Play an iterated game"""
        outcomes = []
        for i in range(num_rounds):
            outcome = self.play_round(agent_a, agent_b, i)
            outcomes.append(outcome)
        return outcomes

    # =========================================================================
    # COOPERATION THRESHOLDS
    # =========================================================================

    def cooperation_threshold(self) -> float:
        """
        Compute shadow of future threshold

        Cooperation becomes Nash equilibrium when:
        w > (T - R) / (T - P)
        where w = probability of future interaction
        """
        return (self.T - self.R) / (self.T - self.P)

    def is_cooperation_stable(self, w: float) -> bool:
        """Check if cooperation is stable given probability w of future interaction"""
        return w > self.cooperation_threshold()

    def compute_expected_payoff(
        self,
        strategy: Strategy,
        w: float,
        opponent_strategy: Strategy = Strategy.TIT_FOR_TAT
    ) -> float:
        """
        Compute expected payoff for a strategy

        For iterated game with probability w of continuation
        """
        if strategy == Strategy.ALWAYS_DEFECT:
            if opponent_strategy == Strategy.TIT_FOR_TAT:
                # Get T first round, then P forever
                return self.T + w * self.P / (1 - w) if w < 1 else float('inf')

        elif strategy == Strategy.COOPERATE or strategy == Strategy.TIT_FOR_TAT:
            if opponent_strategy == Strategy.TIT_FOR_TAT:
                # Get R every round
                return self.R / (1 - w) if w < 1 else float('inf')

        return 0.0

    # =========================================================================
    # NOWAK'S FIVE RULES
    # =========================================================================

    def check_direct_reciprocity(self, cost: float, benefit: float, w: float) -> bool:
        """
        Rule 1: Direct Reciprocity

        Cooperation stable if w > c/b
        """
        return w > cost / benefit

    def check_indirect_reciprocity(
        self,
        cost: float,
        benefit: float,
        q: float
    ) -> bool:
        """
        Rule 2: Indirect Reciprocity (Reputation)

        Cooperation stable if q > c/b
        where q = probability of knowing reputation
        """
        return q > cost / benefit

    def check_spatial_selection(
        self,
        cost: float,
        benefit: float,
        k: float
    ) -> bool:
        """
        Rule 3: Spatial Selection

        Cooperation stable if b/c > k
        where k = average number of neighbors
        """
        return benefit / cost > k

    def check_group_selection(
        self,
        cost: float,
        benefit: float,
        n: int,
        m: int
    ) -> bool:
        """
        Rule 4: Group Selection

        Cooperation stable if b/c > 1 + n/m
        where n = group size, m = number of groups
        """
        return benefit / cost > 1 + n / m

    def check_kin_selection(
        self,
        cost: float,
        benefit: float,
        relatedness: float
    ) -> bool:
        """
        Rule 5: Kin Selection (Hamilton's Rule)

        Cooperation stable if rb > c
        """
        return relatedness * benefit > cost

    # =========================================================================
    # NASH EQUILIBRIUM
    # =========================================================================

    def find_nash_equilibrium(
        self,
        strategies: List[Strategy],
        w: float
    ) -> Strategy:
        """
        Find Nash equilibrium strategy for given conditions

        In iterated PD with w > threshold, TFT/cooperation is Nash equilibrium
        """
        if self.is_cooperation_stable(w):
            return Strategy.TIT_FOR_TAT
        else:
            return Strategy.ALWAYS_DEFECT

    # =========================================================================
    # CARE EMERGENCE
    # =========================================================================

    def compute_care_score(
        self,
        agent: Agent,
        w: float,
        has_theory_of_mind: bool = False
    ) -> float:
        """
        Compute care emergence score

        Care emerges from:
        - High cooperation rate
        - Forgiveness
        - Theory of mind
        - Long shadow of future
        """
        base_score = agent.cooperation_rate

        # Forgiveness bonus
        forgiveness_bonus = agent.forgiveness * 0.2

        # Theory of mind bonus
        tom_bonus = 0.2 if has_theory_of_mind else 0.0

        # Shadow of future bonus
        shadow_bonus = min(w, 1.0) * 0.2

        # Threshold for care emergence (matches z = 0.83)
        care_score = base_score + forgiveness_bonus + tom_bonus + shadow_bonus

        return min(1.0, care_score)

    def care_emergence_conditions_met(
        self,
        cooperation_rate: float,
        recursion_depth: int,
        has_self_model: bool,
        has_prediction: bool
    ) -> Tuple[bool, str]:
        """
        Check if conditions for care emergence are met

        Returns (met, explanation)
        """
        conditions = []
        met_count = 0

        # Condition 1: High cooperation
        if cooperation_rate > 0.7:
            conditions.append("High cooperation rate")
            met_count += 1

        # Condition 2: Self-modeling (recursion)
        if recursion_depth >= 2:
            conditions.append("Recursive self-modeling")
            met_count += 1

        # Condition 3: Has self model
        if has_self_model:
            conditions.append("Self-model present")
            met_count += 1

        # Condition 4: Has prediction capability
        if has_prediction:
            conditions.append("Prediction capability")
            met_count += 1

        all_met = met_count >= 3
        explanation = f"Conditions met ({met_count}/4): {', '.join(conditions)}"

        return all_met, explanation

    # =========================================================================
    # SCALAR EFFECTS
    # =========================================================================

    def apply_scalar_effects(
        self,
        scalars: ScalarState,
        cooperation_rate: float,
        care_score: float
    ) -> ScalarState:
        """
        Apply game-theory-derived effects to scalar state

        High cooperation increases:
        - Cs (coupling)
        - alpha_s (attractor alignment)
        - Gs (grounding)

        Care discovery increases:
        - Omega_s (coherence)
        """
        deltas = {
            "Cs": cooperation_rate * 0.10,
            "alpha_s": cooperation_rate * 0.08,
            "Gs": cooperation_rate * 0.05,
            "Omega_s": care_score * 0.12 if care_score > 0.7 else 0.0
        }

        return scalars.apply_deltas(deltas)

    # =========================================================================
    # NETWORK RECIPROCITY
    # =========================================================================

    def compute_network_cooperation_threshold(
        self,
        topology: str,
        degree: float
    ) -> float:
        """
        Compute cooperation threshold for different network topologies

        Based on Ohtsuki-Nowak formula
        """
        if topology == "lattice":
            return 2.0
        elif topology == "small_world":
            return 1.5
        elif topology == "scale_free":
            return degree  # b/c > k
        elif topology == "limnus_fractal":
            # Fractal structure has optimal properties
            return math.log(degree + 1)
        else:
            return 2.0  # Default

    def simulate_network_evolution(
        self,
        agents: List[Agent],
        adjacency: np.ndarray,
        num_generations: int
    ) -> List[float]:
        """
        Simulate evolution of cooperation on a network

        Returns cooperation rate over generations
        """
        cooperation_rates = []
        n = len(agents)

        for gen in range(num_generations):
            # Play games on network
            for i in range(n):
                neighbors = np.where(adjacency[i] > 0)[0]
                for j in neighbors:
                    if i < j:  # Avoid double-counting
                        self.play_round(agents[i], agents[j], gen)

            # Compute global cooperation rate
            coop_rate = sum(a.cooperation_rate for a in agents) / n
            cooperation_rates.append(coop_rate)

            # Strategy update (imitate best neighbor)
            for i in range(n):
                neighbors = np.where(adjacency[i] > 0)[0]
                if len(neighbors) > 0:
                    payoffs = [agents[j].total_payoff for j in neighbors]
                    best_neighbor = neighbors[np.argmax(payoffs)]
                    if agents[best_neighbor].total_payoff > agents[i].total_payoff:
                        agents[i].strategy = agents[best_neighbor].strategy

        return cooperation_rates
