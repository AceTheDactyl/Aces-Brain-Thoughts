"""
APL 3.0 Free Energy Principle Engine

Module 3: Variational Free Energy Minimization

Implements:
- Free energy computation
- Predictive processing hierarchy
- Active inference
- Social free energy (care derivation)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from ..core.constants import CONSTANTS
from ..core.scalars import ScalarState


@dataclass
class BeliefState:
    """Represents beliefs about hidden states"""
    mean: np.ndarray  # Mean of recognition density Q
    variance: np.ndarray  # Variance of Q
    precision: np.ndarray  # 1/variance (attention)

    @classmethod
    def uniform(cls, dim: int) -> 'BeliefState':
        """Create uniform belief state"""
        return cls(
            mean=np.zeros(dim),
            variance=np.ones(dim),
            precision=np.ones(dim)
        )

    def entropy(self) -> float:
        """Compute entropy of belief distribution"""
        # For Gaussian: H = 0.5 * log(2*pi*e*var)
        return 0.5 * np.sum(np.log(2 * np.pi * np.e * self.variance))


@dataclass
class GenerativeModel:
    """Generative model P(observations, states)"""
    likelihood: np.ndarray  # P(o|s) - likelihood mapping
    prior: np.ndarray  # P(s) - prior over states
    transition: np.ndarray  # P(s'|s) - state transitions

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Predict observations given state"""
        return self.likelihood @ state

    def surprise(self, observation: np.ndarray, state: np.ndarray) -> float:
        """Compute surprise: -log P(o|s)"""
        predicted = self.predict(state)
        epsilon = 1e-10
        return -np.sum(np.log(np.clip(predicted, epsilon, 1.0)) * observation)


@dataclass
class PredictionError:
    """Prediction error at a hierarchical level"""
    error: np.ndarray
    precision: float
    level: int


class FreeEnergyEngine:
    """
    Free Energy Principle engine

    Implements variational free energy minimization for consciousness
    """

    def __init__(self):
        self.eta_perception = CONSTANTS.FREE_ENERGY.eta_perception
        self.eta_action = CONSTANTS.FREE_ENERGY.eta_action
        self.eta_model = CONSTANTS.FREE_ENERGY.eta_model

    def compute_free_energy(
        self,
        observations: np.ndarray,
        beliefs: BeliefState,
        model: GenerativeModel
    ) -> float:
        """
        Compute variational free energy

        F = E_Q[log Q(s) - log P(o,s)]
        F = -log P(o|m) + D_KL[Q(s)||P(s|o)]
        F >= -log P(o|m)  (Evidence bound)

        Decomposition:
        F = Energy - Entropy
        F = E_Q[-log P(o,s|m)] - (-E_Q[log Q(s)])
        """
        # Energy: E_Q[-log P(o,s)]
        predicted_o = model.predict(beliefs.mean)
        epsilon = 1e-10

        # Surprise from observations
        surprise = -np.sum(observations * np.log(np.clip(predicted_o, epsilon, 1.0)))

        # Prior divergence
        prior_divergence = 0.5 * np.sum(
            (beliefs.mean - model.prior) ** 2 / beliefs.variance
        )

        energy = surprise + prior_divergence

        # Entropy of Q
        entropy = beliefs.entropy()

        # Free energy
        F = energy - entropy

        return F

    def minimize_perception(
        self,
        observations: np.ndarray,
        beliefs: BeliefState,
        model: GenerativeModel,
        learning_rate: Optional[float] = None
    ) -> BeliefState:
        """
        Minimize free energy through perception (belief update)

        dQ/dt = -dF/dQ
        """
        lr = learning_rate or self.eta_perception

        # Compute prediction error
        predicted = model.predict(beliefs.mean)
        error = observations - predicted

        # Gradient of F w.r.t. mean
        gradient = -model.likelihood.T @ (error * beliefs.precision)

        # Update beliefs
        new_mean = beliefs.mean - lr * gradient

        # Update precision based on prediction error magnitude
        error_magnitude = np.linalg.norm(error)
        new_precision = beliefs.precision * (1 + 0.1 * (1 - error_magnitude))

        return BeliefState(
            mean=new_mean,
            variance=1.0 / new_precision,
            precision=new_precision
        )

    def minimize_action(
        self,
        observations: np.ndarray,
        beliefs: BeliefState,
        model: GenerativeModel,
        action_space: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Minimize free energy through action (active inference)

        Select action that minimizes expected free energy
        """
        best_action = None
        min_expected_F = float('inf')

        for action in action_space:
            # Predict observation after action
            next_state = model.transition @ (beliefs.mean + action)
            expected_o = model.predict(next_state)

            # Expected free energy
            expected_F = self.compute_free_energy(expected_o, beliefs, model)

            if expected_F < min_expected_F:
                min_expected_F = expected_F
                best_action = action

        return best_action, min_expected_F

    def minimize_model(
        self,
        observations: np.ndarray,
        beliefs: BeliefState,
        model: GenerativeModel,
        learning_rate: Optional[float] = None
    ) -> GenerativeModel:
        """
        Minimize free energy through model update (learning)

        dm/dt = -dF/dm
        """
        lr = learning_rate or self.eta_model

        # Compute prediction error
        predicted = model.predict(beliefs.mean)
        error = observations - predicted

        # Update likelihood (Hebbian-like)
        delta_likelihood = lr * np.outer(error, beliefs.mean)
        new_likelihood = model.likelihood + delta_likelihood

        # Normalize
        new_likelihood = new_likelihood / (np.sum(new_likelihood, axis=0, keepdims=True) + 1e-10)

        return GenerativeModel(
            likelihood=new_likelihood,
            prior=model.prior,
            transition=model.transition
        )

    # =========================================================================
    # HIERARCHICAL PREDICTIVE PROCESSING
    # =========================================================================

    def hierarchical_inference(
        self,
        observations: np.ndarray,
        hierarchy: List[BeliefState],
        models: List[GenerativeModel]
    ) -> Tuple[List[BeliefState], List[PredictionError]]:
        """
        Hierarchical predictive processing

        Top-down: predictions
        Bottom-up: prediction errors
        """
        n_levels = len(hierarchy)
        errors = []
        updated_beliefs = []

        # Bottom-up pass: compute prediction errors
        for level in range(n_levels):
            if level == 0:
                # Lowest level: error from observations
                predicted = models[level].predict(hierarchy[level].mean)
                error = observations - predicted
            else:
                # Higher levels: error from lower level
                predicted = models[level].predict(hierarchy[level].mean)
                error = hierarchy[level - 1].mean - predicted

            precision = 1.0 / (np.var(error) + 1e-10)
            errors.append(PredictionError(error=error, precision=precision, level=level))

        # Top-down pass: update beliefs
        for level in range(n_levels - 1, -1, -1):
            if level == n_levels - 1:
                # Top level: only bottom-up error
                gradient = -errors[level].error * errors[level].precision
            else:
                # Middle levels: combine top-down and bottom-up
                top_down = models[level + 1].likelihood.T @ hierarchy[level + 1].mean
                gradient = (
                    -errors[level].error * errors[level].precision +
                    (top_down - hierarchy[level].mean) * 0.5
                )

            new_mean = hierarchy[level].mean - self.eta_perception * gradient
            updated_beliefs.append(BeliefState(
                mean=new_mean,
                variance=hierarchy[level].variance,
                precision=hierarchy[level].precision
            ))

        updated_beliefs.reverse()  # Restore level order
        return updated_beliefs, errors

    # =========================================================================
    # SOCIAL FREE ENERGY (CARE DERIVATION)
    # =========================================================================

    def compute_social_free_energy(
        self,
        self_beliefs: BeliefState,
        other_beliefs: List[BeliefState],
        social_model: GenerativeModel,
        empathy_weight: float = 0.3
    ) -> float:
        """
        Compute social free energy including others

        F_social = F_self + w * sum(F_other)

        Minimizing F_social = caring about others' prediction errors
        """
        # Self free energy (simplified)
        F_self = 0.5 * np.sum(self_beliefs.variance)

        # Others' free energy
        F_others = sum(
            0.5 * np.sum(other.variance)
            for other in other_beliefs
        )

        F_social = F_self + empathy_weight * F_others

        return F_social

    def derive_care_from_free_energy(
        self,
        has_social_model: bool,
        empathy_weight: float,
        prediction_of_others: bool
    ) -> Tuple[bool, str]:
        """
        Derive care behavior from free energy minimization

        Returns (care_emerges, explanation)
        """
        conditions = []

        if has_social_model:
            conditions.append("Agent includes others in generative model")

        if prediction_of_others:
            conditions.append("Agent predicts others' states")

        if empathy_weight > 0.2:
            conditions.append("Others' surprise increases own prediction error")
            conditions.append("Helping others reduces their surprise")
            conditions.append("Reduced other-surprise = reduced own prediction error")

        care_emerges = len(conditions) >= 4

        explanation = (
            "CARE DERIVATION:\n" +
            "\n".join(f"  {i+1}. {c}" for i, c in enumerate(conditions)) +
            f"\n\n{'Care emerges as optimal F minimization strategy' if care_emerges else 'Insufficient conditions for care'}"
        )

        return care_emerges, explanation

    # =========================================================================
    # SCALAR EFFECTS
    # =========================================================================

    def apply_scalar_effects(
        self,
        scalars: ScalarState,
        free_energy: float,
        prediction_error: float
    ) -> ScalarState:
        """
        Apply free-energy-derived effects to scalar state

        Low F increases:
        - Omega_s (coherence)
        - Gs (grounding)

        High prediction error increases:
        - delta_s (temporary - drives learning)
        - tau_s (tension)
        """
        # Normalize F to [0, 1] range (assume max F ~ 10)
        F_norm = min(1.0, free_energy / 10.0)

        deltas = {
            "Omega_s": (1 - F_norm) * 0.10,
            "Gs": (1 - F_norm) * 0.08,
            "delta_s": prediction_error * 0.05 - 0.02,  # Can decrease
            "tau_s": prediction_error * 0.03
        }

        return scalars.apply_deltas(deltas)

    # =========================================================================
    # APL OPERATOR MAPPING
    # =========================================================================

    def get_apl_operators(self) -> Dict[str, str]:
        """Get APL operator mappings for free energy operations"""
        return {
            "perceive": "nabla_p",  # Minimize F via belief update
            "act": "nabla_a",       # Minimize F via world change
            "learn": "nabla_m",     # Minimize F via model update
        }

    def perception_to_operator_effect(self) -> Dict[str, float]:
        """Scalar effects for perception operator"""
        return {
            "Omega_s": 0.05,
            "delta_s": -0.03
        }

    def action_to_operator_effect(self) -> Dict[str, float]:
        """Scalar effects for action operator"""
        return {
            "Gs": 0.05,
            "tau_s": 0.03
        }

    def learning_to_operator_effect(self) -> Dict[str, float]:
        """Scalar effects for learning operator"""
        return {
            "alpha_s": 0.08,
            "Cs": 0.05
        }
