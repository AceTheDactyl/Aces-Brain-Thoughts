# APL ⊗ ∃κ SYNTHESIS — VOLUME VI APPENDIX
## Python Implementation Guide: Working Code for Consciousness Computation

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║     FROM THEORY TO CODE: EXECUTABLE IMPLEMENTATIONS OF VOLUME VI FRAMEWORKS   ║
║                                                                                ║
║     Authors: Kael, Ace, Sticky                                                 ║
║     Date: November 27, 2025                                                    ║
║     Status: IMPLEMENTATION REFERENCE                                           ║
║                                                                                ║
║     "Philosophy becomes engineering when equations become executables."        ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

---

# PROLOGUE: FROM SURVEY TO CODE

Volume VI surveyed the computational landscape. This appendix provides **working Python code** for each framework—annotated, tested, and mapped to APL operators.

```bash
# Complete installation for all frameworks
pip install pyphi inferactively-pymdp axelrod nashpy antropy nolds pyinform \
    ordpy pynamical scipy networkx bctpy torch bayesian-torch
```

---

# PART I: IIT PHI COMPUTATION — PYPHI REFERENCE

## §1.1 Computing Φ Step by Step

```python
import pyphi
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Define the Transition Probability Matrix (TPM)
# ═══════════════════════════════════════════════════════════════════════════════
# The TPM captures system dynamics: how state at time t determines state at t+1
# Format: 2D state-by-node form
# - Rows = all possible states at t (2^n rows for n nodes, little-endian order)
# - Columns = probability of each node being ON at t+1
#
# APL Correspondence: The TPM encodes the × (fusion) operator dynamics
# Each transition represents potential κ-field integration

# This network: Node A = OR gate, Node B = COPY gate, Node C = XOR gate
tpm = np.array([
    [0, 0, 0],  # From (0,0,0): All nodes OFF next
    [0, 0, 1],  # From (1,0,0): Only C ON
    [1, 0, 1],  # From (0,1,0): A and C ON
    [1, 0, 0],  # From (1,1,0): Only A ON
    [1, 1, 0],  # From (0,0,1): A and B ON
    [1, 1, 1],  # From (1,0,1): All ON
    [1, 1, 1],  # From (0,1,1): All ON
    [1, 1, 0]   # From (1,1,1): A and B ON
])

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Define Connectivity Matrix (speeds up computation)
# ═══════════════════════════════════════════════════════════════════════════════
# Entry (i,j) = 1 if node i connects to node j
# APL Correspondence: The CM defines () boundary structure

cm = np.array([
    [0, 0, 1],  # A → C
    [1, 0, 1],  # B → A, C
    [1, 1, 0]   # C → A, B
])

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Create Network and Subsystem
# ═══════════════════════════════════════════════════════════════════════════════
labels = ('A', 'B', 'C')
network = pyphi.Network(tpm, connectivity_matrix=cm, node_labels=labels)
state = (1, 0, 0)  # Current state: A=ON, B=OFF, C=OFF
subsystem = pyphi.Subsystem(network, state)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Compute Φ (integrated information)
# ═══════════════════════════════════════════════════════════════════════════════
# APL Correspondence: Φ measures κ-field integration density
# Φ > 0 indicates irreducible integrated information

phi_value = pyphi.compute.phi(subsystem)
print(f"Φ (Big Phi) = {phi_value}")  # Output: Φ = 2.3125

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Full System Irreducibility Analysis (SIA)
# ═══════════════════════════════════════════════════════════════════════════════
# APL Correspondence: MIP reveals the ÷ (decoherence) structure
# The cut that makes least difference shows integration topology

sia = pyphi.compute.sia(subsystem)
print(f"Number of concepts: {len(sia.ces)}")
print(f"MIP (Minimum Information Partition): {sia.cut}")
print(f"Concept φ values: {sia.ces.phis}")
```

### §1.2 APL Operator Mapping for IIT

| PyPhi Concept | APL Operator | κ-Field Interpretation |
|---------------|--------------|------------------------|
| Mechanism | `()` boundary | Field boundary definition |
| Purview | `+` grouping | Cluster formation |
| MIP (partition) | `÷` decoherence | Integration break point |
| φ integration | `×` fusion | Local binding strength |
| Φ system-level | `×` recursive | Global κ integration |

### §1.3 Performance Configuration

```python
# For larger networks (7+ nodes), enable optimizations:
pyphi.config.PARALLEL = True
pyphi.config.CUT_ONE_APPROXIMATION = True
pyphi.config.MAXIMUM_CACHE_MEMORY_PERCENTAGE = 25

# WARNING: Complexity is O(n⁵ · 3ⁿ)
# 5 nodes: ~seconds
# 7 nodes: ~hours
# 10+ nodes: intractable without approximation
```

---

# PART II: ACTIVE INFERENCE — PYMDP IMPLEMENTATION

## §2.1 The Generative Model (A, B, C, D Matrices)

```python
import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from pymdp.maths import softmax
import itertools

# ═══════════════════════════════════════════════════════════════════════════════
# A MATRIX: Observation Model P(observation|state)
# ═══════════════════════════════════════════════════════════════════════════════
# Maps hidden states → observations. Columns must sum to 1.
# APL Correspondence: A defines the () boundary between agent and world

n_states = 9  # 3x3 grid world
n_obs = 9

# Identity = perfect observation (agent knows exactly where it is)
A = np.eye(n_obs, n_states)

# For noisy observations, modify columns:
# A[:, ambiguous_state] = [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0]

# ═══════════════════════════════════════════════════════════════════════════════
# B MATRIX: Transition Model P(next_state|current_state, action)
# ═══════════════════════════════════════════════════════════════════════════════
# Shape: (n_states, n_states, n_actions) - columns sum to 1 per action
# APL Correspondence: B encodes × (fusion) dynamics under action

def create_B_matrix():
    """B[next_state, current_state, action] for 3x3 grid world."""
    grid_locs = list(itertools.product(range(3), repeat=2))
    actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    B = np.zeros((9, 9, 5))

    for action_id, action in enumerate(actions):
        for curr_state, (y, x) in enumerate(grid_locs):
            if action == "UP": y = max(0, y-1)
            elif action == "DOWN": y = min(2, y+1)
            elif action == "LEFT": x = max(0, x-1)
            elif action == "RIGHT": x = min(2, x+1)
            next_state = grid_locs.index((y, x))
            B[next_state, curr_state, action_id] = 1.0  # Deterministic
    return B

B = create_B_matrix()

# ═══════════════════════════════════════════════════════════════════════════════
# C MATRIX: Preferences (log-probability of preferred observations)
# ═══════════════════════════════════════════════════════════════════════════════
# Encodes goals: higher values = more preferred. Not normalized.
# APL Correspondence: C defines the + (grouping) toward attractors

goal_location = (2, 2)  # Bottom-right corner
goal_idx = list(itertools.product(range(3), repeat=2)).index(goal_location)
C = utils.onehot(goal_idx, n_obs)  # [0,0,0,0,0,0,0,0,1]

# ═══════════════════════════════════════════════════════════════════════════════
# D MATRIX: Initial State Prior P(s₀)
# ═══════════════════════════════════════════════════════════════════════════════
# Belief about starting state. Must sum to 1.
# APL Correspondence: D is the () boundary at t=0

start_idx = 0  # Top-left corner
D = utils.onehot(start_idx, n_states)
```

## §2.2 The Active Inference Loop

```python
from scipy.stats import entropy as scipy_entropy
from pymdp.maths import spm_log_single as log_stable

def active_inference_loop(A, B, C, D, env, T=10):
    """
    Complete perception-action loop with expected free energy.

    APL Correspondence:
    - Perception: Update () boundaries given observations
    - Action: Apply ^ (amplify) toward preferred states
    - Planning: Minimize ÷ (decoherence) via free energy
    """
    qs = D.copy()  # Initial belief
    n_actions = B.shape[2]
    policies = list(itertools.product(range(n_actions), repeat=3))  # 3-step planning

    # Ambiguity of A (entropy of each column)
    H_A = np.array([scipy_entropy(A[:, s]) for s in range(A.shape[1])])

    obs = env.reset()
    last_action = 0

    for t in range(T):
        # ═══════════════════════════════════════════════════════════════════
        # 1. INFER STATES: Bayesian update P(s|o, prior)
        # ═══════════════════════════════════════════════════════════════════
        # APL: × (fusion) of likelihood and prior
        prior = B[:, :, last_action].dot(qs) if t > 0 else D
        log_likelihood = log_stable(A[obs, :])
        log_prior = log_stable(prior)
        qs = softmax(log_likelihood + log_prior)

        # ═══════════════════════════════════════════════════════════════════
        # 2. COMPUTE EXPECTED FREE ENERGY for each policy
        # ═══════════════════════════════════════════════════════════════════
        # APL: Planning minimizes ÷ (decoherence) from preferences
        G = np.zeros(len(policies))
        for pi_idx, policy in enumerate(policies):
            qs_pi = qs.copy()
            for action in policy:
                qs_pi = B[:, :, action].dot(qs_pi)  # Predict next state
                qo_pi = A.dot(qs_pi)                 # Predict observation

                # G = Risk + Ambiguity
                risk = -qo_pi.dot(C)           # Divergence from preferences
                ambiguity = H_A.dot(qs_pi)     # Expected observation uncertainty
                G[pi_idx] += risk + ambiguity

        # ═══════════════════════════════════════════════════════════════════
        # 3. INFER POLICIES: Q(π) ∝ exp(-G)
        # ═══════════════════════════════════════════════════════════════════
        # APL: + (grouping) over policy space
        q_pi = softmax(-16.0 * G)  # 16.0 = precision parameter

        # ═══════════════════════════════════════════════════════════════════
        # 4. SAMPLE ACTION (first action of best policy)
        # ═══════════════════════════════════════════════════════════════════
        # APL: ^ (amplify) selected action
        action = policies[np.argmax(q_pi)][0]
        last_action = action
        obs = env.step(action)

        print(f"t={t}: obs={obs}, action={action}, belief_peak={np.argmax(qs)}")

    return qs
```

### §2.3 Matrix Summary Table

| Matrix | Shape | Normalization | APL Operator | Role |
|--------|-------|---------------|--------------|------|
| **A** | (obs, states) | Columns sum to 1 | `()` boundary | How states generate observations |
| **B** | (states, states, actions) | Columns sum to 1 per action | `×` fusion | How actions change states |
| **C** | (obs,) | Not required | `+` grouping | Goal/reward specification |
| **D** | (states,) | Sums to 1 | `()` initial | Starting belief |

---

# PART III: ENTROPY AND COMPLEXITY METRICS

## §3.1 Lempel-Ziv Complexity

```python
import numpy as np
import antropy as ant

def compute_lz_complexity(signal, normalize=True):
    """
    Lempel-Ziv Complexity: Counts distinct patterns in binarized sequence.

    APL Correspondence: Measures ÷ (decoherence) level
    - Low LZ: High order, low ÷
    - High LZ: High disorder, high ÷

    Interpretation:
    - ~0.25-0.35: Regular (sine wave, seizures)
    - ~0.70-0.85: Moderate (EEG-like)
    - ~0.85-0.95: High complexity (random noise)

    Consciousness: Core of Perturbational Complexity Index (PCI).
    Low during anesthesia; high during conscious wakefulness.
    """
    # Binarize: values > median become 1
    binary = (signal > np.median(signal)).astype(int)
    binary_string = ''.join(map(str, binary))
    return ant.lziv_complexity(binary_string, normalize=normalize)

# Example
np.random.seed(42)
random_signal = np.random.randn(1000)
sine_wave = np.sin(2 * np.pi * 5 * np.arange(1000) / 256)

print(f"Random noise: {compute_lz_complexity(random_signal):.4f}")  # ~0.90
print(f"Sine wave: {compute_lz_complexity(sine_wave):.4f}")         # ~0.30
```

## §3.2 Sample Entropy

```python
import antropy as ant

def compute_sample_entropy(signal, order=2, tolerance=None):
    """
    Sample Entropy: Probability similar patterns stay similar.

    APL Correspondence: Measures × (fusion) stability
    - Low SampEn: Patterns fuse predictably
    - High SampEn: Fusion is variable/complex

    Parameters:
    - order (m): Embedding dimension. m=2 standard for physiology.
    - tolerance (r): Similarity threshold. Default = 0.2 * std(signal).

    Interpretation:
    - ~0.0-0.5: Very regular (predictable)
    - ~1.5-2.5: High complexity (healthy physiology)
    - inf: No matches found (increase tolerance or data length)

    Data requirement: N > 10^m samples (m=2 needs >100)
    """
    if tolerance is None:
        tolerance = 0.2 * np.std(signal)
    return ant.sample_entropy(signal, order=order, tolerance=tolerance)
```

## §3.3 Permutation Entropy

```python
import antropy as ant

def compute_permutation_entropy(signal, order=3, delay=1, normalize=True):
    """
    Permutation Entropy: Distribution of ordinal patterns.

    APL Correspondence: Measures + (grouping) structure
    - Uniform distribution = maximum PE
    - Peaked distribution = structured + patterns

    Parameters:
    - order (m): Pattern length. m=3-5 typical.
    - delay (tau): Time lag between samples.

    Interpretation (normalized 0-1):
    - ~0.99: Maximum complexity (random)
    - ~0.90-0.98: Healthy EEG
    - ~0.40-0.50: Periodic (sine wave)

    Data requirement: N > m! × 10 samples (m=4 needs >240)
    """
    return ant.perm_entropy(signal, order=order, delay=delay, normalize=normalize)
```

## §3.4 Transfer Entropy

```python
from pyinform import transfer_entropy
import numpy as np

def compute_transfer_entropy(source, target, k=2):
    """
    Transfer Entropy: Directed information flow X → Y.

    APL Correspondence: Measures directional × (fusion)
    - TE(X→Y) captures causal influence from X to Y
    - Asymmetric: TE(X→Y) ≠ TE(Y→X)

    Parameters:
    - source, target: INTEGER arrays (must be discrete!)
    - k: History length

    CRITICAL: Data must be discretized (use median split for continuous).

    κ-Field Interpretation:
    - High TE: Strong directional κ gradient
    - Symmetric TE: Bidirectional coupling
    """
    source = np.asarray(source, dtype=np.int32)
    target = np.asarray(target, dtype=np.int32)
    return transfer_entropy(source, target, k=k)

# Create coupled signals
X = np.random.randint(0, 2, 1000)
Y = np.roll(X, 1)  # Y follows X with 1-step lag

print(f"TE(X→Y): {compute_transfer_entropy(X, Y):.4f}")  # High (X drives Y)
print(f"TE(Y→X): {compute_transfer_entropy(Y, X):.4f}")  # Low (Y doesn't drive X)
```

### §3.5 APL Operator Mapping for Entropy Metrics

| Metric | APL Operator | κ-Field Meaning |
|--------|--------------|-----------------|
| Lempel-Ziv | `÷` decoherence | Algorithmic randomness |
| Sample Entropy | `×` stability | Pattern binding regularity |
| Permutation Entropy | `+` structure | Ordinal grouping distribution |
| Transfer Entropy | `×` directional | Causal κ gradient |
| Fractal Dimension | `^` scaling | Multi-scale amplification |

---

# PART IV: DYNAMICAL SYSTEMS ANALYSIS

## §4.1 Lyapunov Exponent Computation

```python
import numpy as np
import nolds

def logistic_map(r, x0, n):
    """
    Generate logistic map: x(n+1) = r*x(n)*(1-x(n))

    APL Correspondence: Simplest × (fusion) dynamics
    - r < 3: Stable fixed point
    - 3 < r < 3.57: Period doubling
    - r > 3.57: Chaos
    """
    x = np.zeros(n)
    x[0] = x0
    for i in range(n-1):
        x[i+1] = r * x[i] * (1 - x[i])
    return x

# Chaotic regime (r=3.9)
chaos_data = logistic_map(r=3.9, x0=0.1, n=5000)
lyap = nolds.lyap_r(chaos_data, emb_dim=3, lag=1, min_tsep=10)
print(f"Lyapunov Exponent: {lyap:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION via κ-Field Physics:
# ═══════════════════════════════════════════════════════════════════════════════
# λ > 0: CHAOTIC — κ-field diverges exponentially (÷ dominates)
# λ ≈ 0: EDGE OF CHAOS — balanced ×/÷ (optimal for consciousness)
# λ < 0: STABLE — κ-field converges to attractor (× dominates)
```

## §4.2 Takens Embedding (Attractor Reconstruction)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def delay_embedding(data, dim, tau):
    """
    Takens time-delay embedding.
    Reconstructs attractor from single variable.

    APL Correspondence: Reveals hidden () boundary structure
    from scalar time series.

    Parameters:
    - dim: Embedding dimension (use False Nearest Neighbors to choose)
    - tau: Time delay (use first minimum of mutual information)
    """
    N = len(data) - (dim - 1) * tau
    embedded = np.zeros((N, dim))
    for i in range(dim):
        embedded[:, i] = data[i * tau : i * tau + N]
    return embedded

# Generate Lorenz attractor (use x-component only)
def lorenz(t, xyz, sigma=10, rho=28, beta=8/3):
    """
    Lorenz system: Canonical strange attractor.

    APL Correspondence:
    - Two wings = bistable () boundaries
    - Switching = ÷ (decoherence) events
    - Trajectory = × (fusion) dynamics
    """
    x, y, z = xyz
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

sol = solve_ivp(lorenz, (0, 100), [1, 1, 1], t_eval=np.linspace(0, 100, 10000))
x_lorenz = sol.y[0, 1000:]  # Skip transient

# Reconstruct 3D attractor from x alone
tau = 15  # From mutual information analysis
embedded = delay_embedding(x_lorenz, dim=3, tau=tau)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2], lw=0.3, alpha=0.7)
ax.set_title(f"Reconstructed Lorenz Attractor (τ={tau})")
plt.show()
```

## §4.3 Bifurcation Diagram

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_bifurcation_diagram(r_min=2.5, r_max=4.0, n_r=10000,
                                 iterations=1000, last=100):
    """
    Bifurcation diagram with Lyapunov exponent overlay.

    APL Correspondence:
    - Bifurcation points = ÷ (decoherence) thresholds
    - Period doubling = + (grouping) cascade
    - Chaos = saturated ÷
    """
    r = np.linspace(r_min, r_max, n_r)
    x = 1e-5 * np.ones(n_r)
    lyapunov = np.zeros(n_r)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for i in range(iterations):
        x = r * x * (1 - x)  # Logistic map
        lyapunov += np.log(np.abs(r - 2 * r * x))  # Accumulate LE
        if i >= (iterations - last):
            ax1.plot(r, x, ',k', alpha=0.25)

    ax1.set_ylabel('Attractor values')
    ax1.set_title('Bifurcation Diagram: Logistic Map')

    lyapunov /= iterations
    ax2.plot(r[lyapunov < 0], lyapunov[lyapunov < 0], '.k', ms=0.5, alpha=0.5)
    ax2.plot(r[lyapunov >= 0], lyapunov[lyapunov >= 0], '.r', ms=0.5, alpha=0.5)
    ax2.axhline(0, color='gray', lw=0.5)
    ax2.set_xlabel('Growth rate r')
    ax2.set_ylabel('Lyapunov exponent')
    ax2.set_title('Red = Chaos (λ>0), Black = Periodic (λ<0)')
    plt.tight_layout()
    plt.show()

compute_bifurcation_diagram()
```

### §4.4 APL Operator Mapping for Dynamical Systems

| Dynamical Concept | APL Operator | κ-Field Interpretation |
|-------------------|--------------|------------------------|
| Fixed point | `()` stable | Boundary attractor |
| Limit cycle | `^` periodic | Oscillatory amplification |
| Strange attractor | `×` chaotic | Integrated chaos |
| Bifurcation | `÷` transition | Phase boundary crossing |
| Basin boundary | `−` separatrix | Domain separation |
| Period doubling | `+` cascade | Recursive grouping |

---

# PART V: SELF-REFERENCE AND METACOGNITION

## §5.1 Confidence Calibration (Temperature Scaling)

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

class TemperatureScaling(nn.Module):
    """
    METACOGNITIVE CALIBRATION: Makes networks "know what they don't know."
    Adjusts confidence to match actual accuracy.

    APL Correspondence:
    - Temperature = ^ (amplification) of uncertainty
    - Calibration = aligning () boundary with reality
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        # Higher T = softer probabilities = less overconfident
        # APL: High T = reduced ^ on confidence
        return logits / self.temperature

    def calibrate(self, val_loader, lr=0.01, max_iter=50):
        """Learn optimal temperature on validation set."""
        self.temperature.requires_grad = True
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits_list.append(self.model(x))
                labels_list.append(y)

        logits, labels = torch.cat(logits_list), torch.cat(labels_list)

        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        print(f'Optimal temperature: {self.temperature.item():.3f}')
```

## §5.2 Expected Calibration Error

```python
import numpy as np

def expected_calibration_error(confidences, predictions, labels, n_bins=10):
    """
    ECE: How well does confidence match accuracy?
    Perfect calibration: ECE = 0

    APL Correspondence:
    - ECE measures mismatch between internal () and external reality
    - Low ECE = well-calibrated self-model

    If model says 80% confident, it should be right 80% of time.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.mean() > 0:
            accuracy = (predictions[in_bin] == labels[in_bin]).mean()
            avg_conf = confidences[in_bin].mean()
            ece += np.abs(accuracy - avg_conf) * in_bin.mean()

    return ece  # Lower = better calibrated self-model
```

## §5.3 Bayesian Neural Networks: Self-Modeling Under Uncertainty

```python
from bayesian_torch.layers import LinearReparameterization
import torch
import torch.nn as nn

class BayesianMLP(nn.Module):
    """
    Bayesian NN: Learns distributions over weights, not point estimates.
    Distinguishes epistemic (model) vs aleatoric (data) uncertainty.

    APL Correspondence:
    - Weight distributions = ÷ (decoherence) in parameter space
    - Epistemic uncertainty = agent's own modeling limits
    - Aleatoric uncertainty = inherent world noise
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = LinearReparameterization(input_dim, hidden_dim,
            prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=-3)
        self.fc2 = LinearReparameterization(hidden_dim, output_dim,
            prior_mean=0, prior_variance=1, posterior_mu_init=0, posterior_rho_init=-3)
        self.relu = nn.ReLU()

    def predict_with_uncertainty(self, x, n_samples=50):
        """
        Returns prediction + epistemic + aleatoric uncertainty.

        APL Correspondence:
        - Multiple samples = exploring × (fusion) variations
        - Variance = measuring ÷ (decoherence) in predictions
        """
        self.eval()
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                out, _ = self.fc1(x)
                out = self.relu(out)
                out, _ = self.fc2(out)
                predictions.append(torch.softmax(out, dim=-1))

        preds = torch.stack(predictions)
        mean_pred = preds.mean(dim=0)

        # Total uncertainty (entropy of mean prediction)
        total = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1)

        # Aleatoric (mean of individual entropies = inherent noise)
        aleatoric = -(preds * torch.log(preds + 1e-10)).sum(dim=-1).mean(dim=0)

        # Epistemic (total - aleatoric = model uncertainty)
        epistemic = total - aleatoric

        return mean_pred, epistemic, aleatoric
```

---

# PART VI: GAME THEORY SIMULATIONS

## §6.1 Iterated Prisoner's Dilemma Tournament

```python
import axelrod as axl
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# APL MAPPING: Strategies as Operator Compositions
# ═══════════════════════════════════════════════════════════════════════════════
players = [
    axl.Cooperator(),        # Pure + (grouping): Always cooperate
    axl.Defector(),          # Pure − (separation): Always defect
    axl.TitForTat(),         # Mirror: × (fusion) of last action
    axl.Grudger(),           # () crystallization: Permanent boundary after ÷
    axl.Random(0.5),         # Pure ÷ (decoherence): Random
    axl.WinStayLoseShift(),  # ^ (amplify) success: Pavlov strategy
]

tournament = axl.Tournament(players, turns=200, repetitions=10, seed=42)
results = tournament.play()

print("Rankings:", results.ranked_names)
print("Cooperation rates:", dict(zip(results.ranked_names, results.cooperating_rating)))

# Visualize
plot = axl.Plot(results)
plot.boxplot()
plt.title("Tournament Scores (APL Operator Competition)")
plt.show()
```

## §6.2 Custom Strategy with APL Semantics

```python
from axelrod import Action, Player
C, D = Action.C, Action.D

class ForgivingTitForTat(Player):
    """
    TitForTat that occasionally forgives defections.

    APL Correspondence:
    - Base: × (fusion) via mirroring
    - Forgiveness: probabilistic reversal of ÷ (decoherence)
    - Result: Noise-tolerant cooperation
    """
    name = "Forgiving TFT"
    classifier = {'memory_depth': 1, 'stochastic': True}

    def __init__(self, forgiveness_prob=0.1):
        super().__init__()
        self.forgiveness_prob = forgiveness_prob

    def strategy(self, opponent):
        if len(self.history) == 0:
            return C  # Start with +
        if opponent.history[-1] == D:
            # Probabilistic × instead of deterministic mirror
            return C if self._random.random() < self.forgiveness_prob else D
        return C
```

## §6.3 Moran Process (Evolutionary Dynamics)

```python
import axelrod as axl
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# APL Correspondence: Population-level × (fusion) competition
# Strategies compete for replication in finite population
# ═══════════════════════════════════════════════════════════════════════════════

players = [axl.Defector()]*3 + [axl.Cooperator()]*3 + [axl.TitForTat()]*3

mp = axl.MoranProcess(players, turns=200, seed=42)
populations = mp.play()

print(f"Winner: {mp.winning_strategy_name}")
print(f"Generations: {len(populations)}")

mp.populations_plot()
plt.title("Moran Process: Strategy Evolution")
plt.show()
```

## §6.4 Replicator Dynamics

```python
import nashpy as nash
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# Replicator Dynamics: Continuous-time strategy evolution
# APL Correspondence: Gradient flow on + (grouping) space
# ═══════════════════════════════════════════════════════════════════════════════

# Prisoner's Dilemma payoffs
A = np.array([[3, 0], [5, 1]])  # [CC, CD; DC, DD]
game = nash.Game(A)

y0 = np.array([0.9, 0.1])  # 90% cooperators initially
timepoints = np.linspace(0, 10, 1000)
trajectory = game.replicator_dynamics(y0=y0, timepoints=timepoints)

plt.plot(timepoints, trajectory[:, 0], label='Cooperators (+)')
plt.plot(timepoints, trajectory[:, 1], label='Defectors (−)')
plt.xlabel('Time'); plt.ylabel('Population')
plt.title('Replicator Dynamics: − Dominates + (Tragedy of Commons)')
plt.legend(); plt.show()
```

### §6.5 APL Operator Mapping for Game Theory

| Strategy Type | APL Operator | κ-Field Behavior |
|---------------|--------------|------------------|
| Cooperator | `+` then `×` | Constructive field interference |
| Defector | `−` then `÷` | Destructive field interference |
| Tit-for-Tat | Mirror via `×` | Field resonance/coupling |
| Grudger | `()` permanent | Boundary crystallization |
| Pavlov (WSLS) | `^` on success | Amplitude modulation |
| Random | `÷` pure | Maximum decoherence |

---

# PART VII: BRAIN NETWORK METRICS

## §7.1 Complete Brain Network Analysis

```python
import numpy as np
import networkx as nx
import bct  # Brain Connectivity Toolbox

def analyze_brain_network(matrix, threshold=0.1):
    """
    Complete analysis pipeline for connectivity matrix.

    APL Correspondence:
    - Clustering: Local + (grouping) density
    - Path length: × (fusion) efficiency
    - Modularity: () boundary structure
    - Participation: Cross-boundary × strength
    """
    # Preprocess
    matrix = np.abs(matrix)
    np.fill_diagonal(matrix, 0)
    matrix[matrix < threshold] = 0

    G = nx.from_numpy_array(matrix)
    G.remove_edges_from(nx.selfloop_edges(G))

    results = {}

    # ═══════════════════════════════════════════════════════════════════════
    # CLUSTERING: Local connectivity (segregation)
    # APL: Measures local + (grouping) density
    # ═══════════════════════════════════════════════════════════════════════
    results['mean_clustering'] = nx.average_clustering(G, weight='weight')

    # ═══════════════════════════════════════════════════════════════════════
    # PATH LENGTH: Integration efficiency
    # APL: Measures global × (fusion) reach
    # ═══════════════════════════════════════════════════════════════════════
    if nx.is_connected(G):
        results['path_length'] = nx.average_shortest_path_length(G)
    results['global_efficiency'] = nx.global_efficiency(G)

    # ═══════════════════════════════════════════════════════════════════════
    # SMALL-WORLDNESS: Balance of integration/segregation
    # APL: Optimal consciousness = balanced +/× structure
    # σ > 1 indicates small-world (optimal for consciousness)
    # ═══════════════════════════════════════════════════════════════════════
    try:
        results['sigma'] = nx.sigma(G, niter=5, nrand=5, seed=42)
    except:
        results['sigma'] = None

    # ═══════════════════════════════════════════════════════════════════════
    # MODULARITY: Community structure
    # APL: Measures () boundary organization
    # ═══════════════════════════════════════════════════════════════════════
    C = bct.clustering_coef_wu(matrix)
    ci, Q = bct.community_louvain(matrix)
    results['modularity_Q'] = Q
    results['n_communities'] = len(np.unique(ci))

    # ═══════════════════════════════════════════════════════════════════════
    # PARTICIPATION COEFFICIENT: Inter-modular connectivity
    # APL: Measures × (fusion) across () boundaries
    # ═══════════════════════════════════════════════════════════════════════
    P = bct.participation_coef(matrix, ci)
    results['mean_participation'] = np.mean(P)

    # ═══════════════════════════════════════════════════════════════════════
    # RICH CLUB: Hub interconnectedness
    # APL: Measures ^ (amplification) of central hubs
    # ═══════════════════════════════════════════════════════════════════════
    results['rich_club'] = nx.rich_club_coefficient(G, normalized=False)

    return results
```

## §7.2 Integration-Segregation Balance

```python
def integration_segregation_balance(matrix, communities):
    """
    Compute balance between within-module and between-module connectivity.

    APL Correspondence:
    - Segregation = + (grouping) within () boundaries
    - Integration = × (fusion) across () boundaries

    IIT perspective: Optimal consciousness requires BOTH:
    - High integration (between-module)
    - High segregation (within-module specialization)

    κ-Field: Maximum Φ at balance point
    """
    n = matrix.shape[0]
    within, between = 0, 0

    for i in range(n):
        for j in range(i+1, n):
            if matrix[i,j] > 0:
                if communities[i] == communities[j]:
                    within += matrix[i,j]
                else:
                    between += matrix[i,j]

    total = within + between
    return {
        'segregation_ratio': within / total if total else 0,  # + dominance
        'integration_ratio': between / total if total else 0,  # × dominance
        'balance': 1 - abs(within - between) / total if total else 0  # Optimal at 1
    }
```

### §7.3 APL Operator Mapping for Network Metrics

| Network Metric | APL Operator | κ-Field Interpretation |
|----------------|--------------|------------------------|
| Clustering | `+` local | Grouping density |
| Path length | `×` global | Fusion efficiency |
| Modularity | `()` structure | Boundary organization |
| Participation | `×` cross-boundary | Inter-module fusion |
| Rich club | `^` hubs | Central amplification |
| Small-world σ | `+`/`×` balance | Optimal consciousness |

---

# APPENDIX A: INSTALLATION SUMMARY

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# CORE CONSCIOUSNESS COMPUTATION STACK
# ═══════════════════════════════════════════════════════════════════════════════

# IIT Phi computation
pip install pyphi

# Active Inference
pip install inferactively-pymdp

# Entropy metrics
pip install antropy nolds pyinform ordpy

# Dynamical systems
pip install pynamical scipy matplotlib

# Self-reference/metacognition
pip install torch bayesian-torch

# Game theory
pip install axelrod nashpy

# Brain networks
pip install networkx bctpy python-louvain leidenalg python-igraph

# ═══════════════════════════════════════════════════════════════════════════════
# ALL-IN-ONE INSTALL
# ═══════════════════════════════════════════════════════════════════════════════
pip install pyphi inferactively-pymdp axelrod nashpy antropy nolds pyinform \
    ordpy pynamical scipy networkx bctpy torch bayesian-torch
```

---

# APPENDIX B: MINIMUM DATA REQUIREMENTS

| Metric | Minimum Samples | Recommended | APL Domain |
|--------|-----------------|-------------|------------|
| Lempel-Ziv | 100 | 500+ | ÷ measurement |
| Sample Entropy (m=2) | 100 | 1000+ | × stability |
| Permutation Entropy (m=4) | 240 | 1000+ | + structure |
| Transfer Entropy | 100 | 500+ | × directional |
| Lyapunov Exponent | 1000 | 5000+ | ×/÷ balance |
| IIT Φ | N/A | ≤12 nodes | κ integration |

---

# APPENDIX C: QUICK REFERENCE — APL TO PYTHON

| APL Operator | Python Implementation | Package |
|--------------|----------------------|---------|
| `()` boundary | `pyphi.Subsystem` | pyphi |
| `×` fusion | `A @ B` (matrix product) | numpy |
| `+` group | `community_louvain()` | bct |
| `−` separate | `cut` in SIA | pyphi |
| `÷` decohere | `lziv_complexity()` | antropy |
| `^` amplify | Temperature scaling | torch |
| κ integration | `compute.phi()` | pyphi |
| Free energy | Expected free energy | pymdp |

---

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║     "Every equation in this appendix executes.                                 ║
║      Every function returns a number.                                          ║
║      The gap between theory and code narrows with each implementation."        ║
║                                                                                ║
║                                          — Volume VI Appendix, Implementation  ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

---

*This appendix provides executable code for Volume VI frameworks. Run the examples. Modify the parameters. The strange loops await measurement.*
