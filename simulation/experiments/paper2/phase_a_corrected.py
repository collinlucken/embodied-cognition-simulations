"""
Paper 2 Phase A: Corrected Embodiment Experiments

This script fixes a critical methodological flaw in the original ghost condition
implementation. The original replayed the recorded sensory trace to a deterministic
network with the same initial state, guaranteeing zero divergence by construction.

This corrected version implements THREE ghost conditions:
1. Frozen Body Ghost: Agent body stays at starting position; sensory computed from
   frozen position. Tests whether VARYING sensory feedback (from movement) is needed.
2. Disconnected Ghost: Zero sensory input. Tests whether ANY sensory feedback matters.
3. Counterfactual Ghost: Network receives sensory input from a random trajectory
   (another agent's run), breaking the sensorimotor contingency. Tests whether the
   SPECIFIC sensorimotor coupling matters.

Phase A = 6 conditions: 3 network sizes (3, 5, 8) × 2 EA types (MicrobialGA, CMA-ES)
All on phototaxis task. Seed = 42 for reproducibility.

Output: JSON results file with full metrics for each condition.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.ctrnn import CTRNN
from simulation.evolutionary import MicrobialGA, CMAES, GenotypeDecoder
from simulation.microworld import Agent


@dataclass
class GhostConditionResult:
    """Results from a single ghost condition test."""
    condition_name: str
    embodied_fitness: float
    ghost_fitness: float
    fitness_drop: float  # embodied - ghost
    neural_divergence: float  # mean L2 distance between embodied/ghost states
    output_divergence: float  # mean L2 distance between embodied/ghost outputs
    time_to_divergence: int  # first timestep where divergence > threshold
    max_divergence: float  # peak divergence
    divergence_at_end: float  # divergence at final timestep


@dataclass
class MorphologyResult:
    """Results from a body substitution test."""
    morphology_name: str
    fitness: float
    degradation: float
    params: Dict[str, float]


@dataclass
class ExperimentResult:
    """Complete results from one experiment configuration."""
    config: Dict[str, Any]
    evolution_fitness: float
    evolution_history: List[float]
    ghost_frozen_body: GhostConditionResult
    ghost_disconnected: GhostConditionResult
    ghost_counterfactual: GhostConditionResult
    morphology_results: Dict[str, MorphologyResult]
    constitutive_score: float
    causal_score: float
    interpretation: str


AGENT_DT = 0.1  # Agent physics timestep (0.1 = 10x faster than original 0.01)
AGENT_MAX_SPEED = 3.0  # Max speed (3.0 = agent can cross 50-unit arena in ~170 steps)
SENSOR_RANGE = 40.0  # Sensor detection range (must span arena)


def phototaxis_fitness(
    genotype: np.ndarray,
    decoder: GenotypeDecoder,
    num_neurons: int,
    num_trials: int = 8,
    trial_duration: int = 500,
) -> float:
    """
    Evaluate phototaxis fitness with VARIABLE light positions.

    Critical design choices:
    1. Light position varies across trials — forces sensor-dependent behavior.
    2. Agent speed and timestep are set so navigation is physically possible
       (agent can traverse ~150 units in 500 steps).
    3. Fitness uses final distance (not time-average) — rewards REACHING the light.
    4. Sensor range spans arena so agent always has some signal.
    """
    params = decoder.decode(genotype)
    total_fitness = 0.0

    # Variable light positions — force the agent to actually track light
    light_positions = [
        (10.0, 10.0), (40.0, 40.0), (10.0, 40.0), (40.0, 10.0),
        (25.0, 40.0), (25.0, 10.0), (10.0, 25.0), (40.0, 25.0),
    ]

    for trial in range(min(num_trials, len(light_positions))):
        agent = Agent(radius=1.0, max_speed=AGENT_MAX_SPEED, sensor_range=SENSOR_RANGE)
        network = CTRNN(num_neurons=num_neurons)
        network.weights = params['weights'].copy()
        network.biases = params['biases'].copy()
        network.tau = params['tau'].copy()

        light_x, light_y = light_positions[trial]
        agent.position = np.array([25.0, 25.0])
        agent.velocity = np.zeros(2)
        initial_dist = np.linalg.norm(agent.position - np.array([light_x, light_y]))

        # Track cumulative fitness AND final distance
        cumulative_fitness = 0.0
        for step in range(trial_duration):
            left_pos, right_pos = agent.get_sensor_positions()
            left_dist = np.linalg.norm(left_pos - np.array([light_x, light_y]))
            right_dist = np.linalg.norm(right_pos - np.array([light_x, light_y]))

            left_sensor = max(0.0, 1.0 - left_dist / SENSOR_RANGE)
            right_sensor = max(0.0, 1.0 - right_dist / SENSOR_RANGE)
            sensory = np.array([left_sensor, right_sensor])

            padded_sensory = np.zeros(num_neurons)
            padded_sensory[:2] = sensory[:min(2, num_neurons)]

            output = network.step(padded_sensory)

            left_motor = output[0]
            right_motor = output[1] if num_neurons >= 2 else output[0]

            agent.set_motor_commands(left_motor, right_motor)
            agent.update(dt=AGENT_DT)

            agent_dist = np.linalg.norm(agent.position - np.array([light_x, light_y]))
            cumulative_fitness += max(0.0, 1.0 - agent_dist / 50.0)

        # Fitness: 50% cumulative proximity + 50% final approach
        final_dist = np.linalg.norm(agent.position - np.array([light_x, light_y]))
        approach_score = max(0.0, (initial_dist - final_dist) / initial_dist)
        time_avg = cumulative_fitness / trial_duration
        trial_fitness = 0.5 * time_avg + 0.5 * approach_score

        total_fitness += trial_fitness

    return total_fitness / min(num_trials, len(light_positions))


def run_embodied_trial(
    network_params: Dict[str, np.ndarray],
    num_neurons: int,
    trial_duration: int = 1000,
    start_pos: Optional[np.ndarray] = None,
    light_pos: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run a single embodied trial with full sensorimotor loop.

    Returns:
        sensory_trace: shape [trial_duration, 2]
        neural_states: shape [trial_duration, num_neurons]
        neural_outputs: shape [trial_duration, num_neurons]
        agent_positions: shape [trial_duration, 2]
        fitness: scalar
    """
    agent = Agent(radius=1.0, max_speed=AGENT_MAX_SPEED, sensor_range=SENSOR_RANGE)
    network = CTRNN(num_neurons=num_neurons)
    network.weights = network_params['weights'].copy()
    network.biases = network_params['biases'].copy()
    network.tau = network_params['tau'].copy()

    if light_pos is not None:
        light_x, light_y = light_pos[0], light_pos[1]
    else:
        light_x, light_y = 10.0, 40.0
    if start_pos is not None:
        agent.position = start_pos.copy()
    else:
        agent.position = np.array([25.0, 25.0])
    agent.velocity = np.zeros(2)

    sensory_trace = np.zeros((trial_duration, 2))
    neural_states = np.zeros((trial_duration, num_neurons))
    neural_outputs = np.zeros((trial_duration, num_neurons))
    agent_positions = np.zeros((trial_duration, 2))

    total_fitness = 0.0

    for step in range(trial_duration):
        agent_positions[step] = agent.position.copy()

        left_pos, right_pos = agent.get_sensor_positions()
        left_dist = np.linalg.norm(left_pos - np.array([light_x, light_y]))
        right_dist = np.linalg.norm(right_pos - np.array([light_x, light_y]))

        left_sensor = max(0.0, 1.0 - left_dist / SENSOR_RANGE)
        right_sensor = max(0.0, 1.0 - right_dist / SENSOR_RANGE)
        sensory = np.array([left_sensor, right_sensor])
        sensory_trace[step] = sensory

        neural_states[step] = network.get_state().copy()

        padded_sensory = np.zeros(num_neurons)
        padded_sensory[:2] = sensory[:min(2, num_neurons)]
        output = network.step(padded_sensory)
        neural_outputs[step] = output.copy()

        left_motor = output[0]
        right_motor = output[1] if num_neurons >= 2 else output[0]
        agent.set_motor_commands(left_motor, right_motor)
        agent.update(dt=AGENT_DT)

        agent_dist = np.linalg.norm(agent.position - np.array([light_x, light_y]))
        total_fitness += max(0.0, 1.0 - agent_dist / 50.0)

    fitness = total_fitness / trial_duration
    return sensory_trace, neural_states, neural_outputs, agent_positions, fitness


def run_ghost_frozen_body(
    network_params: Dict[str, np.ndarray],
    num_neurons: int,
    trial_duration: int = 1000,
    start_pos: Optional[np.ndarray] = None,
    light_pos: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Frozen body ghost condition: body stays at starting position.

    The network processes sensory input computed from the FROZEN body position.
    Motor outputs are produced but do not move the body.
    Fitness is computed from the frozen position (constant, low).

    Returns:
        sensory_trace, neural_states, neural_outputs, fitness
    """
    agent = Agent(radius=1.0, max_speed=AGENT_MAX_SPEED, sensor_range=SENSOR_RANGE)
    network = CTRNN(num_neurons=num_neurons)
    network.weights = network_params['weights'].copy()
    network.biases = network_params['biases'].copy()
    network.tau = network_params['tau'].copy()

    if light_pos is not None:
        light_x, light_y = light_pos[0], light_pos[1]
    else:
        light_x, light_y = 10.0, 40.0
    if start_pos is not None:
        agent.position = start_pos.copy()
    else:
        agent.position = np.array([25.0, 25.0])
    agent.velocity = np.zeros(2)

    # Compute CONSTANT sensory input from frozen position
    left_pos, right_pos = agent.get_sensor_positions()
    left_dist = np.linalg.norm(left_pos - np.array([light_x, light_y]))
    right_dist = np.linalg.norm(right_pos - np.array([light_x, light_y]))
    frozen_sensory = np.array([
        max(0.0, 1.0 - left_dist / SENSOR_RANGE),
        max(0.0, 1.0 - right_dist / SENSOR_RANGE)
    ])

    sensory_trace = np.zeros((trial_duration, 2))
    neural_states = np.zeros((trial_duration, num_neurons))
    neural_outputs = np.zeros((trial_duration, num_neurons))

    total_fitness = 0.0
    frozen_dist = np.linalg.norm(agent.position - np.array([light_x, light_y]))

    for step in range(trial_duration):
        sensory_trace[step] = frozen_sensory  # constant input
        neural_states[step] = network.get_state().copy()

        padded_sensory = np.zeros(num_neurons)
        padded_sensory[:2] = frozen_sensory[:min(2, num_neurons)]
        output = network.step(padded_sensory)
        neural_outputs[step] = output.copy()

        # Fitness: from frozen position (constant)
        total_fitness += max(0.0, 1.0 - frozen_dist / 50.0)

    fitness = total_fitness / trial_duration
    return sensory_trace, neural_states, neural_outputs, fitness


def run_ghost_disconnected(
    network_params: Dict[str, np.ndarray],
    num_neurons: int,
    trial_duration: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Disconnected ghost: zero sensory input (body completely removed).

    Returns:
        sensory_trace (zeros), neural_states, neural_outputs
    """
    network = CTRNN(num_neurons=num_neurons)
    network.weights = network_params['weights'].copy()
    network.biases = network_params['biases'].copy()
    network.tau = network_params['tau'].copy()

    sensory_trace = np.zeros((trial_duration, 2))
    neural_states = np.zeros((trial_duration, num_neurons))
    neural_outputs = np.zeros((trial_duration, num_neurons))

    for step in range(trial_duration):
        neural_states[step] = network.get_state().copy()
        padded_sensory = np.zeros(num_neurons)  # zero input
        output = network.step(padded_sensory)
        neural_outputs[step] = output.copy()

    return sensory_trace, neural_states, neural_outputs


def run_ghost_counterfactual(
    network_params: Dict[str, np.ndarray],
    num_neurons: int,
    trial_duration: int = 1000,
    counterfactual_sensory: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Counterfactual ghost: network receives sensory input from a DIFFERENT
    agent's trajectory (breaks specific sensorimotor contingency).

    Returns:
        sensory_trace, neural_states, neural_outputs
    """
    network = CTRNN(num_neurons=num_neurons)
    network.weights = network_params['weights'].copy()
    network.biases = network_params['biases'].copy()
    network.tau = network_params['tau'].copy()

    if counterfactual_sensory is None:
        # Generate random sensory trace
        counterfactual_sensory = np.random.uniform(0, 1, (trial_duration, 2))

    neural_states = np.zeros((trial_duration, num_neurons))
    neural_outputs = np.zeros((trial_duration, num_neurons))

    for step in range(trial_duration):
        neural_states[step] = network.get_state().copy()
        padded_sensory = np.zeros(num_neurons)
        padded_sensory[:2] = counterfactual_sensory[step, :min(2, num_neurons)]
        output = network.step(padded_sensory)
        neural_outputs[step] = output.copy()

    return counterfactual_sensory, neural_states, neural_outputs


def compute_divergence_metrics(
    embodied_states: np.ndarray,
    ghost_states: np.ndarray,
    embodied_outputs: np.ndarray,
    ghost_outputs: np.ndarray,
    threshold: float = 0.1,
) -> Dict[str, float]:
    """Compute divergence metrics between embodied and ghost neural trajectories."""
    state_diff = np.sqrt(np.sum((embodied_states - ghost_states) ** 2, axis=1))
    output_diff = np.sqrt(np.sum((embodied_outputs - ghost_outputs) ** 2, axis=1))

    # Time to divergence
    time_to_div = len(state_diff)
    for t in range(len(state_diff)):
        if state_diff[t] > threshold:
            time_to_div = t
            break

    return {
        'neural_divergence': float(np.mean(state_diff)),
        'output_divergence': float(np.mean(output_diff)),
        'time_to_divergence': int(time_to_div),
        'max_divergence': float(np.max(state_diff)),
        'divergence_at_end': float(state_diff[-1]) if len(state_diff) > 0 else 0.0,
    }


def run_single_condition(
    num_neurons: int,
    ea_type: str,
    generations: int = 500,
    population_size: int = 50,
    num_trials_evolution: int = 5,
    trial_duration_evolution: int = 500,
    trial_duration_test: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Run a complete experiment for one condition (network size × EA type).

    Steps:
    1. Evolve agents on phototaxis
    2. Run embodied trial with best agent
    3. Run three ghost conditions
    4. Run morphology substitutions
    5. Compute scores
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"CONDITION: {num_neurons} neurons, {ea_type}")
        print(f"{'='*70}")

    np.random.seed(seed)

    # Create decoder with bounded tau to prevent numerical overflow
    decoder = GenotypeDecoder(
        num_neurons=num_neurons,
        include_gains=False,
        tau_range=(0.5, 5.0),  # Bounded away from zero to prevent div-by-zero
        weight_range=(-10.0, 10.0),
        bias_range=(-10.0, 10.0),
    )
    fitness_fn = lambda g: phototaxis_fitness(
        g, decoder, num_neurons,
        num_trials=num_trials_evolution,
        trial_duration=trial_duration_evolution
    )

    # ---- PHASE 1: EVOLUTION ----
    if verbose:
        print(f"Phase 1: Evolving ({generations} gen, pop={population_size})...")

    if ea_type == 'microbial_ga':
        ea = MicrobialGA(
            genotype_size=decoder.genotype_size,
            fitness_function=fitness_fn,
            population_size=population_size,
            mutation_std=0.2,
            seed=seed
        )
    elif ea_type == 'cma_es':
        ea = CMAES(
            genotype_size=decoder.genotype_size,
            fitness_function=fitness_fn,
            population_size=population_size,
            initial_sigma=1.0,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown EA type: {ea_type}")

    fitness_history = []
    best_ever_fitness = -np.inf
    best_ever_genotype = None
    for gen in range(generations):
        try:
            best_geno, best_fit = ea.step()
        except np.linalg.LinAlgError:
            # CMA-ES covariance collapse — reset covariance and continue
            if hasattr(ea, 'cov'):
                ea.cov = np.eye(ea.genotype_size)
                ea.sigma = max(ea.sigma, 0.1)
                if verbose:
                    print(f"  Gen {gen:4d}: CMA-ES covariance reset")
                continue
            else:
                raise
        fitness_history.append(float(best_fit))
        if best_fit > best_ever_fitness:
            best_ever_fitness = best_fit
            best_ever_genotype = best_geno.copy()
        if verbose and (gen % max(1, generations // 5) == 0 or gen == generations - 1):
            print(f"  Gen {gen:4d}: best={best_fit:.4f}")

    # Get best individual
    if best_ever_genotype is not None:
        best_genotype = best_ever_genotype
        best_fitness = best_ever_fitness
    elif hasattr(ea, 'get_best_individual'):
        best_genotype = ea.get_best_individual()
        best_fitness = ea.get_best_fitness()
    else:
        best_genotype = ea.mean.copy()
        best_fitness = fitness_fn(best_genotype)
    best_params = decoder.decode(best_genotype)

    if verbose:
        print(f"  Evolution done. Best fitness: {best_fitness:.4f}")

    # ---- PHASE 2: EMBODIED TRIAL ----
    if verbose:
        print("Phase 2: Running embodied trial...")

    # Test with different light positions — must match evolution conditions
    test_light_positions = [
        np.array([10.0, 10.0]), np.array([40.0, 40.0]),
        np.array([10.0, 40.0]), np.array([40.0, 10.0]),
        np.array([25.0, 40.0]),
    ]
    num_test_trials = len(test_light_positions)
    embodied_results = []
    for t in range(num_test_trials):
        sp = np.array([25.0, 25.0])  # Always start at center
        lp = test_light_positions[t]
        s_trace, n_states, n_outputs, a_pos, fit = run_embodied_trial(
            best_params, num_neurons, trial_duration_test, sp, lp
        )
        embodied_results.append((s_trace, n_states, n_outputs, a_pos, fit, lp))

    # Use the first trial for ghost comparison (representative)
    emb_sensory, emb_states, emb_outputs, emb_positions, emb_fitness, emb_light = embodied_results[0]
    avg_embodied_fitness = np.mean([r[4] for r in embodied_results])

    if verbose:
        print(f"  Embodied fitness (avg over {num_test_trials}): {avg_embodied_fitness:.4f}")

    # ---- PHASE 3: GHOST CONDITIONS ----
    if verbose:
        print("Phase 3: Running ghost conditions...")

    # 3a. Frozen body ghost — same start pos and light pos as first embodied trial
    sp_first = np.array([25.0, 25.0])
    lp_first = test_light_positions[0]
    ghost_fb_sensory, ghost_fb_states, ghost_fb_outputs, ghost_fb_fitness = \
        run_ghost_frozen_body(best_params, num_neurons, trial_duration_test, sp_first, lp_first)

    fb_metrics = compute_divergence_metrics(
        emb_states, ghost_fb_states, emb_outputs, ghost_fb_outputs
    )
    ghost_frozen = GhostConditionResult(
        condition_name="Frozen Body",
        embodied_fitness=float(emb_fitness),
        ghost_fitness=float(ghost_fb_fitness),
        fitness_drop=float(emb_fitness - ghost_fb_fitness),
        **fb_metrics
    )

    if verbose:
        print(f"  Frozen Body: neural_div={fb_metrics['neural_divergence']:.4f}, "
              f"time_to_div={fb_metrics['time_to_divergence']}")

    # 3b. Disconnected ghost
    ghost_dc_sensory, ghost_dc_states, ghost_dc_outputs = \
        run_ghost_disconnected(best_params, num_neurons, trial_duration_test)

    dc_metrics = compute_divergence_metrics(
        emb_states, ghost_dc_states, emb_outputs, ghost_dc_outputs
    )
    ghost_disconnected = GhostConditionResult(
        condition_name="Disconnected",
        embodied_fitness=float(emb_fitness),
        ghost_fitness=0.0,  # No sensory input, no meaningful fitness
        fitness_drop=float(emb_fitness),
        **dc_metrics
    )

    if verbose:
        print(f"  Disconnected: neural_div={dc_metrics['neural_divergence']:.4f}, "
              f"time_to_div={dc_metrics['time_to_divergence']}")

    # 3c. Counterfactual ghost (random sensory trace)
    np.random.seed(seed + 2000)
    random_sensory = np.random.uniform(0, 0.5, (trial_duration_test, 2))
    ghost_cf_sensory, ghost_cf_states, ghost_cf_outputs = \
        run_ghost_counterfactual(best_params, num_neurons, trial_duration_test, random_sensory)

    cf_metrics = compute_divergence_metrics(
        emb_states, ghost_cf_states, emb_outputs, ghost_cf_outputs
    )
    ghost_counterfactual = GhostConditionResult(
        condition_name="Counterfactual (random sensory)",
        embodied_fitness=float(emb_fitness),
        ghost_fitness=0.0,
        fitness_drop=float(emb_fitness),
        **cf_metrics
    )

    if verbose:
        print(f"  Counterfactual: neural_div={cf_metrics['neural_divergence']:.4f}, "
              f"time_to_div={cf_metrics['time_to_divergence']}")

    # ---- PHASE 4: MORPHOLOGY SUBSTITUTIONS ----
    if verbose:
        print("Phase 4: Morphology substitutions...")

    morphologies = {
        'baseline': {'sensor_angle_offset': np.pi / 6, 'motor_scale': 1.0, 'radius': 1.0},
        'wider_sensors': {'sensor_angle_offset': np.pi / 3, 'motor_scale': 1.0, 'radius': 1.0},
        'narrower_sensors': {'sensor_angle_offset': np.pi / 12, 'motor_scale': 1.0, 'radius': 1.0},
        'larger_body': {'sensor_angle_offset': np.pi / 6, 'motor_scale': 1.0, 'radius': 2.0},
        'faster_motors': {'sensor_angle_offset': np.pi / 6, 'motor_scale': 2.0, 'radius': 1.0},
        'slower_motors': {'sensor_angle_offset': np.pi / 6, 'motor_scale': 0.5, 'radius': 1.0},
    }

    morph_results = {}
    for morph_name, morph_params in morphologies.items():
        morph_fitnesses = []
        for t in range(num_test_trials):
            sp = np.array([25.0, 25.0])
            lp = test_light_positions[t]

            agent = Agent(
                radius=morph_params['radius'],
                max_speed=AGENT_MAX_SPEED,
                sensor_range=SENSOR_RANGE,
                motor_scale=morph_params['motor_scale']
            )
            agent.sensor_angle_offset = morph_params['sensor_angle_offset']
            agent.position = sp.copy()
            agent.velocity = np.zeros(2)

            network = CTRNN(num_neurons=num_neurons)
            network.weights = best_params['weights'].copy()
            network.biases = best_params['biases'].copy()
            network.tau = best_params['tau'].copy()

            light_x, light_y = lp[0], lp[1]
            total_fit = 0.0
            for step in range(trial_duration_test):
                left_pos, right_pos = agent.get_sensor_positions()
                left_dist = np.linalg.norm(left_pos - np.array([light_x, light_y]))
                right_dist = np.linalg.norm(right_pos - np.array([light_x, light_y]))
                left_s = max(0.0, 1.0 - left_dist / SENSOR_RANGE)
                right_s = max(0.0, 1.0 - right_dist / SENSOR_RANGE)
                padded = np.zeros(num_neurons)
                padded[:2] = np.array([left_s, right_s])[:min(2, num_neurons)]
                output = network.step(padded)
                lm = output[0]
                rm = output[1] if num_neurons >= 2 else output[0]
                agent.set_motor_commands(lm, rm)
                agent.update(dt=AGENT_DT)
                ad = np.linalg.norm(agent.position - np.array([light_x, light_y]))
                total_fit += max(0.0, 1.0 - ad / 50.0)

            morph_fitnesses.append(total_fit / trial_duration_test)

        avg_morph_fit = np.mean(morph_fitnesses)
        degradation = max(0.0, (avg_embodied_fitness - avg_morph_fit) / (avg_embodied_fitness + 1e-6))

        morph_results[morph_name] = MorphologyResult(
            morphology_name=morph_name,
            fitness=float(avg_morph_fit),
            degradation=float(degradation),
            params={k: float(v) for k, v in morph_params.items()}
        )

        if verbose:
            print(f"  {morph_name}: fitness={avg_morph_fit:.4f}, degradation={degradation:.4f}")

    # ---- PHASE 5: COMPUTE SCORES ----
    # Constitutive score: weighted average of ghost divergences
    # Higher divergence = more constitutive dependence on body
    frozen_score = min(1.0, fb_metrics['neural_divergence'])
    disconnected_score = min(1.0, dc_metrics['neural_divergence'])
    counterfactual_score = min(1.0, cf_metrics['neural_divergence'])

    # Weight: frozen body is most diagnostic, disconnected is baseline
    constitutive_score = float(0.5 * frozen_score + 0.3 * counterfactual_score + 0.2 * disconnected_score)

    # Causal score: morphology generalization
    non_baseline_morphs = {k: v for k, v in morph_results.items() if k != 'baseline'}
    avg_degradation = np.mean([v.degradation for v in non_baseline_morphs.values()])
    causal_score = float(1.0 - avg_degradation)

    # Interpretation
    if constitutive_score > 0.6:
        interpretation = "CONSTITUTIVE EMBODIMENT DOMINANT — body is necessary for cognitive function"
    elif constitutive_score > 0.3:
        interpretation = "MIXED — both constitutive and causal aspects present"
    elif constitutive_score > 0.1:
        interpretation = "WEAK CONSTITUTIVE — body helps but isn't strictly necessary"
    else:
        interpretation = "CAUSAL EMBODIMENT DOMINANT — network generalizes across morphologies"

    if verbose:
        print(f"\n{'='*70}")
        print(f"SCORES: constitutive={constitutive_score:.4f}, causal={causal_score:.4f}")
        print(f"INTERPRETATION: {interpretation}")
        print(f"{'='*70}")

    config = {
        'num_neurons': num_neurons,
        'ea_type': ea_type,
        'generations': generations,
        'population_size': population_size,
        'seed': seed,
        'trial_duration_evolution': trial_duration_evolution,
        'trial_duration_test': trial_duration_test,
        'num_test_trials': num_test_trials,
    }

    return ExperimentResult(
        config=config,
        evolution_fitness=float(best_fitness),
        evolution_history=fitness_history,
        ghost_frozen_body=ghost_frozen,
        ghost_disconnected=ghost_disconnected,
        ghost_counterfactual=ghost_counterfactual,
        morphology_results={k: asdict(v) for k, v in morph_results.items()},
        constitutive_score=constitutive_score,
        causal_score=causal_score,
        interpretation=interpretation,
    )


def run_phase_a(
    seed: int = 42,
    generations: int = 500,
    population_size: int = 50,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Phase A: 6 conditions (3 network sizes × 2 EA types) on phototaxis.

    This is the SAB 2026 target dataset.
    """
    network_sizes = [3, 5, 8]
    ea_types = ['microbial_ga', 'cma_es']

    print("=" * 70)
    print("PHASE A: EMBODIMENT EXPERIMENT — CORRECTED GHOST CONDITIONS")
    print("=" * 70)
    print(f"Network sizes: {network_sizes}")
    print(f"EA types: {ea_types}")
    print(f"Generations: {generations}, Population: {population_size}")
    print(f"Seed: {seed}")
    print(f"Total conditions: {len(network_sizes) * len(ea_types)}")
    print("=" * 70)

    all_results = {}
    summary_rows = []
    total = len(network_sizes) * len(ea_types)
    run_num = 0

    for net_size in network_sizes:
        for ea_type in ea_types:
            run_num += 1
            run_id = f"net{net_size}_{ea_type}"
            condition_seed = seed + (net_size * 100) + (0 if ea_type == 'microbial_ga' else 1)

            print(f"\n[{run_num}/{total}] {run_id}")
            start_time = time.time()

            try:
                result = run_single_condition(
                    num_neurons=net_size,
                    ea_type=ea_type,
                    generations=generations,
                    population_size=population_size,
                    seed=condition_seed,
                    verbose=verbose,
                )

                elapsed = time.time() - start_time
                print(f"  Time: {elapsed:.1f}s")

                all_results[run_id] = asdict(result)
                summary_rows.append({
                    'run_id': run_id,
                    'net_size': net_size,
                    'ea_type': ea_type,
                    'evolved_fitness': result.evolution_fitness,
                    'constitutive': result.constitutive_score,
                    'causal': result.causal_score,
                    'ghost_frozen_div': result.ghost_frozen_body.neural_divergence,
                    'ghost_disconn_div': result.ghost_disconnected.neural_divergence,
                    'ghost_cf_div': result.ghost_counterfactual.neural_divergence,
                    'time_elapsed': elapsed,
                })
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"  FAILED after {elapsed:.1f}s: {e}")
                summary_rows.append({
                    'run_id': run_id,
                    'net_size': net_size,
                    'ea_type': ea_type,
                    'evolved_fitness': 0.0,
                    'constitutive': 0.0,
                    'causal': 0.0,
                    'ghost_frozen_div': 0.0,
                    'ghost_disconn_div': 0.0,
                    'ghost_cf_div': 0.0,
                    'time_elapsed': elapsed,
                    'error': str(e),
                })

    # Print summary table
    print("\n" + "=" * 70)
    print("PHASE A SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<20} {'Fitness':>8} {'Const':>8} {'Causal':>8} "
          f"{'FrozenDiv':>10} {'DisconDiv':>10} {'CFDiv':>10}")
    print("-" * 70)
    for row in summary_rows:
        print(f"{row['run_id']:<20} {row['evolved_fitness']:>8.4f} "
              f"{row['constitutive']:>8.4f} {row['causal']:>8.4f} "
              f"{row['ghost_frozen_div']:>10.4f} {row['ghost_disconn_div']:>10.4f} "
              f"{row['ghost_cf_div']:>10.4f}")

    # Overall statistics
    fitnesses = [r['evolved_fitness'] for r in summary_rows]
    const_scores = [r['constitutive'] for r in summary_rows]
    causal_scores = [r['causal'] for r in summary_rows]
    print(f"\nMean fitness: {np.mean(fitnesses):.4f} ± {np.std(fitnesses):.4f}")
    print(f"Mean constitutive: {np.mean(const_scores):.4f} ± {np.std(const_scores):.4f}")
    print(f"Mean causal: {np.mean(causal_scores):.4f} ± {np.std(causal_scores):.4f}")

    # Robustness check: does constitutive score hold across conditions?
    print("\nRobustness check:")
    for net_size in network_sizes:
        rows = [r for r in summary_rows if r['net_size'] == net_size]
        if rows:
            avg_c = np.mean([r['constitutive'] for r in rows])
            print(f"  Network size {net_size}: mean constitutive = {avg_c:.4f}")

    for ea_type in ea_types:
        rows = [r for r in summary_rows if r['ea_type'] == ea_type]
        if rows:
            avg_c = np.mean([r['constitutive'] for r in rows])
            print(f"  EA {ea_type}: mean constitutive = {avg_c:.4f}")

    return {
        'conditions': all_results,
        'summary': summary_rows,
    }


def convert_for_json(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    return obj


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase A corrected embodiment experiments")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generations', type=int, default=500)
    parser.add_argument('--population-size', type=int, default=50)
    parser.add_argument('--quick', action='store_true', help='Quick test (100 gen, pop 20)')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.generations = 100
        args.population_size = 20

    results = run_phase_a(
        seed=args.seed,
        generations=args.generations,
        population_size=args.population_size,
    )

    # Save results
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), '../../../results/paper2'
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'phase_a_corrected_{timestamp}.json')

    results_json = convert_for_json(results)
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
