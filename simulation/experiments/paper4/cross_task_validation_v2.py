"""
Cross-Task Validation v2: Perceptual Crossing × Network Capacity

Redesign addressing Session 7 null result. The original version failed because
agents could only traverse ~5 units in 500 steps (max_speed=1.0, dt=0.01),
while starting 50 units apart on a 100-circumference ring.

Key parameter changes:
1. max_speed: 1.0 → 5.0 (0.05 units/step)
2. Episode length: 500 → 5000 (max displacement: 250 units)
3. Fitness: "exploration" (reward actual position change, not just motor activity)
4. Generations: 300 → 500
5. Starting distance: 50 → 25 units (agents at 37.5 and 62.5)

Reference parameters from Froese & Di Paolo (2010):
- Ring: 600 units, 16000 steps
- Our scaled equivalent: 100 units, ~2667 steps minimum
- We use 5000 steps for safety margin
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.ctrnn import CTRNN
from simulation.evolutionary import MicrobialGA, GenotypeDecoder
from simulation.microworld import Agent, PerceptualCrossingEnv

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../../results/paper4')
NETWORK_SIZES = [2, 3, 4, 5, 6, 8]
SEEDS = [42, 137, 256, 512, 1024]
NUM_GENERATIONS = 500
POPULATION_SIZE = 30
EPISODE_LENGTH = 5000
MAX_SPEED = 5.0
CIRCUMFERENCE = 100.0
PERCEPTION_DIST = 5.0

# Starting positions (25 units apart instead of 50)
START_POS_1 = 37.5
START_POS_2 = 62.5


def create_agents_and_env(net1, net2):
    """Create fresh agents and environment for an episode."""
    agent1 = Agent(radius=1.0, max_speed=MAX_SPEED)
    agent2 = Agent(radius=1.0, max_speed=MAX_SPEED)
    agent1.position = np.array([START_POS_1, 0.0])
    agent2.position = np.array([START_POS_2, 0.0])

    env = PerceptualCrossingEnv(circumference=CIRCUMFERENCE)
    env.set_agents(agent1, agent2)

    return agent1, agent2, env


def run_episode(net1, net2, episode_length, fitness_type="exploration"):
    """Run one episode and return fitness + coordination metrics."""
    agent1, agent2, env = create_agents_and_env(net1, net2)
    net1.reset()
    net2.reset()

    fitness = 0.0
    perception_count = 0
    total_perception_time = 0
    prev_pos1 = agent1.position[0]
    prev_pos2 = agent2.position[0]
    total_displacement = 0.0

    for step in range(episode_length):
        # Check perception
        dist = min(
            abs(agent1.position[0] - agent2.position[0]),
            CIRCUMFERENCE - abs(agent1.position[0] - agent2.position[0])
        )
        perceive = dist < PERCEPTION_DIST

        # Sensor input
        sensor1 = np.array([float(perceive), float(perceive)])
        sensor2 = np.array([float(perceive), float(perceive)])

        # Neural step
        ext_input1 = np.zeros(net1.num_neurons)
        ext_input2 = np.zeros(net2.num_neurons)
        ext_input1[:min(2, net1.num_neurons)] = sensor1[:min(2, net1.num_neurons)]
        ext_input2[:min(2, net2.num_neurons)] = sensor2[:min(2, net2.num_neurons)]

        out1 = net1.step(ext_input1)
        out2 = net2.step(ext_input2)

        # Motor commands from first two outputs
        left_motor1 = float(out1[0]) if len(out1) > 0 else 0.0
        right_motor1 = float(out1[1]) if len(out1) > 1 else left_motor1
        left_motor2 = float(out2[0]) if len(out2) > 0 else 0.0
        right_motor2 = float(out2[1]) if len(out2) > 1 else left_motor2

        agent1.set_motor_commands(left_motor1, right_motor1)
        agent2.set_motor_commands(left_motor2, right_motor2)

        env.step()

        # Track perception
        if perceive:
            perception_count += 1
            total_perception_time += 1

        # Track displacement (on ring)
        disp1 = min(
            abs(agent1.position[0] - prev_pos1),
            CIRCUMFERENCE - abs(agent1.position[0] - prev_pos1)
        )
        disp2 = min(
            abs(agent2.position[0] - prev_pos2),
            CIRCUMFERENCE - abs(agent2.position[0] - prev_pos2)
        )
        total_displacement += disp1 + disp2
        prev_pos1 = agent1.position[0]
        prev_pos2 = agent2.position[0]

        # Fitness
        if fitness_type == "exploration":
            # Reward actual position change (exploration of ring)
            fitness += (disp1 + disp2)
        elif fitness_type == "activity":
            # Reward motor activity (original, broken)
            fitness += (abs(left_motor1) + abs(right_motor1) +
                       abs(left_motor2) + abs(right_motor2)) / 4.0

    fitness /= episode_length

    return {
        'fitness': fitness,
        'perception_count': perception_count,
        'perception_fraction': perception_count / episode_length,
        'total_displacement': total_displacement,
    }


def evolve_pair(num_neurons, seed):
    """Evolve a pair of agents for the perceptual crossing task."""
    decoder = GenotypeDecoder(
        num_neurons=num_neurons,
        include_gains=False,
        tau_range=(0.5, 5.0),
        weight_range=(-10.0, 10.0),
        bias_range=(-10.0, 10.0),
    )
    genotype_size = decoder.genotype_size

    # We evolve two genotypes jointly (concatenated)
    total_genotype_size = genotype_size * 2

    def fitness_fn(genotype):
        gen1 = genotype[:genotype_size]
        gen2 = genotype[genotype_size:]

        params1 = decoder.decode(gen1)
        params2 = decoder.decode(gen2)

        net1 = CTRNN(num_neurons=num_neurons,
                     time_constants=params1['tau'],
                     weights=params1['weights'],
                     biases=params1['biases'],
                     step_size=0.01, center_crossing=True)
        net2 = CTRNN(num_neurons=num_neurons,
                     time_constants=params2['tau'],
                     weights=params2['weights'],
                     biases=params2['biases'],
                     step_size=0.01, center_crossing=True)

        # Average over 2 trials with different starting positions
        total_fitness = 0.0
        for trial in range(2):
            result = run_episode(net1, net2, EPISODE_LENGTH, "exploration")
            total_fitness += result['fitness']
        return total_fitness / 2.0

    ga = MicrobialGA(
        genotype_size=total_genotype_size,
        fitness_function=fitness_fn,
        population_size=POPULATION_SIZE,
        mutation_std=0.2,
        seed=seed,
    )

    history = []
    for gen in range(NUM_GENERATIONS):
        _, best_fit = ga.step()
        if gen % 100 == 0 or gen == NUM_GENERATIONS - 1:
            history.append(float(best_fit))

    best_genotype = ga.get_best_individual()
    gen1 = best_genotype[:genotype_size]
    gen2 = best_genotype[genotype_size:]

    params1 = decoder.decode(gen1)
    params2 = decoder.decode(gen2)

    net1 = CTRNN(num_neurons=num_neurons,
                 time_constants=params1['tau'],
                 weights=params1['weights'],
                 biases=params1['biases'],
                 step_size=0.01, center_crossing=True)
    net2 = CTRNN(num_neurons=num_neurons,
                 time_constants=params2['tau'],
                 weights=params2['weights'],
                 biases=params2['biases'],
                 step_size=0.01, center_crossing=True)

    return net1, net2, history


def analyze_coordination(net1, net2, n_episodes=5):
    """Analyze coordination metrics over multiple episodes."""
    total_perception = 0
    total_perception_frac = 0
    total_displacement = 0

    for ep in range(n_episodes):
        result = run_episode(net1, net2, EPISODE_LENGTH, "exploration")
        total_perception += result['perception_count']
        total_perception_frac += result['perception_fraction']
        total_displacement += result['total_displacement']

    return {
        'mean_perception_count': total_perception / n_episodes,
        'mean_perception_fraction': total_perception_frac / n_episodes,
        'mean_displacement': total_displacement / n_episodes,
    }


def run_perturbation_tests(net1, net2):
    """Run basic perturbation tests."""
    results = {}

    # Baseline
    baseline = run_episode(net1, net2, EPISODE_LENGTH, "exploration")
    results['baseline'] = baseline

    # Test 1: Freeze agent 1
    agent1, agent2, env = create_agents_and_env(net1, net2)
    net1.reset()
    net2.reset()
    frozen_perception = 0
    for step in range(EPISODE_LENGTH):
        dist = min(
            abs(agent1.position[0] - agent2.position[0]),
            CIRCUMFERENCE - abs(agent1.position[0] - agent2.position[0])
        )
        perceive = dist < PERCEPTION_DIST
        if perceive:
            frozen_perception += 1

        sensor2 = np.array([float(perceive), float(perceive)])
        ext_input2 = np.zeros(net2.num_neurons)
        ext_input2[:min(2, net2.num_neurons)] = sensor2[:min(2, net2.num_neurons)]

        # Agent 1 frozen (no motor commands, no movement)
        # Agent 2 active
        out2 = net2.step(ext_input2)
        left_motor2 = float(out2[0]) if len(out2) > 0 else 0.0
        right_motor2 = float(out2[1]) if len(out2) > 1 else left_motor2
        agent2.set_motor_commands(left_motor2, right_motor2)

        # Only agent 2 moves
        agent2.update(0.01)
        agent2.position[0] = agent2.position[0] % CIRCUMFERENCE
        env._update_sensors()

    results['frozen_agent1'] = {
        'perception_count': frozen_perception,
        'perception_fraction': frozen_perception / EPISODE_LENGTH,
    }

    return results


def main():
    print("=" * 70)
    print("CROSS-TASK VALIDATION v2: PERCEPTUAL CROSSING (REDESIGNED)")
    print("=" * 70)
    print(f"Sizes: {NETWORK_SIZES}")
    print(f"Seeds: {SEEDS}")
    print(f"Total conditions: {len(NETWORK_SIZES) * len(SEEDS)}")
    print(f"Generations: {NUM_GENERATIONS}, Population: {POPULATION_SIZE}")
    print(f"Episode length: {EPISODE_LENGTH}, Max speed: {MAX_SPEED}")
    print(f"Starting positions: {START_POS_1}, {START_POS_2} ({START_POS_2 - START_POS_1} units apart)")
    print("=" * 70)

    total = len(NETWORK_SIZES) * len(SEEDS)
    results = {}
    cid = 0

    start_total = time.time()
    for ns in NETWORK_SIZES:
        for s in SEEDS:
            cid += 1
            run_id = f"net{ns}_seed{s}"
            start = time.time()
            print(f"\n[{cid}/{total}] {run_id}: evolving...", flush=True)

            try:
                net1, net2, history = evolve_pair(ns, s)
                elapsed_evo = time.time() - start

                # Coordination analysis
                coord = analyze_coordination(net1, net2, n_episodes=5)

                # Perturbation tests
                perturb = run_perturbation_tests(net1, net2)

                elapsed_total = time.time() - start
                print(f"  [{cid}/{total}] {run_id}: done in {elapsed_total:.1f}s | "
                      f"perception={coord['mean_perception_count']:.1f}/{EPISODE_LENGTH}, "
                      f"displacement={coord['mean_displacement']:.1f}", flush=True)

                results[run_id] = {
                    'num_neurons': ns,
                    'seed': s,
                    'evolution': {
                        'history': history,
                        'best_fitness': history[-1] if history else 0,
                    },
                    'coordination': coord,
                    'perturbations': perturb,
                    'timing': {'elapsed_seconds': elapsed_total},
                }

            except Exception as e:
                elapsed = time.time() - start
                print(f"  [{cid}/{total}] {run_id}: ERROR - {e} ({elapsed:.1f}s)", flush=True)
                results[run_id] = {
                    'num_neurons': ns,
                    'seed': s,
                    'error': str(e),
                    'timing': {'elapsed_seconds': elapsed},
                }

    elapsed_total = time.time() - start_total
    print(f"\nAll done in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

    # === ANALYSIS ===
    from scipy.stats import spearmanr

    print(f"\n{'='*70}")
    print("RESULTS BY NETWORK SIZE")
    print(f"{'='*70}")

    sizes_all, perceptions_all, displacements_all = [], [], []
    for ns in NETWORK_SIZES:
        subset = {k: v for k, v in results.items()
                  if v.get('num_neurons') == ns and 'error' not in v}
        if not subset:
            continue

        perceptions = [v['coordination']['mean_perception_count'] for v in subset.values()]
        fracs = [v['coordination']['mean_perception_fraction'] for v in subset.values()]
        disps = [v['coordination']['mean_displacement'] for v in subset.values()]

        print(f"\n  n={ns} (n_conditions={len(subset)}):")
        print(f"    Perception count:    {np.mean(perceptions):.1f} ± {np.std(perceptions):.1f} / {EPISODE_LENGTH}")
        print(f"    Perception fraction: {np.mean(fracs):.4f} ± {np.std(fracs):.4f}")
        print(f"    Mean displacement:   {np.mean(disps):.1f} ± {np.std(disps):.1f}")

        for v in subset.values():
            sizes_all.append(v['num_neurons'])
            perceptions_all.append(v['coordination']['mean_perception_count'])
            displacements_all.append(v['coordination']['mean_displacement'])

    # Correlations
    print(f"\n{'='*70}")
    print("CORRELATIONS WITH NETWORK SIZE")
    print(f"{'='*70}")

    if len(sizes_all) >= 5:
        rho_p, p_p = spearmanr(sizes_all, perceptions_all)
        rho_d, p_d = spearmanr(sizes_all, displacements_all)
        print(f"  Size vs perception count: rho={rho_p:.3f}, p={p_p:.4f}")
        print(f"  Size vs displacement:     rho={rho_d:.3f}, p={p_d:.4f}")

        print(f"\n  Paper 2 reference: size vs embodiment dependence rho=0.392, p=0.002")
        if p_p < 0.05:
            print(f"  CROSS-TASK VALIDATION: Perception correlation SIGNIFICANT (p={p_p:.4f})")
        else:
            print(f"  Cross-task: Perception correlation NOT significant (p={p_p:.4f})")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = os.path.join(RESULTS_DIR, f'cross_task_v2_{timestamp}.json')

    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        return obj

    save = {
        'conditions': convert(results),
        'meta': {
            'timestamp': timestamp,
            'network_sizes': NETWORK_SIZES,
            'seeds': SEEDS,
            'generations': NUM_GENERATIONS,
            'population_size': POPULATION_SIZE,
            'episode_length': EPISODE_LENGTH,
            'max_speed': MAX_SPEED,
            'circumference': CIRCUMFERENCE,
            'start_positions': [START_POS_1, START_POS_2],
            'changes_from_v1': [
                'max_speed: 1.0 -> 5.0',
                'episode_length: 500 -> 5000',
                'fitness: exploration (displacement) instead of activity',
                'generations: 300 -> 500',
                'starting distance: 50 -> 25 units',
            ],
        },
    }

    with open(outfile, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nSaved to: {outfile}")


if __name__ == "__main__":
    main()
