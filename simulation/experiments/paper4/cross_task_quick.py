"""
Quick Cross-Task Validation: Perceptual Crossing × Network Capacity
Reduced scope version for feasible runtime.

Changes from v2:
- 3 seeds instead of 5 (18 conditions total)
- 200 generations instead of 500
- 2000-step episodes during evolution (5000 for analysis)
- 3 analysis episodes instead of 5
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
SEEDS = [42, 137, 256]  # Reduced from 5 to 3
NUM_GENERATIONS = 200  # Reduced from 500
POPULATION_SIZE = 30
EVO_EPISODE_LENGTH = 2000  # Shorter for evolution
ANALYSIS_EPISODE_LENGTH = 5000  # Longer for analysis
MAX_SPEED = 5.0
CIRCUMFERENCE = 100.0
PERCEPTION_DIST = 5.0
START_POS_1, START_POS_2 = 37.5, 62.5


def run_episode(net1, net2, episode_length):
    """Run one episode, return metrics."""
    agent1 = Agent(radius=1.0, max_speed=MAX_SPEED)
    agent2 = Agent(radius=1.0, max_speed=MAX_SPEED)
    agent1.position = np.array([START_POS_1, 0.0])
    agent2.position = np.array([START_POS_2, 0.0])
    env = PerceptualCrossingEnv(circumference=CIRCUMFERENCE)
    env.set_agents(agent1, agent2)
    net1.reset()
    net2.reset()

    fitness = 0.0
    perception_count = 0
    prev_pos1, prev_pos2 = agent1.position[0], agent2.position[0]

    for step in range(episode_length):
        dist = min(
            abs(agent1.position[0] - agent2.position[0]),
            CIRCUMFERENCE - abs(agent1.position[0] - agent2.position[0])
        )
        perceive = dist < PERCEPTION_DIST

        sensor = np.array([float(perceive), float(perceive)])
        ext1 = np.zeros(net1.num_neurons)
        ext2 = np.zeros(net2.num_neurons)
        ext1[:min(2, net1.num_neurons)] = sensor[:min(2, net1.num_neurons)]
        ext2[:min(2, net2.num_neurons)] = sensor[:min(2, net2.num_neurons)]

        out1 = net1.step(ext1)
        out2 = net2.step(ext2)

        lm1, rm1 = float(out1[0]), float(out1[1]) if len(out1) > 1 else float(out1[0])
        lm2, rm2 = float(out2[0]), float(out2[1]) if len(out2) > 1 else float(out2[0])

        agent1.set_motor_commands(lm1, rm1)
        agent2.set_motor_commands(lm2, rm2)
        env.step()

        if perceive:
            perception_count += 1

        disp1 = min(abs(agent1.position[0] - prev_pos1), CIRCUMFERENCE - abs(agent1.position[0] - prev_pos1))
        disp2 = min(abs(agent2.position[0] - prev_pos2), CIRCUMFERENCE - abs(agent2.position[0] - prev_pos2))
        fitness += disp1 + disp2
        prev_pos1, prev_pos2 = agent1.position[0], agent2.position[0]

    return fitness / episode_length, perception_count, perception_count / episode_length


def evolve_pair(num_neurons, seed):
    """Evolve a pair of agents."""
    decoder = GenotypeDecoder(num_neurons=num_neurons, include_gains=False,
                               tau_range=(0.5, 5.0), weight_range=(-10.0, 10.0), bias_range=(-10.0, 10.0))
    gs = decoder.genotype_size
    total_gs = gs * 2

    def fitness_fn(genotype):
        p1 = decoder.decode(genotype[:gs])
        p2 = decoder.decode(genotype[gs:])
        n1 = CTRNN(num_neurons, p1['tau'], p1['weights'], p1['biases'], step_size=0.01, center_crossing=True)
        n2 = CTRNN(num_neurons, p2['tau'], p2['weights'], p2['biases'], step_size=0.01, center_crossing=True)
        fit, _, _ = run_episode(n1, n2, EVO_EPISODE_LENGTH)
        return fit

    ga = MicrobialGA(total_gs, fitness_fn, POPULATION_SIZE, 0.2, seed=seed)
    for _ in range(NUM_GENERATIONS):
        ga.step()

    best = ga.get_best_individual()
    p1 = decoder.decode(best[:gs])
    p2 = decoder.decode(best[gs:])
    n1 = CTRNN(num_neurons, p1['tau'], p1['weights'], p1['biases'], step_size=0.01, center_crossing=True)
    n2 = CTRNN(num_neurons, p2['tau'], p2['weights'], p2['biases'], step_size=0.01, center_crossing=True)
    return n1, n2, float(ga.get_best_fitness())


def main():
    print("=" * 70)
    print("QUICK CROSS-TASK VALIDATION: PERCEPTUAL CROSSING (REDESIGNED)")
    print("=" * 70)
    total = len(NETWORK_SIZES) * len(SEEDS)
    print(f"Total conditions: {total} ({len(NETWORK_SIZES)} sizes × {len(SEEDS)} seeds)")
    print(f"Evolution: {NUM_GENERATIONS} gen, {EVO_EPISODE_LENGTH} steps/episode")
    print(f"Analysis: {ANALYSIS_EPISODE_LENGTH} steps/episode, 3 episodes")
    print(f"Max speed: {MAX_SPEED}, starting dist: {START_POS_2 - START_POS_1}")
    print("=" * 70)

    results = {}
    cid = 0
    start_total = time.time()

    for ns in NETWORK_SIZES:
        for s in SEEDS:
            cid += 1
            run_id = f"net{ns}_seed{s}"
            start = time.time()
            print(f"\n[{cid}/{total}] {run_id}: evolving...", end=" ", flush=True)

            try:
                n1, n2, best_fit = evolve_pair(ns, s)
                evo_time = time.time() - start
                print(f"({evo_time:.0f}s) analyzing...", end=" ", flush=True)

                # Analysis: 3 episodes at full length
                perc_counts = []
                perc_fracs = []
                for ep in range(3):
                    _, pc, pf = run_episode(n1, n2, ANALYSIS_EPISODE_LENGTH)
                    perc_counts.append(pc)
                    perc_fracs.append(pf)

                elapsed = time.time() - start
                mean_pc = np.mean(perc_counts)
                mean_pf = np.mean(perc_fracs)
                print(f"done ({elapsed:.0f}s) | perception={mean_pc:.0f}/{ANALYSIS_EPISODE_LENGTH} ({mean_pf:.3f})")

                results[run_id] = {
                    'num_neurons': ns, 'seed': s,
                    'best_fitness': best_fit,
                    'mean_perception_count': float(mean_pc),
                    'mean_perception_fraction': float(mean_pf),
                    'perception_counts': [int(p) for p in perc_counts],
                }
            except Exception as e:
                elapsed = time.time() - start
                print(f"ERROR: {e} ({elapsed:.0f}s)")
                results[run_id] = {'num_neurons': ns, 'seed': s, 'error': str(e)}

    total_time = time.time() - start_total
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Analysis
    from scipy.stats import spearmanr, mannwhitneyu

    print(f"\n{'='*70}")
    print("RESULTS BY NETWORK SIZE")
    print(f"{'='*70}")

    sizes_all, percs_all = [], []
    for ns in NETWORK_SIZES:
        subset = {k: v for k, v in results.items() if v.get('num_neurons') == ns and 'error' not in v}
        if not subset:
            continue
        percs = [v['mean_perception_count'] for v in subset.values()]
        fracs = [v['mean_perception_fraction'] for v in subset.values()]
        print(f"  n={ns}: perception={np.mean(percs):.1f} ± {np.std(percs):.1f} ({np.mean(fracs):.4f})")
        for v in subset.values():
            sizes_all.append(v['num_neurons'])
            percs_all.append(v['mean_perception_count'])

    if len(sizes_all) >= 5:
        rho, p = spearmanr(sizes_all, percs_all)
        print(f"\n  Size vs perception: rho={rho:.3f}, p={p:.4f}")
        print(f"  Paper 2 reference: size vs ED rho=0.392, p=0.002")
        if p < 0.05:
            print(f"  CROSS-TASK VALIDATION SUCCESS")
        else:
            print(f"  Cross-task validation: NOT significant")

        # Small vs large
        small = [percs_all[i] for i in range(len(sizes_all)) if sizes_all[i] <= 3]
        large = [percs_all[i] for i in range(len(sizes_all)) if sizes_all[i] >= 6]
        if len(small) >= 3 and len(large) >= 3:
            stat, p_mw = mannwhitneyu(large, small, alternative='greater')
            print(f"  Large (n≥6) vs Small (n≤3) perception: U={stat:.0f}, p={p_mw:.4f}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = os.path.join(RESULTS_DIR, f'cross_task_v2_{timestamp}.json')
    save = {'conditions': results, 'meta': {'timestamp': timestamp, 'sizes': NETWORK_SIZES,
            'seeds': SEEDS, 'generations': NUM_GENERATIONS, 'evo_episode': EVO_EPISODE_LENGTH,
            'analysis_episode': ANALYSIS_EPISODE_LENGTH, 'max_speed': MAX_SPEED}}

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        return obj

    with open(outfile, 'w') as f:
        json.dump(convert(save), f, indent=2, default=str)
    print(f"\nSaved to: {outfile}")


if __name__ == "__main__":
    main()
