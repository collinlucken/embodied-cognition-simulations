"""
Paper 4: Enriched Perceptual Crossing × Network Capacity

Enriched with Froese & Di Paolo (2010) features:
1. Shadow objects: passive entities that mirror one agent's movement
   - Same perception distance as agents
   - Agent perceives shadow identically to perceiving other agent
   - Shadow does NOT perceive (no neural controller)
   - Creates discrimination challenge: agent must distinguish
     real partner (who responds to interaction) from shadow (who doesn't)

2. Fitness = proximity_to_partner - proximity_to_shadow + exploration
   - Rewards approaching the real partner
   - Penalizes approaching the shadow
   - Includes exploration bonus to prevent stasis

Hypothesis: Larger networks should better discriminate partner from shadow,
because discrimination requires maintaining a history of interaction dynamics
(the partner responds; the shadow doesn't) — requiring more computational capacity.
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.ctrnn import CTRNN
from simulation.evolutionary import MicrobialGA, GenotypeDecoder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../../results/paper4')
NETWORK_SIZES = [2, 3, 4, 5, 6, 8]
SEEDS = [42, 137, 256]
NUM_GENERATIONS = 300
POPULATION_SIZE = 30
EVO_EPISODE_LENGTH = 3000
ANALYSIS_EPISODE_LENGTH = 5000
MAX_SPEED = 5.0
CIRCUMFERENCE = 100.0
PERCEPTION_DIST = 5.0
DT = 0.01


def wrap_distance(pos1, pos2, circ):
    """Compute distance on 1D ring."""
    raw = abs(pos1 - pos2)
    return min(raw, circ - raw)


def run_enriched_episode(net1, net2, episode_length, shadow_mode='mirror'):
    """
    Run one enriched perceptual crossing episode.

    Entities on the ring:
    - Agent 1 (position a1, controlled by net1)
    - Agent 2 (position a2, controlled by net2)
    - Shadow 1 (mirrors Agent 1's position offset by 50 units)
    - Shadow 2 (mirrors Agent 2's position offset by 50 units)

    Each agent senses: [is_something_nearby] (binary)
    The "something" could be the other agent OR either shadow.
    Agents cannot distinguish partner from shadow via perception alone.

    Returns: dict with fitness, discrimination metrics, perception events
    """
    # Starting positions
    a1_pos = 25.0
    a2_pos = 75.0
    n1, n2 = net1.num_neurons, net2.num_neurons

    net1.reset()
    net2.reset()

    # Tracking
    partner_time = 0  # steps where agents are near each other
    shadow_time = 0   # steps where agents are near a shadow
    exploration = 0.0
    prev_a1, prev_a2 = a1_pos, a2_pos

    # Detailed tracking for analysis
    partner_encounters = 0
    shadow_encounters = 0
    total_steps = 0

    for step in range(episode_length):
        # Shadow positions (mirror with offset)
        s1_pos = (a1_pos + CIRCUMFERENCE/2) % CIRCUMFERENCE  # Shadow of agent 1
        s2_pos = (a2_pos + CIRCUMFERENCE/2) % CIRCUMFERENCE  # Shadow of agent 2

        # Agent 1 sensing: detect agent 2 OR shadow 2
        d_a1_a2 = wrap_distance(a1_pos, a2_pos, CIRCUMFERENCE)
        d_a1_s2 = wrap_distance(a1_pos, s2_pos, CIRCUMFERENCE)
        a1_perceives_partner = d_a1_a2 < PERCEPTION_DIST
        a1_perceives_shadow = d_a1_s2 < PERCEPTION_DIST
        a1_perceives = a1_perceives_partner or a1_perceives_shadow

        # Agent 2 sensing: detect agent 1 OR shadow 1
        d_a2_a1 = d_a1_a2  # symmetric
        d_a2_s1 = wrap_distance(a2_pos, s1_pos, CIRCUMFERENCE)
        a2_perceives_partner = d_a2_a1 < PERCEPTION_DIST
        a2_perceives_shadow = d_a2_s1 < PERCEPTION_DIST
        a2_perceives = a2_perceives_partner or a2_perceives_shadow

        # Neural input (binary perception)
        ext1 = np.zeros(n1)
        ext2 = np.zeros(n2)
        ext1[0] = float(a1_perceives)
        ext2[0] = float(a2_perceives)

        # Step networks
        out1 = net1.step(ext1)
        out2 = net2.step(ext2)

        # Motor: use first output neuron as velocity on ring
        v1 = float(out1[0]) * MAX_SPEED
        v2 = float(out2[0]) * MAX_SPEED

        # Update positions on ring
        a1_pos = (a1_pos + v1 * DT) % CIRCUMFERENCE
        a2_pos = (a2_pos + v2 * DT) % CIRCUMFERENCE

        # Track metrics
        if a1_perceives_partner or a2_perceives_partner:
            partner_time += 1
            if a1_perceives_partner:
                partner_encounters += 1
        if a1_perceives_shadow or a2_perceives_shadow:
            shadow_time += 1
            if a1_perceives_shadow:
                shadow_encounters += 1

        # Exploration (displacement)
        d1 = wrap_distance(a1_pos, prev_a1, CIRCUMFERENCE)
        d2 = wrap_distance(a2_pos, prev_a2, CIRCUMFERENCE)
        exploration += d1 + d2
        prev_a1, prev_a2 = a1_pos, a2_pos
        total_steps += 1

    # Compute fitness
    partner_frac = partner_time / episode_length
    shadow_frac = shadow_time / episode_length
    exploration_norm = exploration / episode_length

    # Discrimination score: partner_time - shadow_time (higher = better discrimination)
    discrimination = (partner_time - shadow_time) / max(1, partner_time + shadow_time)

    # Fitness: reward partner interaction, penalize shadow interaction, add exploration
    fitness = partner_frac - 0.5 * shadow_frac + 0.3 * exploration_norm

    return {
        'fitness': float(fitness),
        'partner_fraction': float(partner_frac),
        'shadow_fraction': float(shadow_frac),
        'discrimination': float(discrimination),
        'exploration': float(exploration_norm),
        'partner_encounters': int(partner_encounters),
        'shadow_encounters': int(shadow_encounters),
        'total_steps': total_steps,
    }


def evolve_pair(num_neurons, seed):
    """Evolve a pair of agents on the enriched perceptual crossing task."""
    decoder = GenotypeDecoder(
        num_neurons=num_neurons, include_gains=False,
        tau_range=(0.5, 5.0), weight_range=(-10.0, 10.0),
        bias_range=(-10.0, 10.0)
    )
    gs = decoder.genotype_size
    total_gs = gs * 2

    def fitness_fn(genotype):
        p1 = decoder.decode(genotype[:gs])
        p2 = decoder.decode(genotype[gs:])
        n1 = CTRNN(num_neurons, p1['tau'], p1['weights'], p1['biases'],
                    step_size=DT, center_crossing=True)
        n2 = CTRNN(num_neurons, p2['tau'], p2['weights'], p2['biases'],
                    step_size=DT, center_crossing=True)
        result = run_enriched_episode(n1, n2, EVO_EPISODE_LENGTH)
        return result['fitness']

    ga = MicrobialGA(total_gs, fitness_fn, POPULATION_SIZE, 0.2, seed=seed)
    for _ in range(NUM_GENERATIONS):
        ga.step()

    best = ga.get_best_individual()
    p1 = decoder.decode(best[:gs])
    p2 = decoder.decode(best[gs:])
    n1 = CTRNN(num_neurons, p1['tau'], p1['weights'], p1['biases'],
                step_size=DT, center_crossing=True)
    n2 = CTRNN(num_neurons, p2['tau'], p2['weights'], p2['biases'],
                step_size=DT, center_crossing=True)
    return n1, n2, float(ga.get_best_fitness())


def main():
    print("=" * 70)
    print("ENRICHED PERCEPTUAL CROSSING × NETWORK CAPACITY")
    print("=" * 70)
    total = len(NETWORK_SIZES) * len(SEEDS)
    print(f"Total conditions: {total}")
    print(f"Evolution: {NUM_GENERATIONS} gen, pop {POPULATION_SIZE}, {EVO_EPISODE_LENGTH} steps/ep")
    print(f"Analysis: {ANALYSIS_EPISODE_LENGTH} steps/ep, 3 episodes")
    print(f"Features: shadows (mirrored position +50 units)")
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

                # Analysis: 3 episodes
                episode_results = []
                for ep in range(3):
                    ep_result = run_enriched_episode(n1, n2, ANALYSIS_EPISODE_LENGTH)
                    episode_results.append(ep_result)

                elapsed = time.time() - start
                mean_disc = np.mean([r['discrimination'] for r in episode_results])
                mean_partner = np.mean([r['partner_fraction'] for r in episode_results])
                mean_shadow = np.mean([r['shadow_fraction'] for r in episode_results])

                print(f"done ({elapsed:.0f}s) | disc={mean_disc:.3f} partner={mean_partner:.3f} shadow={mean_shadow:.3f}")

                results[run_id] = {
                    'num_neurons': ns, 'seed': s,
                    'best_fitness': best_fit,
                    'mean_discrimination': float(mean_disc),
                    'mean_partner_fraction': float(mean_partner),
                    'mean_shadow_fraction': float(mean_shadow),
                    'mean_exploration': float(np.mean([r['exploration'] for r in episode_results])),
                    'episodes': episode_results,
                }
            except Exception as e:
                elapsed = time.time() - start
                print(f"ERROR: {e} ({elapsed:.0f}s)")
                results[run_id] = {'num_neurons': ns, 'seed': s, 'error': str(e)}

    total_time = time.time() - start_total
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # === ANALYSIS ===
    print(f"\n{'='*70}")
    print("RESULTS BY NETWORK SIZE")
    print(f"{'='*70}")

    sizes_all, disc_all, partner_all, shadow_all = [], [], [], []
    for ns in NETWORK_SIZES:
        subset = {k: v for k, v in results.items() if v.get('num_neurons') == ns and 'error' not in v}
        if not subset:
            continue
        discs = [v['mean_discrimination'] for v in subset.values()]
        partners = [v['mean_partner_fraction'] for v in subset.values()]
        shadows = [v['mean_shadow_fraction'] for v in subset.values()]
        print(f"  n={ns}: disc={np.mean(discs):.3f}±{np.std(discs):.3f}  "
              f"partner={np.mean(partners):.4f}  shadow={np.mean(shadows):.4f}")
        for v in subset.values():
            sizes_all.append(v['num_neurons'])
            disc_all.append(v['mean_discrimination'])
            partner_all.append(v['mean_partner_fraction'])
            shadow_all.append(v['mean_shadow_fraction'])

    if len(sizes_all) >= 5:
        print(f"\n  Correlations with network size:")
        rho_d, p_d = spearmanr(sizes_all, disc_all)
        rho_p, p_p = spearmanr(sizes_all, partner_all)
        rho_s, p_s = spearmanr(sizes_all, shadow_all)
        print(f"    Discrimination: rho={rho_d:.3f}, p={p_d:.4f} {'*' if p_d < 0.05 else ''}")
        print(f"    Partner time:   rho={rho_p:.3f}, p={p_p:.4f} {'*' if p_p < 0.05 else ''}")
        print(f"    Shadow time:    rho={rho_s:.3f}, p={p_s:.4f} {'*' if p_s < 0.05 else ''}")

        # Small vs large
        small_d = [disc_all[i] for i in range(len(sizes_all)) if sizes_all[i] <= 3]
        large_d = [disc_all[i] for i in range(len(sizes_all)) if sizes_all[i] >= 6]
        if len(small_d) >= 3 and len(large_d) >= 3:
            stat, p_mw = mannwhitneyu(large_d, small_d, alternative='greater')
            print(f"\n    Large vs Small discrimination: U={stat:.0f}, p={p_mw:.4f}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = os.path.join(RESULTS_DIR, f'cross_task_enriched_{timestamp}.json')

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        return obj

    with open(outfile, 'w') as f:
        json.dump(convert({'conditions': results, 'meta': {
            'timestamp': timestamp, 'sizes': NETWORK_SIZES, 'seeds': SEEDS,
            'generations': NUM_GENERATIONS, 'evo_episode': EVO_EPISODE_LENGTH,
            'analysis_episode': ANALYSIS_EPISODE_LENGTH, 'max_speed': MAX_SPEED,
            'features': 'shadow_mirror'
        }}), f, indent=2, default=str)
    print(f"\nSaved to: {outfile}")


if __name__ == "__main__":
    main()
