"""
Paper 4: Enriched Perceptual Crossing × Network Capacity (10 Seeds per Size)

EXPANDED EXPERIMENTAL DESIGN:
- 6 network sizes × 10 seeds = 60 conditions (vs 18 in original)
- Provides statistical power matching Paper 2 (1980 conditions)
- Includes ghost conditions (freeze one agent's sensory input)
- Behavioral characterization (movement patterns, approach velocity, dwelling time)

KEY ANALYSES:
1. Size vs Discrimination/Fitness: Spearman correlation
2. Small (n≤3) vs Large (n≥6): Mann-Whitney U test
3. Ghost conditions: ED score correlates with discrimination
4. Behavioral strategies: approach velocity, dwelling time, movement classification

Hypothesis: Larger networks discriminate partner from shadow better via
maintaining interaction history (partner responds; shadow doesn't).
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
SEEDS = [42, 137, 256, 314, 500, 628, 777, 888, 999, 1234]
NUM_GENERATIONS = 300
POPULATION_SIZE = 30
EVO_EPISODE_LENGTH = 3000
ANALYSIS_EPISODE_LENGTH = 5000
MAX_SPEED = 5.0
CIRCUMFERENCE = 100.0
PERCEPTION_DIST = 5.0
DT = 0.01


def wrap_distance(pos1, pos2, circ):
    """Compute distance on 1D ring, always taking shorter path."""
    raw = abs(pos1 - pos2)
    return min(raw, circ - raw)


def run_enriched_episode(net1, net2, episode_length, shadow_mode='mirror', ghost_agent=None):
    """
    Run one enriched perceptual crossing episode.

    Args:
        net1, net2: CTRNN networks controlling agent 1 and 2
        episode_length: Number of simulation steps
        shadow_mode: 'mirror' (shadows at +50 offset) or other modes
        ghost_agent: None (normal), 1 (freeze agent 1's input), or 2 (freeze agent 2's input)

    Returns:
        dict with fitness, discrimination metrics, perception events, behavioral metrics
    """
    # Starting positions
    a1_pos = 25.0
    a2_pos = 75.0
    n1, n2 = net1.num_neurons, net2.num_neurons

    net1.reset()
    net2.reset()

    # Tracking basic metrics
    partner_time = 0
    shadow_time = 0
    exploration = 0.0
    prev_a1, prev_a2 = a1_pos, a2_pos

    # Tracking encounters
    partner_encounters = 0
    shadow_encounters = 0

    # Behavioral tracking
    velocities_1 = []
    velocities_2 = []
    distances_to_partner = []
    distances_to_shadow_1 = []
    distances_to_shadow_2 = []
    near_partner_1 = []  # boolean per step: agent 1 near partner
    near_partner_2 = []  # boolean per step: agent 2 near partner
    near_shadow_1 = []
    near_shadow_2 = []

    total_steps = 0

    for step in range(episode_length):
        # Shadow positions (mirror with offset)
        s1_pos = (a1_pos + CIRCUMFERENCE/2) % CIRCUMFERENCE
        s2_pos = (a2_pos + CIRCUMFERENCE/2) % CIRCUMFERENCE

        # Distances
        d_a1_a2 = wrap_distance(a1_pos, a2_pos, CIRCUMFERENCE)
        d_a1_s2 = wrap_distance(a1_pos, s2_pos, CIRCUMFERENCE)
        d_a2_a1 = d_a1_a2
        d_a2_s1 = wrap_distance(a2_pos, s1_pos, CIRCUMFERENCE)

        # Perception
        a1_perceives_partner = d_a1_a2 < PERCEPTION_DIST
        a1_perceives_shadow = d_a1_s2 < PERCEPTION_DIST
        a1_perceives = a1_perceives_partner or a1_perceives_shadow

        a2_perceives_partner = d_a2_a1 < PERCEPTION_DIST
        a2_perceives_shadow = d_a2_s1 < PERCEPTION_DIST
        a2_perceives = a2_perceives_partner or a2_perceives_shadow

        # Store distances for behavioral analysis
        distances_to_partner.append(float(d_a1_a2))
        distances_to_shadow_1.append(float(d_a1_s2))
        distances_to_shadow_2.append(float(d_a2_s1))
        near_partner_1.append(a1_perceives_partner)
        near_partner_2.append(a2_perceives_partner)
        near_shadow_1.append(a1_perceives_shadow)
        near_shadow_2.append(a2_perceives_shadow)

        # GHOST CONDITION: optionally freeze one agent's sensory input
        if ghost_agent == 1:
            a1_perceives = False  # Agent 1 perceives nothing
        elif ghost_agent == 2:
            a2_perceives = False  # Agent 2 perceives nothing

        # Neural input (binary perception)
        ext1 = np.zeros(n1)
        ext2 = np.zeros(n2)
        ext1[0] = float(a1_perceives)
        ext2[0] = float(a2_perceives)

        # Step networks
        out1 = net1.step(ext1)
        out2 = net2.step(ext2)

        # Motor: velocity
        v1 = float(out1[0]) * MAX_SPEED
        v2 = float(out2[0]) * MAX_SPEED
        velocities_1.append(abs(v1))
        velocities_2.append(abs(v2))

        # Update positions
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

        # Exploration
        d1 = wrap_distance(a1_pos, prev_a1, CIRCUMFERENCE)
        d2 = wrap_distance(a2_pos, prev_a2, CIRCUMFERENCE)
        exploration += d1 + d2
        prev_a1, prev_a2 = a1_pos, a2_pos
        total_steps += 1

    # Compute basic metrics
    partner_frac = partner_time / episode_length
    shadow_frac = shadow_time / episode_length
    exploration_norm = exploration / episode_length
    discrimination = (partner_time - shadow_time) / max(1, partner_time + shadow_time)
    fitness = partner_frac - 0.5 * shadow_frac + 0.3 * exploration_norm

    # BEHAVIORAL ANALYSIS
    # 1. Approach velocity: average velocity when close to partner vs shadow
    approach_vel_partner = np.mean([velocities_1[i] for i in range(len(near_partner_1))
                                   if near_partner_1[i]]) if any(near_partner_1) else 0.0
    approach_vel_shadow = np.mean([velocities_1[i] for i in range(len(near_shadow_1))
                                  if near_shadow_1[i]]) if any(near_shadow_1) else 0.0

    # 2. Dwelling time: how long does agent stay within perception range after encounter?
    def compute_dwelling(perceives_list, perception_dist_data):
        """Compute average dwelling time (steps within perception range per encounter)."""
        if not any(perceives_list):
            return 0.0
        dwelling_times = []
        current_dwelling = 0
        for i, is_perceiving in enumerate(perceives_list):
            if is_perceiving:
                current_dwelling += 1
            else:
                if current_dwelling > 0:
                    dwelling_times.append(current_dwelling)
                current_dwelling = 0
        if current_dwelling > 0:
            dwelling_times.append(current_dwelling)
        return np.mean(dwelling_times) if dwelling_times else 0.0

    dwelling_partner = compute_dwelling(near_partner_1, distances_to_partner)
    dwelling_shadow = compute_dwelling(near_shadow_1, distances_to_shadow_1)

    # 3. Movement classification: oscillating vs scanning vs stationary
    mean_vel = np.mean(velocities_1)
    std_vel = np.std(velocities_1)
    # Oscillating: high std in velocity, low mean (reversing direction)
    # Scanning: moderate mean, moderate std
    # Stationary: low mean and std
    vel_cv = std_vel / (mean_vel + 0.01)  # coefficient of variation
    if mean_vel < 0.5 and std_vel < 0.5:
        movement_class = 'stationary'
    elif vel_cv > 1.5:
        movement_class = 'oscillating'
    else:
        movement_class = 'scanning'

    return {
        'fitness': float(fitness),
        'partner_fraction': float(partner_frac),
        'shadow_fraction': float(shadow_frac),
        'discrimination': float(discrimination),
        'exploration': float(exploration_norm),
        'partner_encounters': int(partner_encounters),
        'shadow_encounters': int(shadow_encounters),
        'total_steps': total_steps,
        'approach_velocity_partner': float(approach_vel_partner),
        'approach_velocity_shadow': float(approach_vel_shadow),
        'dwelling_time_partner': float(dwelling_partner),
        'dwelling_time_shadow': float(dwelling_shadow),
        'movement_class': movement_class,
        'mean_velocity': float(mean_vel),
        'std_velocity': float(std_vel),
        'velocity_cv': float(vel_cv),
        'distances_to_partner': [float(d) for d in distances_to_partner[:100]],  # sample
    }


def compute_ed_score(net1, net2, episode_length=2000):
    """
    Compute ED (Embodied Discrimination) score via ghost condition.

    ED = |discrimination_normal - discrimination_ghost1 - discrimination_ghost2| / 2

    The idea: if an agent's perceptual input is frozen (ghost), the ED score measures
    how much the pair's discrimination performance drops. Higher ED means the agent
    was critical for discrimination.
    """
    # Normal condition
    result_normal = run_enriched_episode(net1, net2, episode_length, ghost_agent=None)
    disc_normal = result_normal['discrimination']

    # Ghost agent 1 (freeze agent 1's perception)
    result_ghost1 = run_enriched_episode(net1, net2, episode_length, ghost_agent=1)
    disc_ghost1 = result_ghost1['discrimination']

    # Ghost agent 2 (freeze agent 2's perception)
    result_ghost2 = run_enriched_episode(net1, net2, episode_length, ghost_agent=2)
    disc_ghost2 = result_ghost2['discrimination']

    # ED: average drop from normal when either agent is ghosted
    ed_score = (disc_normal - (disc_ghost1 + disc_ghost2) / 2.0)
    ed_score = max(0, ed_score)  # ED should be non-negative

    return {
        'ed_score': float(ed_score),
        'discrimination_normal': float(disc_normal),
        'discrimination_ghost1': float(disc_ghost1),
        'discrimination_ghost2': float(disc_ghost2),
    }


def evolve_pair(num_neurons, seed):
    """Evolve a pair of agents on enriched perceptual crossing."""
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
    print("=" * 80)
    print("PAPER 4: ENRICHED PERCEPTUAL CROSSING × NETWORK CAPACITY (10 SEEDS)")
    print("=" * 80)
    total = len(NETWORK_SIZES) * len(SEEDS)
    print(f"Total conditions: {total} (6 sizes × 10 seeds)")
    print(f"Evolution: {NUM_GENERATIONS} gen, pop {POPULATION_SIZE}, {EVO_EPISODE_LENGTH} steps/ep")
    print(f"Analysis: {ANALYSIS_EPISODE_LENGTH} steps/ep, 3 episodes + ghost conditions")
    print(f"Seeds: {SEEDS}")
    print("=" * 80)

    results = {}
    cid = 0
    start_total = time.time()

    for ns in NETWORK_SIZES:
        for s in SEEDS:
            cid += 1
            run_id = f"net{ns}_seed{s}"
            start = time.time()
            print(f"\n[{cid:2d}/{total}] {run_id}: evolving...", end=" ", flush=True)

            try:
                n1, n2, best_fit = evolve_pair(ns, s)
                evo_time = time.time() - start
                print(f"({evo_time:.0f}s) analyzing...", end=" ", flush=True)

                # Analysis: 3 normal episodes
                episode_results = []
                for ep in range(3):
                    ep_result = run_enriched_episode(n1, n2, ANALYSIS_EPISODE_LENGTH)
                    episode_results.append(ep_result)

                mean_disc = np.mean([r['discrimination'] for r in episode_results])
                mean_partner = np.mean([r['partner_fraction'] for r in episode_results])
                mean_shadow = np.mean([r['shadow_fraction'] for r in episode_results])
                mean_fitness = np.mean([r['fitness'] for r in episode_results])

                print(f"ghost conditions...", end=" ", flush=True)

                # Ghost conditions (ED score) - run 2 episodes for speed
                ghost_results = []
                for ep in range(2):
                    ghost_result = compute_ed_score(n1, n2, ANALYSIS_EPISODE_LENGTH)
                    ghost_results.append(ghost_result)

                mean_ed = np.mean([r['ed_score'] for r in ghost_results])

                elapsed = time.time() - start
                print(f"done ({elapsed:.0f}s)")
                print(f"     disc={mean_disc:.3f} partner={mean_partner:.3f} shadow={mean_shadow:.3f} ed={mean_ed:.3f}")

                results[run_id] = {
                    'num_neurons': ns,
                    'seed': s,
                    'best_fitness': best_fit,
                    'mean_discrimination': float(mean_disc),
                    'std_discrimination': float(np.std([r['discrimination'] for r in episode_results])),
                    'mean_partner_fraction': float(mean_partner),
                    'mean_shadow_fraction': float(mean_shadow),
                    'mean_exploration': float(np.mean([r['exploration'] for r in episode_results])),
                    'mean_fitness': float(mean_fitness),
                    'mean_ed_score': float(mean_ed),
                    'episodes': episode_results,
                    'ghost_episodes': ghost_results,
                }
            except Exception as e:
                elapsed = time.time() - start
                print(f"ERROR: {e} ({elapsed:.0f}s)")
                results[run_id] = {'num_neurons': ns, 'seed': s, 'error': str(e)}

    total_time = time.time() - start_total
    print(f"\n{'='*80}")
    print(f"Total evolution time: {total_time:.0f}s ({total_time/60:.1f} min, {total_time/3600:.2f} hours)")
    print(f"{'='*80}")

    # === COMPREHENSIVE ANALYSIS ===
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}")

    # Aggregate data
    sizes_all, disc_all, fitness_all, partner_all, shadow_all, ed_all = [], [], [], [], [], []
    for ns in NETWORK_SIZES:
        subset = {k: v for k, v in results.items() if v.get('num_neurons') == ns and 'error' not in v}
        if not subset:
            continue
        discs = [v['mean_discrimination'] for v in subset.values()]
        fitnesses = [v['mean_fitness'] for v in subset.values()]
        partners = [v['mean_partner_fraction'] for v in subset.values()]
        shadows = [v['mean_shadow_fraction'] for v in subset.values()]
        eds = [v['mean_ed_score'] for v in subset.values()]

        for v in subset.values():
            sizes_all.append(v['num_neurons'])
            disc_all.append(v['mean_discrimination'])
            fitness_all.append(v['mean_fitness'])
            partner_all.append(v['mean_partner_fraction'])
            shadow_all.append(v['mean_shadow_fraction'])
            ed_all.append(v['mean_ed_score'])

    # Per-size summary statistics
    print("\n1. SUMMARY BY NETWORK SIZE:")
    print("-" * 80)
    for ns in NETWORK_SIZES:
        subset = {k: v for k, v in results.items() if v.get('num_neurons') == ns and 'error' not in v}
        if not subset:
            print(f"  n={ns}: (no results)")
            continue
        discs = [v['mean_discrimination'] for v in subset.values()]
        fitnesses = [v['mean_fitness'] for v in subset.values()]
        partners = [v['mean_partner_fraction'] for v in subset.values()]
        shadows = [v['mean_shadow_fraction'] for v in subset.values()]
        eds = [v['mean_ed_score'] for v in subset.values()]

        print(f"\n  n={ns} (k={len(discs)} seeds):")
        print(f"    Discrimination: {np.mean(discs):.4f} ± {np.std(discs):.4f}")
        print(f"    Fitness:        {np.mean(fitnesses):.4f} ± {np.std(fitnesses):.4f}")
        print(f"    Partner time:   {np.mean(partners):.4f} ± {np.std(partners):.4f}")
        print(f"    Shadow time:    {np.mean(shadows):.4f} ± {np.std(shadows):.4f}")
        print(f"    ED score:       {np.mean(eds):.4f} ± {np.std(eds):.4f}")

    # Spearman correlations
    print(f"\n2. SIZE-DEPENDENT CAPACITY EFFECTS (Spearman correlation):")
    print("-" * 80)
    if len(sizes_all) >= 5:
        rho_d, p_d = spearmanr(sizes_all, disc_all)
        rho_f, p_f = spearmanr(sizes_all, fitness_all)
        rho_p, p_p = spearmanr(sizes_all, partner_all)
        rho_s, p_s = spearmanr(sizes_all, shadow_all)
        rho_ed, p_ed = spearmanr(sizes_all, ed_all)

        print(f"  Discrimination vs Size:     ρ={rho_d:+.4f}, p={p_d:.4f} {'***' if p_d < 0.001 else '**' if p_d < 0.01 else '*' if p_d < 0.05 else 'ns'}")
        print(f"  Fitness vs Size:            ρ={rho_f:+.4f}, p={p_f:.4f} {'***' if p_f < 0.001 else '**' if p_f < 0.01 else '*' if p_f < 0.05 else 'ns'}")
        print(f"  Partner time vs Size:       ρ={rho_p:+.4f}, p={p_p:.4f} {'***' if p_p < 0.001 else '**' if p_p < 0.01 else '*' if p_p < 0.05 else 'ns'}")
        print(f"  Shadow time vs Size:        ρ={rho_s:+.4f}, p={p_s:.4f} {'***' if p_s < 0.001 else '**' if p_s < 0.01 else '*' if p_s < 0.05 else 'ns'}")
        print(f"  ED score vs Size:           ρ={rho_ed:+.4f}, p={p_ed:.4f} {'***' if p_ed < 0.001 else '**' if p_ed < 0.01 else '*' if p_ed < 0.05 else 'ns'}")

        # Small vs Large group tests
        print(f"\n3. SMALL vs LARGE CAPACITY (Mann-Whitney U test):")
        print("-" * 80)
        small_d = [disc_all[i] for i in range(len(sizes_all)) if sizes_all[i] <= 3]
        large_d = [disc_all[i] for i in range(len(sizes_all)) if sizes_all[i] >= 6]
        small_f = [fitness_all[i] for i in range(len(sizes_all)) if sizes_all[i] <= 3]
        large_f = [fitness_all[i] for i in range(len(sizes_all)) if sizes_all[i] >= 6]
        small_ed = [ed_all[i] for i in range(len(sizes_all)) if sizes_all[i] <= 3]
        large_ed = [ed_all[i] for i in range(len(sizes_all)) if sizes_all[i] >= 6]

        if len(small_d) >= 3 and len(large_d) >= 3:
            stat_d, p_mw_d = mannwhitneyu(large_d, small_d, alternative='greater')
            print(f"  Discrimination (Large > Small): U={stat_d:.0f}, p={p_mw_d:.4f} {'*' if p_mw_d < 0.05 else 'ns'}")
            print(f"    Small (n≤3): median={np.median(small_d):.4f}, mean={np.mean(small_d):.4f} ± {np.std(small_d):.4f}")
            print(f"    Large (n≥6): median={np.median(large_d):.4f}, mean={np.mean(large_d):.4f} ± {np.std(large_d):.4f}")

        if len(small_f) >= 3 and len(large_f) >= 3:
            stat_f, p_mw_f = mannwhitneyu(large_f, small_f, alternative='greater')
            print(f"\n  Fitness (Large > Small):       U={stat_f:.0f}, p={p_mw_f:.4f} {'*' if p_mw_f < 0.05 else 'ns'}")
            print(f"    Small (n≤3): median={np.median(small_f):.4f}, mean={np.mean(small_f):.4f} ± {np.std(small_f):.4f}")
            print(f"    Large (n≥6): median={np.median(large_f):.4f}, mean={np.mean(large_f):.4f} ± {np.std(large_f):.4f}")

        if len(small_ed) >= 3 and len(large_ed) >= 3:
            stat_ed, p_mw_ed = mannwhitneyu(large_ed, small_ed, alternative='greater')
            print(f"\n  ED score (Large > Small):      U={stat_ed:.0f}, p={p_mw_ed:.4f} {'*' if p_mw_ed < 0.05 else 'ns'}")
            print(f"    Small (n≤3): median={np.median(small_ed):.4f}, mean={np.mean(small_ed):.4f} ± {np.std(small_ed):.4f}")
            print(f"    Large (n≥6): median={np.median(large_ed):.4f}, mean={np.mean(large_ed):.4f} ± {np.std(large_ed):.4f}")

    # ED score analysis
    print(f"\n4. GHOST CONDITIONS (Embodied Discrimination):")
    print("-" * 80)
    if ed_all:
        print(f"  Mean ED score across all conditions: {np.mean(ed_all):.4f} ± {np.std(ed_all):.4f}")
        print(f"  Correlation: ED score vs Discrimination")
        if len(ed_all) >= 5:
            rho_ed_disc, p_ed_disc = spearmanr(ed_all, disc_all)
            print(f"    ρ={rho_ed_disc:+.4f}, p={p_ed_disc:.4f} {'*' if p_ed_disc < 0.05 else 'ns'}")

    # Save comprehensive results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save full results
    outfile = os.path.join(RESULTS_DIR, f'paper4_10seeds_full_{timestamp}.json')

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        return obj

    with open(outfile, 'w') as f:
        json.dump(convert({'conditions': results, 'meta': {
            'timestamp': timestamp,
            'total_time_sec': total_time,
            'sizes': NETWORK_SIZES,
            'seeds': SEEDS,
            'num_conditions': total,
            'generations': NUM_GENERATIONS,
            'evo_episode': EVO_EPISODE_LENGTH,
            'analysis_episode': ANALYSIS_EPISODE_LENGTH,
            'max_speed': MAX_SPEED,
            'circumference': CIRCUMFERENCE,
            'perception_dist': PERCEPTION_DIST,
            'features': 'enriched_shadows_ghost_behavioral'
        }}), f, indent=2, default=str)
    print(f"\nSaved full results to: {outfile}")

    # Save summary statistics
    summary_file = os.path.join(RESULTS_DIR, f'paper4_10seeds_summary_{timestamp}.json')
    summary = {
        'meta': {
            'timestamp': timestamp,
            'total_time_sec': total_time,
            'total_conditions': total,
            'successful_conditions': len([v for v in results.values() if 'error' not in v]),
        },
        'by_size': {},
        'correlations': {},
        'group_comparisons': {},
    }

    for ns in NETWORK_SIZES:
        subset = {k: v for k, v in results.items() if v.get('num_neurons') == ns and 'error' not in v}
        if subset:
            discs = [v['mean_discrimination'] for v in subset.values()]
            fitnesses = [v['mean_fitness'] for v in subset.values()]
            partners = [v['mean_partner_fraction'] for v in subset.values()]
            shadows = [v['mean_shadow_fraction'] for v in subset.values()]
            eds = [v['mean_ed_score'] for v in subset.values()]

            summary['by_size'][str(ns)] = {
                'discrimination': {
                    'mean': float(np.mean(discs)),
                    'std': float(np.std(discs)),
                    'median': float(np.median(discs)),
                    'min': float(np.min(discs)),
                    'max': float(np.max(discs)),
                },
                'fitness': {
                    'mean': float(np.mean(fitnesses)),
                    'std': float(np.std(fitnesses)),
                },
                'partner_fraction': {
                    'mean': float(np.mean(partners)),
                    'std': float(np.std(partners)),
                },
                'shadow_fraction': {
                    'mean': float(np.mean(shadows)),
                    'std': float(np.std(shadows)),
                },
                'ed_score': {
                    'mean': float(np.mean(eds)),
                    'std': float(np.std(eds)),
                },
            }

    if len(sizes_all) >= 5:
        rho_d, p_d = spearmanr(sizes_all, disc_all)
        rho_f, p_f = spearmanr(sizes_all, fitness_all)
        rho_ed, p_ed = spearmanr(sizes_all, ed_all)

        summary['correlations'] = {
            'discrimination_vs_size': {'rho': float(rho_d), 'p': float(p_d)},
            'fitness_vs_size': {'rho': float(rho_f), 'p': float(p_f)},
            'ed_vs_size': {'rho': float(rho_ed), 'p': float(p_ed)},
        }

        small_d = [disc_all[i] for i in range(len(sizes_all)) if sizes_all[i] <= 3]
        large_d = [disc_all[i] for i in range(len(sizes_all)) if sizes_all[i] >= 6]
        if len(small_d) >= 3 and len(large_d) >= 3:
            stat_d, p_mw_d = mannwhitneyu(large_d, small_d, alternative='greater')
            summary['group_comparisons'] = {
                'small_vs_large_discrimination': {
                    'u_statistic': float(stat_d),
                    'p_value': float(p_mw_d),
                    'small_mean': float(np.mean(small_d)),
                    'large_mean': float(np.mean(large_d)),
                }
            }

    with open(summary_file, 'w') as f:
        json.dump(convert(summary), f, indent=2, default=str)
    print(f"Saved summary to: {summary_file}")

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
