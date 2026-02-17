"""
Cross-Task Validation: Perceptual Crossing × Network Capacity

Tests whether Paper 2's capacity-dependence finding (larger networks → higher
embodiment dependence) holds across a fundamentally different task.

Paper 2: Individual phototaxis (single agent, sensorimotor coupling)
Paper 4: Perceptual crossing (agent pairs, emergent coordination)

If coordination quality also scales with network capacity in the perceptual
crossing task, the capacity-dependence finding is cross-task robust.
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.experiments.paper4.perceptual_crossing import PerceptualCrossingExperiment

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../../results/paper4')
NETWORK_SIZES = [2, 3, 4, 5, 6, 8]
SEEDS = [42, 137, 256, 512, 1024]  # 5 seeds per size (30 total conditions)
NUM_GENERATIONS = 300
POPULATION_SIZE = 30


def run_single(net_size, seed, condition_id, total):
    """Run one condition: evolve + analyze coordination."""
    run_id = f"net{net_size}_seed{seed}"
    start = time.time()
    print(f"\n[{condition_id}/{total}] {run_id}: evolving ({NUM_GENERATIONS} gen)...", flush=True)

    try:
        exp = PerceptualCrossingExperiment(
            population_size=POPULATION_SIZE,
            num_generations=NUM_GENERATIONS,
            num_neurons=net_size,
            seed=seed,
        )

        # Phase 1: Evolution
        gen1, gen2, history = exp.phase1_evolution_for_task(
            task="activity",
            episode_length=500,
            verbose=False,
        )
        best_fitness = history[-1] if history else 0.0

        # Phase 2: Coordination analysis
        metrics = exp.phase2_analyze_coordination(episode_length=1000)

        # Phase 3: Perturbation robustness
        perturbations = exp.phase3_asymmetry_perturbation()

        # Phase 4: Individuation
        individuation = exp.phase4_individuation_analysis()

        elapsed = time.time() - start
        print(f"  [{condition_id}/{total}] {run_id}: done in {elapsed:.1f}s | "
              f"perception={metrics.mutual_perception_events}/1000, "
              f"sync={metrics.synchronization_index:.3f}", flush=True)

        return run_id, {
            'num_neurons': net_size,
            'seed': seed,
            'evolution': {
                'best_fitness': best_fitness,
                'generations': NUM_GENERATIONS,
            },
            'coordination': {
                'mutual_perception_events': metrics.mutual_perception_events,
                'mutual_perception_duration_total': metrics.mutual_perception_duration_total,
                'coordination_duration_avg': metrics.coordination_duration_avg,
                'coordination_stability': metrics.coordination_stability,
                'synchronization_index': metrics.synchronization_index,
                'joint_behavior_entropy': metrics.joint_behavior_entropy,
                'phase_coupling': metrics.phase_coupling,
            },
            'perturbations': perturbations,
            'individuation': individuation,
            'timing': {'elapsed_seconds': elapsed},
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"  [{condition_id}/{total}] {run_id}: ERROR - {e} ({elapsed:.1f}s)", flush=True)
        return run_id, {
            'num_neurons': net_size,
            'seed': seed,
            'error': str(e),
            'timing': {'elapsed_seconds': elapsed},
        }


def main():
    print("=" * 70)
    print("CROSS-TASK VALIDATION: PERCEPTUAL CROSSING × NETWORK CAPACITY")
    print("=" * 70)
    print(f"Sizes: {NETWORK_SIZES}")
    print(f"Seeds: {SEEDS}")
    print(f"Total conditions: {len(NETWORK_SIZES) * len(SEEDS)}")
    print(f"Generations: {NUM_GENERATIONS}, Population: {POPULATION_SIZE}")
    print("=" * 70)

    total = len(NETWORK_SIZES) * len(SEEDS)
    results = {}
    cid = 0

    start_total = time.time()
    for ns in NETWORK_SIZES:
        for s in SEEDS:
            cid += 1
            run_id, result = run_single(ns, s, cid, total)
            results[run_id] = result

    elapsed_total = time.time() - start_total
    print(f"\nAll done in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

    # --- Analyze results ---
    from scipy.stats import spearmanr

    print(f"\n{'='*70}")
    print("RESULTS BY NETWORK SIZE")
    print(f"{'='*70}")

    size_data = {}
    for ns in NETWORK_SIZES:
        subset = {k: v for k, v in results.items()
                  if v.get('num_neurons') == ns and 'error' not in v}
        if not subset:
            continue

        perceptions = [v['coordination']['mutual_perception_events'] for v in subset.values()]
        syncs = [v['coordination']['synchronization_index'] for v in subset.values()]
        couplings = [v['coordination']['phase_coupling'] for v in subset.values()]
        entropies = [v['coordination']['joint_behavior_entropy'] for v in subset.values()]
        info_couplings = [v['individuation']['informational_coupling_strength']
                         for v in subset.values()
                         if 'individuation' in v and 'informational_coupling_strength' in v.get('individuation', {})]

        size_data[ns] = {
            'perception_mean': np.mean(perceptions),
            'perception_std': np.std(perceptions),
            'sync_mean': np.mean(syncs),
            'sync_std': np.std(syncs),
            'coupling_mean': np.mean(couplings),
            'entropy_mean': np.mean(entropies),
            'info_coupling_mean': np.mean(info_couplings) if info_couplings else 0.0,
        }

        print(f"\n  n={ns} (n_conditions={len(subset)}):")
        print(f"    Perception events:  {np.mean(perceptions):.1f} ± {np.std(perceptions):.1f} / 1000")
        print(f"    Synchronization:    {np.mean(syncs):.3f} ± {np.std(syncs):.3f}")
        print(f"    Phase coupling:     {np.mean(couplings):.3f}")
        print(f"    Joint entropy:      {np.mean(entropies):.3f}")
        if info_couplings:
            print(f"    Info coupling:      {np.mean(info_couplings):.3f}")

    # Correlations
    print(f"\n{'='*70}")
    print("CORRELATIONS WITH NETWORK SIZE")
    print(f"{'='*70}")

    sizes_all, perceptions_all, syncs_all = [], [], []
    for k, v in results.items():
        if 'error' not in v:
            sizes_all.append(v['num_neurons'])
            perceptions_all.append(v['coordination']['mutual_perception_events'])
            syncs_all.append(v['coordination']['synchronization_index'])

    if len(sizes_all) >= 5:
        rho_p, p_p = spearmanr(sizes_all, perceptions_all)
        rho_s, p_s = spearmanr(sizes_all, syncs_all)
        print(f"  Size vs perception events: rho={rho_p:.3f}, p={p_p:.4f}")
        print(f"  Size vs synchronization:   rho={rho_s:.3f}, p={p_s:.4f}")

        # Compare with Paper 2 finding (rho=0.39 for size vs embodiment dependence)
        print(f"\n  Paper 2 reference: size vs embodiment dependence rho=0.392, p=0.002")
        if p_p < 0.05:
            print(f"  CROSS-TASK VALIDATION: Perception correlation SIGNIFICANT (p={p_p:.4f})")
        else:
            print(f"  Cross-task validation: Perception correlation NOT significant (p={p_p:.4f})")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = os.path.join(RESULTS_DIR, f'cross_task_validation_{timestamp}.json')

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
        'size_aggregates': convert(size_data),
        'correlations': {
            'size_vs_perception': {'rho': float(rho_p), 'p': float(p_p)} if len(sizes_all) >= 5 else {},
            'size_vs_sync': {'rho': float(rho_s), 'p': float(p_s)} if len(sizes_all) >= 5 else {},
        },
        'meta': {
            'timestamp': timestamp,
            'network_sizes': NETWORK_SIZES,
            'seeds': SEEDS,
            'generations': NUM_GENERATIONS,
            'population_size': POPULATION_SIZE,
        },
    }
    with open(outfile, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nSaved to: {outfile}")


if __name__ == "__main__":
    main()
