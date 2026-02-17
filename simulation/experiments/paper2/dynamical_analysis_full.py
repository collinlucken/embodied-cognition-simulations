"""
Full Dynamical Analysis: All Network Sizes × Multiple Seeds

Addresses the limitation that dynamical analysis was only done for 3 sizes with 1 seed.
Now runs all 6 network sizes × 3 seeds (minimum), using genotypes from the expanded
experiment results if available, or re-evolving if needed.

Computes:
- Participation ratio (effective dimensionality)
- Perturbation sensitivity (growth rate, fraction amplifying)
- Lyapunov spectrum estimates
- Autocorrelation timescale
- Inter-trial trajectory distance

Uses multiprocessing for speed.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from simulation.experiments.paper2.dynamical_analysis import (
    compute_lyapunov_spectrum,
    compute_trajectory_complexity,
    compute_perturbation_sensitivity,
    evolve_network,
)
from simulation.evolutionary import GenotypeDecoder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../../results/paper2')
NETWORK_SIZES = (2, 3, 4, 5, 6, 8)
SEEDS = (42, 137, 256)  # Match first 3 seeds for comparison with expanded Phase A
NUM_WORKERS = min(4, cpu_count())


def load_genotype(net_size, seed):
    """Try to load a genotype from the 10-seed results file."""
    results_path = Path(RESULTS_DIR)
    # Try 10-seed file first, then expanded
    for pattern in ['phase_a_10seeds_*.json', 'phase_a_expanded_*.json']:
        json_files = sorted(results_path.glob(pattern), reverse=True)
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                run_id = f"net{net_size}_seed{seed}"
                condition = data.get('conditions', {}).get(run_id, {})
                genotype = condition.get('evolution', {}).get('best_genotype', None)
                if genotype is not None:
                    return np.array(genotype)
            except Exception:
                continue
    return None


def analyze_single_condition(args):
    """Worker function: run full dynamical analysis for one condition."""
    net_size, seed = args
    run_id = f"net{net_size}_seed{seed}"
    print(f"  Analyzing {run_id}...", flush=True)
    start_time = time.time()

    # Try to load genotype from results
    genotype = load_genotype(net_size, seed)

    decoder = GenotypeDecoder(
        num_neurons=net_size,
        include_gains=False,
        tau_range=(0.5, 5.0),
        weight_range=(-10.0, 10.0),
        bias_range=(-10.0, 10.0),
    )

    if genotype is not None:
        params = decoder.decode(genotype)
        fitness = None  # Not re-evaluated
        print(f"    Loaded genotype from saved results", flush=True)
    else:
        print(f"    No saved genotype, re-evolving (2000 gen)...", flush=True)
        params, fitness = evolve_network(net_size, generations=2000, seed=seed)

    # Compute all dynamical measures
    try:
        lyap = compute_lyapunov_spectrum(params, net_size, trial_duration=500)
    except Exception as e:
        lyap = {'max_lyapunov': 0.0, 'mean_lyapunov': 0.0, 'error': str(e)}

    try:
        complexity = compute_trajectory_complexity(params, net_size)
    except Exception as e:
        complexity = {'participation_ratio': 0.0, 'error': str(e)}

    try:
        perturbation = compute_perturbation_sensitivity(params, net_size)
    except Exception as e:
        perturbation = {'mean_growth_rate': 0.0, 'error': str(e)}

    elapsed = time.time() - start_time
    print(f"  {run_id}: PR={complexity.get('participation_ratio', 0):.3f}, "
          f"growth={perturbation.get('mean_growth_rate', 0):.3f}, "
          f"lyap={lyap.get('max_lyapunov', 0):.3f}, "
          f"time={elapsed:.1f}s", flush=True)

    return run_id, {
        'num_neurons': net_size,
        'seed': seed,
        'genotype_source': 'loaded' if genotype is not None else 'evolved',
        'fitness': float(fitness) if fitness is not None else None,
        'lyapunov': lyap,
        'trajectory_complexity': complexity,
        'perturbation_sensitivity': perturbation,
        'timing': {'elapsed_seconds': elapsed},
    }


def main():
    print("=" * 70)
    print("FULL DYNAMICAL ANALYSIS: ALL SIZES × SEEDS")
    print("=" * 70)
    print(f"Network sizes: {NETWORK_SIZES}")
    print(f"Seeds: {SEEDS}")
    print(f"Workers: {NUM_WORKERS}")
    print("=" * 70)

    tasks = [(ns, s) for ns in NETWORK_SIZES for s in SEEDS]
    total = len(tasks)
    print(f"Total conditions: {total}")

    start_time = time.time()

    results = {}
    with Pool(processes=NUM_WORKERS) as pool:
        outputs = pool.map(analyze_single_condition, tasks)

    for run_id, result in outputs:
        results[run_id] = result

    elapsed = time.time() - start_time
    print(f"\nAll analyses completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Size':>5} {'Seed':>6} {'PR':>8} {'GrowthRate':>12} {'FracAmpl':>10} "
          f"{'MaxLyap':>9} {'Entropy':>8} {'AutoCorr':>9} {'InterTrial':>11}")
    print("-" * 90)

    for ns in NETWORK_SIZES:
        for s in SEEDS:
            r = results.get(f"net{ns}_seed{s}", {})
            tc = r.get('trajectory_complexity', {})
            ps = r.get('perturbation_sensitivity', {})
            ly = r.get('lyapunov', {})
            print(f"{ns:>5} {s:>6} "
                  f"{tc.get('participation_ratio', 0):>8.3f} "
                  f"{ps.get('mean_growth_rate', 0):>12.4f} "
                  f"{ps.get('fraction_amplifying', 0):>10.2f} "
                  f"{ly.get('max_lyapunov', 0):>9.4f} "
                  f"{tc.get('state_entropy', 0):>8.3f} "
                  f"{tc.get('autocorrelation_timescale', 0):>9.1f} "
                  f"{tc.get('inter_trial_distance_mean', 0):>11.4f}")

    # Aggregate by size
    print("\n" + "=" * 70)
    print("AGGREGATE BY NETWORK SIZE")
    print("=" * 70)
    aggregate = {}
    for ns in NETWORK_SIZES:
        prs = [results[f"net{ns}_seed{s}"]['trajectory_complexity'].get('participation_ratio', 0)
               for s in SEEDS if f"net{ns}_seed{s}" in results]
        grs = [results[f"net{ns}_seed{s}"]['perturbation_sensitivity'].get('mean_growth_rate', 0)
               for s in SEEDS if f"net{ns}_seed{s}" in results]
        fas = [results[f"net{ns}_seed{s}"]['perturbation_sensitivity'].get('fraction_amplifying', 0)
               for s in SEEDS if f"net{ns}_seed{s}" in results]
        lyaps = [results[f"net{ns}_seed{s}"]['lyapunov'].get('max_lyapunov', 0)
                 for s in SEEDS if f"net{ns}_seed{s}" in results]

        if prs:
            agg = {
                'participation_ratio_mean': float(np.mean(prs)),
                'participation_ratio_std': float(np.std(prs)),
                'growth_rate_mean': float(np.mean(grs)),
                'growth_rate_std': float(np.std(grs)),
                'fraction_amplifying_mean': float(np.mean(fas)),
                'fraction_amplifying_std': float(np.std(fas)),
                'max_lyapunov_mean': float(np.mean(lyaps)),
                'max_lyapunov_std': float(np.std(lyaps)),
            }
            aggregate[f'net{ns}'] = agg
            print(f"\n  {ns} neurons:")
            print(f"    Participation ratio: {agg['participation_ratio_mean']:.3f} ± {agg['participation_ratio_std']:.3f}")
            print(f"    Growth rate:         {agg['growth_rate_mean']:.4f} ± {agg['growth_rate_std']:.4f}")
            print(f"    Frac amplifying:     {agg['fraction_amplifying_mean']:.2f} ± {agg['fraction_amplifying_std']:.2f}")
            print(f"    Max Lyapunov:        {agg['max_lyapunov_mean']:.4f} ± {agg['max_lyapunov_std']:.4f}")

    # Correlation: participation ratio vs constitutive score (if available)
    # Load constitutive scores from expanded results
    const_scores = {}
    results_path = Path(RESULTS_DIR)
    for pattern in ['phase_a_10seeds_*.json', 'phase_a_expanded_*.json']:
        json_files = sorted(results_path.glob(pattern), reverse=True)
        if json_files:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            for k, v in data.get('conditions', {}).items():
                if 'error' not in v:
                    const_scores[k] = v['scores']['constitutive']
            break

    if const_scores:
        pr_vals = []
        cs_vals = []
        for run_id, r in results.items():
            if run_id in const_scores:
                pr_vals.append(r['trajectory_complexity'].get('participation_ratio', 0))
                cs_vals.append(const_scores[run_id])

        if len(pr_vals) > 2:
            from scipy.stats import spearmanr
            rho, p = spearmanr(pr_vals, cs_vals)
            print(f"\n  Correlation (participation ratio vs constitutive score):")
            print(f"    Spearman: rho={rho:.4f}, p={p:.6f}, n={len(pr_vals)}")

    # Save results
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(RESULTS_DIR, f'dynamical_analysis_full_{timestamp}.json')

    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        return obj

    save_data = {
        'conditions': convert(results),
        'aggregate': convert(aggregate),
        'meta': {
            'network_sizes': list(NETWORK_SIZES),
            'seeds': list(SEEDS),
            'timestamp': timestamp,
        }
    }

    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return results, aggregate


if __name__ == "__main__":
    main()
