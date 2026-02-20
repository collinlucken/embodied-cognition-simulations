"""
Behavioral Strategy Characterization Analysis for Paper 4
Analyzes embodied cognition & sensorimotor coupling in perceptual crossing

Uses existing 60-condition results to characterize behavioral strategies
without requiring re-evolution.
"""

import json
import numpy as np
from scipy.stats import spearmanr, fisher_exact, chi2_contingency
from datetime import datetime
import os

RESULTS_FILE = "/sessions/clever-epic-dirac/mnt/Robotics Program/results/paper4/paper4_10seeds_full_20260218_211006.json"
OUTPUT_FILE = "/sessions/clever-epic-dirac/mnt/Robotics Program/results/paper4/behavioral_strategy_analysis.json"


def classify_strategy(discrimination, partner_fraction, exploration):
    """
    Classify strategy based on available metrics.

    Returns: strategy_name (str)
    """
    # Threshold values
    high_disc = 0.7
    high_partner = 0.3
    low_partner = 0.15
    high_expl = 0.15
    low_disc = 0.3
    low_expl = 0.08

    # Active Discriminator: high discrimination AND high partner_fraction
    if discrimination > high_disc and partner_fraction > high_partner:
        return "Active Discriminator"

    # Passive Discriminator: high discrimination but low partner_fraction
    if discrimination > high_disc and partner_fraction <= low_partner:
        return "Passive Discriminator"

    # Indiscriminate Explorer: high exploration, low discrimination
    if exploration > high_expl and discrimination <= low_disc:
        return "Indiscriminate Explorer"

    # Partner-Fixated: very high partner_fraction but moderate discrimination
    if partner_fraction > high_partner and discrimination <= high_disc:
        return "Partner-Fixated"

    # Inactive: low everything
    if discrimination <= low_disc and partner_fraction <= low_partner and exploration <= low_expl:
        return "Inactive"

    # Default fallback
    return "Mixed"


def compute_ed_score(embodied_metrics, ghost_metrics):
    """
    Compute ED (Embodied Difference) score.
    ED = mean(discrimination_embodied) - mean(discrimination_ghost)

    Positive ED indicates dependence on sensorimotor coupling.

    Ghost metrics are in a different format (discrimination_normal, discrimination_ghost1, etc.)
    """
    embodied_disc = np.mean([e['discrimination'] for e in embodied_metrics])

    # Ghost episodes have different structure
    ghost_disc_values = []
    for e in ghost_metrics:
        if 'ed_score' in e:
            # This is already an ED score, extract the ghost component
            # ed_score = discrimination_normal - mean(ghost discriminations)
            # We want just the ghost discrimination
            if 'discrimination_ghost1' in e:
                ghost_disc_values.append(e['discrimination_ghost1'])
            if 'discrimination_ghost2' in e:
                ghost_disc_values.append(e['discrimination_ghost2'])

    if ghost_disc_values:
        ghost_disc = np.mean(ghost_disc_values)
    else:
        return None

    return float(embodied_disc - ghost_disc)


def main():
    print("=" * 80)
    print("BEHAVIORAL STRATEGY CHARACTERIZATION ANALYSIS - PAPER 4")
    print("=" * 80)

    # Load data
    print("\nLoading results from:", RESULTS_FILE)
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    conditions = data['conditions']
    print(f"Loaded {len(conditions)} conditions")

    # Parse all conditions
    condition_data = []
    for cond_id, cond_info in conditions.items():
        if 'error' in cond_info:
            continue

        num_neurons = cond_info['num_neurons']
        seed = cond_info['seed']

        # Embodied episodes
        episodes = cond_info.get('episodes', [])
        ghost_episodes = cond_info.get('ghost_episodes', [])

        if not episodes:
            continue

        # Compute metrics
        embodied_disc = np.mean([e['discrimination'] for e in episodes])
        embodied_partner = np.mean([e['partner_fraction'] for e in episodes])
        embodied_shadow = np.mean([e['shadow_fraction'] for e in episodes])
        embodied_expl = np.mean([e['exploration'] for e in episodes])
        embodied_fitness = np.mean([e['fitness'] for e in episodes])

        # ED score
        if ghost_episodes:
            ed_score = compute_ed_score(episodes, ghost_episodes)
            # Ghost discrimination from the ghost episodes
            ghost_disc_values = []
            for e in ghost_episodes:
                if 'discrimination_ghost1' in e:
                    ghost_disc_values.append(e['discrimination_ghost1'])
                if 'discrimination_ghost2' in e:
                    ghost_disc_values.append(e['discrimination_ghost2'])
            ghost_disc = np.mean(ghost_disc_values) if ghost_disc_values else None
        else:
            ed_score = None
            ghost_disc = None

        # Strategy classification
        strategy = classify_strategy(embodied_disc, embodied_partner, embodied_expl)

        # Discrimination reliability (correlation between episodes)
        if len(episodes) > 1:
            disc_values = [e['discrimination'] for e in episodes]
            if len(set(disc_values)) > 1:  # If there's variance
                disc_reliability = np.corrcoef(range(len(disc_values)), disc_values)[0, 1]
            else:
                disc_reliability = 1.0  # Perfect reliability if constant
        else:
            disc_reliability = None

        condition_data.append({
            'condition_id': cond_id,
            'num_neurons': num_neurons,
            'seed': seed,
            'strategy': strategy,
            'embodied_discrimination': embodied_disc,
            'embodied_partner_fraction': embodied_partner,
            'embodied_shadow_fraction': embodied_shadow,
            'embodied_exploration': embodied_expl,
            'embodied_fitness': embodied_fitness,
            'ghost_discrimination': ghost_disc,
            'ed_score': ed_score,
            'discrimination_reliability': disc_reliability,
            'num_embodied_episodes': len(episodes),
            'num_ghost_episodes': len(ghost_episodes),
        })

    print(f"Analyzed {len(condition_data)} valid conditions\n")

    # === STRATEGY CLASSIFICATION ANALYSIS ===
    print("=" * 80)
    print("STRATEGY CLASSIFICATION")
    print("=" * 80)

    strategies = [c['strategy'] for c in condition_data]
    strategy_counts = {}
    for s in strategies:
        strategy_counts[s] = strategy_counts.get(s, 0) + 1

    print("\nStrategy Distribution (n=60):")
    for strategy in sorted(strategy_counts.keys()):
        count = strategy_counts[strategy]
        pct = 100 * count / len(condition_data)
        print(f"  {strategy:30s}: {count:3d} ({pct:5.1f}%)")

    # === STRATEGY DISTRIBUTION BY NETWORK SIZE ===
    print("\n" + "=" * 80)
    print("STRATEGY × NETWORK SIZE")
    print("=" * 80)

    network_sizes = sorted(set(c['num_neurons'] for c in condition_data))
    strategy_by_size = {}

    for size in network_sizes:
        size_conditions = [c for c in condition_data if c['num_neurons'] == size]
        size_strategies = [c['strategy'] for c in size_conditions]
        print(f"\nNetwork Size n={size} (N={len(size_conditions)}):")

        size_counts = {}
        for s in size_strategies:
            size_counts[s] = size_counts.get(s, 0) + 1

        for strategy in sorted(size_counts.keys()):
            count = size_counts[strategy]
            pct = 100 * count / len(size_conditions)
            print(f"  {strategy:30s}: {count:2d} ({pct:5.1f}%)")

        strategy_by_size[size] = size_counts

    # Contingency table for Fisher's exact test (simplified 2x2 for Active vs others)
    print("\n" + "-" * 80)
    print("Contingency Table: Active Discriminator vs Others")
    print("-" * 80)

    active_by_size = {}
    contingency_table = []

    for size in network_sizes:
        size_conditions = [c for c in condition_data if c['num_neurons'] == size]
        n_active = sum(1 for c in size_conditions if c['strategy'] == 'Active Discriminator')
        n_other = len(size_conditions) - n_active
        active_by_size[size] = n_active
        contingency_table.append([n_active, n_other])
        print(f"n={size}: Active={n_active:2d}, Other={n_other:2d}")

    # Chi-square test
    if len(contingency_table) > 1:
        ct_array = np.array(contingency_table)
        chi2, p_chi, dof, expected = chi2_contingency(ct_array)
        print(f"\nChi-square test: χ²={chi2:.2f}, p={p_chi:.4f}, dof={dof}")

    # === GHOST CONDITION ANALYSIS ===
    print("\n" + "=" * 80)
    print("GHOST CONDITION ANALYSIS (Embodied vs Ghost)")
    print("=" * 80)

    conditions_with_ghost = [c for c in condition_data if c['ed_score'] is not None]
    print(f"\nConditions with ghost data: {len(conditions_with_ghost)}/60")

    if conditions_with_ghost:
        ed_scores = [c['ed_score'] for c in conditions_with_ghost]
        print(f"\nED Score Distribution (Embodied - Ghost discrimination):")
        print(f"  Mean ED:     {np.mean(ed_scores):7.4f}")
        print(f"  Std ED:      {np.std(ed_scores):7.4f}")
        print(f"  Min ED:      {np.min(ed_scores):7.4f}")
        print(f"  Max ED:      {np.max(ed_scores):7.4f}")

        # Positive ED indicates embodiment advantage
        n_positive = sum(1 for e in ed_scores if e > 0)
        print(f"\nConditions with embodiment advantage (ED > 0): {n_positive}/{len(ed_scores)} ({100*n_positive/len(ed_scores):.1f}%)")

        # ED score by strategy
        print("\nED Score by Strategy:")
        strategies_with_ghost = set(c['strategy'] for c in conditions_with_ghost)
        for strategy in sorted(strategies_with_ghost):
            strat_conditions = [c for c in conditions_with_ghost if c['strategy'] == strategy]
            strat_ed = [c['ed_score'] for c in strat_conditions]
            print(f"  {strategy:30s}: {np.mean(strat_ed):7.4f} ± {np.std(strat_ed):6.4f} (n={len(strat_ed)})")

    # === STATISTICAL ANALYSIS ===
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    # Discrimination by network size
    print("\nDiscrimination vs Network Size (Spearman correlation):")
    sizes_all = [c['num_neurons'] for c in condition_data]
    disc_all = [c['embodied_discrimination'] for c in condition_data]

    rho_disc, p_disc = spearmanr(sizes_all, disc_all)
    print(f"  ρ = {rho_disc:6.3f}, p = {p_disc:.4f} {'*' if p_disc < 0.05 else ''}")

    # Partner fraction by network size
    partner_all = [c['embodied_partner_fraction'] for c in condition_data]
    rho_partner, p_partner = spearmanr(sizes_all, partner_all)
    print(f"\nPartner Fraction vs Network Size (Spearman correlation):")
    print(f"  ρ = {rho_partner:6.3f}, p = {p_partner:.4f} {'*' if p_partner < 0.05 else ''}")

    # Exploration by network size
    expl_all = [c['embodied_exploration'] for c in condition_data]
    rho_expl, p_expl = spearmanr(sizes_all, expl_all)
    print(f"\nExploration vs Network Size (Spearman correlation):")
    print(f"  ρ = {rho_expl:6.3f}, p = {p_expl:.4f} {'*' if p_expl < 0.05 else ''}")

    # ED score vs discrimination (if available)
    if conditions_with_ghost:
        ed_all = [c['ed_score'] for c in conditions_with_ghost]
        disc_ghost_conditions = [c['embodied_discrimination'] for c in conditions_with_ghost]

        rho_ed_disc, p_ed_disc = spearmanr(ed_all, disc_ghost_conditions)
        print(f"\nED Score vs Embodied Discrimination (Spearman correlation):")
        print(f"  ρ = {rho_ed_disc:6.3f}, p = {p_ed_disc:.4f} {'*' if p_ed_disc < 0.05 else ''}")

        # Discrimination reliability
        disc_rel_all = [c['discrimination_reliability'] for c in condition_data if c['discrimination_reliability'] is not None]
        if disc_rel_all:
            print(f"\nDiscrimination Reliability (correlation within episodes):")
            print(f"  Mean: {np.mean(disc_rel_all):6.3f}")
            print(f"  Std:  {np.std(disc_rel_all):6.3f}")
            print(f"  Min:  {np.min(disc_rel_all):6.3f}")
            print(f"  Max:  {np.max(disc_rel_all):6.3f}")

    # === SUMMARY STATISTICS ===
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print("\nEmbodied Discrimination:")
    print(f"  Mean:  {np.mean(disc_all):.4f}")
    print(f"  Std:   {np.std(disc_all):.4f}")
    print(f"  Range: [{np.min(disc_all):.4f}, {np.max(disc_all):.4f}]")

    print("\nEmbodied Partner Fraction:")
    print(f"  Mean:  {np.mean(partner_all):.4f}")
    print(f"  Std:   {np.std(partner_all):.4f}")
    print(f"  Range: [{np.min(partner_all):.4f}, {np.max(partner_all):.4f}]")

    print("\nEmbodied Exploration:")
    print(f"  Mean:  {np.mean(expl_all):.4f}")
    print(f"  Std:   {np.std(expl_all):.4f}")
    print(f"  Range: [{np.min(expl_all):.4f}, {np.max(expl_all):.4f}]")

    # === PREPARE OUTPUT ===
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'source_file': RESULTS_FILE,
            'num_conditions': len(condition_data),
            'network_sizes': network_sizes,
        },
        'strategy_distribution': strategy_counts,
        'strategy_by_size': {str(k): v for k, v in strategy_by_size.items()},
        'active_discriminator_by_size': {str(k): v for k, v in active_by_size.items()},
        'statistics': {
            'discrimination': {
                'mean': float(np.mean(disc_all)),
                'std': float(np.std(disc_all)),
                'min': float(np.min(disc_all)),
                'max': float(np.max(disc_all)),
                'spearman_with_size': {
                    'rho': float(rho_disc),
                    'p': float(p_disc),
                }
            },
            'partner_fraction': {
                'mean': float(np.mean(partner_all)),
                'std': float(np.std(partner_all)),
                'min': float(np.min(partner_all)),
                'max': float(np.max(partner_all)),
                'spearman_with_size': {
                    'rho': float(rho_partner),
                    'p': float(p_partner),
                }
            },
            'exploration': {
                'mean': float(np.mean(expl_all)),
                'std': float(np.std(expl_all)),
                'min': float(np.min(expl_all)),
                'max': float(np.max(expl_all)),
                'spearman_with_size': {
                    'rho': float(rho_expl),
                    'p': float(p_expl),
                }
            },
        },
        'condition_data': condition_data,
    }

    # Add ghost analysis if available
    if conditions_with_ghost:
        output['ghost_analysis'] = {
            'num_conditions_with_ghost': len(conditions_with_ghost),
            'ed_score_distribution': {
                'mean': float(np.mean(ed_scores)),
                'std': float(np.std(ed_scores)),
                'min': float(np.min(ed_scores)),
                'max': float(np.max(ed_scores)),
            },
            'ed_score_by_strategy': {},
            'ed_vs_discrimination': {
                'spearman_rho': float(rho_ed_disc),
                'spearman_p': float(p_ed_disc),
            }
        }

        for strategy in sorted(strategies_with_ghost):
            strat_conditions = [c for c in conditions_with_ghost if c['strategy'] == strategy]
            strat_ed = [c['ed_score'] for c in strat_conditions]
            output['ghost_analysis']['ed_score_by_strategy'][strategy] = {
                'mean': float(np.mean(strat_ed)),
                'std': float(np.std(strat_ed)),
                'n': len(strat_ed),
            }

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
