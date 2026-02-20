#!/usr/bin/env python3
"""
Regenerate Appendix Tables from Actual Simulation Data

This script loads divergence data from the phase_a simulation results and
generates LaTeX tables for the appendix showing raw uncapped divergence values
across network sizes and seeds.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Data file paths
DATA_DIR = Path("/sessions/lucid-beautiful-babbage/mnt/Robotics Program/results/paper2")
PRIMARY_DATA = DATA_DIR / "phase_a_10seeds_20260216_224044.json"
ALTERNATIVE_DATA = DATA_DIR / "phase_a_10seeds_with_all_genotypes.json"

# Configuration
NETWORK_SIZES = [2, 3, 4, 5, 6, 8]
SEEDS = [42, 137, 256, 512, 1024, 2048, 3141, 4096, 5555, 7777]
GHOST_CONDITIONS = ['ghost_frozen_body', 'ghost_disconnected', 'ghost_counterfactual']
DIVERGENCE_METRIC = 'neural_divergence'  # Can be: neural_divergence, output_divergence, max_divergence, divergence_at_end


def load_data(data_path: Path) -> Dict:
    """Load the simulation data from JSON file."""
    with open(data_path, 'r') as f:
        return json.load(f)


def extract_divergence_values(data: Dict, network_size: int, seed: int) -> Dict[str, float]:
    """
    Extract raw divergence values for a given network size and seed.

    Returns a dict with the average divergence across the three ghost conditions.
    """
    key = f"net{network_size}_seed{seed}"

    if key not in data['conditions']:
        return None

    entry = data['conditions'][key]

    # Collect divergence values from all three ghost conditions
    divergences = []
    for ghost_condition in GHOST_CONDITIONS:
        if ghost_condition in entry:
            ghost_data = entry[ghost_condition]
            if DIVERGENCE_METRIC in ghost_data:
                divergences.append(ghost_data[DIVERGENCE_METRIC])

    if not divergences:
        return None

    return {
        'values': divergences,
        'mean': np.mean(divergences),
        'std': np.std(divergences),
        'min': np.min(divergences),
        'max': np.max(divergences)
    }


def generate_table_data(data: Dict) -> Dict[int, Dict[int, Dict]]:
    """
    Generate table data organized by network size and seed.

    Returns: {network_size: {seed: {'mean': float, 'std': float, ...}}}
    """
    table_data = {}

    for net_size in NETWORK_SIZES:
        table_data[net_size] = {}
        for seed in SEEDS:
            divergence_info = extract_divergence_values(data, net_size, seed)
            if divergence_info:
                table_data[net_size][seed] = divergence_info
            else:
                print(f"Warning: No data for net{net_size}_seed{seed}")

    return table_data


def format_table_latex_by_network(table_data: Dict) -> str:
    """
    Generate LaTeX table with one table per network size (seeds across columns).
    Matches the format: Seeds are columns, each network size gets its own table.
    """
    latex_output = []
    latex_output.append("% Regenerated Appendix Tables - Divergence Analysis")
    latex_output.append("% Generated from actual simulation data")
    latex_output.append("")

    for net_size in NETWORK_SIZES:
        latex_output.append(f"% Network Size: {net_size} neurons")
        latex_output.append(f"\\begin{{table}}[htbp]")
        latex_output.append(f"\\centering")
        latex_output.append(f"\\caption{{Raw uncapped neural divergence for network size N={net_size} across all seeds}}")
        latex_output.append(f"\\label{{tab:divergence_n{net_size}}}")
        latex_output.append(f"\\begin{{tabular}}{{c{'r' * len(SEEDS)}}}")
        latex_output.append(f"\\toprule")

        # Header with seeds
        seed_header = " & ".join([f"Seed {s}" for s in SEEDS])
        latex_output.append(f"Network Size & {seed_header} \\\\")
        latex_output.append(f"\\midrule")

        # Data row for this network size
        row_values = [f"N={net_size}"]
        for seed in SEEDS:
            if seed in table_data[net_size]:
                mean_val = table_data[net_size][seed]['mean']
                std_val = table_data[net_size][seed]['std']
                # Format as mean ± std
                row_values.append(f"${mean_val:.4f} \\pm {std_val:.4f}$")
            else:
                row_values.append("---")

        latex_output.append(" & ".join(row_values) + " \\\\")
        latex_output.append(f"\\bottomrule")
        latex_output.append(f"\\end{{tabular}}")
        latex_output.append(f"\\end{{table}}")
        latex_output.append("")

    return "\n".join(latex_output)


def format_table_latex_comprehensive(table_data: Dict) -> str:
    """
    Generate a comprehensive LaTeX table with network sizes as rows and seeds as columns.
    """
    latex_output = []
    latex_output.append("% Comprehensive Appendix Table - Raw Uncapped Neural Divergence")
    latex_output.append("% Generated from actual simulation data")
    latex_output.append("")
    latex_output.append("\\begin{table*}[htbp]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{Raw uncapped neural divergence (mean ± std) across all network sizes and seeds}")
    latex_output.append("\\label{tab:divergence_comprehensive}")

    # Build table with network sizes as rows and seeds as columns
    col_spec = "c" + "r" * len(SEEDS)
    latex_output.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_output.append("\\toprule")

    # Header with seeds
    seed_header = " & ".join([f"Seed {s}" for s in SEEDS])
    latex_output.append(f"Network & {seed_header} \\\\")
    latex_output.append("Size &" + " & ".join([""] * len(SEEDS)) + " \\\\")
    latex_output.append("\\midrule")

    # Data rows for each network size
    for net_size in NETWORK_SIZES:
        row_values = [f"N={net_size}"]
        for seed in SEEDS:
            if seed in table_data[net_size]:
                mean_val = table_data[net_size][seed]['mean']
                std_val = table_data[net_size][seed]['std']
                row_values.append(f"${mean_val:.4f} \\pm {std_val:.4f}$")
            else:
                row_values.append("---")

        latex_output.append(" & ".join(row_values) + " \\\\")

    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table*}")

    return "\n".join(latex_output)


def format_table_plain_text(table_data: Dict) -> str:
    """
    Generate a plain text version for easy viewing.
    """
    output = []
    output.append("=" * 120)
    output.append(f"RAW UNCAPPED NEURAL DIVERGENCE DATA")
    output.append(f"Metric: {DIVERGENCE_METRIC}")
    output.append(f"Organized by Network Size (rows) and Seed (columns)")
    output.append("=" * 120)
    output.append("")

    # Header with seeds
    header = f"{'Network':<10}" + "".join([f"{s:>14}" for s in SEEDS])
    output.append(header)
    output.append("-" * 120)

    # Data rows
    for net_size in NETWORK_SIZES:
        row = f"N={net_size:<7}"
        for seed in SEEDS:
            if seed in table_data[net_size]:
                mean_val = table_data[net_size][seed]['mean']
                std_val = table_data[net_size][seed]['std']
                row += f"{mean_val:7.4f}±{std_val:.3f}  "
            else:
                row += f"{'---':>14}"
        output.append(row)

    output.append("=" * 120)

    # Statistics summary
    output.append("\nSUMMARY STATISTICS")
    output.append("-" * 120)
    for net_size in NETWORK_SIZES:
        means = [table_data[net_size][s]['mean'] for s in SEEDS if s in table_data[net_size]]
        if means:
            output.append(f"N={net_size}: Mean divergence = {np.mean(means):.4f} ± {np.std(means):.4f} "
                         f"(range: {np.min(means):.4f} to {np.max(means):.4f})")

    output.append("=" * 120)

    return "\n".join(output)


def compare_data_files() -> None:
    """Compare the two available data files."""
    print("=" * 80)
    print("DATA FILE COMPARISON")
    print("=" * 80)

    primary_data = load_data(PRIMARY_DATA)
    alt_data = load_data(ALTERNATIVE_DATA)

    primary_count = len(primary_data['conditions'])
    alt_count = len(alt_data['conditions'])

    print(f"\nPrimary file ({PRIMARY_DATA.name}):")
    print(f"  - Number of conditions: {primary_count}")

    print(f"\nAlternative file ({ALTERNATIVE_DATA.name}):")
    print(f"  - Number of conditions: {alt_count}")

    print(f"\nAssessment:")
    if alt_count > primary_count:
        print(f"  ✓ Alternative file has MORE data ({alt_count} vs {primary_count})")
        print(f"    Recommendation: Use alternative file for more comprehensive results")
    else:
        print(f"  ✓ Primary file has {primary_count} conditions")

    # Check data integrity for a sample
    sample_key = "net2_seed42"
    print(f"\nData integrity check for {sample_key}:")

    for ghost_cond in GHOST_CONDITIONS:
        primary_val = primary_data['conditions'][sample_key][ghost_cond].get(DIVERGENCE_METRIC, 'N/A')
        alt_val = alt_data['conditions'][sample_key][ghost_cond].get(DIVERGENCE_METRIC, 'N/A')
        match = "✓" if primary_val == alt_val else "✗"
        print(f"  {ghost_cond}: {match} Primary={primary_val:.6f}, Alt={alt_val:.6f}")

    print("\n" + "=" * 80)


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("APPENDIX TABLE REGENERATION")
    print("=" * 80)

    # Compare data files
    compare_data_files()

    # Load primary data
    print(f"\nLoading data from {PRIMARY_DATA.name}...")
    data = load_data(PRIMARY_DATA)

    # Generate table data
    print("Extracting divergence values...")
    table_data = generate_table_data(data)

    # Check completeness
    total_expected = len(NETWORK_SIZES) * len(SEEDS)
    total_found = sum(len(seeds) for seeds in table_data.values())
    print(f"Data extraction complete: {total_found}/{total_expected} entries found")

    # Generate output formats
    print("\n" + "=" * 80)
    print("PLAIN TEXT TABLE")
    print("=" * 80)
    plain_text = format_table_plain_text(table_data)
    print(plain_text)

    print("\n" + "=" * 80)
    print("LATEX TABLE (COMPREHENSIVE FORMAT)")
    print("=" * 80)
    latex_comprehensive = format_table_latex_comprehensive(table_data)
    print(latex_comprehensive)

    print("\n" + "=" * 80)
    print("LATEX TABLES (BY NETWORK SIZE)")
    print("=" * 80)
    latex_by_network = format_table_latex_by_network(table_data)
    print(latex_by_network)

    # Save outputs to files
    output_dir = Path(__file__).parent.resolve()

    plain_output_file = output_dir / "appendix_tables_plaintext.txt"
    latex_comprehensive_file = output_dir / "appendix_table_comprehensive.tex"
    latex_by_network_file = output_dir / "appendix_tables_by_network.tex"

    print(f"\nSaving outputs to: {output_dir}")

    with open(str(plain_output_file), 'w') as f:
        f.write(plain_text)
    print(f"✓ Plain text table saved to: {plain_output_file}")

    with open(str(latex_comprehensive_file), 'w') as f:
        f.write(latex_comprehensive)
    print(f"✓ Comprehensive LaTeX table saved to: {latex_comprehensive_file}")

    with open(str(latex_by_network_file), 'w') as f:
        f.write(latex_by_network)
    print(f"✓ LaTeX tables by network saved to: {latex_by_network_file}")

    print("\n" + "=" * 80)
    print("REGENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
