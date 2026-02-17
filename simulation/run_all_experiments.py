"""
Master script for running all Paper experiments.

This script provides a unified interface for running and comparing results
from Papers 2, 3, and 4 experiments with options for:
- Selecting which papers to run
- Reproducibility via seed control
- Summary statistics across experiments

Usage:
    python run_all_experiments.py --paper 2 --seed 42 --generations 100
    python run_all_experiments.py --all-papers --seed 42
    python run_all_experiments.py --paper 3 --seed 42

References:
    Paper 2: Constitutive vs. Causal Embodiment
    Paper 3: Representation Criteria in Minimal Agents
    Paper 4: Perceptual Crossing and Dynamical Coupling
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import experiment modules
from experiments.paper2.constitutive_vs_causal import (
    ConstitutiveEmbodimentExperiment,
    run_robustness_matrix
)
from experiments.paper3.representation_criteria import RepresentationCriteriaExperiment
from experiments.paper4.perceptual_crossing import PerceptualCrossingExperiment


def run_paper2_experiments(
    seed: Optional[int] = None,
    generations: int = 200,
    population_size: int = 30,
    num_neurons: int = 4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Paper 2 experiments: Constitutive vs. Causal Embodiment.
    
    Args:
        seed: Random seed for reproducibility.
        generations: Number of evolutionary generations.
        population_size: Population size for evolution.
        num_neurons: Number of neurons in CTRNN.
        verbose: If True, print detailed progress.
    
    Returns:
        Dictionary of results from all tasks.
    """
    print("\n" + "=" * 70)
    print("PAPER 2: CONSTITUTIVE VS. CAUSAL EMBODIMENT")
    print("=" * 70)
    
    results = {}
    
    # Test on all three tasks
    for task_type in ['phototaxis', 'categorical_perception', 'delayed_response']:
        print(f"\n>>> Running {task_type.upper()} task...")
        
        experiment = ConstitutiveEmbodimentExperiment(
            num_evolutionary_generations=generations,
            population_size=population_size,
            num_neurons=num_neurons,
            num_sensors=2,
            num_motors=2,
            num_trials_per_evaluation=3,
            trial_duration=500,
            seed=seed
        )
        
        try:
            result = experiment.run_full_experiment(task_type=task_type, verbose=verbose)
            results[task_type] = result
        except Exception as e:
            print(f"ERROR in {task_type}: {str(e)}")
            results[task_type] = {'error': str(e)}
    
    return results


def run_paper3_experiments(
    seed: Optional[int] = None,
    num_neurons: int = 4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Paper 3 experiments: Representation Criteria.
    
    Args:
        seed: Random seed for reproducibility.
        num_neurons: Number of neurons in CTRNN.
        verbose: If True, print detailed progress.
    
    Returns:
        Dictionary of results from representation tests.
    """
    print("\n" + "=" * 70)
    print("PAPER 3: REPRESENTATION CRITERIA IN MINIMAL AGENTS")
    print("=" * 70)
    
    # Create a pre-trained network (use Paper 2 result as example)
    from ctrnn import CTRNN
    from evolutionary import GenotypeDecoder
    
    decoder = GenotypeDecoder(num_neurons=num_neurons)
    
    # Create a random network for testing
    if seed is not None:
        np.random.seed(seed)
    
    random_genotype = np.random.uniform(-1, 1, decoder.genotype_size)
    params = decoder.decode(random_genotype)
    
    network = CTRNN(num_neurons=num_neurons)
    network.weights = params['weights']
    network.biases = params['biases']
    network.tau = params['tau']
    
    # Run representation criteria tests
    experiment = RepresentationCriteriaExperiment(network, environment=None)
    
    try:
        test_results = experiment.run_all_criteria()
        comparison = experiment.compare_criteria()
        
        return {
            'test_results': {k: {
                'criterion': v.criterion_name,
                'hypothesis': v.hypothesis,
                'passed': v.test_passed,
                'evidence_strength': v.evidence_strength,
                'details': v.details
            } for k, v in test_results.items()},
            'comparison': comparison
        }
    except Exception as e:
        print(f"ERROR in representation tests: {str(e)}")
        return {'error': str(e)}


def run_paper4_experiments(
    seed: Optional[int] = None,
    generations: int = 200,
    population_size: int = 30,
    num_neurons: int = 4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Paper 4 experiments: Perceptual Crossing.
    
    Args:
        seed: Random seed for reproducibility.
        generations: Number of evolutionary generations.
        population_size: Population size for evolution.
        num_neurons: Number of neurons per agent.
        verbose: If True, print detailed progress.
    
    Returns:
        Dictionary of results from perceptual crossing analysis.
    """
    print("\n" + "=" * 70)
    print("PAPER 4: PERCEPTUAL CROSSING AND DYNAMICAL COUPLING")
    print("=" * 70)
    
    results = {}
    
    # Test on different tasks
    for task_type in ['longevity', 'activity']:
        print(f"\n>>> Running {task_type.upper()} task...")
        
        experiment = PerceptualCrossingExperiment(
            population_size=population_size,
            num_generations=generations,
            circumference=100.0,
            num_neurons=num_neurons,
            seed=seed
        )
        
        try:
            result = experiment.run_full_experiment(task=task_type, verbose=verbose)
            results[task_type] = result
        except Exception as e:
            print(f"ERROR in {task_type}: {str(e)}")
            results[task_type] = {'error': str(e)}
    
    return results


def run_robustness_tests(
    seed: Optional[int] = None,
    generations: int = 100,
    population_size: int = 20
) -> Dict[str, Any]:
    """
    Run the robustness matrix from Paper 2.
    
    Tests embodiment findings across different network sizes, EA types, and tasks.
    
    Args:
        seed: Random seed for reproducibility.
        generations: Generations per experiment.
        population_size: Population size per experiment.
    
    Returns:
        Robustness matrix results.
    """
    print("\n" + "=" * 70)
    print("ROBUSTNESS MATRIX: Testing Embodiment Findings")
    print("=" * 70)
    
    try:
        results = run_robustness_matrix(
            network_sizes=[3, 5, 8],
            ea_types=['microbial_ga'],  # Add 'novelty_search' if Priority 2 implemented
            task_types=['phototaxis', 'categorical_perception'],
            generations=generations,
            population_size=population_size,
            seed=seed
        )
        return results
    except Exception as e:
        print(f"ERROR in robustness matrix: {str(e)}")
        return {'error': str(e)}


def save_results(results: Dict[str, Any], output_dir: Optional[str] = None) -> str:
    """
    Save all results to JSON file.
    
    Args:
        results: Dictionary of all experiment results.
        output_dir: Directory to save results (default: results/).
    
    Returns:
        Path to saved results file.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__),
            '../results'
        )
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'all_experiments_{timestamp}.json')
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
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
    
    results_serializable = convert_for_json(results)
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2, default=str)
    
    return output_file


def print_summary(results: Dict[str, Any]) -> None:
    """
    Print summary of all results.
    
    Args:
        results: Dictionary of all experiment results.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENTAL SUMMARY")
    print("=" * 70)
    
    if 'paper2' in results:
        print("\nPAPER 2: Embodiment Dependence Scores")
        for task, task_results in results['paper2'].items():
            if isinstance(task_results, dict) and 'embodiment_analysis' in task_results:
                scores = task_results['embodiment_analysis']
                print(f"\n  {task.upper()}:")
                print(f"    Constitutive score: {scores.get('constitutive_score', 0):.3f}")
                print(f"    Causal score: {scores.get('causal_score', 0):.3f}")
    
    if 'paper3' in results:
        print("\nPAPER 3: Representation Criteria")
        if 'test_results' in results['paper3']:
            for criterion, result in results['paper3']['test_results'].items():
                status = "PASS" if result.get('passed', False) else "FAIL"
                print(f"  {criterion}: {status} (evidence: {result.get('evidence_strength', 0):.3f})")
    
    if 'paper4' in results:
        print("\nPAPER 4: Coordination Analysis")
        for task, task_results in results['paper4'].items():
            if isinstance(task_results, dict) and 'coordination_metrics' in task_results:
                metrics = task_results['coordination_metrics']
                print(f"\n  {task.upper()}:")
                print(f"    Synchronization: {metrics.get('synchronization_index', 0):.3f}")
                print(f"    Stability: {metrics.get('coordination_stability', 0):.3f}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Run simulation experiments for Papers 2, 3, and 4'
    )
    
    parser.add_argument(
        '--paper',
        type=int,
        choices=[2, 3, 4],
        default=None,
        help='Run only Paper N experiments (default: all papers)'
    )
    
    parser.add_argument(
        '--all-papers',
        action='store_true',
        help='Run all paper experiments'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--generations',
        type=int,
        default=200,
        help='Number of evolutionary generations (default: 200)'
    )
    
    parser.add_argument(
        '--population-size',
        type=int,
        default=30,
        help='Population size for evolution (default: 30)'
    )
    
    parser.add_argument(
        '--num-neurons',
        type=int,
        default=4,
        help='Number of neurons in CTRNN (default: 4)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed progress output'
    )
    
    parser.add_argument(
        '--robustness',
        action='store_true',
        help='Run robustness matrix tests'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # Determine which papers to run
    papers_to_run = []
    if args.all_papers or (args.paper is None and not args.robustness):
        papers_to_run = [2, 3, 4]
    elif args.paper is not None:
        papers_to_run = [args.paper]
    
    all_results = {}
    
    # Run selected papers
    if 2 in papers_to_run:
        all_results['paper2'] = run_paper2_experiments(
            seed=args.seed,
            generations=args.generations,
            population_size=args.population_size,
            num_neurons=args.num_neurons,
            verbose=not args.quiet
        )
    
    if 3 in papers_to_run:
        all_results['paper3'] = run_paper3_experiments(
            seed=args.seed,
            num_neurons=args.num_neurons,
            verbose=not args.quiet
        )
    
    if 4 in papers_to_run:
        all_results['paper4'] = run_paper4_experiments(
            seed=args.seed,
            generations=args.generations,
            population_size=args.population_size,
            num_neurons=args.num_neurons,
            verbose=not args.quiet
        )
    
    # Run robustness matrix if requested
    if args.robustness:
        all_results['robustness_matrix'] = run_robustness_tests(
            seed=args.seed,
            generations=args.generations,
            population_size=args.population_size
        )
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    output_file = save_results(all_results, args.output_dir)
    print(f"\n\nResults saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
