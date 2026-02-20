"""
Quick test of categorical perception experiment components.
Tests just 2 agents (2 sizes × 1 seed) to verify functionality.
"""

import sys
import os
from typing import List
import numpy as np
from datetime import datetime
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from categorical_perception_criteria import (
    evolve_categorical_perception_agent,
    test_ramsey_criterion,
    test_shea_criterion,
    test_gm_mutual_information,
    test_gm_transfer_entropy,
    test_ghost_condition,
    test_noise_injection,
    AgentResults
)

from simulation.microworld import CategoricalPerceptionEnv, Agent
from dataclasses import asdict

def run_minimal_test():
    """Run a quick test with 2 agents."""

    print("\n" + "="*80)
    print("MINIMAL TEST: Categorical Perception Criteria (2 agents)")
    print("="*80)

    results = []

    for size in [2, 4]:
        for seed in [42]:
            run_id = f"net{size}_seed{seed}"
            print(f"\n[{run_id}]")

            try:
                # EVOLUTION
                print("  1. Evolving network (100 generations)...", end=' ', flush=True)
                network, fitness, _ = evolve_categorical_perception_agent(
                    num_neurons=size,
                    seed=seed,
                    population_size=20,
                    generations=100,
                    num_trials=5,
                    verbose=False
                )
                print(f"fitness={fitness:.4f}")

                # Environment for testing
                env = CategoricalPerceptionEnv()
                agent = Agent()
                env.set_agent(agent)

                # TEST CRITERIA
                print("  2. Testing criteria...")
                ramsey = test_ramsey_criterion(network, env, num_trials=5)
                shea = test_shea_criterion(network, env, num_trials=5)
                gm_mi = test_gm_mutual_information(network, env, num_trials=10)
                gm_te = test_gm_transfer_entropy(network, env, num_trials=10)

                # EMBODIMENT
                print("  3. Testing embodiment...")
                ghost_result, decoupling = test_ghost_condition(network, env, num_trials=3)
                noise_results = test_noise_injection(network, env, [0.1, 0.3], num_trials=1)

                # COMPILE
                num_passed = sum([
                    ramsey.passed,
                    shea.passed,
                    gm_mi.passed,
                    gm_te.passed,
                    decoupling.passed
                ])

                mean_score = np.mean([
                    ramsey.score,
                    shea.score,
                    gm_mi.score,
                    gm_te.score,
                    decoupling.score
                ])

                result = AgentResults(
                    run_id=run_id,
                    num_neurons=size,
                    seed=seed,
                    evolved_fitness=float(fitness),
                    evolution_generations=100,
                    ramsey=ramsey,
                    shea=shea,
                    gm_mutual_info=gm_mi,
                    gm_transfer_entropy=gm_te,
                    decoupling=decoupling,
                    ghost_condition=ghost_result,
                    noise_invariance=noise_results,
                    num_criteria_passed=num_passed,
                    mean_criterion_score=float(mean_score)
                )

                results.append(result)

                print(f"  RESULT: {num_passed}/5 criteria passed, mean={mean_score:.3f}")

            except Exception as e:
                print(f"  ERROR: {str(e)}")
                import traceback
                traceback.print_exc()

    # Save results
    output = {
        'meta': {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'test_type': 'minimal_test',
            'n_agents': len(results),
            'notes': 'Quick validation test'
        },
        'conditions': [asdict(r) for r in results]
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'/sessions/clever-epic-dirac/mnt/Robotics Program/results/paper3/categorical_perception_test_{timestamp}.json'

    def convert_to_serializable(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    output_serializable = convert_to_serializable(output)

    with open(output_path, 'w') as f:
        json.dump(output_serializable, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for result in results:
        print(f"\n{result.run_id}:")
        print(f"  Fitness: {result.evolved_fitness:.4f}")
        print(f"  Criteria passed: {result.num_criteria_passed}/5")
        print(f"  Mean criterion score: {result.mean_criterion_score:.3f}")
        print(f"  Embodiment dependence: {result.ghost_condition.ed_score:.3f}")

if __name__ == '__main__':
    run_minimal_test()
