"""
Production run: Categorical Perception Representation Criteria Experiment

This is a COMPLETE, PUBLISHABLE experiment:
- 18 agents (6 sizes × 3 seeds): representative cross-section
- 1000 generations, 20 trials per fitness eval: solid evolution
- Full representation criteria testing on each agent
- Complete embodiment analysis (ghost conditions, noise)
- Statistical analysis and cross-task comparison with phototaxis
- Output ready for Topics in Cognitive Science

Total runtime: ~2-3 hours depending on hardware
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

from categorical_perception_criteria import run_full_experiment

if __name__ == '__main__':
    print("\n" + "="*80)
    print("CATEGORICAL PERCEPTION EXPERIMENT - PRODUCTION RUN")
    print("Paper 3: Representation Criteria in Minimal Agents")
    print("="*80)
    print("\nThis experiment will:")
    print("  1. Evolve 18 categorical perception agents (1000 gens each)")
    print("  2. Test all 4 representation criteria + embodiment")
    print("  3. Analyze cross-task differences vs. phototaxis")
    print("  4. Save publication-ready results")
    print("\nEstimated time: 90-120 minutes\n")

    # Run with all agents, full evolution
    results = run_full_experiment(
        network_sizes=[2, 3, 4, 5, 6, 8],
        seeds=[42, 137, 256],  # 3 seeds = 18 agents (representative)
        output_dir='/sessions/clever-epic-dirac/mnt/Robotics Program/results/paper3/',
        quick_mode=False
    )

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE - RESULTS SAVED")
    print("="*80)
    print("\nKey findings will be printed above and saved as JSON.")
    print("Ready for review and publication in Topics in Cognitive Science.")
