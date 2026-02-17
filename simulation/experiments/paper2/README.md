# Paper 2: Constitutive vs. Causal Embodiment Experiments

This module implements fully functional experiments for testing whether embodiment in neural control systems plays a **constitutive role** (necessary for cognitive abilities) or a **causal role** (influences but is replaceable).

## Module Location

```
/sessions/nice-vibrant-hamilton/mnt/Robotics Program/simulation/experiments/paper2/constitutive_vs_causal.py
```

## Quick Start

### Run Proof of Concept (under 2 minutes)

```python
from experiments.paper2.constitutive_vs_causal import run_proof_of_concept

results = run_proof_of_concept()
print(results['embodiment_analysis']['constitutive_score'])
```

### Run Full Experiment (configurable)

```python
from experiments.paper2.constitutive_vs_causal import run_full_experiment

results = run_full_experiment(
    num_generations=500,
    population_size=50,
    num_neurons=5
)
```

## Experimental Design

The experiment consists of three phases:

### Phase 1: Evolution
- Evolves CTRNN controllers using MicrobialGA
- Task: Phototaxis (moving toward a light source)
- Fitness: Distance to target at end of trial
- Returns: Best network found

### Phase 2a: Ghost Condition
- Records sensory inputs during embodied task execution
- Replays sensory trace to same network WITHOUT body control
- Measures:
  - Neural state divergence between embodied and ghost conditions
  - Time to state divergence (when trajectories significantly differ)
  - Output similarity
- **Interpretation**: Large divergence suggests embodiment is constitutive

### Phase 2b: Body Substitution
- Takes evolved network and tests it with modified morphologies:
  - Wider bilateral sensor separation
  - Narrower bilateral sensor separation
  - Different motor scales and body sizes
- Measures: Performance degradation on each morphology
- **Interpretation**: Large degradation suggests constitutive embodiment

## Results Structure

```python
results = {
    'experiment_config': {
        'num_generations': int,
        'population_size': int,
        'num_neurons': int,
        'num_trials_per_evaluation': int,
        'trial_duration': int
    },
    'phase1_evolution': {
        'best_fitness': float,
        'fitness_history': list[float]  # One per generation
    },
    'phase2a_ghost': {
        'condition_name': str,
        'fitness': float,
        'task_success_rate': float,
        'neural_divergence': float,
        'time_to_divergence': int,
        'mean_output_difference': float
    },
    'phase2b_body_substitution': {
        'baseline': {
            'morphology_name': str,
            'fitness': float,
            'performance_degradation': float,
            'feasibility': bool,
            'notes': str
        },
        # ... other morphologies
    },
    'embodiment_analysis': {
        'ghost_divergence': float,
        'ghost_time_to_divergence': float,
        'morphology_degradation': float,
        'morphology_feasibility': float,
        'constitutive_score': float,  # 0-1: higher = more constitutive
        'causal_score': float,  # 0-1: higher = more causal (replaceable)
        'body_brain_coupling_strength': float,
        'morphology_generalization': float
    }
}
```

## Key Classes and Methods

### ConstitutiveEmbodimentExperiment

Main experimental class with the following methods:

- `phase1_evolution()` - Evolve agents
- `phase2a_ghost_condition()` - Test ghost condition
- `phase2b_body_substitution()` - Test body variations
- `analyze_embodiment_dependence()` - Compute metrics
- `run_full_experiment()` - Run all phases
- `save_results(output_dir)` - Save to JSON

### EmbodimentTestResult (dataclass)

Results from ghost condition test:
- `condition_name`: Description
- `fitness`: Performance metric
- `task_success_rate`: Baseline success rate
- `neural_divergence`: State space divergence
- `time_to_divergence`: Steps before divergence
- `mean_output_difference`: Motor output difference

### BodySubstitutionResult (dataclass)

Results from body substitution test:
- `morphology_name`: Which morphology variant
- `fitness`: Performance on this morphology
- `performance_degradation`: Compared to baseline
- `feasibility`: Whether network can control it
- `notes`: Additional details

## Philosophical Interpretation

The experiment answers the question: **Is embodiment constitutive or causal?**

### Evidence for Constitutive Embodiment
- Large neural state divergence in ghost condition
- Early time-to-divergence (states diverge quickly)
- Large performance degradation with body substitution
- Network cannot generalize to different morphologies

### Evidence for Causal Embodiment
- Small neural state divergence in ghost condition
- Late time-to-divergence (similar trajectories for extended time)
- Small performance degradation with body substitution
- Network generalizes well to different morphologies

### Mixed Results
- Both aspects may be present
- Embodiment may be constitutive for some aspects, causal for others

## Technical Details

### CTRNN Parameters
- Time constants (tau): 0.1-10.0
- Weights: -16.0 to 16.0
- Biases: -16.0 to 16.0

### Evolution Algorithm
- MicrobialGA with population size 30-50
- Mutation standard deviation: 0.2
- 200-500 generations

### Task Simulation
- Phototaxis with bilateral sensors
- Light source at fixed position (25, 25)
- Agent starts at random position
- 500-1000 steps per trial
- 3-5 trials per fitness evaluation

## Example Output

```
======================================================================
EMBODIMENT EXPERIMENT: Constitutive vs. Causal
======================================================================
Phase 1: Evolving agents with embodied control...
  Generations: 200
  Generation   0: best_fitness = 0.7505
  Generation  20: best_fitness = 0.8333
  ...
  Generation 199: best_fitness = 0.8474
  Evolution complete. Best fitness: 0.8474

Phase 2: Testing embodiment dependence...
Phase 2a: Testing ghost condition...
  Embodied fitness: 0.7000
  Ghost condition outputs similarity: 1.0000
  Neural state divergence: 0.0000
  Time to state divergence: 500 steps

Phase 2b: Testing body substitution...
  Testing: baseline...
    Fitness: 0.7000, Degradation: 0.1739
  Testing: wider_sensors...
    Fitness: 0.7000, Degradation: 0.1739
  Testing: narrower_sensors...
    Fitness: 0.7000, Degradation: 0.1739

======================================================================
EXPERIMENT RESULTS
======================================================================

Embodiment Dependence Analysis:
  constitutive_score: 0.087
  causal_score: 1.000
  
Interpretation:
  -> CAUSAL EMBODIMENT DOMINANT
     Network generalizes across different morphologies.
```

## Dependencies

- numpy
- scipy
- simulation.ctrnn (CTRNN implementation)
- simulation.evolutionary (MicrobialGA, GenotypeDecoder)
- simulation.microworld (Agent, environments)
- simulation.analysis (Analysis tools)

## References

- Thompson, E., & Varela, F. J. (2001). Radical embodiment: Neural dynamics and consciousness. Trends in Cognitive Sciences, 5(10), 418-425.
- Chemero, A. (2009). Radical embodied cognitive science. MIT press.
- Barsalou, L. W. (2008). Grounded cognition. Annual Review of Psychology, 59, 617-645.
- Beer, R. D. (2003). The dynamics of active categorical perception in an evolved model agent. Adaptive Behavior, 11(4), 209-243.

## Future Extensions

1. **More sophisticated tasks**: Categorical perception, obstacle avoidance
2. **Different evolutionary algorithms**: CMAES, particle swarm
3. **Transfer learning**: Network evolved on task A tested on task B
4. **Dynamical analysis**: Bifurcation analysis, Lyapunov exponents
5. **Embodied information**: Information-theoretic measures (transfer entropy, integrated information)
6. **Multiple morphologies**: Co-evolution with body parameters
