# CTRNN-Based Evolutionary Robotics Research Framework

A comprehensive Python simulation scaffold for embodied cognition and evolutionary robotics research, designed for philosophical investigations at the intersection of dynamical systems and minimal agent simulation.

## Overview

This framework provides scientifically rigorous tools for evolving and analyzing Continuous-Time Recurrent Neural Networks (CTRNNs) controlling embodied agents in simulated microworlds. It implements theoretical frameworks from:

- **Dynamical systems neuroscience** (Beer 1995, 2003)
- **Embodied cognition** (Thompson & Varela, Chemero)
- **Perceptual crossing paradigm** (Froese & Di Paolo 2008)
- **Information-theoretic analysis** (Tononi, Schreiber)
- **Philosophical theories of representation** (Ramsey, Shea, Gładziejewski & Miłkowski)

## Core Modules

### 1. `ctrnn.py` - Continuous-Time Recurrent Neural Networks

Implements the CTRNN model from Beer (1995) with full dynamical systems analysis capabilities.

**Key Classes:**
- `CTRNN`: Main network class with Euler integration, Jacobian computation, Lyapunov exponent estimation
- Full support for time constants, connection weights, biases, and gains
- Center-crossing sigmoid option for biological plausibility
- Methods for direct state access and manipulation (for perturbation studies)

**Features:**
- Configurable step size for integration accuracy
- Multiple sigmoid variants (standard and center-crossing)
- Dynamical analysis: Jacobian, eigenvalues, Lyapunov exponents
- Batch processing with `run()` method

**Usage:**
```python
from ctrnn import CTRNN
import numpy as np

network = CTRNN(num_neurons=4, step_size=0.01, center_crossing=True)
output = network.step(external_inputs=np.array([0.5, 0.2, 0.0, 0.0]))
states = network.get_state()
jacobian = network.get_jacobian()
```

### 2. `evolutionary.py` - Evolutionary Optimization

Implements multiple evolutionary algorithms for parameter evolution.

**Key Classes:**
- `MicrobialGA`: Microbial Genetic Algorithm (Harvey 2009) - simple, effective, biologically plausible
- `CMAES`: Covariance Matrix Adaptation Evolution Strategy - sophisticated black-box optimizer
- `GenotypeDecoder`: Maps flat genotype vectors to CTRNN parameters with appropriate scaling

**Features:**
- Population statistics tracking
- Fitness logging and best-solution tracking
- Configurable parameter ranges with automatic scaling (log-scale for time constants, linear for weights)
- Ready for integration with task evaluation functions

**Usage:**
```python
from evolutionary import MicrobialGA, GenotypeDecoder

def fitness_function(genotype):
    # Run simulation and compute fitness
    return fitness_value

decoder = GenotypeDecoder(num_neurons=4)
ga = MicrobialGA(
    genotype_size=decoder.genotype_size,
    fitness_function=fitness_function,
    population_size=50
)

for generation in range(300):
    best_genotype, best_fitness = ga.step()
    print(f"Gen {generation}: {best_fitness:.4f}")
```

### 3. `microworld.py` - 2D Agent-Environment Simulations

Implements minimal but complete agent morphologies and multiple task environments.

**Agent Model:**
- Circular body with configurable radius
- Bilateral sensory system (left/right sensors)
- Differential drive motors (left/right)
- Continuous physics simulation with friction

**Environments:**

1. **CategoricalPerceptionEnv** (Beer 2003)
   - Objects fall from above
   - Agent must catch small objects and avoid large ones
   - Tests categorical perception and decision-making

2. **PhototaxisEnv**
   - Simple light-seeking task
   - Tests basic sensorimotor coordination
   - Useful for quick validation

3. **PerceptualCrossingEnv** (Froese & Di Paolo 2008)
   - Two agents in 1D ring
   - Perception possible only during bilateral coordination
   - Tests emergence of coordination from dynamical coupling

**Usage:**
```python
from microworld import Agent, CategoricalPerceptionEnv
from ctrnn import CTRNN

agent = Agent(radius=1.0, max_speed=1.0)
env = CategoricalPerceptionEnv()
env.set_agent(agent)

network = CTRNN(num_neurons=2)

for t in range(1000):
    sensors = env.get_sensor_readings()
    motors = network.step(sensors)
    agent.set_motor_commands(motors[0], motors[1])
    env.step()

fitness = env.evaluate_fitness()
```

### 4. `analysis.py` - Dynamical Systems and Information-Theoretic Analysis

Tools for understanding evolved networks through dynamical systems and information theory.

**Key Classes:**
- `PhasePortrait`: Generate and visualize neural state trajectories
- `BifurcationAnalyzer`: Detect bifurcations and parameter-dependent behavior
- `InformationAnalyzer`: Compute mutual information, transfer entropy, integrated information
- `EmbodimentAnalyzer`: Ghost conditions, perturbation analysis, causal decomposition

**Features:**
- Phase portrait generation with fixed point detection
- Vector field analysis for understanding flow
- Transfer entropy for directional information analysis
- Lyapunov exponent estimation (chaos detection)
- Embodiment-dependence quantification

**Note:** Requires scipy, scikit-learn, matplotlib, seaborn
```bash
pip install scipy scikit-learn matplotlib seaborn
```

### 5. Experiment Scaffolds

Three comprehensive experiment frameworks organized by research paper:

#### `experiments/paper2/constitutive_vs_causal.py`

Tests whether embodiment plays a **constitutive** role (necessary for cognition) or **causal** role (influences but replaceable).

**Phases:**
1. Evolution with normal embodiment
2. Ghost condition (disembodied playback)
3. Body substitution (different morphologies)
4. Network transfer (task generalization)

**Key Methods:**
- `run_full_experiment()`: Execute complete investigation
- `analyze_embodiment_dependence()`: Quantify role of embodiment

#### `experiments/paper3/representation_criteria.py`

Compares multiple philosophical theories of representation:
- **Ramsey (1997)**: Causal role semantics
- **Shea (2018)**: Teleosemantics (evolutionary function)
- **Gładziejewski & Miłkowski (2017)**: Information-theoretic criteria

**Tests:**
- State causal role
- Spurious correlation resistance
- Mutual information
- Transfer entropy

#### `experiments/paper4/perceptual_crossing.py`

Investigates emergence of coordination and autonomy in coupled agent systems.

**Phases:**
1. Evolution for task without explicit coordination fitness
2. Coordination analysis (do agents naturally perceive each other?)
3. Perturbation robustness (asymmetries, noise, morphology)
4. Individuation analysis (one system or two agents?)
5. Philosophical interpretation

## Installation

### Basic (core functionality only)
```bash
pip install numpy
```

### Full (with analysis tools)
```bash
pip install numpy scipy scikit-learn matplotlib seaborn
```

### Optional (advanced information theory)
```bash
pip install jpype1  # For JIDT library integration
```

## Quick Start Example

```python
import numpy as np
from ctrnn import CTRNN
from evolutionary import MicrobialGA, GenotypeDecoder
from microworld import Agent, CategoricalPerceptionEnv

# Setup
decoder = GenotypeDecoder(num_neurons=4)

def evaluate_agent(genotype):
    """Fitness evaluation function."""
    params = decoder.decode(genotype)
    network = CTRNN(num_neurons=4)
    network.tau = params['tau']
    network.weights = params['weights']
    network.biases = params['biases']
    
    agent = Agent()
    env = CategoricalPerceptionEnv()
    env.set_agent(agent)
    env.reset()
    
    # Run task for 10,000 timesteps
    for _ in range(10000):
        sensors = env.get_sensor_readings()
        motors = network.step(sensors)
        agent.set_motor_commands(motors[0], motors[1])
        env.step()
    
    return env.evaluate_fitness()

# Evolve
ga = MicrobialGA(
    genotype_size=decoder.genotype_size,
    fitness_function=evaluate_agent,
    population_size=50,
    seed=42
)

for generation in range(100):
    best_genotype, best_fitness = ga.step()
    if generation % 10 == 0:
        print(f"Generation {generation}: Fitness = {best_fitness:.4f}")

# Analyze best individual
best_params = decoder.decode(ga.get_best_individual())
print(f"\nEvolved network time constants: {best_params['tau']}")
print(f"Best fitness achieved: {ga.get_best_fitness():.4f}")
```

## Scientific References

### Neural Models
- Beer, R. D. (1995). On the dynamics of small continuous-time recurrent neural networks. *Adaptive Behavior*, 3(4), 469-509.

### Tasks & Environments
- Beer, R. D. (2003). The dynamics of active categorical perception in an evolved model agent. *Adaptive Behavior*, 11(4), 209-243.
- Froese, T., & Di Paolo, E. A. (2008). Emergence of joint action in two robotic agents coupled through kinetic energy transfer. *New Ideas in Psychology*, 26(3), 384-401.

### Evolutionary Algorithms
- Harvey, I. (2009). The microbial genetic algorithm. In *Advances in Artificial Life* (pp. 126-133).
- Beyer, H. G., & Schwefel, H. P. (2002). Evolution strategies – a comprehensive introduction. *Natural Computing*, 1(1), 3-52.

### Information Theory
- Tononi, G., Edelman, G. M., & Sporns, O. (1998). Complexity and coherency: Integrating information in the brain. *Trends in Cognitive Sciences*, 2(12), 474-484.
- Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461.

### Embodied Cognition
- Thompson, E., & Varela, F. J. (2001). Radical embodiment: Neural dynamics and consciousness. *Trends in Cognitive Sciences*, 5(10), 418-425.
- Chemero, A. (2009). *Radical embodied cognitive science*. MIT Press.
- Di Paolo, E. A. (2009). Extended life. *Topoi*, 28(1), 9-21.

### Representation
- Ramsey, W. (1997). *Representing the world: Words, theories, and things*.
- Shea, N. (2018). *Representation in cognitive science*. Oxford University Press.
- Gładziejewski, P., & Miłkowski, M. (2017). Informational semantics, mathematical functions, and computationalism. *Journal of Cognitive Science*, 18(2), 261-313.

## File Structure

```
simulation/
├── ctrnn.py                          # CTRNN core implementation
├── evolutionary.py                    # GA and CMA-ES
├── microworld.py                      # Agents and environments
├── analysis.py                        # Dynamical systems & info theory
├── requirements.txt                   # Dependencies
├── README.md                          # This file
└── experiments/
    ├── __init__.py
    ├── paper2/
    │   ├── __init__.py
    │   └── constitutive_vs_causal.py  # Embodiment tests
    ├── paper3/
    │   ├── __init__.py
    │   └── representation_criteria.py  # Representation tests
    └── paper4/
        ├── __init__.py
        └── perceptual_crossing.py      # Coordination studies
```

## Design Philosophy

This framework embodies several design principles:

1. **Scientific Rigor**: All implementations follow primary literature with detailed citations
2. **Clarity**: Extensive docstrings, type hints, and comments for reproducibility
3. **Modularity**: Clean separation between neural models, evolution, environments, and analysis
4. **Functionality**: Core modules (CTRNN, GA, environments) are immediately usable for research
5. **Extensibility**: Scaffolds for custom experiments and variants
6. **Philosophical Grounding**: Experiment designs motivated by actual philosophical problems

## Notes for Researchers

### For Evolution Studies
- MicrobialGA is recommended for small-to-medium problems (up to ~100 parameters)
- CMAES is better for higher-dimensional optimization (>50 parameters)
- Use `GenotypeDecoder` for automatic parameter scaling

### For Dynamical Analysis
- Phase portraits are most informative for 2-neuron systems
- Bifurcation analysis reveals parameter-dependent qualitative changes
- Jacobian eigenvalues indicate local stability

### For Embodiment Studies
- Use `EmbodimentAnalyzer.ghost_condition()` to test disembodiment
- Compare embodied vs. ghost neural states for divergence metrics
- Mutual information often increases with embodiment

### For Philosophical Work
- Each experiment scaffold includes philosophical interpretation methods
- Cross-reference results with primary literature cited in docstrings
- Consider multiple competing hypotheses when interpreting results

## License

This scaffold is provided for research use.

## Contact & Support

For questions about the implementation, refer to the comprehensive docstrings in each module.
Each function includes:
- Purpose and scientific context
- Arguments and return types
- Example usage patterns
- References to primary literature
