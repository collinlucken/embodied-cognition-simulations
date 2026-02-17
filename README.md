# Embodied Cognition Simulation Framework

A research framework for investigating embodiment, representation, and social cognition through evolutionary robotics simulation. Uses Continuous-Time Recurrent Neural Networks (CTRNNs) controlling minimal agents in simulated microworlds, analyzed with dynamical systems theory and information-theoretic methods.

## Research Program

This framework supports a multi-paper research program examining what causal patterns are revealed by minimal agent simulations — drawing on philosophy of science , embodied cognitive science , and enactivist theory.

### Core Questions

1. **Causal patterns in minimal models.** What kind of knowledge do minimal agent simulations produce? Under what conditions do their results reveal genuine features of the causal landscape rather than modeling artifacts?

2. **Constitutive vs. causal embodiment.** Does the body play a constitutive role in cognition (part of the cognitive system itself) or a merely causal role (input channel to the brain)? Ghost conditions, body substitutions, and Woodwardian interventions operationalize this question.

3. **Representation criteria.** Do evolved neural agents develop internal representations? Formal philosophical criteria from Ramsey, Shea, and Gładziejewski & Miłkowski are operationalized and applied to the same agents, testing whether "representation" is a natural kind.

4. **Social cognition and coordination.** Does genuine social understanding emerge from dynamical coupling between agents, without explicit cooperation pressure? Perceptual crossing experiments test the participatory sense-making hypothesis.

5. **Active inference bridge.** Do evolved CTRNN agents and active inference agents converge on the same causal patterns in identical environments?

## Architecture

### Core Modules

- **`ctrnn.py`** — Continuous-Time Recurrent Neural Networks (Beer 1995). Euler integration, Jacobian computation, Lyapunov exponent estimation, center-crossing sigmoid option.
- **`evolutionary.py`** — Microbial Genetic Algorithm (Harvey 2009), CMA-ES, Novelty Search, and genotype-phenotype decoding with automatic parameter scaling.
- **`microworld.py`** — 2D embodied agents with bilateral sensors and differential drive. Environments: categorical perception (Beer 2003), phototaxis, perceptual crossing (Froese & Di Paolo 2008).
- **`analysis.py`** — Phase portraits, bifurcation analysis, mutual information, transfer entropy, integrated information, embodiment dependence quantification (ghost conditions, perturbation analysis).

### Experiments

- **Paper 2** (`experiments/paper2/`) — Constitutive vs. causal embodiment. Ghost conditions, body substitution, network transfer across phototaxis, categorical perception, and delayed response tasks. Full robustness matrix across network sizes, evolutionary algorithms, and task environments.
- **Paper 3** (`experiments/paper3/`) — Representation criteria. Ramsey's causal role test, Shea's teleosemantic test, Gładziejewski & Miłkowski's information-theoretic criteria. Cross-criteria comparison and philosophical interpretation.
- **Paper 4** (`experiments/paper4/`) — Perceptual crossing. Evolution of agent pairs, coordination analysis, asymmetry perturbations, information-theoretic individuation analysis.

### Testing & Reproducibility

- Unit tests for CTRNN dynamics (analytical solutions, Jacobian verification)
- Master experiment runner with seed control (`run_all_experiments.py`)
- JSON output for downstream analysis

## Quick Start

```bash
# Install dependencies
pip install numpy scipy scikit-learn matplotlib seaborn

# Run all experiments with reproducible seed
python simulation/run_all_experiments.py --all-papers --seed 42

# Run Paper 2 robustness matrix
python simulation/run_all_experiments.py --robustness --seed 42

# Run individual paper
python simulation/run_all_experiments.py --paper 3 --generations 200
```

## Repository Structure

```
├── simulation/
│   ├── ctrnn.py                    # CTRNN implementation
│   ├── evolutionary.py             # GA, CMA-ES, Novelty Search
│   ├── microworld.py               # Agents and environments
│   ├── analysis.py                 # Dynamical + info-theoretic analysis
│   ├── run_all_experiments.py      # Master experiment runner
│   ├── requirements.txt
│   └── experiments/
│       ├── paper2/                 # Constitutive embodiment
│       ├── paper3/                 # Representation criteria
│       └── paper4/                 # Perceptual crossing
├── Paper_1_Methodology/            # Philosophical framework
├── Paper_2_Constitutive_vs_Causal/ # Embodiment experiments
├── Paper_3_Representation_Revisited/
├── Paper_4_Social_Interaction/
├── Papers_5_6_Active_Inference/
└── Literature_Tracker.xlsx
```

## Key References

### Neural Models & Methodology
- Beer, R. D. (1995). On the dynamics of small continuous-time recurrent neural networks. *Adaptive Behavior*, 3(4), 469–509.
- Beer, R. D. (2003). The dynamics of active categorical perception in an evolved model agent. *Adaptive Behavior*, 11(4), 209–243.
- Beer, R. D. (2021). Some historical context for minimal cognition. *Adaptive Behavior*.
- Potochnik, A. (2017). *Idealization and the Aims of Science*. University of Chicago Press.
- Wimsatt, W. C. (2007). *Re-Engineering Philosophy for Limited Beings*. Harvard University Press.

### Embodied Cognition
- Chemero, A. (2009). *Radical Embodied Cognitive Science*. MIT Press.
- Di Paolo, E. A. (2005). Autopoiesis, adaptivity, teleology, agency. *Phenomenology and the Cognitive Sciences*, 4(4), 429–452.
- Di Paolo, E. A. (2025). Sensorimotor incorporation. *Phenomenology and the Cognitive Sciences*.
- Thompson, E. (2007). *Mind in Life*. Harvard University Press.

### Social Cognition
- De Jaegher, H. & Di Paolo, E. A. (2007). Participatory sense-making. *Phenomenology and the Cognitive Sciences*, 6(4), 485–507.
- Froese, T. & Di Paolo, E. A. (2011). The enactive approach. *Pragmatics & Cognition*, 19(1), 1–36.
- Froese, T. et al. (2024). An open-source perceptual crossing device. *PLOS One*.

### Representation
- Ramsey, W. (2007). *Representation Reconsidered*. Cambridge University Press.
- Shea, N. (2018). *Representation in Cognitive Science*. Oxford University Press.
- Gładziejewski, P. & Miłkowski, M. (2017). Structural representations. *Synthese*, 194(12), 4901–4932.

### Evolutionary Algorithms
- Harvey, I. (2009). The microbial genetic algorithm. In *Advances in Artificial Life*, 126–133.

## Related Publications

- Collin Lucken & Tim Elmo Feiten. "Leveraging participatory sense-making and public engagement with science for AI democratization." *Studies in History and Philosophy of Science*, 110, 55–64, 2025. [DOI](https://doi.org/10.1016/j.shpsa.2025.02.003)
- Collin Lucken. *Engineering Progress in Science*. PhD Dissertation, University of Cincinnati, 2025.

## Author

**Collin Lucken**
Hastings Postdoctoral Scholar in AI and Humanity
Bowdoin College
[collin-lucken.com](https://www.collin-lucken.com)

## License

MIT

