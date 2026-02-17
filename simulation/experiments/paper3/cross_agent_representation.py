"""
Paper 3: Cross-Agent Representation Criteria Analysis

Uses the 42 decoded genotypes from Paper 2 (phase_a_10seeds with saved genotypes)
to test whether representation criteria scores correlate with embodiment dependence.

Key questions:
1. Do high-ED agents show MORE representational content (input-driven computation)?
2. Do high-ED agents show LESS representational content (purely dynamical, non-representational)?
3. Are representation criteria orthogonal to embodiment dependence?

Tests run per agent:
- Ramsey: Causal role test (targeted vs random perturbation of each neuron)
- Shea: Teleosemantic test (response to evolved vs spurious stimuli)
- G&M mutual information: I(stimulus; state) vs baseline
- G&M transfer entropy: TE(stimulus→state→action) mediation
- Additional: Decoupling invariance test (novel for this paper)
  Tests whether internal state structure persists when decoupled from input

This produces the core dataset for Paper 3: a matrix of [42 agents × 5+ criteria scores]
that can be correlated with ED scores, network size, mechanistic type, etc.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.ctrnn import CTRNN
from simulation.evolutionary import GenotypeDecoder
from simulation.analysis import InformationAnalyzer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../../results/paper3')
PAPER2_RESULTS = os.path.join(os.path.dirname(__file__), '../../../results/paper2')
NETWORK_SIZES = (2, 3, 4, 5, 6, 8)
ALL_SEEDS = (42, 137, 256, 512, 1024, 2048, 3141, 4096, 5555, 7777)

HIGH_THRESHOLD = 0.70
LOW_THRESHOLD = 0.30


def build_ctrnn(genotype_list, num_neurons):
    """Build a CTRNN from a genotype vector."""
    decoder = GenotypeDecoder(
        num_neurons=num_neurons,
        include_gains=False,
        tau_range=(0.5, 5.0),
        weight_range=(-10.0, 10.0),
        bias_range=(-10.0, 10.0),
    )
    genotype = np.array(genotype_list)
    params = decoder.decode(genotype)
    net = CTRNN(
        num_neurons=num_neurons,
        time_constants=params['tau'],
        weights=params['weights'],
        biases=params['biases'],
        step_size=0.01,
        center_crossing=True
    )
    return net, params


def compute_bilateral_sensors(agent_x, agent_y, agent_heading, light_x, light_y,
                              body_radius=1.0, sensor_offset_angle=np.pi/6, max_range=40.0):
    """Compute bilateral photosensor readings."""
    left_angle = agent_heading + sensor_offset_angle
    right_angle = agent_heading - sensor_offset_angle
    left_x = agent_x + body_radius * np.cos(left_angle)
    left_y = agent_y + body_radius * np.sin(left_angle)
    right_x = agent_x + body_radius * np.cos(right_angle)
    right_y = agent_y + body_radius * np.sin(right_angle)
    left_dist = np.sqrt((left_x - light_x)**2 + (left_y - light_y)**2)
    right_dist = np.sqrt((right_x - light_x)**2 + (right_y - light_y)**2)
    left_sensor = max(0.0, 1.0 - left_dist / max_range)
    right_sensor = max(0.0, 1.0 - right_dist / max_range)
    return left_sensor, right_sensor


def run_embodied_episode(net, num_neurons, episode_steps=300, rng=None,
                         light_x=40.0, light_y=40.0):
    """Run one embodied phototaxis episode, recording all data."""
    if rng is None:
        rng = np.random.RandomState(42)
    net.reset()
    agent_x, agent_y = 25.0, 25.0
    agent_heading = rng.uniform(0, 2*np.pi)
    max_speed = 3.0

    stimuli = np.zeros((episode_steps, 2))
    states = np.zeros((episode_steps, num_neurons))
    outputs = np.zeros((episode_steps, num_neurons))

    for t in range(episode_steps):
        left_s, right_s = compute_bilateral_sensors(
            agent_x, agent_y, agent_heading, light_x, light_y
        )
        stimuli[t] = [left_s, right_s]

        ext_input = np.zeros(num_neurons)
        ext_input[0] = left_s
        if num_neurons > 1:
            ext_input[1] = right_s

        states[t] = net.get_state()
        output = net.step(ext_input)
        outputs[t] = output

        if num_neurons >= 2:
            left_motor = float(output[0])
            right_motor = float(output[1])
        else:
            left_motor = float(output[0])
            right_motor = float(output[0])

        forward_speed = (left_motor + right_motor) / 2.0 * max_speed
        turn_rate = (right_motor - left_motor) * 2.0
        agent_heading += turn_rate * 0.01
        agent_x += forward_speed * np.cos(agent_heading) * 0.01
        agent_y += forward_speed * np.sin(agent_heading) * 0.01
        agent_x = np.clip(agent_x, 0, 50)
        agent_y = np.clip(agent_y, 0, 50)

    return stimuli, states, outputs


# ============================================================
# CRITERION 1: Ramsey Causal Role Test
# ============================================================
def test_ramsey_causal_role(net, num_neurons, n_trials=20, rng=None):
    """
    Ramsey (2007): A state is representational if it plays a distinctive
    causal role — perturbation produces systematic, context-dependent effects.

    For each neuron: perturb it at mid-trial under different stimulus contexts.
    Measure whether the behavioral disruption varies systematically with context.
    Compare targeted perturbation to random perturbation (null model).

    Returns: dict with per-neuron causal role scores and summary.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    neuron_scores = []

    for ni in range(num_neurons):
        context_disruptions = []

        for trial in range(n_trials):
            # Random stimulus context
            stimulus_strength = rng.uniform(0.0, 1.0)
            ext_input = np.zeros(num_neurons)
            ext_input[0] = stimulus_strength
            if num_neurons > 1:
                ext_input[1] = rng.uniform(0.0, 1.0)

            # Run network to settle
            net.reset()
            for _ in range(100):
                net.step(ext_input)
            baseline_state = net.get_state()

            # Measure baseline behavior (20 steps)
            net.set_state(baseline_state)
            baseline_outputs = []
            for _ in range(20):
                out = net.step(ext_input)
                baseline_outputs.append(out.copy())
            baseline_motor = np.mean([np.abs(o[:min(2, num_neurons)]).mean() for o in baseline_outputs])

            # Targeted perturbation of neuron ni
            perturbed_state = baseline_state.copy()
            perturbed_state[ni] += 0.5  # moderate perturbation
            net.set_state(perturbed_state)
            perturbed_outputs = []
            for _ in range(20):
                out = net.step(ext_input)
                perturbed_outputs.append(out.copy())
            perturbed_motor = np.mean([np.abs(o[:min(2, num_neurons)]).mean() for o in perturbed_outputs])

            disruption = abs(perturbed_motor - baseline_motor)
            context_disruptions.append((stimulus_strength, disruption))

        # Causal role score: correlation between context and disruption
        contexts = np.array([c[0] for c in context_disruptions])
        disruptions = np.array([c[1] for c in context_disruptions])

        # Context-dependence: does disruption magnitude depend on stimulus?
        if np.std(disruptions) > 1e-10 and np.std(contexts) > 1e-10:
            context_dep, _ = spearmanr(contexts, disruptions)
            context_dep = abs(context_dep)
        else:
            context_dep = 0.0

        # Disruption magnitude (normalized)
        mean_disruption = np.mean(disruptions)

        neuron_scores.append({
            'neuron': ni,
            'mean_disruption': float(mean_disruption),
            'context_dependence': float(context_dep),
            'causal_role_score': float(context_dep * min(1.0, mean_disruption * 10)),
        })

    # Summary: best neuron's causal role score
    best_score = max(ns['causal_role_score'] for ns in neuron_scores)
    mean_score = np.mean([ns['causal_role_score'] for ns in neuron_scores])

    return {
        'criterion': 'Ramsey (2007)',
        'best_causal_role': float(best_score),
        'mean_causal_role': float(mean_score),
        'mean_disruption': float(np.mean([ns['mean_disruption'] for ns in neuron_scores])),
        'per_neuron': neuron_scores,
        'passed': best_score > 0.15,
    }


# ============================================================
# CRITERION 2: Shea Teleosemantic Test
# ============================================================
def test_shea_teleosemantic(net, num_neurons, n_trials=20, rng=None):
    """
    Shea (2018): A state represents X if it was selected (by evolution) to
    indicate X. Test: response to evolved stimulus vs spurious correlate.

    Evolved stimulus: structured bilateral input (as in phototaxis)
    Spurious correlate: random noise with same mean/variance

    Dissociation score: how much more does network respond to structured
    vs unstructured input?
    """
    if rng is None:
        rng = np.random.RandomState(42)

    responses_structured = []
    responses_random = []

    for trial in range(n_trials):
        # Structured input: bilateral sensor-like (correlated pair)
        base_intensity = rng.uniform(0.2, 0.8)
        differential = rng.uniform(-0.2, 0.2)
        left_s = base_intensity + differential
        right_s = base_intensity - differential

        net.reset()
        structured_response = 0.0
        for step in range(150):
            ext_input = np.zeros(num_neurons)
            ext_input[0] = left_s
            if num_neurons > 1:
                ext_input[1] = right_s
            out = net.step(ext_input)
            if step >= 50:  # skip transient
                structured_response += np.abs(out[:min(2, num_neurons)]).mean()
        structured_response /= 100.0
        responses_structured.append(structured_response)

        # Random input: same mean but uncorrelated noise
        net.reset()
        random_response = 0.0
        for step in range(150):
            ext_input = np.zeros(num_neurons)
            ext_input[0] = rng.uniform(0.0, 1.0)
            if num_neurons > 1:
                ext_input[1] = rng.uniform(0.0, 1.0)
            out = net.step(ext_input)
            if step >= 50:
                random_response += np.abs(out[:min(2, num_neurons)]).mean()
        random_response /= 100.0
        responses_random.append(random_response)

    mean_struct = np.mean(responses_structured)
    mean_random = np.mean(responses_random)

    # Dissociation: does the network preferentially respond to structured input?
    max_resp = max(mean_struct, mean_random, 1e-10)
    dissociation = (mean_struct - mean_random) / max_resp

    # Consistency: does structured input produce more consistent responses?
    struct_cv = np.std(responses_structured) / (mean_struct + 1e-10)
    random_cv = np.std(responses_random) / (mean_random + 1e-10)
    consistency_advantage = random_cv - struct_cv

    return {
        'criterion': 'Shea (2018)',
        'structured_response': float(mean_struct),
        'random_response': float(mean_random),
        'dissociation_score': float(dissociation),
        'consistency_advantage': float(consistency_advantage),
        'passed': dissociation > 0.1,
    }


# ============================================================
# CRITERION 3: Gładziejewski & Miłkowski Information Test
# ============================================================
def test_gm_information(net, num_neurons, n_episodes=5, episode_steps=200, rng=None):
    """
    Gładziejewski & Miłkowski (2017): Representational states carry systematic
    information. Tests:
    1. Mutual information I(stimulus; state)
    2. Transfer entropy TE(stimulus→state) and TE(state→action)
    3. Mediation: does state mediate stimulus→action?

    Uses embodied phototaxis episodes for ecological validity.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    all_stimuli = []
    all_states = []
    all_actions = []

    for ep in range(n_episodes):
        light_angle = rng.uniform(0, 2*np.pi)
        light_x = 25.0 + 20.0 * np.cos(light_angle)
        light_y = 25.0 + 20.0 * np.sin(light_angle)

        stimuli, states, outputs = run_embodied_episode(
            net, num_neurons, episode_steps=episode_steps,
            rng=rng, light_x=light_x, light_y=light_y
        )

        all_stimuli.append(stimuli[:, 0])  # left sensor
        all_states.append(states[:, 0])    # first neuron state
        # Action = motor differential
        if num_neurons >= 2:
            all_actions.append(outputs[:, 0] - outputs[:, 1])
        else:
            all_actions.append(outputs[:, 0])

    # Concatenate across episodes
    stim_series = np.concatenate(all_stimuli)
    state_series = np.concatenate(all_states)
    action_series = np.concatenate(all_actions)

    # 1. Mutual information
    mi_stim_state = InformationAnalyzer.mutual_information(stim_series, state_series, bins=8)
    mi_stim_action = InformationAnalyzer.mutual_information(stim_series, action_series, bins=8)

    # Baseline: MI with shuffled state
    shuffled = rng.permutation(state_series)
    mi_baseline = InformationAnalyzer.mutual_information(stim_series, shuffled, bins=8)

    mi_gain = mi_stim_state - mi_baseline

    # 2. Transfer entropy
    te_stim_state = InformationAnalyzer.transfer_entropy(stim_series, state_series, lag=1, bins=5)
    te_state_action = InformationAnalyzer.transfer_entropy(state_series, action_series, lag=1, bins=5)
    te_stim_action = InformationAnalyzer.transfer_entropy(stim_series, action_series, lag=1, bins=5)

    # Mediation score
    if te_stim_action > 1e-6:
        mediation = (te_stim_state * te_state_action) / (te_stim_action + 1e-6)
    else:
        mediation = 0.0
    mediation = min(1.0, max(0.0, mediation))

    return {
        'criterion': 'Gładziejewski & Miłkowski (2017)',
        'mi_stim_state': float(mi_stim_state),
        'mi_stim_action': float(mi_stim_action),
        'mi_baseline': float(mi_baseline),
        'mi_gain': float(mi_gain),
        'te_stim_state': float(te_stim_state),
        'te_state_action': float(te_state_action),
        'te_stim_action': float(te_stim_action),
        'mediation_score': float(mediation),
        'passed': mi_gain > 0.05 and mediation > 0.1,
    }


# ============================================================
# CRITERION 4: Decoupling Invariance (Novel)
# ============================================================
def test_decoupling_invariance(net, num_neurons, n_trials=10, rng=None):
    """
    Novel criterion for Paper 3: Does internal state structure persist
    when decoupled from sensory input?

    If a state is truly representational, it should maintain some structure
    even when decoupled — it carries information that outlasts the stimulus.
    If a state is purely reactive/dynamical, decoupling should immediately
    destroy its informational content.

    Method:
    1. Run embodied for 200 steps to build up state
    2. Decouple (zero input) for 100 steps
    3. Re-couple with same input
    4. Measure: does state after re-coupling match state before decoupling?
    5. Compare to baseline: state from scratch with same input for 300 steps

    High invariance = state structure persists → more representational
    Low invariance = state collapses → more dynamical/reactive
    """
    if rng is None:
        rng = np.random.RandomState(42)

    invariance_scores = []

    for trial in range(n_trials):
        # Random input context
        left_s = rng.uniform(0.2, 0.8)
        right_s = rng.uniform(0.2, 0.8)
        ext_input = np.zeros(num_neurons)
        ext_input[0] = left_s
        if num_neurons > 1:
            ext_input[1] = right_s

        # Phase 1: Coupled (200 steps)
        net.reset()
        for _ in range(200):
            net.step(ext_input)
        state_before_decoupling = net.get_state()

        # Phase 2: Decoupled (100 steps with zero input)
        for _ in range(100):
            net.step(np.zeros(num_neurons))

        # Phase 3: Re-coupled (100 steps with same input)
        for _ in range(100):
            net.step(ext_input)
        state_after_recoupling = net.get_state()

        # Baseline: run from scratch for 300 steps coupled
        net.reset()
        for _ in range(300):
            net.step(ext_input)
        state_baseline = net.get_state()

        # Invariance: how similar is recoupled state to baseline?
        # (baseline represents "ideal" state under this input)
        dist_recoupled = np.linalg.norm(state_after_recoupling - state_baseline)
        dist_decoupled = np.linalg.norm(state_before_decoupling - state_baseline)

        # Normalize by state magnitude
        state_scale = np.linalg.norm(state_baseline) + 1e-10

        # Recovery score: 1.0 = perfect recovery, 0.0 = no recovery
        recovery = 1.0 - min(1.0, dist_recoupled / (state_scale + 1e-10))

        # Persistence: how much structure survived decoupling?
        # Compare state before decoupling to state after decoupling (before recoupling)
        net.reset()
        for _ in range(200):
            net.step(ext_input)
        net_state_pre = net.get_state()
        for _ in range(100):
            net.step(np.zeros(num_neurons))
        net_state_post_decouple = net.get_state()

        persistence = 1.0 - min(1.0, np.linalg.norm(net_state_post_decouple - net_state_pre) / (state_scale + 1e-10))

        invariance_scores.append({
            'recovery': float(recovery),
            'persistence': float(persistence),
        })

    mean_recovery = np.mean([s['recovery'] for s in invariance_scores])
    mean_persistence = np.mean([s['persistence'] for s in invariance_scores])

    return {
        'criterion': 'Decoupling Invariance (Novel)',
        'mean_recovery': float(mean_recovery),
        'mean_persistence': float(mean_persistence),
        'invariance_score': float((mean_recovery + mean_persistence) / 2.0),
        'per_trial': invariance_scores,
        'passed': mean_recovery > 0.5 and mean_persistence > 0.3,
    }


# ============================================================
# MAIN: Run across all 42 decoded genotypes
# ============================================================
def main():
    print("=" * 70)
    print("PAPER 3: CROSS-AGENT REPRESENTATION CRITERIA ANALYSIS")
    print("=" * 70)

    # Load Paper 2 phase data
    results_path = Path(PAPER2_RESULTS)
    phase_files = sorted(results_path.glob('phase_a_10seeds_*.json'), reverse=True)
    if not phase_files:
        raise FileNotFoundError("No phase_a_10seeds results found")
    with open(phase_files[0], 'r') as f:
        phase_data = json.load(f)

    # Load mechanistic analysis for weight data
    mech_files = sorted(results_path.glob('mechanistic_analysis_*.json'), reverse=True)
    mech_data = None
    if mech_files:
        with open(mech_files[0], 'r') as f:
            mech_data = json.load(f)

    # Load attractor geometry for type classification
    attr_files = sorted(results_path.glob('attractor_geometry_*.json'), reverse=True)
    attr_data = None
    if attr_files:
        with open(attr_files[0], 'r') as f:
            attr_data = json.load(f)

    conditions = phase_data['conditions']
    all_results = []
    skipped = 0

    for ns in NETWORK_SIZES:
        for s in ALL_SEEDS:
            run_id = f"net{ns}_seed{s}"
            cond = conditions.get(run_id, {})

            if 'error' in cond or 'scores' not in cond:
                skipped += 1
                continue

            genotype = cond.get('evolution', {}).get('best_genotype', None)
            if genotype is None:
                skipped += 1
                continue

            ed_score = cond['scores']['constitutive']
            print(f"\n  Analyzing {run_id} (ED={ed_score:.3f}, n={ns})...", flush=True)

            try:
                net, params = build_ctrnn(genotype, ns)
                rng = np.random.RandomState(42)

                # Run all criteria
                ramsey = test_ramsey_causal_role(net, ns, n_trials=15, rng=rng)
                print(f"    Ramsey: causal_role={ramsey['best_causal_role']:.3f}", end="", flush=True)

                net2, _ = build_ctrnn(genotype, ns)
                shea = test_shea_teleosemantic(net2, ns, n_trials=15, rng=np.random.RandomState(137))
                print(f"  Shea: dissoc={shea['dissociation_score']:.3f}", end="", flush=True)

                net3, _ = build_ctrnn(genotype, ns)
                gm = test_gm_information(net3, ns, n_episodes=4, episode_steps=150,
                                         rng=np.random.RandomState(256))
                print(f"  G&M: MI={gm['mi_gain']:.3f}", end="", flush=True)

                net4, _ = build_ctrnn(genotype, ns)
                decoupling = test_decoupling_invariance(net4, ns, n_trials=8,
                                                        rng=np.random.RandomState(512))
                print(f"  Decoupling: inv={decoupling['invariance_score']:.3f}")

                result = {
                    'run_id': run_id,
                    'num_neurons': ns,
                    'seed': s,
                    'ed_score': ed_score,
                    'ramsey': ramsey,
                    'shea': shea,
                    'gm': gm,
                    'decoupling': decoupling,
                }
                all_results.append(result)

            except Exception as e:
                print(f"  ERROR: {e}")
                skipped += 1

    print(f"\n\nAnalyzed: {len(all_results)} agents (skipped: {skipped})")

    # === STATISTICAL ANALYSIS ===
    print(f"\n{'='*70}")
    print("CORRELATIONS: REPRESENTATION CRITERIA vs EMBODIMENT DEPENDENCE")
    print(f"{'='*70}")

    ed_scores = np.array([r['ed_score'] for r in all_results])

    criteria_metrics = {
        'ramsey_best_causal_role': [r['ramsey']['best_causal_role'] for r in all_results],
        'ramsey_mean_causal_role': [r['ramsey']['mean_causal_role'] for r in all_results],
        'ramsey_mean_disruption': [r['ramsey']['mean_disruption'] for r in all_results],
        'shea_dissociation': [r['shea']['dissociation_score'] for r in all_results],
        'shea_consistency_advantage': [r['shea']['consistency_advantage'] for r in all_results],
        'gm_mi_gain': [r['gm']['mi_gain'] for r in all_results],
        'gm_mediation': [r['gm']['mediation_score'] for r in all_results],
        'gm_te_stim_state': [r['gm']['te_stim_state'] for r in all_results],
        'gm_te_state_action': [r['gm']['te_state_action'] for r in all_results],
        'decoupling_invariance': [r['decoupling']['invariance_score'] for r in all_results],
        'decoupling_recovery': [r['decoupling']['mean_recovery'] for r in all_results],
        'decoupling_persistence': [r['decoupling']['mean_persistence'] for r in all_results],
    }

    correlation_results = {}
    for name, vals in criteria_metrics.items():
        vals_arr = np.array(vals)
        finite_mask = np.isfinite(vals_arr)
        if np.sum(finite_mask) < 10:
            continue
        rho, p = spearmanr(vals_arr[finite_mask], ed_scores[finite_mask])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:<35} rho={rho:+.3f} (p={p:.4f}) {sig}")
        correlation_results[name] = {'rho': float(rho), 'p': float(p)}

    # === HIGH vs LOW ED COMPARISON ===
    print(f"\n{'='*70}")
    print("HIGH vs LOW EMBODIMENT: REPRESENTATION CRITERIA")
    print(f"{'='*70}")

    high = [r for r in all_results if r['ed_score'] >= HIGH_THRESHOLD]
    low = [r for r in all_results if r['ed_score'] < LOW_THRESHOLD]

    print(f"\n  High ED (n={len(high)}), Low ED (n={len(low)})")

    for name, vals in criteria_metrics.items():
        h_vals = [vals[all_results.index(r)] for r in high]
        l_vals = [vals[all_results.index(r)] for r in low]
        if len(h_vals) >= 3 and len(l_vals) >= 3:
            h_m, l_m = np.mean(h_vals), np.mean(l_vals)
            try:
                stat, p = mannwhitneyu(h_vals, l_vals, alternative='two-sided')
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {name:<35} H={h_m:.3f} L={l_m:.3f} p={p:.4f} {sig}")
            except Exception:
                print(f"  {name:<35} H={h_m:.3f} L={l_m:.3f} (test failed)")

    # === CRITERIA AGREEMENT ===
    print(f"\n{'='*70}")
    print("CRITERIA AGREEMENT ANALYSIS")
    print(f"{'='*70}")

    criteria_pass = {
        'Ramsey': [r['ramsey']['passed'] for r in all_results],
        'Shea': [r['shea']['passed'] for r in all_results],
        'G&M': [r['gm']['passed'] for r in all_results],
        'Decoupling': [r['decoupling']['passed'] for r in all_results],
    }

    for crit_name, passes in criteria_pass.items():
        n_pass = sum(passes)
        pct = 100 * n_pass / len(passes)
        print(f"  {crit_name:<15} passed: {n_pass}/{len(passes)} ({pct:.0f}%)")

    # Agreement matrix
    crit_names = list(criteria_pass.keys())
    print(f"\n  Pairwise agreement rates:")
    for i, c1 in enumerate(crit_names):
        for j, c2 in enumerate(crit_names):
            if j > i:
                agree = sum(1 for a, b in zip(criteria_pass[c1], criteria_pass[c2]) if a == b)
                rate = agree / len(all_results)
                print(f"    {c1} vs {c2}: {rate:.2f}")

    # === BY NETWORK SIZE ===
    print(f"\n{'='*70}")
    print("REPRESENTATION BY NETWORK SIZE")
    print(f"{'='*70}")

    for ns in NETWORK_SIZES:
        subset = [r for r in all_results if r['num_neurons'] == ns]
        if not subset:
            continue
        eds = [r['ed_score'] for r in subset]
        ramsey_scores = [r['ramsey']['best_causal_role'] for r in subset]
        mi_scores = [r['gm']['mi_gain'] for r in subset]
        decouple_scores = [r['decoupling']['invariance_score'] for r in subset]
        print(f"  n={ns}: ED={np.mean(eds):.3f}±{np.std(eds):.3f}  "
              f"Ramsey={np.mean(ramsey_scores):.3f}  MI_gain={np.mean(mi_scores):.3f}  "
              f"Decouple={np.mean(decouple_scores):.3f}")

    # === KEY FINDING: ED vs REPRESENTATION DISSOCIATION ===
    print(f"\n{'='*70}")
    print("KEY FINDING: RELATIONSHIP BETWEEN ED AND REPRESENTATION")
    print(f"{'='*70}")

    # Create composite representation score
    composite = []
    for r in all_results:
        # Normalize each criterion to [0,1] and average
        scores = [
            min(1.0, r['ramsey']['best_causal_role'] * 3),  # scale up
            max(0, r['shea']['dissociation_score']),
            min(1.0, max(0, r['gm']['mi_gain'] * 5)),  # scale up
            r['decoupling']['invariance_score'],
        ]
        composite.append(np.mean(scores))

    composite_arr = np.array(composite)
    rho_composite, p_composite = spearmanr(composite_arr, ed_scores)
    print(f"\n  Composite representation score vs ED: rho={rho_composite:+.3f} (p={p_composite:.4f})")

    if rho_composite > 0.3:
        print("  → High-ED agents show MORE representational features")
    elif rho_composite < -0.3:
        print("  → High-ED agents show FEWER representational features (more dynamical)")
    else:
        print("  → Representation criteria are largely ORTHOGONAL to embodiment dependence")

    # === SAVE RESULTS ===
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, float) and not np.isfinite(obj): return str(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        return obj

    save_data = {
        'meta': {
            'timestamp': timestamp,
            'n_agents': len(all_results),
            'skipped': skipped,
        },
        'correlations': convert(correlation_results),
        'composite_vs_ed': {
            'rho': float(rho_composite),
            'p': float(p_composite),
        },
        'conditions': convert(all_results),
    }

    outfile = os.path.join(RESULTS_DIR, f'representation_criteria_{timestamp}.json')
    with open(outfile, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to: {outfile}")


if __name__ == "__main__":
    main()
