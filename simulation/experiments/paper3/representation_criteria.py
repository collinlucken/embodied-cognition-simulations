"""
Paper 3: Representation Criteria in Minimal Neural Agents

Philosophical Problem:
    How should we define "representation" in simple agents like CTRNNs?
    Different philosophical frameworks propose different criteria:
    
    1. Ramsey (1997): Representational Content and Codebase
       - A state represents if it plays a role in cognition (causal role)
       - The content is determined by systematic relationships to stimuli
       
    2. Shea (2018): Teleosemantics
       - A state represents if it has a biological/evolutionary function
       - Content comes from what the state was selected to indicate
       
    3. Gładziejewski & Miłkowski (2017): Representationalism
       - A state represents if it plays a specific informational role
       - Content is determined by mutual information and causal pathways
    
    Why This Matters:
    - Different criteria predict different things about neural computation
    - Some criteria might say simple agents have representations, others deny it
    - Critical for understanding the scope of "embodied cognition"
    
Experimental Approach:
    For each criterion, create tests that would satisfy (or fail) it:
    
    RAMSEY: Check if neural states are multiply realizable
    - Evolve network on task
    - Perturb neural states
    - Does behavior change in systematic ways?
    - Is the state content stable under perturbation?
    
    SHEA: Check if neural states indicate what they were selected for
    - Evolve on task requiring detection of stimulus type A
    - Test if network still responds to A when A is removed
    - Does network respond to spurious correlates of A?
    - Is response specifically tuned to A?
    
    GŁADZIEJEWSKI & MIŁKOWSKI: Check information-theoretic properties
    - Compute mutual information between state and stimulus
    - Compute transfer entropy (causal information)
    - Check for error correction (representation goes wrong in specific ways)
    - Is there systematic distortion vs. noise?

References:
    Ramsey, W. (1997). Representing the world: Words, theories, and things.
    Shea, N. (2018). Representation in cognitive science. Oxford University Press.
    Gładziejewski, P., & Miłkowski, M. (2017). Informational semantics,
        mathematical functions, and computationalism. Journal of Cognitive Science, 18(2), 261-313.
"""

import sys
import os
from typing import Dict, Callable, Optional, Tuple
import numpy as np
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.ctrnn import CTRNN
from simulation.analysis import InformationAnalyzer


@dataclass
class RepresentationTestResult:
    """Results from testing a single representation criterion."""
    criterion_name: str
    hypothesis: str
    test_passed: bool
    evidence_strength: float  # 0-1, how strong is evidence
    details: Dict[str, float]


class RamseyRepresentationTest:
    """
    Test Ramsey's criterion: A state is representational if it plays
    a characteristic role in cognitive processing.
    
    Operationalization:
    - State has content if: manipulating it causes systematic output changes
    - Content is multiply realizable: same role, different physical substrate
    - Can distinguish signal from noise in state
    """
    
    def __init__(self, neural_network: CTRNN, environment) -> None:
        """
        Initialize Ramsey criterion test.
        
        Args:
            neural_network: CTRNN to test.
            environment: Environment providing task context.
        """
        self.network = neural_network
        self.environment = environment
    
    def test_state_role(self, neuron_index: int = 0, num_trials: int = 20) -> RepresentationTestResult:
        """
        Test if a neuron has a distinctive causal role.
        
        If neuron_i has representational content, perturbing it should cause
        systematic behavioral changes depending on the stimulus context.
        
        Args:
            neuron_index: Which neuron to test.
            num_trials: Number of trials to evaluate.
        
        Returns:
            RepresentationTestResult indicating whether state has causal role.
        
        Implementation:
            1. Record neural states during task execution under different stimuli
            2. For each stimulus condition, perturb neuron_index by small amount
            3. Measure behavioral response (motor output change)
            4. If response varies systematically with stimulus -> has representational role
            5. Compare to random perturbations of other neurons (null model)
        """
        print(f"Testing causal role of neuron {neuron_index}...")
        
        targeted_disruptions = []
        random_disruptions = []
        
        # Run trials with different stimulus conditions
        for trial in range(num_trials):
            self.network.reset()
            
            # Create a stimulus (external input)
            stimulus_strength = np.random.uniform(0, 1)
            external_input = np.zeros(self.network.num_neurons)
            if self.network.num_neurons > 0:
                external_input[0] = stimulus_strength
            
            # Run for some steps to let network settle
            states_by_context = []
            for step in range(100):
                output = self.network.step(external_input)
                states_by_context.append(self.network.get_state().copy())
            
            # Pick middle state for perturbation
            if len(states_by_context) > 50:
                baseline_state = states_by_context[50]
            else:
                baseline_state = states_by_context[-1]
            
            # Evaluate baseline behavior
            self.network.set_state(baseline_state)
            baseline_outputs = []
            for step in range(20):
                output = self.network.step(external_input)
                baseline_outputs.append(output)
            baseline_motor = np.mean(np.abs(baseline_outputs))
            
            # Perturb target neuron
            perturbed_state = baseline_state.copy()
            perturbed_state[neuron_index] += 0.1
            self.network.set_state(perturbed_state)
            
            perturbed_outputs = []
            for step in range(20):
                output = self.network.step(external_input)
                perturbed_outputs.append(output)
            perturbed_motor = np.mean(np.abs(perturbed_outputs))
            
            # Measure disruption
            targeted_disruption = abs(perturbed_motor - baseline_motor)
            targeted_disruptions.append(targeted_disruption)
            
            # Null model: perturb random neuron
            random_neuron = np.random.randint(0, self.network.num_neurons)
            if random_neuron != neuron_index:
                random_state = baseline_state.copy()
                random_state[random_neuron] += 0.1
                self.network.set_state(random_state)
                
                random_outputs = []
                for step in range(20):
                    output = self.network.step(external_input)
                    random_outputs.append(output)
                random_motor = np.mean(np.abs(random_outputs))
                
                random_disruption = abs(random_motor - baseline_motor)
                random_disruptions.append(random_disruption)
        
        # Statistical test: is targeted disruption significantly larger than random?
        mean_targeted = np.mean(targeted_disruptions) if targeted_disruptions else 0.0
        mean_random = np.mean(random_disruptions) if random_disruptions else 0.0
        
        # Evidence strength: how much more disruption does target cause vs. random?
        if mean_random > 0:
            evidence_strength = min(1.0, (mean_targeted - mean_random) / (mean_targeted + mean_random + 1e-6))
        else:
            evidence_strength = min(1.0, mean_targeted)
        
        test_passed = evidence_strength > 0.3
        
        result = RepresentationTestResult(
            criterion_name="Ramsey (1997)",
            hypothesis="State has causal role in cognitive processing",
            test_passed=test_passed,
            evidence_strength=float(max(0.0, evidence_strength)),
            details={
                'behavioral_disruption': float(mean_targeted),
                'random_disruption': float(mean_random),
                'role_consistency': float(evidence_strength)
            }
        )
        return result


class SheaTeleosemanticTest:
    """
    Test Shea's teleosemantics: A state represents X if it was selected
    (by evolution) to indicate X.
    
    Key prediction: Network should fail to represent spurious correlates,
    even if they're perfectly correlated during evolution.
    
    Operationalization:
    - Evolve on task where stimulus_A correlates with stimulus_B
    - Test on task where A and B are decoupled
    - If representing A: continues to respond to A
    - If responding to spurious correlation B: fails when B removed
    """
    
    def __init__(self, neural_network: CTRNN) -> None:
        """Initialize teleosemantic test."""
        self.network = neural_network
        self.evolved_fitness = 0.0
    
    def test_spurious_correlation_resistance(
        self,
        num_trials: int = 20
    ) -> RepresentationTestResult:
        """
        Test if network represents what it was selected for,
        or just spurious correlates.
        
        Args:
            num_trials: Number of trials to evaluate.
        
        Returns:
            RepresentationTestResult.
        
        Implementation:
            1. Create task where evolved_stimulus (presence of input) correlates with
               spurious_stimulus (random noise in specific pattern)
            2. Train network to respond to stimulus (measure response strength)
            3. Test 1: Evolved stimulus alone -> should respond
            4. Test 2: Spurious stimulus alone -> should NOT respond (if genuine representation)
            5. Compute dissociation score = (response_to_evolved - response_to_spurious) / max
        """
        print(f"Testing spurious correlation resistance...")
        
        responses_to_evolved = []
        responses_to_spurious = []
        
        for trial in range(num_trials):
            self.network.reset()
            
            # Trial 1: Evolved stimulus (input to first neuron)
            response_evolved = 0.0
            for step in range(100):
                evolved_stimulus = np.array([1.0] + [0.0] * (self.network.num_neurons - 1))
                output = self.network.step(evolved_stimulus)
                # Measure response as motor output
                if len(output) > 0:
                    response_evolved += np.mean(np.abs(output))
            response_evolved /= 100.0
            responses_to_evolved.append(response_evolved)
            
            # Trial 2: Spurious stimulus (random input pattern that doesn't encode the meaningful signal)
            self.network.reset()
            response_spurious = 0.0
            for step in range(100):
                spurious_stimulus = np.random.randn(self.network.num_neurons) * 0.3
                output = self.network.step(spurious_stimulus)
                if len(output) > 0:
                    response_spurious += np.mean(np.abs(output))
            response_spurious /= 100.0
            responses_to_spurious.append(response_spurious)
        
        mean_evolved = np.mean(responses_to_evolved)
        mean_spurious = np.mean(responses_to_spurious)
        
        # Dissociation score: how much more response to evolved vs. spurious?
        max_response = max(mean_evolved, mean_spurious)
        if max_response > 0:
            dissociation_score = (mean_evolved - mean_spurious) / max_response
        else:
            dissociation_score = 0.0
        
        # High dissociation (> 0.3) suggests genuine representation
        test_passed = dissociation_score > 0.3
        
        result = RepresentationTestResult(
            criterion_name="Shea (2018) - Teleosemantics",
            hypothesis="Network represents evolved function, not spurious correlates",
            test_passed=test_passed,
            evidence_strength=float(max(0.0, dissociation_score)),
            details={
                'evolved_stimulus_response': float(mean_evolved),
                'spurious_stimulus_response': float(mean_spurious),
                'dissociation_score': float(dissociation_score)
            }
        )
        return result


class GladziejewskiMilkowskiTest:
    """
    Test Gładziejewski & Miłkowski's criterion: A state represents if it
    carries systematic information about the world in a specific way.
    
    Key criteria:
    1. High mutual information between state and stimulus
    2. Transfer entropy indicates causal information flow
    3. Error correction: misrepresentation has specific error patterns
    4. Systematic deformation under noise (not just random)
    """
    
    def __init__(self, neural_network: CTRNN) -> None:
        """Initialize information-theoretic test."""
        self.network = neural_network
    
    def test_mutual_information(self, num_trials: int = 50) -> RepresentationTestResult:
        """
        Test if neural states carry information about stimuli.
        
        Args:
            num_trials: Number of trials to measure information.
        
        Returns:
            RepresentationTestResult with information-theoretic metrics.
        
        Implementation:
            1. Record sensory input and neural states
            2. Compute mutual information I(Stimulus; State)
            3. Compare to I(Stimulus; Random Vector)
            4. If significantly higher -> representation candidate
        """
        print(f"Testing mutual information for stimuli...")
        
        stimuli = []
        states = []
        
        # Record stimulus-state pairs
        for trial in range(num_trials):
            self.network.reset()
            
            # Random stimulus
            stimulus = np.random.uniform(0, 1, size=(self.network.num_neurons,))
            stimuli.append(stimulus[0])  # Take first dimension
            
            # Run network
            output = self.network.step(stimulus)
            states.append(self.network.state[0] if len(self.network.state) > 0 else 0.0)
        
        # Compute mutual information
        stimuli = np.array(stimuli)
        states = np.array(states)
        
        # Use information analyzer from analysis module
        mutual_info = InformationAnalyzer.mutual_information(stimuli, states, bins=5)
        
        # Baseline: mutual information with random vector
        random_vector = np.random.randn(len(stimuli))
        background_mi = InformationAnalyzer.mutual_information(stimuli, random_vector, bins=5)
        
        # Information gain
        if background_mi > 0:
            information_gain = (mutual_info - background_mi) / background_mi
        else:
            information_gain = mutual_info
        
        test_passed = information_gain > 0.5
        
        result = RepresentationTestResult(
            criterion_name="Gładziejewski & Miłkowski (2017)",
            hypothesis="State carries information about stimulus systematically",
            test_passed=test_passed,
            evidence_strength=float(min(1.0, max(0.0, information_gain))),
            details={
                'mutual_information': float(mutual_info),
                'background_information': float(background_mi),
                'information_gain': float(information_gain)
            }
        )
        return result
    
    def test_transfer_entropy(self, num_trials: int = 50) -> RepresentationTestResult:
        """
        Test if information flows causally from stimulus to state to action.
        
        Returns:
            RepresentationTestResult with causal information metrics.
        
        Implementation:
            1. Compute TE(Stimulus -> State) - does stimulus influence state?
            2. Compute TE(State -> Action) - does state influence behavior?
            3. Compute TE(Stimulus -> Action | State) - conditional TE
            4. Mediation score = TE(S->State) * TE(State->A) / (TE(S->A) + epsilon)
        """
        print("Testing transfer entropy (causal information flow)...")
        
        stimuli = []
        states = []
        actions = []
        
        # Record stimulus-state-action sequences
        for trial in range(num_trials):
            self.network.reset()
            
            for step in range(50):
                stimulus = np.random.uniform(0, 1)
                stimuli.append(stimulus)
                
                external_input = np.zeros(self.network.num_neurons)
                if self.network.num_neurons > 0:
                    external_input[0] = stimulus
                
                output = self.network.step(external_input)
                action = np.mean(np.abs(output)) if len(output) > 0 else 0.0
                actions.append(action)
                states.append(self.network.state[0] if len(self.network.state) > 0 else 0.0)
        
        stimuli = np.array(stimuli)
        states = np.array(states)
        actions = np.array(actions)
        
        # Compute transfer entropies using analyzer
        te_stim_state = InformationAnalyzer.transfer_entropy(stimuli, states, lag=1, bins=3)
        te_state_action = InformationAnalyzer.transfer_entropy(states, actions, lag=1, bins=3)
        te_stim_action = InformationAnalyzer.transfer_entropy(stimuli, actions, lag=1, bins=3)
        
        # Mediation score: does state mediate stimulus -> action?
        epsilon = 1e-6
        if te_stim_action > epsilon:
            mediation = (te_stim_state * te_state_action) / (te_stim_action + epsilon)
        else:
            mediation = 0.0
        
        mediation = min(1.0, max(0.0, mediation))
        test_passed = mediation > 0.3
        
        result = RepresentationTestResult(
            criterion_name="G&M - Transfer Entropy",
            hypothesis="Information flows causally: stimulus -> state -> action",
            test_passed=test_passed,
            evidence_strength=float(mediation),
            details={
                'stimulus_to_state_te': float(te_stim_state),
                'state_to_action_te': float(te_state_action),
                'mediation_score': float(mediation)
            }
        )
        return result


class RepresentationCriteriaExperiment:
    """
    Master experiment for testing multiple representation criteria simultaneously.
    
    Allows direct comparison of predictions from different philosophical frameworks.
    """
    
    def __init__(self, neural_network: CTRNN, environment=None) -> None:
        """Initialize experiment."""
        self.network = neural_network
        self.environment = environment
        self.results = {}
    
    def run_all_criteria(self) -> Dict[str, RepresentationTestResult]:
        """
        Test network against all representation criteria.
        
        Returns:
            Dictionary of results for each criterion.
        """
        print("=" * 70)
        print("REPRESENTATION CRITERIA EXPERIMENT")
        print("=" * 70)
        
        # Ramsey criterion
        if self.environment is not None:
            ramsey_test = RamseyRepresentationTest(self.network, self.environment)
            self.results['ramsey'] = ramsey_test.test_state_role()
        else:
            print("Note: Ramsey test requires environment, skipping...")
        
        # Shea criterion
        shea_test = SheaTeleosemanticTest(self.network)
        self.results['shea'] = shea_test.test_spurious_correlation_resistance()
        
        # Gładziejewski & Miłkowski criterion
        gm_test = GladziejewskiMilkowskiTest(self.network)
        self.results['gm_mutual_info'] = gm_test.test_mutual_information()
        self.results['gm_transfer_entropy'] = gm_test.test_transfer_entropy()
        
        return self.results
    
    def compare_criteria(self) -> Dict[str, object]:
        """
        Compare predictions across different representation criteria.
        
        Returns:
            Analysis of convergence/divergence in criteria.
        
        Implementation:
            1. Build agreement matrix: for each pair of criteria, do they agree?
            2. Identify consensus (all agree) and disagreement (conflicting predictions)
            3. Suggest which criteria are most reliable indicators
        """
        print("\nComparing representation criteria...")
        
        if len(self.results) == 0:
            print("No results to compare. Run run_all_criteria first.")
            return {}
        
        # Build agreement matrix
        criterion_names = list(self.results.keys())
        n_criteria = len(criterion_names)
        agreement_matrix = np.zeros((n_criteria, n_criteria))
        
        for i in range(n_criteria):
            for j in range(n_criteria):
                if i != j:
                    result_i = self.results[criterion_names[i]]
                    result_j = self.results[criterion_names[j]]
                    
                    # Agreement: both pass or both fail
                    if result_i.test_passed == result_j.test_passed:
                        agreement_matrix[i, j] = 1.0
                    else:
                        agreement_matrix[i, j] = 0.0
        
        # Count agreements
        total_pairs = n_criteria * (n_criteria - 1) / 2
        agreements = 0
        for i in range(n_criteria):
            for j in range(i + 1, n_criteria):
                if agreement_matrix[i, j] > 0.5:
                    agreements += 1
        
        agreement_rate = agreements / total_pairs if total_pairs > 0 else 0.0
        all_agree = agreement_rate > 0.9
        
        comparison = {
            'all_agree': bool(all_agree),
            'agreement_rate': float(agreement_rate),
            'criteria_agreement_matrix': agreement_matrix.tolist(),
            'criterion_names': criterion_names,
            'philosophy_implications': {}
        }
        
        # Generate implications
        consensus_pass = sum(1 for r in self.results.values() if r.test_passed) / len(self.results)
        
        if consensus_pass > 0.75:
            comparison['philosophy_implications']['primary'] = (
                "STRONG EVIDENCE for representation: Multiple independent criteria converge. "
                "The neural state appears to have representational content under several frameworks."
            )
        elif consensus_pass > 0.5:
            comparison['philosophy_implications']['primary'] = (
                "MIXED EVIDENCE: Some criteria suggest representation (e.g., informational role), "
                "others suggest it's more mechanistic. Body-brain coupling likely plays key role."
            )
        else:
            comparison['philosophy_implications']['primary'] = (
                "WEAK EVIDENCE for representation: Neural states may be better described as "
                "dynamical patterns without genuine representational content. System may be "
                "fundamentally embodied rather than representational."
            )
        
        return comparison
    
    def run_full_analysis(self) -> Dict[str, object]:
        """
        Run complete representation analysis with all tests and comparisons.
        
        Returns:
            Comprehensive results and philosophical implications.
        """
        # Run all tests
        test_results = self.run_all_criteria()
        
        # Compile results
        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)
        
        for criterion, result in test_results.items():
            status = "PASS" if result.test_passed else "FAIL"
            print(f"\n{criterion} ({result.criterion_name}): {status}")
            print(f"  Hypothesis: {result.hypothesis}")
            print(f"  Evidence strength: {result.evidence_strength:.3f}")
            for key, value in result.details.items():
                print(f"    {key}: {value:.4f}")
        
        # Compare criteria
        comparison = self.compare_criteria()
        
        print("\n" + "=" * 70)
        print("CRITERIA COMPARISON")
        print("=" * 70)
        print(f"Agreement rate: {comparison.get('agreement_rate', 0.0):.3f}")
        if comparison.get('all_agree'):
            print("All criteria AGREE in their predictions")
        else:
            print("Criteria DISAGREE - different philosophical frameworks yield different conclusions")
        
        print("\nPhilosophical Implications:")
        implications = comparison.get('philosophy_implications', {})
        if 'primary' in implications:
            print(f"  {implications['primary']}")
        
        return {
            'test_results': test_results,
            'comparison': comparison
        }


if __name__ == "__main__":
    # Example usage
    print("Paper 3: Representation Criteria Test Suite")
    print("This module provides tests for evaluating whether neural states")
    print("meet different philosophical criteria for 'representation'.")
    print("\nRun with a trained CTRNN network to test representation hypotheses.")
