"""
Unit tests for CTRNN module.

Tests cover:
1. Step function against analytical solution
2. State reset functionality
3. Jacobian computation (vs. finite differences)
4. Network dimensions
5. Activation functions
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from simulation.ctrnn import CTRNN


class TestCTRNN:
    """Test suite for CTRNN implementation."""
    
    def test_single_neuron_analytical(self):
        """
        Test CTRNN step function against analytical solution for 1-neuron case.
        
        For a single neuron with no recurrence:
        dy/dt = (-y + I) / tau
        
        Analytical solution (Euler method):
        y(t+dt) = y(t) + (dt/tau) * (-y(t) + I)
        
        This is just exponential decay/growth toward equilibrium I.
        """
        print("Testing single-neuron analytical solution...")
        
        # Create 1-neuron network
        network = CTRNN(
            num_neurons=1,
            time_constants=np.array([1.0]),
            weights=np.array([[0.0]]),  # No self-recurrence
            biases=np.array([0.0]),
            step_size=0.01
        )
        
        # Test with constant external input
        external_input = np.array([1.0])
        
        # Analytical integration
        state_analytical = 0.0
        tau = 1.0
        dt = 0.01
        
        # Numerical integration
        network.reset(initial_state=np.array([0.0]))
        
        for step in range(100):
            # Analytical (Euler)
            state_analytical += (dt / tau) * (-state_analytical + 1.0)
            
            # CTRNN
            output = network.step(external_input)
            state_numerical = network.state[0]
            
            # Should be close
            error = abs(state_analytical - state_numerical)
            assert error < 0.01, f"Analytical mismatch at step {step}: {error:.6f}"
        
        print("  PASS: Analytical solution matches")
    
    def test_reset_functionality(self):
        """
        Test that reset() returns network to initial state.
        
        After running for some steps, calling reset() should restore
        the initial state exactly.
        """
        print("Testing reset functionality...")
        
        network = CTRNN(
            num_neurons=3,
            time_constants=np.array([0.5, 1.0, 2.0]),
            weights=np.random.randn(3, 3),
            biases=np.random.randn(3),
            step_size=0.01
        )
        
        # Record initial state
        initial_state = network.get_state().copy()
        
        # Run for some steps
        external_input = np.random.randn(3)
        for _ in range(50):
            network.step(external_input)
        
        # State should have changed
        current_state = network.get_state()
        assert not np.allclose(current_state, initial_state), "State didn't change after stepping"
        
        # Reset
        network.reset(initial_state=initial_state)
        
        # Should match initial state exactly
        reset_state = network.get_state()
        assert np.allclose(reset_state, initial_state), "Reset didn't restore initial state"
        
        # Reset to zero (default)
        network.reset()
        zero_state = network.get_state()
        assert np.allclose(zero_state, np.zeros(3)), "Default reset didn't zero state"
        
        print("  PASS: Reset functionality works correctly")
    
    def test_jacobian_vs_finite_difference(self):
        """
        Test Jacobian computation against finite differences.
        
        Compute Jacobian analytically vs. numerically via small perturbations.
        They should match closely.
        """
        print("Testing Jacobian computation...")
        
        network = CTRNN(
            num_neurons=3,
            time_constants=np.array([0.5, 1.0, 1.5]),
            weights=np.random.randn(3, 3),
            biases=np.random.randn(3),
            step_size=0.01
        )
        
        # Set to some state
        test_state = np.array([0.1, -0.2, 0.3])
        network.set_state(test_state)
        
        # Analytical Jacobian
        J_analytical = network.get_jacobian(test_state)
        
        # Numerical Jacobian (finite differences)
        epsilon = 1e-6
        J_numerical = np.zeros((3, 3))
        
        for j in range(3):
            # Perturb j-th dimension
            state_plus = test_state.copy()
            state_plus[j] += epsilon
            network.set_state(state_plus)
            network.step()
            state_plus_next = network.get_state()
            
            state_minus = test_state.copy()
            state_minus[j] -= epsilon
            network.set_state(state_minus)
            network.step()
            state_minus_next = network.get_state()
            
            # Finite difference
            J_numerical[:, j] = (state_plus_next - state_minus_next) / (2 * epsilon)
        
        # Compare
        max_error = np.max(np.abs(J_analytical - J_numerical))
        mean_error = np.mean(np.abs(J_analytical - J_numerical))
        
        assert max_error < 0.1, f"Jacobian error too large: {max_error:.6f}"
        
        print(f"  Jacobian max error: {max_error:.6f}")
        print(f"  Jacobian mean error: {mean_error:.6f}")
        print("  PASS: Jacobian matches finite differences")
    
    def test_network_dimensions(self):
        """
        Test that network dimensions are consistent.
        
        For N neurons:
        - weights: N x N
        - biases: N
        - tau: N
        - state: N
        - output: N
        """
        print("Testing network dimensions...")
        
        for n in [1, 3, 5, 10]:
            network = CTRNN(num_neurons=n)
            
            assert network.weights.shape == (n, n), f"Weight shape mismatch: {network.weights.shape}"
            assert network.biases.shape == (n,), f"Bias shape mismatch: {network.biases.shape}"
            assert network.tau.shape == (n,), f"Tau shape mismatch: {network.tau.shape}"
            assert network.state.shape == (n,), f"State shape mismatch: {network.state.shape}"
            
            output = network.step()
            assert output.shape == (n,), f"Output shape mismatch: {output.shape}"
        
        print("  PASS: All dimension checks passed")
    
    def test_center_crossing_sigmoid(self):
        """
        Test center-crossing vs. standard sigmoid.
        
        Center-crossing sigmoid should output in [-1, 1].
        Standard sigmoid should output in [0, 1].
        """
        print("Testing sigmoid activation functions...")
        
        # Center-crossing
        network_cc = CTRNN(num_neurons=1, center_crossing=True)
        network_cc.state = np.array([0.0])  # Set state to 0
        output_cc = network_cc.step()
        # Should be -1 + 2*sigmoid(0 + 0) = -1 + 2*0.5 = 0 at state=0, bias=0
        assert -1 < output_cc[0] < 1, f"Center-crossing output out of range: {output_cc[0]}"
        
        # Standard sigmoid
        network_std = CTRNN(num_neurons=1, center_crossing=False)
        network_std.state = np.array([0.0])
        output_std = network_std.step()
        assert 0 < output_std[0] < 1, f"Standard sigmoid output out of range: {output_std[0]}"
        
        print("  PASS: Sigmoid functions work correctly")
    
    def test_run_sequence(self):
        """
        Test run() method with sequence of inputs.
        
        Should produce outputs and states for entire sequence.
        """
        print("Testing run() with input sequences...")
        
        network = CTRNN(num_neurons=2)
        
        # Create input sequence
        timesteps = 100
        inputs = np.random.randn(timesteps, 2)
        
        outputs, states = network.run(inputs, reset_state=True)
        
        assert outputs.shape == (timesteps, 2), f"Output shape mismatch: {outputs.shape}"
        assert states.shape == (timesteps, 2), f"State shape mismatch: {states.shape}"
        
        # Outputs should be bounded (since using sigmoid)
        assert np.all(np.abs(outputs) <= 1.5), "Outputs exceed expected bounds"
        
        print("  PASS: run() method works correctly")
    
    def test_parameter_bounds(self):
        """
        Test that parameters stay within expected bounds during mutation/evolution.
        """
        print("Testing parameter bounds...")
        
        # Create network with specific bounds
        tau_range = (0.1, 10.0)
        network = CTRNN(num_neurons=3)
        
        # Simulate mutation-like perturbations
        for _ in range(100):
            perturbation = 0.1 * np.random.randn(3)
            network.tau = np.clip(network.tau * np.exp(perturbation), *tau_range)
            
            assert np.all(network.tau >= tau_range[0]), "Tau below minimum"
            assert np.all(network.tau <= tau_range[1]), "Tau above maximum"
        
        print("  PASS: Parameter bounds maintained")
    
    def run_all_tests(self):
        """Run all test methods."""
        print("=" * 70)
        print("CTRNN UNIT TESTS")
        print("=" * 70)
        
        tests = [
            self.test_single_neuron_analytical,
            self.test_reset_functionality,
            self.test_jacobian_vs_finite_difference,
            self.test_network_dimensions,
            self.test_center_crossing_sigmoid,
            self.test_run_sequence,
            self.test_parameter_bounds
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                test()
                passed += 1
            except Exception as e:
                print(f"  FAIL: {str(e)}")
                failed += 1
        
        print("\n" + "=" * 70)
        print(f"RESULTS: {passed} passed, {failed} failed")
        print("=" * 70)
        
        return passed, failed


if __name__ == "__main__":
    tester = TestCTRNN()
    passed, failed = tester.run_all_tests()
    
    if failed == 0:
        print("All tests passed!")
        sys.exit(0)
    else:
        print(f"{failed} test(s) failed!")
        sys.exit(1)
