"""
Paper 4: Perceptual Crossing, Dynamical Coupling, and Minimal Agency

Philosophical Problem:
    In Froese & Di Paolo's (2008) perceptual crossing paradigm, two agents
    exist in a constrained environment where each can perceive the other
    only under specific conditions. Paradoxically:
    
    1. No agent can tell if it's being perceived (unilateral perception is impossible)
    2. Both agents must coordinate their movements to achieve mutual perception
    3. This requires coupled dynamical interaction, not individual learning
    
    This challenges traditional cognitive science because:
    - Understanding emerges from INTERACTION, not individual computation
    - Autonomy requires being part of a system, not controlling it
    - Cognition is a property of coupled agent-environment system, not brain
    
    Philosophical Implications:
    - Neoenabledist: cognitive properties are relational, not intrinsic
    - Autonomy-based account of agency (Di Paolo)
    - Extended mind, but also "embedded mind" in strong sense
    
Experimental Approach:
    We implement the perceptual crossing setup and study:
    
    1. COORDINATION EMERGENCE
       - Do agents evolve to coordinate despite no explicit fitness for coordination?
       - What dynamical patterns enable mutual perception?
       - Is coordination chaotic, periodic, or at bifurcation?
    
    2. ASYMMETRY AND BREAKDOWN
       - One agent freezes behavior -> can other still engage?
       - Morphology asymmetry -> does one dominate?
       - Lagged sensory feedback -> does coordination still emerge?
    
    3. MINIMAL AUTONOMY
       - Define autonomy as "maintaining viability in face of perturbations"
       - Test if coupled system is more autonomous than individuals
       - Does autonomy emerge from coupling?
    
    4. INDIVIDUATION PROBLEM
       - When two agents are tightly coupled, are they one system or two?
       - Can we decompose the system into individuals?
       - What level of description is fundamental?

References:
    Froese, T., & Di Paolo, E. A. (2008). Emergence of joint action in two
        robotic agents coupled through kinetic energy transfer. New Ideas in
        Psychology, 26(3), 384-401.
    Di Paolo, E. A. (2009). Extended life. Topoi, 28(1), 9-21.
    Froese, T., Di Paolo, E. A., & Izquierdo, E. J. (2014). Collective
        behavior and shared intentionality. Frontiers in Human Neuroscience, 8, 909.
"""

import sys
import os
from typing import Dict, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.ctrnn import CTRNN
from simulation.evolutionary import MicrobialGA, GenotypeDecoder
from simulation.microworld import Agent, PerceptualCrossingEnv
from simulation.analysis import InformationAnalyzer


@dataclass
class CoordinationMetrics:
    """Metrics quantifying coordination between agents."""
    mutual_perception_events: int
    mutual_perception_duration_total: int
    coordination_duration_avg: float
    coordination_stability: float  # 0=chaotic, 1=stable
    synchronization_index: float
    joint_behavior_entropy: float
    phase_coupling: float


class PerceptualCrossingExperiment:
    """
    Full experiment for studying perceptual crossing and emergence of coordination.
    
    The experimental setup:
    - Two agents in 1D ring environment
    - Each agent has 2 motors and 2 sensors
    - Agent i can perceive agent j only when:
      * Both are within perception distance
      * Agent i is "oriented" toward agent j
    - Agents receive no explicit reward for perceiving each other
    
    Evolution objective:
    - Fitness can be based on simple task (e.g., moving in pattern)
    - Or based on longevity (staying alive without collisions)
    - Or based on activity level (keeping motors active)
    
    Key question: Despite no explicit fitness for coordination,
    do agents evolve to enable mutual perception?
    """
    
    def __init__(
        self,
        population_size: int = 30,
        num_generations: int = 300,
        circumference: float = 100.0,
        num_neurons: int = 4,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize perceptual crossing experiment.
        
        Args:
            population_size: Number of agent pairs to evolve.
            num_generations: Generations of evolution.
            circumference: Environment circumference.
            num_neurons: Number of neurons per agent.
            seed: Random seed for reproducibility.
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.circumference = circumference
        self.num_neurons = num_neurons
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        self.agent_pairs = []  # Evolved agent pairs
        self.evolution_history = []
        self.best_coordination = None
        self.decoder = GenotypeDecoder(num_neurons=num_neurons, include_gains=False)
    
    def phase1_evolution_for_task(
        self,
        task: str = "longevity",
        episode_length: int = 500,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Phase 1: Evolve agent pairs on task WITHOUT explicit coordination objective.
        
        Tasks:
        - "longevity": Maximize time alive without collision
        - "activity": Maintain high motor activity (keep moving)
        - "pattern": Move in specific spatial pattern
        
        Despite no coordination objective, we examine post-evolution
        whether agents have learned to enable mutual perception.
        
        Args:
            task: Which task to evolve for ('longevity', 'activity', or 'pattern').
            episode_length: Length of each episode in timesteps.
            verbose: If True, print progress.
        
        Returns:
            (best_genotype_pair, best_network_pair, evolution_history)
        
        Implementation:
            1. Create population of agent pairs (genotypes represent networks for both agents)
            2. For each generation:
               a. Run each pair for episode_length timesteps
               b. Evaluate fitness based on task
               c. Reproduce best pairs (with mutations)
            3. Record evolution history
        """
        if verbose:
            print(f"Phase 1: Evolving agents for '{task}' task...")
            print(f"  Generations: {self.num_generations}")
            print(f"  Population: {self.population_size}")
            print(f"  Episode length: {episode_length}")
        
        # Genotype size is 2x (both agents)
        genotype_size = self.decoder.genotype_size * 2
        
        def fitness_function(genotype: np.ndarray) -> float:
            """Evaluate fitness for agent pair."""
            # Split genotype into two networks
            gen1 = genotype[:self.decoder.genotype_size]
            gen2 = genotype[self.decoder.genotype_size:]
            
            # Decode to network parameters
            params1 = self.decoder.decode(gen1)
            params2 = self.decoder.decode(gen2)
            
            # Create networks
            net1 = CTRNN(num_neurons=self.num_neurons)
            net1.weights = params1['weights']
            net1.biases = params1['biases']
            net1.tau = params1['tau']
            
            net2 = CTRNN(num_neurons=self.num_neurons)
            net2.weights = params2['weights']
            net2.biases = params2['biases']
            net2.tau = params2['tau']
            
            # Create agents
            agent1 = Agent(radius=1.0, max_speed=1.0)
            agent2 = Agent(radius=1.0, max_speed=1.0)
            
            # Random positions on ring
            agent1.position = np.array([np.random.uniform(0, self.circumference), 0.0])
            agent2.position = np.array([np.random.uniform(0, self.circumference), 0.0])
            
            # Create environment
            env = PerceptualCrossingEnv(circumference=self.circumference)
            env.set_agents(agent1, agent2)
            
            fitness = 0.0
            collision_count = 0
            
            # Run episode
            for step in range(episode_length):
                # Get sensor readings
                # Check mutual perception
                dist = min(
                    abs(agent1.position[0] - agent2.position[0]),
                    self.circumference - abs(agent1.position[0] - agent2.position[0])
                )
                
                perception_dist = 5.0
                perceive = dist < perception_dist
                
                sensor1 = np.array([float(perceive), float(perceive)])
                sensor2 = np.array([float(perceive), float(perceive)])
                
                # Pad to network size
                padded_sensor1 = np.zeros(self.num_neurons)
                padded_sensor1[:2] = sensor1
                
                padded_sensor2 = np.zeros(self.num_neurons)
                padded_sensor2[:2] = sensor2
                
                # Neural control
                output1 = net1.step(padded_sensor1)
                output2 = net2.step(padded_sensor2)
                
                # Motor commands
                left_motor1 = output1[0] if len(output1) > 0 else 0.0
                right_motor1 = output1[1] if len(output1) > 1 else output1[0] if len(output1) > 0 else 0.0
                
                left_motor2 = output2[0] if len(output2) > 0 else 0.0
                right_motor2 = output2[1] if len(output2) > 1 else output2[0] if len(output2) > 0 else 0.0
                
                agent1.set_motor_commands(left_motor1, right_motor1)
                agent2.set_motor_commands(left_motor2, right_motor2)
                
                # Update environment
                env.step()
                
                # Check for collisions
                agent_dist = min(
                    abs(agent1.position[0] - agent2.position[0]),
                    self.circumference - abs(agent1.position[0] - agent2.position[0])
                )
                if agent_dist < 2.0:
                    collision_count += 1
                
                # Evaluate task
                if task == "longevity":
                    # Penalize collisions, reward survival
                    if agent_dist >= 2.0:
                        fitness += 1.0 / episode_length
                elif task == "activity":
                    # Reward motor activity
                    fitness += (np.abs(left_motor1) + np.abs(right_motor1) +
                               np.abs(left_motor2) + np.abs(right_motor2)) / 4.0
                elif task == "pattern":
                    # Reward specific pattern: agents oscillate in position
                    pos_variance = np.std([agent1.position[0], agent2.position[0]])
                    fitness += pos_variance / 50.0
            
            return fitness / episode_length
        
        # Evolutionary algorithm
        ga = MicrobialGA(
            genotype_size=genotype_size,
            fitness_function=fitness_function,
            population_size=self.population_size,
            mutation_std=0.2,
            seed=self.seed
        )
        
        # Run evolution
        for gen in range(self.num_generations):
            best_genotype, best_fitness = ga.step()
            self.evolution_history.append(best_fitness)
            
            if verbose and (gen % max(1, self.num_generations // 10) == 0 or gen == self.num_generations - 1):
                print(f"  Generation {gen:3d}: best_fitness = {best_fitness:.4f}")
        
        # Get best pair
        best_genotype = ga.get_best_individual()
        gen1 = best_genotype[:self.decoder.genotype_size]
        gen2 = best_genotype[self.decoder.genotype_size:]
        
        params1 = self.decoder.decode(gen1)
        params2 = self.decoder.decode(gen2)
        
        net1 = CTRNN(num_neurons=self.num_neurons)
        net1.weights = params1['weights']
        net1.biases = params1['biases']
        net1.tau = params1['tau']
        
        net2 = CTRNN(num_neurons=self.num_neurons)
        net2.weights = params2['weights']
        net2.biases = params2['biases']
        net2.tau = params2['tau']
        
        self.best_network_pair = (net1, net2)
        self.best_genotype_pair = (gen1, gen2)
        
        return gen1, gen2, self.evolution_history
    
    def phase2_analyze_coordination(self, episode_length: int = 1000) -> CoordinationMetrics:
        """
        Phase 2: Analyze coordination patterns in evolved agents.
        
        Questions:
        - Do agents perceive each other more than would occur by chance?
        - What dynamical patterns enable perception?
        - Is coordination predictable or chaotic?
        - How stable is the coordination?
        
        Args:
            episode_length: Length of analysis episode.
        
        Returns:
            CoordinationMetrics summarizing coordination quality.
        
        Implementation:
            1. Run best evolved pair for long episode
            2. Record when mutual perception occurs
            3. Analyze state-space trajectories during coordination
            4. Compute synchronization metrics
        """
        print("Phase 2: Analyzing coordination patterns...")
        
        if self.best_network_pair is None:
            raise ValueError("Must run phase1_evolution first")
        
        net1, net2 = self.best_network_pair
        
        agent1 = Agent(radius=1.0, max_speed=1.0)
        agent2 = Agent(radius=1.0, max_speed=1.0)
        
        agent1.position = np.array([25.0, 0.0])
        agent2.position = np.array([75.0, 0.0])
        
        env = PerceptualCrossingEnv(circumference=self.circumference)
        env.set_agents(agent1, agent2)
        
        # Record data
        perception_events = []
        positions1 = []
        positions2 = []
        states1 = []
        states2 = []
        
        mutual_perception_count = 0
        mutual_perception_duration = 0
        
        for step in range(episode_length):
            # Check perception
            dist = min(
                abs(agent1.position[0] - agent2.position[0]),
                self.circumference - abs(agent1.position[0] - agent2.position[0])
            )
            
            perceive = dist < 5.0
            perception_events.append(perceive)
            if perceive:
                mutual_perception_count += 1
                mutual_perception_duration += 1
            elif mutual_perception_duration > 0:
                mutual_perception_duration = 0
            
            sensor1 = np.array([float(perceive), float(perceive)])
            sensor2 = np.array([float(perceive), float(perceive)])
            
            padded_sensor1 = np.zeros(self.num_neurons)
            padded_sensor1[:2] = sensor1
            
            padded_sensor2 = np.zeros(self.num_neurons)
            padded_sensor2[:2] = sensor2
            
            output1 = net1.step(padded_sensor1)
            output2 = net2.step(padded_sensor2)
            
            left_motor1 = output1[0] if len(output1) > 0 else 0.0
            right_motor1 = output1[1] if len(output1) > 1 else output1[0] if len(output1) > 0 else 0.0
            
            left_motor2 = output2[0] if len(output2) > 0 else 0.0
            right_motor2 = output2[1] if len(output2) > 1 else output2[0] if len(output2) > 0 else 0.0
            
            agent1.set_motor_commands(left_motor1, right_motor1)
            agent2.set_motor_commands(left_motor2, right_motor2)
            
            env.step()
            
            positions1.append(agent1.position[0])
            positions2.append(agent2.position[0])
            states1.append(net1.get_state().copy())
            states2.append(net2.get_state().copy())
        
        # Compute metrics
        positions1 = np.array(positions1)
        positions2 = np.array(positions2)
        states1 = np.array(states1)
        states2 = np.array(states2)
        
        perception_rate = mutual_perception_count / episode_length
        random_expectation = 5.0 / self.circumference  # Expected rate if random
        
        # Average coordination duration (consecutive perception events)
        coordination_durations = []
        current_duration = 0
        for perceive in perception_events:
            if perceive:
                current_duration += 1
            elif current_duration > 0:
                coordination_durations.append(current_duration)
                current_duration = 0
        
        avg_coordination_duration = (
            np.mean(coordination_durations) if coordination_durations else 0.0
        )
        
        # Stability: how predictable are the dynamics?
        # Use Lyapunov-like measure: sensitivity to initial conditions
        pos_variance = np.var(positions1 - positions2)
        stability = 1.0 / (1.0 + pos_variance)
        
        # Synchronization: phase coupling between oscillatory behavior
        # Use cross-correlation of velocities
        vel1 = np.diff(positions1)
        vel2 = np.diff(positions2)
        
        if len(vel1) > 0 and len(vel2) > 0:
            correlation = np.corrcoef(vel1, vel2)[0, 1]
            if not np.isnan(correlation):
                sync_index = (correlation + 1.0) / 2.0  # Map [-1,1] to [0,1]
            else:
                sync_index = 0.0
        else:
            sync_index = 0.0
        
        # Joint behavior entropy: how much information is in the joint state?
        joint_states = np.concatenate([states1, states2], axis=1)
        # Discretize and compute entropy
        discretized = np.digitize(joint_states.flatten(), np.linspace(-1, 1, 10))
        unique_states, counts = np.unique(discretized, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropy_normalized = entropy / np.log(len(unique_states) + 1)
        
        metrics = CoordinationMetrics(
            mutual_perception_events=mutual_perception_count,
            mutual_perception_duration_total=mutual_perception_duration,
            coordination_duration_avg=float(avg_coordination_duration),
            coordination_stability=float(stability),
            synchronization_index=float(sync_index),
            joint_behavior_entropy=float(entropy_normalized),
            phase_coupling=float(np.clip((perception_rate - random_expectation) / 0.1, 0, 1))
        )
        
        print(f"  Mutual perception events: {mutual_perception_count}")
        print(f"  Average coordination duration: {avg_coordination_duration:.2f} steps")
        print(f"  Synchronization index: {sync_index:.3f}")
        
        return metrics
    
    def phase3_asymmetry_perturbation(self) -> Dict[str, float]:
        """
        Phase 3: Test robustness to asymmetries and perturbations.
        
        Questions:
        - What happens if one agent has frozen behavior?
        - What if sensors are delayed/noisy?
        - What if morphology differs?
        - Does coordination require symmetry?
        
        Returns:
            Dictionary of metrics for different perturbations.
        
        Implementation:
            1. Take best evolved pair
            2. Apply perturbations:
               a. Freeze one agent's motors
               b. Add noise to sensors
               c. Make one agent faster/slower
               d. Misalign sensor orientations
            3. Measure if coordination still possible
        """
        print("Phase 3: Testing robustness to perturbations...")
        
        if self.best_network_pair is None:
            raise ValueError("Must run phase1_evolution first")
        
        perturbations = {}
        episode_length = 500
        
        # Test 1: Freeze one agent's motors
        print("  Testing frozen agent...")
        net1, net2 = self.best_network_pair
        net1_frozen = CTRNN(num_neurons=self.num_neurons)
        net1_frozen.weights = net1.weights.copy()
        net1_frozen.biases = net1.biases.copy()
        net1_frozen.tau = net1.tau.copy()
        
        agent1 = Agent(radius=1.0, max_speed=1.0)
        agent2 = Agent(radius=1.0, max_speed=1.0)
        agent1.position = np.array([25.0, 0.0])
        agent2.position = np.array([75.0, 0.0])
        env_test = PerceptualCrossingEnv(circumference=self.circumference)
        env_test.set_agents(agent1, agent2)

        coordination_score = 0.0
        for step in range(episode_length):
            dist = min(
                abs(agent1.position[0] - agent2.position[0]),
                self.circumference - abs(agent1.position[0] - agent2.position[0])
            )
            perceive = dist < 5.0

            sensor1 = np.array([float(perceive), float(perceive)])
            sensor2 = np.array([float(perceive), float(perceive)])

            padded_sensor1 = np.zeros(self.num_neurons)
            padded_sensor1[:2] = sensor1
            padded_sensor2 = np.zeros(self.num_neurons)
            padded_sensor2[:2] = sensor2

            output1 = net1_frozen.step(padded_sensor1)
            output2 = net2.step(padded_sensor2)
            
            # Agent 1 is frozen (no motors)
            agent1.set_motor_commands(0.0, 0.0)
            left_motor2 = output2[0] if len(output2) > 0 else 0.0
            right_motor2 = output2[1] if len(output2) > 1 else output2[0] if len(output2) > 0 else 0.0
            agent2.set_motor_commands(left_motor2, right_motor2)

            env_test.step()
            
            if perceive:
                coordination_score += 1.0
        
        perturbations['frozen_agent'] = coordination_score / episode_length
        print(f"    Frozen agent coordination: {perturbations['frozen_agent']:.3f}")
        
        # Test 2: Sensor noise
        print("  Testing sensor noise...")
        coordination_score = 0.0
        agent1 = Agent(radius=1.0, max_speed=1.0)
        agent2 = Agent(radius=1.0, max_speed=1.0)
        agent1.position = np.array([25.0, 0.0])
        agent2.position = np.array([75.0, 0.0])
        env_test = PerceptualCrossingEnv(circumference=self.circumference)
        env_test.set_agents(agent1, agent2)
        net1.reset()
        net2.reset()

        for step in range(episode_length):
            dist = min(
                abs(agent1.position[0] - agent2.position[0]),
                self.circumference - abs(agent1.position[0] - agent2.position[0])
            )
            perceive = dist < 5.0

            # Add noise
            sensor1 = np.array([float(perceive), float(perceive)]) + 0.2 * np.random.randn(2)
            sensor1 = np.clip(sensor1, 0, 1)
            sensor2 = np.array([float(perceive), float(perceive)]) + 0.2 * np.random.randn(2)
            sensor2 = np.clip(sensor2, 0, 1)

            padded_sensor1 = np.zeros(self.num_neurons)
            padded_sensor1[:2] = sensor1
            padded_sensor2 = np.zeros(self.num_neurons)
            padded_sensor2[:2] = sensor2

            output1 = net1.step(padded_sensor1)
            output2 = net2.step(padded_sensor2)

            left_motor1 = output1[0] if len(output1) > 0 else 0.0
            right_motor1 = output1[1] if len(output1) > 1 else output1[0] if len(output1) > 0 else 0.0
            left_motor2 = output2[0] if len(output2) > 0 else 0.0
            right_motor2 = output2[1] if len(output2) > 1 else output2[0] if len(output2) > 0 else 0.0

            agent1.set_motor_commands(left_motor1, right_motor1)
            agent2.set_motor_commands(left_motor2, right_motor2)

            env_test.step()

            if perceive:
                coordination_score += 1.0

        perturbations['sensor_noise'] = coordination_score / episode_length
        print(f"    Sensor noise coordination: {perturbations['sensor_noise']:.3f}")
        
        # Test 3: Speed mismatch
        print("  Testing speed mismatch...")
        coordination_score = 0.0
        agent1 = Agent(radius=1.0, max_speed=0.5)  # 0.5x speed
        agent2 = Agent(radius=1.0, max_speed=2.0)  # 2x speed
        agent1.position = np.array([25.0, 0.0])
        agent2.position = np.array([75.0, 0.0])
        env_test = PerceptualCrossingEnv(circumference=self.circumference)
        env_test.set_agents(agent1, agent2)
        net1.reset()
        net2.reset()

        for step in range(episode_length):
            dist = min(
                abs(agent1.position[0] - agent2.position[0]),
                self.circumference - abs(agent1.position[0] - agent2.position[0])
            )
            perceive = dist < 5.0

            sensor1 = np.array([float(perceive), float(perceive)])
            sensor2 = np.array([float(perceive), float(perceive)])

            padded_sensor1 = np.zeros(self.num_neurons)
            padded_sensor1[:2] = sensor1
            padded_sensor2 = np.zeros(self.num_neurons)
            padded_sensor2[:2] = sensor2

            output1 = net1.step(padded_sensor1)
            output2 = net2.step(padded_sensor2)

            left_motor1 = output1[0] if len(output1) > 0 else 0.0
            right_motor1 = output1[1] if len(output1) > 1 else output1[0] if len(output1) > 0 else 0.0
            left_motor2 = output2[0] if len(output2) > 0 else 0.0
            right_motor2 = output2[1] if len(output2) > 1 else output2[0] if len(output2) > 0 else 0.0

            agent1.set_motor_commands(left_motor1, right_motor1)
            agent2.set_motor_commands(left_motor2, right_motor2)

            env_test.step()

            if perceive:
                coordination_score += 1.0

        perturbations['speed_mismatch'] = coordination_score / episode_length
        print(f"    Speed mismatch coordination: {perturbations['speed_mismatch']:.3f}")

        return perturbations
    
    def phase4_individuation_analysis(self) -> Dict[str, object]:
        """
        Phase 4: Analyze individuation - is the coupled system one or two agents?
        
        Approaches:
        1. INFORMATIONAL INDIVIDUATION: Can we decompose information flow?
           - Compute mutual information within vs. between agents
           - If MI(agent1_internal) >> MI(agent1-agent2), they're separate
        
        2. DYNAMICAL INDIVIDUATION: Can we decompose state space?
           - Try to decompose joint state space into independent subspaces
           - Measures: Lyapunov exponents, dynamical decoupling
        
        3. FUNCTIONAL INDIVIDUATION: Do agents have independent functions?
           - Can each agent function independently during coordination?
           - How much mutual dependence is there?
        
        Returns:
            Dictionary with individuation metrics.
        """
        print("Phase 4: Analyzing individuation...")
        
        if self.best_network_pair is None:
            raise ValueError("Must run phase1_evolution first")
        
        net1, net2 = self.best_network_pair
        
        # Record state evolution
        agent1 = Agent(radius=1.0, max_speed=1.0)
        agent2 = Agent(radius=1.0, max_speed=1.0)
        agent1.position = np.array([25.0, 0.0])
        agent2.position = np.array([75.0, 0.0])
        env_test = PerceptualCrossingEnv(circumference=self.circumference)
        env_test.set_agents(agent1, agent2)

        states1 = []
        states2 = []

        for step in range(500):
            dist = min(
                abs(agent1.position[0] - agent2.position[0]),
                self.circumference - abs(agent1.position[0] - agent2.position[0])
            )
            perceive = dist < 5.0

            sensor1 = np.array([float(perceive), float(perceive)])
            sensor2 = np.array([float(perceive), float(perceive)])

            padded_sensor1 = np.zeros(self.num_neurons)
            padded_sensor1[:2] = sensor1
            padded_sensor2 = np.zeros(self.num_neurons)
            padded_sensor2[:2] = sensor2

            output1 = net1.step(padded_sensor1)
            output2 = net2.step(padded_sensor2)

            left_motor1 = output1[0] if len(output1) > 0 else 0.0
            right_motor1 = output1[1] if len(output1) > 1 else output1[0] if len(output1) > 0 else 0.0
            left_motor2 = output2[0] if len(output2) > 0 else 0.0
            right_motor2 = output2[1] if len(output2) > 1 else output2[0] if len(output2) > 0 else 0.0

            agent1.set_motor_commands(left_motor1, right_motor1)
            agent2.set_motor_commands(left_motor2, right_motor2)

            env_test.step()
            
            states1.append(net1.get_state().copy())
            states2.append(net2.get_state().copy())
        
        states1 = np.array(states1)
        states2 = np.array(states2)
        
        # Compute information-theoretic individuation
        # MI within each agent
        mi_within1 = 0.0
        mi_within2 = 0.0
        
        for i in range(min(self.num_neurons, 2)):
            for j in range(i + 1, min(self.num_neurons, 2)):
                mi1 = InformationAnalyzer.mutual_information(states1[:, i], states1[:, j], bins=3)
                mi2 = InformationAnalyzer.mutual_information(states2[:, i], states2[:, j], bins=3)
                mi_within1 += mi1
                mi_within2 += mi2
        
        # MI between agents
        mi_between = 0.0
        for i in range(min(self.num_neurons, 2)):
            for j in range(min(self.num_neurons, 2)):
                mi_between += InformationAnalyzer.mutual_information(states1[:, i], states2[:, j], bins=3)
        
        # Ratio
        total_within = mi_within1 + mi_within2
        if total_within > 0:
            individuation_ratio = (total_within - mi_between) / total_within
        else:
            individuation_ratio = 0.0
        
        # Dynamical coupling: cross-correlation of state trajectories
        coupling_strength = 0.0
        if len(states1) > 1 and len(states2) > 1:
            for i in range(min(self.num_neurons, 2)):
                for j in range(min(self.num_neurons, 2)):
                    corr = np.corrcoef(states1[:, i], states2[:, j])[0, 1]
                    if not np.isnan(corr):
                        coupling_strength += abs(corr)
        
        coupling_strength /= max(1, min(self.num_neurons, 2) ** 2)
        
        # Autonomy: how much do agents depend on each other?
        # High mutual information between agents -> low autonomy
        autonomy_of_pair = 1.0 - np.clip(coupling_strength, 0, 1)
        autonomy_of_individuals = np.clip((total_within - mi_between) / (total_within + 1e-6), 0, 1)
        
        individuation = {
            'informational_coupling_strength': float(coupling_strength),
            'informational_individuation': float(np.clip(individuation_ratio, 0, 1)),
            'dynamical_coupling_strength': float(coupling_strength),
            'functional_interdependence': float(coupling_strength),
            'autonomy_of_pair': float(autonomy_of_pair),
            'autonomy_of_individuals': float(autonomy_of_individuals)
        }
        
        print(f"  Informational individuation: {individuation['informational_individuation']:.3f}")
        print(f"  Coupling strength: {coupling_strength:.3f}")
        
        return individuation
    
    def run_full_experiment(
        self,
        task: str = "longevity",
        verbose: bool = True
    ) -> Dict[str, object]:
        """
        Execute complete perceptual crossing experiment.
        
        Args:
            task: Task to evolve agents for ('longevity', 'activity', or 'pattern').
            verbose: If True, print detailed progress.
        
        Returns:
            Complete results from all phases.
        """
        print("=" * 70)
        print("PERCEPTUAL CROSSING EXPERIMENT")
        print(f"Task: {task}")
        print("=" * 70)
        
        # Phase 1: Evolution
        gen1, gen2, evolution_hist = self.phase1_evolution_for_task(task, verbose=verbose)
        
        # Phase 2: Coordination analysis
        print("\nPhase 2: Analyzing coordination...")
        coordination_metrics = self.phase2_analyze_coordination()
        
        # Phase 3: Perturbations
        print("\nPhase 3: Testing robustness...")
        perturbation_results = self.phase3_asymmetry_perturbation()
        
        # Phase 4: Individuation
        print("\nPhase 4: Analyzing individuation...")
        individuation = self.phase4_individuation_analysis()
        
        results = {
            'task': task,
            'evolution_history': evolution_hist,
            'coordination_metrics': {
                'mutual_perception_events': coordination_metrics.mutual_perception_events,
                'coordination_duration_avg': coordination_metrics.coordination_duration_avg,
                'coordination_stability': coordination_metrics.coordination_stability,
                'synchronization_index': coordination_metrics.synchronization_index,
                'joint_behavior_entropy': coordination_metrics.joint_behavior_entropy,
                'phase_coupling': coordination_metrics.phase_coupling
            },
            'perturbation_results': perturbation_results,
            'individuation_analysis': individuation
        }
        
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"\nCoordination:")
        print(f"  Synchronization: {coordination_metrics.synchronization_index:.3f}")
        print(f"  Stability: {coordination_metrics.coordination_stability:.3f}")
        print(f"\nIndividuation:")
        print(f"  Coupling strength: {individuation['dynamical_coupling_strength']:.3f}")
        print(f"  Individual autonomy: {individuation['autonomy_of_individuals']:.3f}")
        
        return results


if __name__ == "__main__":
    print("Paper 4: Perceptual Crossing Experiment")
