"""
Curriculum-Based MARL Training for Renewable Energy Integration

Implements two-phase curriculum learning to gradually introduce renewable energy
complexity, improving agent adaptation to variable renewable resources.

Based on "The AI Economist" curriculum approach adapted for smart grids.
"""

import asyncio
import numpy as np
import torch
import logging
import sys
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.pre_training import AgentPreTrainer
from src.agents.generator_agent import GeneratorAgent
from src.agents.storage_agent import StorageAgent
from src.agents.consumer_agent import ConsumerAgent
from src.agents.grid_operator_agent import GridOperatorAgent
from src.coordination.multi_agent_system import SmartGridSimulation


@dataclass
class CurriculumConfig:
    """Configuration for curriculum-based training"""
    # Phase 1: Stable foundation training
    phase1_steps: int = 50_000_000  # 50M steps like AI Economist
    phase1_renewable_penetration: float = 0.05  # Start with 5% renewables
    phase1_weather_variability: float = 0.1  # Minimal weather variation
    
    # Phase 2: Gradual complexity introduction
    phase2_steps: int = 400_000_000  # 400M steps
    annealing_steps: int = 54_000_000  # 54M steps for full annealing
    max_renewable_penetration: float = 0.8  # Up to 80% renewables
    max_weather_variability: float = 1.0  # Full weather variation
    
    # Curriculum schedules
    renewable_schedule: str = "linear"  # "linear", "exponential", "step"
    weather_schedule: str = "linear"
    demand_schedule: str = "linear"
    
    # Training parameters
    entropy_coefficient_agent: float = 0.025
    entropy_coefficient_grid_op: float = 0.1
    learning_rate_agent: float = 0.0003
    learning_rate_grid_op: float = 0.0001


class RenewableCurriculumTrainer:
    """Curriculum-based trainer for renewable energy integration"""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.logger = logging.getLogger("CurriculumTrainer")
        self.training_metrics = {
            "renewable_penetration_history": [],
            "weather_variability_history": [],
            "performance_metrics": [],
            "violation_counts": [],
            "learning_curves": {}
        }
        
    def create_curriculum_scenario(self, step: int, phase: str) -> Dict[str, Any]:
        """Create training scenario based on curriculum step"""
        
        if phase == "phase1":
            # Stable foundation training
            return {
                "renewable_penetration": self.config.phase1_renewable_penetration,
                "weather_variability": self.config.phase1_weather_variability,
                "demand_variability": 0.1,
                "intermittency_factor": 0.1,
                "ramping_events": False,
                "extreme_weather": False
            }
        
        elif phase == "phase2":
            # Calculate annealing progress
            annealing_progress = min(1.0, step / self.config.annealing_steps)
            
            # Renewable penetration curriculum
            renewable_penetration = self._calculate_curriculum_value(
                start_value=self.config.phase1_renewable_penetration,
                end_value=self.config.max_renewable_penetration,
                progress=annealing_progress,
                schedule=self.config.renewable_schedule
            )
            
            # Weather variability curriculum
            weather_variability = self._calculate_curriculum_value(
                start_value=self.config.phase1_weather_variability,
                end_value=self.config.max_weather_variability,
                progress=annealing_progress,
                schedule=self.config.weather_schedule
            )
            
            # Demand variability curriculum
            demand_variability = self._calculate_curriculum_value(
                start_value=0.1,
                end_value=0.8,
                progress=annealing_progress,
                schedule=self.config.demand_schedule
            )
            
            return {
                "renewable_penetration": renewable_penetration,
                "weather_variability": weather_variability,
                "demand_variability": demand_variability,
                "intermittency_factor": weather_variability,
                "ramping_events": annealing_progress > 0.5,
                "extreme_weather": annealing_progress > 0.7,
                "duck_curve_intensity": annealing_progress * 0.8
            }
        
        # Default fallback scenario
        return {
            "renewable_penetration": 0.3,
            "weather_variability": 0.5,
            "demand_variability": 0.3,
            "intermittency_factor": 0.5,
            "ramping_events": False,
            "extreme_weather": False
        }
    
    def _calculate_curriculum_value(self, start_value: float, end_value: float, 
                                   progress: float, schedule: str) -> float:
        """Calculate curriculum value based on schedule type"""
        
        if schedule == "linear":
            return start_value + (end_value - start_value) * progress
        
        elif schedule == "exponential":
            # Exponential annealing for more gradual introduction
            exp_progress = (np.exp(progress * 3) - 1) / (np.exp(3) - 1)
            return start_value + (end_value - start_value) * exp_progress
        
        elif schedule == "step":
            # Step function for discrete complexity increases
            if progress < 0.25:
                return start_value
            elif progress < 0.5:
                return start_value + (end_value - start_value) * 0.25
            elif progress < 0.75:
                return start_value + (end_value - start_value) * 0.5
            else:
                return end_value
        
        return start_value + (end_value - start_value) * progress
    
    async def train_with_curriculum(self, simulation: SmartGridSimulation) -> Dict[str, Any]:
        """Main curriculum training loop"""
        
        self.logger.info("Starting curriculum-based training for renewable integration")
        
        # Phase 1: Foundation training
        self.logger.info("Phase 1: Foundation training with stable grid conditions")
        phase1_results = await self._run_training_phase(
            simulation, "phase1", self.config.phase1_steps
        )
        
        # Phase 2: Gradual complexity introduction
        self.logger.info("Phase 2: Gradual renewable complexity introduction")
        phase2_results = await self._run_training_phase(
            simulation, "phase2", self.config.phase2_steps
        )
        
        # Compile final results
        training_results = {
            "phase1_results": phase1_results,
            "phase2_results": phase2_results,
            "curriculum_config": self.config.__dict__,
            "training_metrics": self.training_metrics,
            "final_performance": await self._evaluate_final_performance(simulation)
        }
        
        # Save training results
        self._save_training_results(training_results)
        
        return training_results
    
    async def _run_training_phase(self, simulation: SmartGridSimulation, 
                                 phase: str, total_steps: int) -> Dict[str, Any]:
        """Run a single training phase"""
        
        phase_metrics = {
            "total_steps": total_steps,
            "performance_history": [],
            "scenario_history": [],
            "violation_history": []
        }
        
        for step in range(total_steps):
            # Create curriculum scenario for this step
            scenario = self.create_curriculum_scenario(step, phase)
            
            # Apply scenario to simulation
            await self._apply_scenario_to_simulation(simulation, scenario)
            
            # Run training step
            step_results = await simulation.run_simulation_step()
            
            # ✅ ACTUAL RL TRAINING: Call agent learning methods
            if step_results:
                await self._train_agents_from_results(simulation, step_results)
            
            # Record metrics every 10,000 steps
            if step % 10_000 == 0:
                metrics = await self._collect_training_metrics(simulation, scenario, step)
                phase_metrics["performance_history"].append(metrics)
                phase_metrics["scenario_history"].append(scenario)
                
                # Log progress
                self.logger.info(
                    f"{phase} Step {step:,}: "
                    f"Renewable: {scenario['renewable_penetration']:.2%}, "
                    f"Weather Var: {scenario['weather_variability']:.2f}, "
                    f"Performance: {metrics.get('overall_score', 0):.2f}"
                )
            
            # Update training metrics
            self.training_metrics["renewable_penetration_history"].append(
                scenario["renewable_penetration"]
            )
            self.training_metrics["weather_variability_history"].append(
                scenario["weather_variability"]
            )
        
        return phase_metrics
    
    async def _train_agents_from_results(self, simulation: SmartGridSimulation, 
                                       step_results: Dict[str, Any]) -> None:
        """Call actual RL training for each agent after simulation step"""
        
        market_result = step_results.get("market_result", {})
        
        for agent_id, agent in simulation.agents.items():
            if hasattr(agent, 'learn_from_market_result'):
                try:
                    if isinstance(agent, GeneratorAgent):
                        # DQN training for generators
                        agent.learn_from_market_result(market_result)
                    elif isinstance(agent, StorageAgent):
                        # Actor-Critic training for storage
                        agent.learn_from_market_result(market_result)
                    elif isinstance(agent, ConsumerAgent):
                        # MADDPG training for consumers (needs other agent actions)
                        other_actions = self._get_other_agent_actions(simulation, agent_id)
                        agent.learn_from_market_result(market_result, other_actions)
                    
                    self.logger.debug(f"Trained agent {agent_id} from market result")
                except Exception as e:
                    self.logger.warning(f"Failed to train agent {agent_id}: {e}")
    
    def _get_other_agent_actions(self, simulation: SmartGridSimulation, 
                               current_agent_id: str) -> List[np.ndarray]:
        """Get actions from other agents for MADDPG training"""
        
        other_actions = []
        for agent_id, agent in simulation.agents.items():
            if agent_id != current_agent_id and hasattr(agent, 'current_action'):
                # Get the agent's last action if available
                current_action = getattr(agent, 'current_action', None)
                if current_action is not None:
                    other_actions.append(current_action)
                else:
                    # Create dummy action if no action recorded
                    other_actions.append(np.random.rand(4))
        
        # Ensure we have at least 2 other agent actions for MADDPG
        while len(other_actions) < 2:
            other_actions.append(np.random.rand(4))
        
        return other_actions[:2]  # Return max 2 other agents
    
    async def _apply_scenario_to_simulation(self, simulation: SmartGridSimulation, 
                                          scenario: Dict[str, Any]) -> None:
        """Apply curriculum scenario parameters to simulation"""
        
        # Update renewable penetration
        await self._set_renewable_penetration(simulation, scenario["renewable_penetration"])
        
        # Update weather variability
        await self._set_weather_variability(simulation, scenario["weather_variability"])
        
        # Update demand patterns
        await self._set_demand_variability(simulation, scenario["demand_variability"])
        
        # Enable/disable specific challenges
        if scenario.get("ramping_events", False):
            await self._enable_ramping_events(simulation)
        
        if scenario.get("extreme_weather", False):
            await self._enable_extreme_weather(simulation)
    
    async def _set_renewable_penetration(self, simulation: SmartGridSimulation, 
                                       penetration: float) -> None:
        """Set renewable energy penetration level"""
        
        # Calculate renewable capacity needed  
        total_capacity = sum(
            getattr(agent, 'generator_state').max_capacity_mw 
            for agent in simulation.agents.values() 
            if hasattr(agent, 'generator_state')
        )
        
        renewable_capacity = total_capacity * penetration
        
        # Update renewable agents
        renewable_agents = [
            agent for agent in simulation.agents.values()
            if hasattr(agent, 'generator_state') and 
            getattr(agent, 'generator_state').emissions_rate_kg_co2_per_mwh == 0
        ]
        
        if renewable_agents:
            capacity_per_agent = renewable_capacity / len(renewable_agents)
            for agent in renewable_agents:
                gen_state = getattr(agent, 'generator_state')
                gen_state.max_capacity_mw = capacity_per_agent
                gen_state.online_status = True
    
    async def _set_weather_variability(self, simulation: SmartGridSimulation, 
                                     variability: float) -> None:
        """Set weather variability level"""
        
        # Update weather simulation parameters
        weather_params = {
            "temperature_std": 5.0 * variability,
            "wind_speed_std": 3.0 * variability,
            "solar_irradiance_std": 200.0 * variability,
            "cloud_frequency": 0.3 * variability
        }
        
        # Apply to simulation market data (ensure market_data exists)
        if not hasattr(simulation, 'market_data') or simulation.market_data is None:
            simulation.market_data = {}
        simulation.market_data["weather_variability"] = weather_params
    
    async def _set_demand_variability(self, simulation: SmartGridSimulation, 
                                     variability: float) -> None:
        """Set demand variability level"""
        
        # Update demand simulation parameters
        demand_params = {
            "daily_demand_std": 0.1 * variability,
            "hourly_demand_std": 0.05 * variability,
            "peak_demand_std": 0.2 * variability
        }
        
        # Apply to simulation market data (ensure market_data exists)
        if not hasattr(simulation, 'market_data') or simulation.market_data is None:
            simulation.market_data = {}
        simulation.market_data["demand_variability"] = demand_params
    
    async def _enable_ramping_events(self, simulation: SmartGridSimulation) -> None:
        """Enable ramping events in the simulation"""
        
        # This is a placeholder. In a real simulation, you'd modify agent behaviors
        # or market data to simulate sudden changes in demand/generation.
        # For example, you might increase demand by 50% for a short period.
        self.logger.info("Enabling ramping events in simulation.")
        # Example: Increase demand by 50% for 10 steps
        for i in range(10):
            await simulation.run_simulation_step()
            await asyncio.sleep(0.1) # Simulate time passing
    
    async def _enable_extreme_weather(self, simulation: SmartGridSimulation) -> None:
        """Enable extreme weather conditions in the simulation"""
        
        # This is a placeholder. In a real simulation, you'd modify agent behaviors
        # or market data to simulate extreme weather events.
        # For example, you might increase wind speed by 100% for a short period.
        self.logger.info("Enabling extreme weather in simulation.")
        # Example: Increase wind speed by 100% for 10 steps
        for i in range(10):
            await simulation.run_simulation_step()
            await asyncio.sleep(0.1) # Simulate time passing
    
    async def _collect_training_metrics(self, simulation: SmartGridSimulation, 
                                      scenario: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Collect training performance metrics"""
        
        # Get current system metrics
        system_metrics = await simulation.get_real_time_metrics()
        
        # Calculate performance scores
        renewable_utilization = self._calculate_renewable_utilization(system_metrics)
        grid_stability = self._calculate_grid_stability(system_metrics)
        economic_efficiency = self._calculate_economic_efficiency(system_metrics)
        
        overall_score = (
            renewable_utilization * 0.4 +
            grid_stability * 0.4 +
            economic_efficiency * 0.2
        )
        
        return {
            "step": step,
            "renewable_utilization": renewable_utilization,
            "grid_stability": grid_stability,
            "economic_efficiency": economic_efficiency,
            "overall_score": overall_score,
            "frequency_violations": system_metrics.get("frequency_violations", 0),
            "voltage_violations": system_metrics.get("voltage_violations", 0),
            "scenario": scenario
        }
    
    def _calculate_renewable_utilization(self, metrics: Dict[str, Any]) -> float:
        """Calculate renewable energy utilization score"""
        
        renewable_generation = metrics.get("renewable_generation_mw", 0)
        total_generation = metrics.get("total_generation_mw", 1)
        
        if total_generation == 0:
            return 0.0
        
        utilization_ratio = renewable_generation / total_generation
        return min(1.0, utilization_ratio * 2.0)  # Scale to 0-1
    
    def _calculate_grid_stability(self, metrics: Dict[str, Any]) -> float:
        """Calculate grid stability score"""
        
        frequency = metrics.get("frequency_hz", 50.0)
        voltage = metrics.get("voltage_pu", 1.0)
        
        # Frequency stability (target: 50.0 ± 0.1 Hz)
        freq_deviation = abs(frequency - 50.0)
        freq_score = max(0.0, 1.0 - freq_deviation / 0.5)
        
        # Voltage stability (target: 1.0 ± 0.05 pu)
        volt_deviation = abs(voltage - 1.0)
        volt_score = max(0.0, 1.0 - volt_deviation / 0.2)
        
        return (freq_score + volt_score) / 2.0
    
    def _calculate_economic_efficiency(self, metrics: Dict[str, Any]) -> float:
        """Calculate economic efficiency score"""
        
        system_cost = metrics.get("system_cost_per_hour", 1000)
        target_cost = 200  # Target cost per hour
        
        if system_cost <= target_cost:
            return 1.0
        else:
            return max(0.0, 1.0 - (system_cost - target_cost) / target_cost)
    
    async def _evaluate_final_performance(self, simulation: SmartGridSimulation) -> Dict[str, Any]:
        """Evaluate final performance after curriculum training"""
        
        # Run comprehensive test scenarios
        test_scenarios = [
            {"name": "high_renewable", "renewable_penetration": 0.8, "weather_variability": 1.0},
            {"name": "duck_curve", "renewable_penetration": 0.6, "duck_curve_intensity": 0.8},
            {"name": "wind_ramping", "renewable_penetration": 0.5, "ramping_events": True},
            {"name": "extreme_weather", "renewable_penetration": 0.7, "extreme_weather": True}
        ]
        
        final_results = {}
        
        for scenario in test_scenarios:
            await self._apply_scenario_to_simulation(simulation, scenario)
            
            # Run test for 1 hour
            test_metrics = []
            for _ in range(12):  # 12 steps = 1 hour at 5-min intervals
                await simulation.run_simulation_step()
                metrics = await simulation.get_real_time_metrics()
                test_metrics.append(metrics)
            
            # Analyze test results
            scenario_score = self._analyze_test_scenario(test_metrics)
            final_results[scenario["name"]] = scenario_score
        
        return final_results
    
    def _analyze_test_scenario(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results from a test scenario"""
        
        # Calculate averages and violations
        avg_renewable_utilization = np.mean([
            self._calculate_renewable_utilization(m) for m in metrics_list
        ])
        
        avg_grid_stability = np.mean([
            self._calculate_grid_stability(m) for m in metrics_list
        ])
        
        total_violations = sum([
            m.get("frequency_violations", 0) + m.get("voltage_violations", 0)
            for m in metrics_list
        ])
        
        return {
            "renewable_utilization": avg_renewable_utilization,
            "grid_stability": avg_grid_stability,
            "total_violations": total_violations,
            "success": avg_renewable_utilization > 0.5 and avg_grid_stability > 0.8 and total_violations < 3
        }
    
    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """Save curriculum training results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"curriculum_training_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Training results saved to {filename}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        else:
            return obj


# Usage example and integration functions

async def run_curriculum_training():
    """Example of how to run curriculum training"""
    
    # Create curriculum configuration
    config = CurriculumConfig(
        phase1_steps=10_000,  # Reduced for testing
        phase2_steps=40_000,  # Reduced for testing
        annealing_steps=15_000,
        max_renewable_penetration=0.8
    )
    
    # Initialize trainer
    trainer = RenewableCurriculumTrainer(config)
    
    # Create simulation
    simulation = SmartGridSimulation()
    await simulation.create_sample_scenario()
    
    # Run curriculum training
    results = await trainer.train_with_curriculum(simulation)
    
    print("Curriculum training completed!")
    print(f"Final performance: {results['final_performance']}")
    
    return results


if __name__ == "__main__":
    # Run curriculum training
    import asyncio
    asyncio.run(run_curriculum_training()) 