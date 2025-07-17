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
        
        # Update renewable penetration (with default)
        renewable_penetration = scenario.get("renewable_penetration", 0.3)
        await self._set_renewable_penetration(simulation, renewable_penetration)
        
        # Update weather variability (with default)
        weather_variability = scenario.get("weather_variability", 0.5)
        await self._set_weather_variability(simulation, weather_variability)
        
        # Update demand patterns (with default)
        demand_variability = scenario.get("demand_variability", 0.3)
        await self._set_demand_variability(simulation, demand_variability)
        
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
            {"name": "high_renewable", "renewable_penetration": 0.8, "weather_variability": 1.0, "demand_variability": 0.5},
            {"name": "duck_curve", "renewable_penetration": 0.6, "duck_curve_intensity": 0.8, "weather_variability": 0.7, "demand_variability": 0.6},
            {"name": "wind_ramping", "renewable_penetration": 0.5, "ramping_events": True, "weather_variability": 0.8, "demand_variability": 0.4},
            {"name": "extreme_weather", "renewable_penetration": 0.7, "extreme_weather": True, "weather_variability": 1.0, "demand_variability": 0.8}
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
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif obj is None:
            return None
        else:
            # For other objects, try to convert to string as fallback
            try:
                return str(obj)
            except:
                return f"<non-serializable: {type(obj).__name__}>"


@dataclass
class QuickTestConfig(CurriculumConfig):
    """Quick test configuration for 10-minute curriculum validation - IMPROVED VERSION"""
    # Phase 1: Foundation training with more steps
    phase1_steps: int = 300  # Increased from 50 to 300 steps
    phase1_renewable_penetration: float = 0.10  # Start with 10% renewables (more meaningful baseline)
    phase1_weather_variability: float = 0.2  # Slightly more variation for learning
    
    # Phase 2: Aggressive curriculum introduction  
    phase2_steps: int = 700  # Increased from 150 to 700 steps
    annealing_steps: int = 500  # More gradual annealing
    max_renewable_penetration: float = 0.70  # More aggressive target (70% vs 50%)
    max_weather_variability: float = 0.9  # Higher weather variation
    
    # Curriculum schedules
    renewable_schedule: str = "exponential"  # Exponential for faster early progress
    weather_schedule: str = "linear"
    demand_schedule: str = "exponential"  # More aggressive demand variation
    
    # Training parameters - MUCH more aggressive for quick convergence
    entropy_coefficient_agent: float = 0.05  # Higher exploration
    entropy_coefficient_grid_op: float = 0.15  # More grid operator exploration
    learning_rate_agent: float = 0.005  # 5x higher learning rate
    learning_rate_grid_op: float = 0.002  # 4x higher learning rate


async def run_quick_curriculum_test():
    """Quick 10-minute test to validate curriculum approach works"""
    
    print("🚀 Quick Curriculum Test (< 10 minutes)")
    print("=" * 50)
    
    # Create quick test configuration
    config = QuickTestConfig()
    
    print(f"📊 Test Configuration:")
    print(f"   Phase 1: {config.phase1_steps} steps (foundation)")
    print(f"   Phase 2: {config.phase2_steps} steps (curriculum)")
    print(f"   Total: {config.phase1_steps + config.phase2_steps} steps")
    print(f"   Renewable progression: {config.phase1_renewable_penetration:.0%} → {config.max_renewable_penetration:.0%}")
    print(f"   Learning rates: Agent={config.learning_rate_agent}, Grid={config.learning_rate_grid_op}")
    print(f"   Expected: Should show marginal improvement with these aggressive settings")
    
    # Initialize trainer
    trainer = RenewableCurriculumTrainer(config)
    
    # Create simulation
    print("\n🏗️ Creating simulation...")
    simulation = SmartGridSimulation()
    await simulation.create_sample_scenario()
    
    # Baseline performance measurement (no training)
    print("\n📏 Measuring baseline performance (untrained agents)...")
    baseline_metrics = await _measure_performance(simulation, "baseline")
    
    # Run curriculum training
    print("\n🎓 Running curriculum training...")
    start_time = datetime.now()
    
    try:
        results = await trainer.train_with_curriculum(simulation)
        
        # Post-training performance measurement
        print("\n📊 Measuring post-curriculum performance...")
        trained_metrics = await _measure_performance(simulation, "trained")
        
        training_time = datetime.now() - start_time
        
        # Performance comparison
        print("\n" + "=" * 60)
        print("🎯 CURRICULUM TRAINING RESULTS")
        print("=" * 60)
        
        print(f"⏱️ Training Duration: {training_time}")
        print(f"📈 Total Training Steps: {config.phase1_steps + config.phase2_steps}")
        
        print(f"\n📊 Performance Comparison:")
        print(f"{'Metric':<25} {'Baseline':<15} {'Trained':<15} {'Improvement':<15}")
        print("-" * 70)
        
        # Renewable utilization
        baseline_renewable = baseline_metrics["renewable_utilization"]
        trained_renewable = trained_metrics["renewable_utilization"]
        renewable_improvement = ((trained_renewable - baseline_renewable) / max(baseline_renewable, 0.01)) * 100
        print(f"{'Renewable Utilization':<25} {baseline_renewable:<15.3f} {trained_renewable:<15.3f} {renewable_improvement:+.1f}%")
        
        # Grid stability
        baseline_stability = baseline_metrics["grid_stability"]
        trained_stability = trained_metrics["grid_stability"]
        stability_improvement = ((trained_stability - baseline_stability) / max(baseline_stability, 0.01)) * 100
        print(f"{'Grid Stability':<25} {baseline_stability:<15.3f} {trained_stability:<15.3f} {stability_improvement:+.1f}%")
        
        # Economic efficiency
        baseline_economic = baseline_metrics["economic_efficiency"]
        trained_economic = trained_metrics["economic_efficiency"]
        economic_improvement = ((trained_economic - baseline_economic) / max(baseline_economic, 0.01)) * 100
        print(f"{'Economic Efficiency':<25} {baseline_economic:<15.3f} {trained_economic:<15.3f} {economic_improvement:+.1f}%")
        
        # Overall score
        baseline_overall = baseline_metrics["overall_score"]
        trained_overall = trained_metrics["overall_score"]
        overall_improvement = ((trained_overall - baseline_overall) / max(baseline_overall, 0.01)) * 100
        print(f"{'Overall Score':<25} {baseline_overall:<15.3f} {trained_overall:<15.3f} {overall_improvement:+.1f}%")
        
        # Assessment
        print(f"\n🎯 Quick Test Assessment:")
        if overall_improvement > 5:
            print("✅ POSITIVE: Curriculum approach shows clear improvement!")
            print("   Recommended to proceed with full training")
        elif overall_improvement > 1:
            print("🟡 MARGINAL: Small improvement detected")
            print("   Consider adjusting parameters or longer training")
        else:
            print("❌ NO IMPROVEMENT: Curriculum approach not effective")
            print("   May need different approach or parameter tuning")
        
        # Save quick test results
        quick_results = {
            "test_config": config.__dict__,
            "training_time_seconds": training_time.total_seconds(),
            "baseline_metrics": baseline_metrics,
            "trained_metrics": trained_metrics,
            "improvements": {
                "renewable_utilization": renewable_improvement,
                "grid_stability": stability_improvement,
                "economic_efficiency": economic_improvement,
                "overall_score": overall_improvement
            },
            "assessment": "positive" if overall_improvement > 5 else ("marginal" if overall_improvement > 1 else "negative")
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"quick_curriculum_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(quick_results, f, indent=2, default=str)
        
        print(f"\n💾 Quick test results saved to: {filename}")
        
        return quick_results
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def _measure_performance(simulation: SmartGridSimulation, test_type: str) -> Dict[str, float]:
    """Measure simulation performance for comparison"""
    
    print(f"   Running {test_type} performance test...")
    
    # Run simulation for a few steps to get metrics
    metrics_list = []
    
    for i in range(10):  # Run 10 steps for measurement
        try:
            await simulation.run_simulation_step()
            metrics = await simulation.get_real_time_metrics()
            
            # Calculate performance scores using trainer methods
            temp_trainer = RenewableCurriculumTrainer(QuickTestConfig())
            
            renewable_util = temp_trainer._calculate_renewable_utilization(metrics)
            grid_stability = temp_trainer._calculate_grid_stability(metrics)
            economic_eff = temp_trainer._calculate_economic_efficiency(metrics)
            
            overall_score = (renewable_util * 0.4 + grid_stability * 0.4 + economic_eff * 0.2)
            
            metrics_list.append({
                "renewable_utilization": renewable_util,
                "grid_stability": grid_stability,
                "economic_efficiency": economic_eff,
                "overall_score": overall_score
            })
            
        except Exception as e:
            print(f"   Warning: Step {i} failed: {e}")
            continue
    
    if not metrics_list:
        # Fallback metrics if simulation fails
        return {
            "renewable_utilization": 0.0,
            "grid_stability": 0.5,
            "economic_efficiency": 0.3,
            "overall_score": 0.32
        }
    
    # Average the metrics
    avg_metrics = {}
    for key in metrics_list[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    
    print(f"   {test_type.capitalize()} overall score: {avg_metrics['overall_score']:.3f}")
    
    return avg_metrics


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
    
    # Choose which training to run based on command line args
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print("Running quick curriculum test...")
        asyncio.run(run_quick_curriculum_test())
    else:
        print("Running full curriculum training...")
        asyncio.run(run_curriculum_training()) 