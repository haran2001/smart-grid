#!/usr/bin/env python3
"""
Curriculum Learning Integration for Renewable Energy Studies

Integrates curriculum-based MARL training into existing renewable stress tests
to improve agent performance on renewable energy integration challenges.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging

from renewable_stress_tests import RenewableStressTestFramework, StressTestConfig
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.coordination.multi_agent_system import SmartGridSimulation


class CurriculumRenewableTrainer:
    """Applies curriculum learning to renewable integration training"""
    
    def __init__(self):
        self.logger = logging.getLogger("CurriculumRenewable")
        
    async def train_with_curriculum(self, simulation: SmartGridSimulation) -> Dict[str, Any]:
        """Main curriculum training method"""
        
        self.logger.info("Starting curriculum-based renewable integration training")
        
        # Phase 1: Stable grid training (Traditional generators only)
        self.logger.info("Phase 1: Training on stable traditional grid...")
        await self._phase1_stable_training(simulation)
        
        # Phase 2: Gradual renewable introduction with annealing
        self.logger.info("Phase 2: Gradual renewable complexity introduction...")
        results = await self._phase2_renewable_annealing(simulation)
        
        return results
    
    async def _phase1_stable_training(self, simulation: SmartGridSimulation, steps: int = 50000):
        """Phase 1: Train on stable traditional grid"""
        
        # Configure stable scenario
        stable_config = {
            "renewable_capacity_factor": 0.0,  # No renewables initially
            "weather_variability": 0.1,        # Minimal weather variation  
            "demand_patterns": "stable",       # Predictable demand
            "market_volatility": 0.2          # Low market volatility
        }
        
        self.logger.info(f"Phase 1: {steps:,} steps with stable traditional grid")
        
        for step in range(steps):
            # Apply stable configuration
            await self._apply_training_config(simulation, stable_config)
            
            # Run simulation step
            await simulation.run_simulation_step()
            
            if step % 10000 == 0:
                metrics = await simulation.get_real_time_metrics()
                self.logger.info(f"Step {step:,}: Stability = {metrics.get('frequency_hz', 50.0):.3f} Hz")
    
    async def _phase2_renewable_annealing(self, simulation: SmartGridSimulation, steps: int = 200000):
        """Phase 2: Gradually introduce renewable complexity"""
        
        annealing_steps = 50000  # 50K steps to reach full complexity
        results = {"training_history": [], "final_performance": {}}
        
        self.logger.info(f"Phase 2: {steps:,} steps with renewable annealing over {annealing_steps:,} steps")
        
        for step in range(steps):
            # Calculate annealing progress
            annealing_progress = min(1.0, step / annealing_steps)
            
            # Create curriculum configuration
            curriculum_config = self._create_curriculum_config(annealing_progress)
            
            # Apply configuration
            await self._apply_training_config(simulation, curriculum_config)
            
            # Run simulation step
            await simulation.run_simulation_step()
            
            # Record progress every 5K steps
            if step % 5000 == 0:
                metrics = await self._collect_metrics(simulation, curriculum_config, step)
                results["training_history"].append(metrics)
                
                self.logger.info(
                    f"Step {step:,}: Renewable={curriculum_config['renewable_capacity_factor']:.1%}, "
                    f"Weather Var={curriculum_config['weather_variability']:.2f}, "
                    f"Performance={metrics['performance_score']:.2f}"
                )
        
        # Final evaluation
        results["final_performance"] = await self._evaluate_final_performance(simulation)
        
        return results
    
    def _create_curriculum_config(self, progress: float) -> Dict[str, Any]:
        """Create training configuration based on curriculum progress"""
        
        # Linear annealing schedules
        renewable_capacity = 0.0 + progress * 0.8  # 0% → 80% renewable capacity
        weather_variability = 0.1 + progress * 0.9  # Low → High weather variation
        demand_variability = 0.2 + progress * 0.6   # Stable → Variable demand
        
        # Enable advanced challenges at different progress levels
        enable_intermittency = progress > 0.3
        enable_ramping = progress > 0.5
        enable_duck_curve = progress > 0.7
        
        return {
            "renewable_capacity_factor": renewable_capacity,
            "weather_variability": weather_variability,
            "demand_variability": demand_variability,
            "enable_intermittency": enable_intermittency,
            "enable_ramping": enable_ramping,
            "enable_duck_curve": enable_duck_curve,
            "market_volatility": 0.2 + progress * 0.3
        }
    
    async def _apply_training_config(self, simulation: SmartGridSimulation, config: Dict[str, Any]):
        """Apply curriculum configuration to simulation"""
        
        # Update renewable capacity factors
        renewable_factor = config["renewable_capacity_factor"]
        await self._set_renewable_generation(simulation, renewable_factor)
        
        # Update weather variability
        weather_var = config["weather_variability"]
        await self._set_weather_parameters(simulation, weather_var)
        
        # Update demand patterns
        demand_var = config["demand_variability"]
        await self._set_demand_variability(simulation, demand_var)
    
    async def _set_renewable_generation(self, simulation: SmartGridSimulation, factor: float):
        """Set renewable generation capacity factor"""
        
        from src.agents.generator_agent import GeneratorAgent
        
        # Find renewable agents (zero emissions)
        for agent in simulation.agents.values():
            # Check if this is a generator agent
            if isinstance(agent, GeneratorAgent):
                if agent.generator_state.emissions_rate_kg_co2_per_mwh == 0:
                    # This is a renewable generator
                    base_capacity = agent.generator_state.max_capacity_mw
                    current_output = base_capacity * factor
                    
                    # Update weather-dependent output
                    weather = simulation.market_data.get("weather", {})
                    if "solar" in agent.agent_id.lower():
                        solar_irradiance = weather.get("solar_irradiance", 500)
                        agent.generator_state.current_output_mw = current_output * (solar_irradiance / 1000.0)
                    elif "wind" in agent.agent_id.lower():
                        wind_speed = weather.get("wind_speed", 10)
                        # Wind power curve approximation
                        wind_factor = min(1.0, max(0.0, (wind_speed - 3) / 12))
                        agent.generator_state.current_output_mw = current_output * wind_factor
                    
                    agent.generator_state.online_status = factor > 0.1
    
    async def _set_weather_parameters(self, simulation: SmartGridSimulation, variability: float):
        """Set weather variability parameters"""
        
        base_weather = {
            "temperature": 20.0,
            "wind_speed": 10.0,
            "solar_irradiance": 500.0
        }
        
        # Add variability
        weather_noise = {
            "temperature": base_weather["temperature"] + np.random.normal(0, 5 * variability),
            "wind_speed": max(0, base_weather["wind_speed"] + np.random.normal(0, 3 * variability)),
            "solar_irradiance": max(0, base_weather["solar_irradiance"] + np.random.normal(0, 200 * variability))
        }
        
        simulation.market_data["weather"] = weather_noise
    
    async def _set_demand_variability(self, simulation: SmartGridSimulation, variability: float):
        """Set demand variability parameters"""
        
        base_demand = 1000.0  # Base demand in MW
        demand_noise = np.random.normal(0, base_demand * variability * 0.2)
        
        # Update demand forecast
        simulation.market_data["demand_forecast"] = {
            "expected_peak": base_demand + demand_noise,
            "variability_factor": variability
        }
    
    async def _collect_metrics(self, simulation: SmartGridSimulation, 
                              config: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Collect training performance metrics"""
        
        metrics = await simulation.get_real_time_metrics()
        
        # Calculate performance score
        renewable_score = self._score_renewable_utilization(metrics)
        stability_score = self._score_grid_stability(metrics)
        economic_score = self._score_economic_performance(metrics)
        
        performance_score = (renewable_score + stability_score + economic_score) / 3.0
        
        return {
            "step": step,
            "config": config,
            "renewable_score": renewable_score,
            "stability_score": stability_score,
            "economic_score": economic_score,
            "performance_score": performance_score,
            "raw_metrics": metrics
        }
    
    def _score_renewable_utilization(self, metrics: Dict[str, Any]) -> float:
        """Score renewable energy utilization (0-1)"""
        renewable_mw = metrics.get("renewable_generation_mw", 0)
        total_mw = metrics.get("total_generation_mw", 1)
        
        if total_mw == 0:
            return 0.0
        
        utilization = renewable_mw / total_mw
        return min(1.0, utilization * 1.25)  # Bonus for high utilization
    
    def _score_grid_stability(self, metrics: Dict[str, Any]) -> float:
        """Score grid stability (0-1)"""
        frequency = metrics.get("frequency_hz", 50.0)
        voltage = metrics.get("voltage_pu", 1.0)
        
        # Frequency stability (50.0 ± 0.2 Hz acceptable)
        freq_error = abs(frequency - 50.0)
        freq_score = max(0.0, 1.0 - freq_error / 0.5)
        
        # Voltage stability (1.0 ± 0.1 pu acceptable)
        volt_error = abs(voltage - 1.0)
        volt_score = max(0.0, 1.0 - volt_error / 0.2)
        
        return (freq_score + volt_score) / 2.0
    
    def _score_economic_performance(self, metrics: Dict[str, Any]) -> float:
        """Score economic performance (0-1)"""
        cost_per_hour = metrics.get("system_cost_per_hour", 1000)
        target_cost = 250  # Target $250/hour
        
        if cost_per_hour <= target_cost:
            return 1.0
        elif cost_per_hour <= target_cost * 2:
            return 1.0 - (cost_per_hour - target_cost) / target_cost
        else:
            return 0.0
    
    async def _evaluate_final_performance(self, simulation: SmartGridSimulation) -> Dict[str, Any]:
        """Evaluate final performance on challenging scenarios"""
        
        test_scenarios = [
            {
                "name": "high_solar_intermittency",
                "renewable_capacity_factor": 0.7,
                "weather_variability": 1.0,
                "enable_intermittency": True
            },
            {
                "name": "wind_ramping_events", 
                "renewable_capacity_factor": 0.6,
                "weather_variability": 0.8,
                "enable_ramping": True
            },
            {
                "name": "duck_curve_challenge",
                "renewable_capacity_factor": 0.8,
                "weather_variability": 0.5,
                "enable_duck_curve": True
            }
        ]
        
        final_results = {}
        
        for scenario in test_scenarios:
            self.logger.info(f"Testing final performance: {scenario['name']}")
            
            # Apply test scenario
            await self._apply_training_config(simulation, scenario)
            
            # Run test for 1 hour (12 steps at 5-min intervals)
            test_metrics = []
            for _ in range(12):
                await simulation.run_simulation_step()
                metrics = await simulation.get_real_time_metrics()
                test_metrics.append(metrics)
            
            # Calculate scenario performance
            avg_renewable = np.mean([self._score_renewable_utilization(m) for m in test_metrics])
            avg_stability = np.mean([self._score_grid_stability(m) for m in test_metrics])
            avg_economic = np.mean([self._score_economic_performance(m) for m in test_metrics])
            
            violations = sum([
                m.get("frequency_violations", 0) + m.get("voltage_violations", 0) 
                for m in test_metrics
            ])
            
            overall_score = (avg_renewable + avg_stability + avg_economic) / 3.0
            success = overall_score > 0.6 and violations < 3
            
            final_results[scenario["name"]] = {
                "renewable_score": avg_renewable,
                "stability_score": avg_stability,
                "economic_score": avg_economic,
                "overall_score": overall_score,
                "violations": violations,
                "success": success
            }
        
        return final_results


# Integration with existing stress test framework
async def run_curriculum_enhanced_stress_tests():
    """Run stress tests with curriculum-trained agents"""
    
    # Initialize simulation
    simulation = SmartGridSimulation()
    await simulation.create_sample_scenario()
    
    # Apply curriculum training
    trainer = CurriculumRenewableTrainer()
    training_results = await trainer.train_with_curriculum(simulation)
    
    print("Curriculum training completed!")
    print("Final Performance Results:")
    
    for scenario, results in training_results["final_performance"].items():
        status = "✅ PASS" if results["success"] else "❌ FAIL"
        print(f"{scenario}: {status} (Score: {results['overall_score']:.2f})")
    
    # Now run standard stress tests with trained agents
    stress_framework = RenewableStressTestFramework()
    
    # Run enhanced stress tests
    enhanced_results = {}
    test_configs = [
        StressTestConfig(
            test_name="curriculum_enhanced_solar_intermittency",
            description="Solar intermittency test with curriculum-trained agents",
            duration_hours=2
        ),
        StressTestConfig(
            test_name="curriculum_enhanced_wind_ramping", 
            description="Wind ramping test with curriculum-trained agents",
            duration_hours=2
        ),
        StressTestConfig(
            test_name="curriculum_enhanced_duck_curve",
            description="Duck curve test with curriculum-trained agents", 
            duration_hours=3
        )
    ]
    
    for config in test_configs:
        print(f"\nRunning enhanced stress test: {config.test_name}")
        result = await stress_framework.run_stress_test(config)
        enhanced_results[config.test_name] = result
    
    return {
        "curriculum_training": training_results,
        "enhanced_stress_tests": enhanced_results
    }


if __name__ == "__main__":
    # Run curriculum-enhanced renewable integration training and testing
    results = asyncio.run(run_curriculum_enhanced_stress_tests())
    
    print("\n" + "="*60)
    print("CURRICULUM-ENHANCED RENEWABLE INTEGRATION COMPLETE")
    print("="*60) 