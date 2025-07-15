#!/usr/bin/env python3
"""
Renewable Energy Integration Stress Test Framework
Tests the smart grid system against various renewable energy integration challenges
"""

import asyncio
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import pandas as pd

# Import the smart grid system
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # Go up to project root

from src.coordination.multi_agent_system import SmartGridSimulation

from src.coordination.multi_agent_system import SmartGridSimulation
from src.agents.generator_agent import GeneratorAgent
from src.agents.storage_agent import StorageAgent
from src.agents.grid_operator_agent import GridOperatorAgent

@dataclass
class StressTestConfig:
    """Configuration for stress test scenarios"""
    test_name: str
    description: str
    duration_hours: int = 24
    time_step_minutes: int = 5
    agents_config: Dict[str, Any] = field(default_factory=dict)
    disturbance_patterns: List[Dict[str, Any]] = field(default_factory=list)
    validation_metrics: List[str] = field(default_factory=list)
    expected_challenges: List[str] = field(default_factory=list)

class RenewableStressTestFramework:
    """Framework for conducting renewable energy integration stress tests"""
    
    def __init__(self, results_dir: str = "renewable_stress_results"):
        self.results_dir = results_dir
        self.simulation = None
        self.test_results = {}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{results_dir}/stress_test.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("RenewableStressTest")
    
    async def setup_test_scenario(self, config: StressTestConfig) -> SmartGridSimulation:
        """Setup simulation with stress test configuration"""
        self.simulation = SmartGridSimulation()
        
        # Add grid operator
        await self.simulation.add_grid_operator()
        
        # Add agents based on configuration
        if "generators" in config.agents_config:
            for gen_config in config.agents_config["generators"]:
                self.simulation.add_generator_agent(
                    gen_config["agent_id"], 
                    gen_config["params"]
                )
        
        if "storage" in config.agents_config:
            for storage_config in config.agents_config["storage"]:
                self.simulation.add_storage_agent(
                    storage_config["agent_id"], 
                    storage_config["params"]
                )
        
        if "consumers" in config.agents_config:
            for consumer_config in config.agents_config["consumers"]:
                self.simulation.add_consumer_agent(
                    consumer_config["agent_id"], 
                    consumer_config["params"]
                )
        
        return self.simulation
    
    async def apply_disturbance_patterns(self, patterns: List[Dict[str, Any]]) -> None:
        """Apply disturbance patterns during simulation"""
        for pattern in patterns:
            if pattern["type"] == "solar_variability":
                await self._apply_solar_disturbance(pattern)
            elif pattern["type"] == "wind_variability":
                await self._apply_wind_disturbance(pattern)
            elif pattern["type"] == "demand_spike":
                await self._apply_demand_disturbance(pattern)
            elif pattern["type"] == "frequency_event":
                await self._apply_frequency_disturbance(pattern)
            elif pattern["type"] == "voltage_event":
                await self._apply_voltage_disturbance(pattern)
    
    async def _apply_solar_disturbance(self, pattern: Dict[str, Any]) -> None:
        """Apply solar irradiance disturbances"""
        if "solar_agents" in pattern:
            for agent_id in pattern["solar_agents"]:
                if agent_id in self.simulation.agents:
                    agent = self.simulation.agents[agent_id]
                    # Modify solar irradiance in market data
                    irradiance_values = pattern.get("irradiance_pattern", [500] * 24)
                    for i, irradiance in enumerate(irradiance_values):
                        # Schedule irradiance changes
                        self.simulation.market_data["weather"]["solar_irradiance"] = irradiance
                        await asyncio.sleep(0.1)  # Small delay for simulation
    
    async def _apply_wind_disturbance(self, pattern: Dict[str, Any]) -> None:
        """Apply wind speed disturbances"""
        if "wind_agents" in pattern:
            for agent_id in pattern["wind_agents"]:
                if agent_id in self.simulation.agents:
                    agent = self.simulation.agents[agent_id]
                    # Modify wind speed in market data
                    wind_speeds = pattern.get("wind_speed_pattern", [10] * 24)
                    for i, wind_speed in enumerate(wind_speeds):
                        self.simulation.market_data["weather"]["wind_speed"] = wind_speed
                        await asyncio.sleep(0.1)
    
    async def _apply_demand_disturbance(self, pattern: Dict[str, Any]) -> None:
        """Apply demand disturbances"""
        demand_changes = pattern.get("demand_multipliers", [1.0] * 24)
        base_demand = self.simulation.market_data["demand_forecast"]["expected_peak"]
        
        for multiplier in demand_changes:
            self.simulation.market_data["demand_forecast"]["expected_peak"] = base_demand * multiplier
            await asyncio.sleep(0.1)
    
    async def _apply_frequency_disturbance(self, pattern: Dict[str, Any]) -> None:
        """Apply frequency disturbances"""
        frequency_values = pattern.get("frequency_pattern", [50.0] * 24)
        for frequency in frequency_values:
            self.simulation.market_data["frequency_hz"] = frequency
            await asyncio.sleep(0.1)
    
    async def _apply_voltage_disturbance(self, pattern: Dict[str, Any]) -> None:
        """Apply voltage disturbances"""
        voltage_values = pattern.get("voltage_pattern", [1.0] * 24)
        for voltage in voltage_values:
            self.simulation.market_data["voltage_pu"] = voltage
            await asyncio.sleep(0.1)
    
    async def run_stress_test(self, config: StressTestConfig) -> Dict[str, Any]:
        """Run a complete stress test"""
        self.logger.info(f"Starting stress test: {config.test_name}")
        
        # Setup simulation
        await self.setup_test_scenario(config)
        
        # Record initial state
        initial_state = await self.simulation.get_real_time_metrics()
        
        # Run simulation with disturbances
        test_results = {
            "test_name": config.test_name,
            "start_time": datetime.now().isoformat(),
            "initial_state": initial_state,
            "metrics_timeline": [],
            "violations": [],
            "performance_summary": {}
        }
        
        # Start simulation
        simulation_task = asyncio.create_task(
            self.simulation.run_simulation(
                duration_hours=config.duration_hours, 
                real_time=False
            )
        )
        
        # Apply disturbances concurrently
        disturbance_task = asyncio.create_task(
            self.apply_disturbance_patterns(config.disturbance_patterns)
        )
        
        # Monitor simulation
        monitoring_task = asyncio.create_task(
            self._monitor_simulation(test_results, config.validation_metrics)
        )
        
        # Wait for all tasks to complete
        await asyncio.gather(simulation_task, disturbance_task, monitoring_task)
        
        # Calculate final metrics
        final_state = await self.simulation.get_real_time_metrics()
        test_results["final_state"] = final_state
        test_results["end_time"] = datetime.now().isoformat()
        
        # Analyze results
        performance_analysis = self._analyze_performance(test_results, config)
        test_results["performance_analysis"] = performance_analysis
        
        # Save results
        self._save_test_results(test_results, config.test_name)
        
        self.logger.info(f"Completed stress test: {config.test_name}")
        return test_results
    
    async def _monitor_simulation(self, test_results: Dict[str, Any], metrics: List[str]) -> None:
        """Monitor simulation and record metrics"""
        while self.simulation.is_running:
            try:
                # Get current metrics
                current_metrics = await self.simulation.get_real_time_metrics()
                
                # Record timestamp
                current_metrics["timestamp"] = datetime.now().isoformat()
                test_results["metrics_timeline"].append(current_metrics)
                
                # Check for violations
                violations = self._check_violations(current_metrics)
                if violations:
                    test_results["violations"].extend(violations)
                
                # Wait before next measurement
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring: {e}")
                break
    
    def _check_violations(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for system violations"""
        violations = []
        
        # Frequency violations
        frequency = metrics.get("frequency_hz", 50.0)
        if frequency < 49.8 or frequency > 50.2:
            violations.append({
                "type": "frequency_violation",
                "value": frequency,
                "severity": "high" if abs(frequency - 50.0) > 0.5 else "medium",
                "timestamp": datetime.now().isoformat()
            })
        
        # Voltage violations
        voltage = metrics.get("voltage_pu", 1.0)
        if voltage < 0.95 or voltage > 1.05:
            violations.append({
                "type": "voltage_violation",
                "value": voltage,
                "severity": "high" if abs(voltage - 1.0) > 0.1 else "medium",
                "timestamp": datetime.now().isoformat()
            })
        
        # Reserve margin violations
        reserves = metrics.get("reserve_margin_mw", 0.0)
        if reserves < 50.0:
            violations.append({
                "type": "reserve_shortage",
                "value": reserves,
                "severity": "critical" if reserves < 10.0 else "high",
                "timestamp": datetime.now().isoformat()
            })
        
        return violations
    
    def _analyze_performance(self, test_results: Dict[str, Any], config: StressTestConfig) -> Dict[str, Any]:
        """Analyze test performance"""
        metrics_timeline = test_results["metrics_timeline"]
        
        if not metrics_timeline:
            return {"error": "No metrics data available"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(metrics_timeline)
        
        analysis = {
            "stability_metrics": {
                "frequency_std": df["frequency_hz"].std() if "frequency_hz" in df.columns else 0,
                "voltage_std": df["voltage_pu"].std() if "voltage_pu" in df.columns else 0,
                "frequency_violations": len([v for v in test_results["violations"] if v["type"] == "frequency_violation"]),
                "voltage_violations": len([v for v in test_results["violations"] if v["type"] == "voltage_violation"])
            },
            "economic_metrics": {
                "average_price": df["market_price"].mean() if "market_price" in df.columns else 0,
                "price_volatility": df["market_price"].std() if "market_price" in df.columns else 0,
                "total_cost": df["total_cost"].sum() if "total_cost" in df.columns else 0
            },
            "renewable_metrics": {
                "average_penetration": df["renewable_penetration"].mean() if "renewable_penetration" in df.columns else 0,
                "curtailment_events": len([v for v in test_results["violations"] if v["type"] == "curtailment"]),
                "storage_utilization": df["storage_utilization"].mean() if "storage_utilization" in df.columns else 0
            },
            "reliability_metrics": {
                "total_violations": len(test_results["violations"]),
                "critical_violations": len([v for v in test_results["violations"] if v["severity"] == "critical"]),
                "system_recovery_time": self._calculate_recovery_time(test_results["violations"])
            }
        }
        
        return analysis
    
    def _calculate_recovery_time(self, violations: List[Dict[str, Any]]) -> float:
        """Calculate average system recovery time"""
        if not violations:
            return 0.0
        
        # Simple recovery time calculation
        # In a real implementation, this would be more sophisticated
        return len(violations) * 5.0  # Assume 5 minutes per violation
    
    def _save_test_results(self, results: Dict[str, Any], test_name: str) -> None:
        """Save test results to file"""
        filename = f"{self.results_dir}/{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Test results saved to: {filename}")
    
    def create_stress_test_scenarios(self) -> List[StressTestConfig]:
        """Create predefined stress test scenarios"""
        scenarios = []
        
        # Scenario 1: Solar Intermittency
        scenarios.append(StressTestConfig(
            test_name="solar_intermittency_stress",
            description="Test rapid solar power fluctuations due to cloud cover",
            duration_hours=12,
            agents_config={
                "generators": [
                    {"agent_id": "solar_farm_1", "params": {"max_capacity_mw": 200, "fuel_cost_per_mwh": 0}},
                    {"agent_id": "solar_farm_2", "params": {"max_capacity_mw": 150, "fuel_cost_per_mwh": 0}},
                    {"agent_id": "gas_backup", "params": {"max_capacity_mw": 100, "fuel_cost_per_mwh": 80}}
                ],
                "storage": [
                    {"agent_id": "battery_1", "params": {"max_capacity_mwh": 200, "max_power_mw": 50}}
                ]
            },
            disturbance_patterns=[
                {
                    "type": "solar_variability",
                    "solar_agents": ["solar_farm_1", "solar_farm_2"],
                    "irradiance_pattern": [800, 200, 900, 100, 850, 50, 750, 900, 100, 800, 200, 750]
                }
            ],
            validation_metrics=["frequency_stability", "voltage_stability", "storage_utilization"],
            expected_challenges=["rapid_ramping", "storage_cycling", "backup_activation"]
        ))
        
        # Scenario 2: High Wind Variability
        scenarios.append(StressTestConfig(
            test_name="wind_variability_stress",
            description="Test wind power variability and ramping events",
            duration_hours=24,
            agents_config={
                "generators": [
                    {"agent_id": "wind_farm_1", "params": {"max_capacity_mw": 250, "fuel_cost_per_mwh": 0}},
                    {"agent_id": "wind_farm_2", "params": {"max_capacity_mw": 200, "fuel_cost_per_mwh": 0}},
                    {"agent_id": "coal_plant", "params": {"max_capacity_mw": 200, "fuel_cost_per_mwh": 60}}
                ],
                "storage": [
                    {"agent_id": "battery_2", "params": {"max_capacity_mwh": 300, "max_power_mw": 75}}
                ]
            },
            disturbance_patterns=[
                {
                    "type": "wind_variability",
                    "wind_agents": ["wind_farm_1", "wind_farm_2"],
                    "wind_speed_pattern": [15, 5, 20, 3, 18, 8, 2, 25, 12, 6, 22, 4] * 2
                }
            ],
            validation_metrics=["grid_stability", "market_efficiency", "ramping_adequacy"],
            expected_challenges=["wind_forecast_errors", "ramping_constraints", "grid_inertia"]
        ))
        
        # Scenario 3: Duck Curve Challenge
        scenarios.append(StressTestConfig(
            test_name="duck_curve_stress",
            description="Test evening ramping with high solar penetration",
            duration_hours=24,
            agents_config={
                "generators": [
                    {"agent_id": "solar_fleet", "params": {"max_capacity_mw": 400, "fuel_cost_per_mwh": 0}},
                    {"agent_id": "gas_peaker_1", "params": {"max_capacity_mw": 150, "fuel_cost_per_mwh": 120}},
                    {"agent_id": "gas_peaker_2", "params": {"max_capacity_mw": 150, "fuel_cost_per_mwh": 120}}
                ],
                "storage": [
                    {"agent_id": "battery_storage", "params": {"max_capacity_mwh": 400, "max_power_mw": 100}}
                ]
            },
            disturbance_patterns=[
                {
                    "type": "solar_variability",
                    "solar_agents": ["solar_fleet"],
                    "irradiance_pattern": [0, 0, 0, 0, 0, 100, 400, 700, 900, 1000, 800, 600, 300, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                },
                {
                    "type": "demand_spike",
                    "demand_multipliers": [0.6, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.9, 1.0, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6]
                }
            ],
            validation_metrics=["ramping_capability", "storage_arbitrage", "price_volatility"],
            expected_challenges=["evening_ramp", "storage_optimization", "peaker_coordination"]
        ))
        
        # Scenario 4: Extreme Weather Event
        scenarios.append(StressTestConfig(
            test_name="extreme_weather_stress",
            description="Test system resilience during extreme weather",
            duration_hours=48,
            agents_config={
                "generators": [
                    {"agent_id": "solar_1", "params": {"max_capacity_mw": 200, "fuel_cost_per_mwh": 0}},
                    {"agent_id": "wind_1", "params": {"max_capacity_mw": 250, "fuel_cost_per_mwh": 0}},
                    {"agent_id": "gas_1", "params": {"max_capacity_mw": 200, "fuel_cost_per_mwh": 80}},
                    {"agent_id": "coal_1", "params": {"max_capacity_mw": 150, "fuel_cost_per_mwh": 60}}
                ],
                "storage": [
                    {"agent_id": "battery_large", "params": {"max_capacity_mwh": 500, "max_power_mw": 125}}
                ]
            },
            disturbance_patterns=[
                {
                    "type": "solar_variability",
                    "solar_agents": ["solar_1"],
                    "irradiance_pattern": [200] * 12 + [50] * 12 + [800] * 12 + [100] * 12  # Storm then clear
                },
                {
                    "type": "wind_variability", 
                    "wind_agents": ["wind_1"],
                    "wind_speed_pattern": [25] * 6 + [5] * 6 + [2] * 12 + [15] * 12 + [8] * 12  # High winds then calm
                },
                {
                    "type": "demand_spike",
                    "demand_multipliers": [1.3] * 24 + [1.1] * 24  # High demand during storm
                }
            ],
            validation_metrics=["system_resilience", "emergency_response", "blackout_prevention"],
            expected_challenges=["compound_events", "resource_adequacy", "emergency_procedures"]
        ))
        
        return scenarios

# Example usage functions
async def run_individual_stress_test(test_name: str = "solar_intermittency_stress"):
    """Run a single stress test"""
    framework = RenewableStressTestFramework()
    scenarios = framework.create_stress_test_scenarios()
    
    # Find the requested test
    test_config = None
    for scenario in scenarios:
        if scenario.test_name == test_name:
            test_config = scenario
            break
    
    if not test_config:
        print(f"Test '{test_name}' not found")
        return
    
    # Run the test
    results = await framework.run_stress_test(test_config)
    
    print(f"\n=== Stress Test Results: {test_name} ===")
    print(f"Total Violations: {len(results['violations'])}")
    print(f"Performance Analysis: {results['performance_analysis']}")
    
    return results

async def run_all_stress_tests():
    """Run all predefined stress tests"""
    framework = RenewableStressTestFramework()
    scenarios = framework.create_stress_test_scenarios()
    
    all_results = {}
    
    for scenario in scenarios:
        print(f"\nRunning {scenario.test_name}...")
        results = await framework.run_stress_test(scenario)
        all_results[scenario.test_name] = results
    
    # Generate summary report
    _generate_summary_report(all_results)
    
    return all_results

def _generate_summary_report(all_results: Dict[str, Any]):
    """Generate a summary report of all tests"""
    print("\n" + "="*50)
    print("RENEWABLE ENERGY INTEGRATION STRESS TEST SUMMARY")
    print("="*50)
    
    for test_name, results in all_results.items():
        print(f"\n{test_name.upper()}:")
        print(f"  Total Violations: {len(results['violations'])}")
        
        analysis = results.get('performance_analysis', {})
        stability = analysis.get('stability_metrics', {})
        economic = analysis.get('economic_metrics', {})
        renewable = analysis.get('renewable_metrics', {})
        
        print(f"  Frequency Std: {stability.get('frequency_std', 0):.4f}")
        print(f"  Voltage Std: {stability.get('voltage_std', 0):.4f}")
        print(f"  Avg Renewable Penetration: {renewable.get('average_penetration', 0):.2%}")
        print(f"  Price Volatility: {economic.get('price_volatility', 0):.2f}")

if __name__ == "__main__":
    # Run a single test
    # asyncio.run(run_individual_stress_test("solar_intermittency_stress"))
    
    # Or run all tests
    asyncio.run(run_all_stress_tests()) 