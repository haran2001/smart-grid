#!/usr/bin/env python3
"""
Demonstration script for renewable energy integration stress testing
Shows how to use the framework to test specific challenges
"""

import asyncio
import sys
import os

# Import utilities are handled by the renewable_stress_tests module

from renewable_stress_tests import RenewableStressTestFramework, StressTestConfig

async def demo_solar_intermittency():
    """Demonstrate solar intermittency stress test"""
    print("🌞 DEMO: Solar Intermittency Stress Test")
    print("="*50)
    
    framework = RenewableStressTestFramework()
    
    # Create a focused solar intermittency scenario
    config = StressTestConfig(
        test_name="demo_solar_intermittency",
        description="Demonstrate rapid solar power fluctuations from cloud cover",
        duration_hours=6,  # Shorter for demo
        agents_config={
            "generators": [
                {
                    "agent_id": "solar_farm_main", 
                    "params": {
                        "max_capacity_mw": 300,
                        "fuel_cost_per_mwh": 0,
                        "emissions_rate_kg_co2_per_mwh": 0,
                        "efficiency": 1.0
                    }
                },
                {
                    "agent_id": "gas_backup", 
                    "params": {
                        "max_capacity_mw": 150,
                        "fuel_cost_per_mwh": 85,
                        "emissions_rate_kg_co2_per_mwh": 400,
                        "efficiency": 0.45
                    }
                }
            ],
            "storage": [
                {
                    "agent_id": "grid_battery", 
                    "params": {
                        "max_capacity_mwh": 400,
                        "max_power_mw": 100,
                        "round_trip_efficiency": 0.88
                    }
                }
            ]
        },
        disturbance_patterns=[
            {
                "type": "solar_variability",
                "solar_agents": ["solar_farm_main"],
                "irradiance_pattern": [
                    900,  # Morning: Full sun
                    850,  # Partial clouds
                    200,  # Heavy cloud cover
                    800,  # Clouds pass
                    100,  # Dense clouds
                    950,  # Clear again
                    300,  # Patchy clouds
                    750   # Evening sun
                ]
            }
        ],
        validation_metrics=[
            "frequency_stability",
            "voltage_stability", 
            "storage_utilization",
            "backup_activation",
            "market_price_volatility"
        ],
        expected_challenges=[
            "rapid_output_changes",
            "storage_response_time",
            "backup_generator_ramping",
            "frequency_regulation"
        ]
    )
    
    print("Configuration:")
    print(f"  - Solar Farm: {config.agents_config['generators'][0]['params']['max_capacity_mw']} MW")
    print(f"  - Gas Backup: {config.agents_config['generators'][1]['params']['max_capacity_mw']} MW")  
    print(f"  - Battery Storage: {config.agents_config['storage'][0]['params']['max_capacity_mwh']} MWh")
    print(f"  - Test Duration: {config.duration_hours} hours")
    print(f"  - Disturbance: {len(config.disturbance_patterns[0]['irradiance_pattern'])} cloud events")
    
    print("\n⚡ Starting simulation...")
    results = await framework.run_stress_test(config)
    
    print("\n📊 Results Summary:")
    print(f"  - Total Violations: {len(results['violations'])}")
    print(f"  - Test Duration: {config.duration_hours} hours")
    
    if results['violations']:
        print("  - Violation Types:")
        for violation in results['violations'][:3]:  # Show first 3
            print(f"    • {violation['type']}: {violation['severity']} ({violation['value']:.2f})")
    
    analysis = results.get('performance_analysis', {})
    if analysis:
        stability = analysis.get('stability_metrics', {})
        economic = analysis.get('economic_metrics', {})
        
        print("  - Stability Metrics:")
        print(f"    • Frequency Std: {stability.get('frequency_std', 0):.4f} Hz")
        print(f"    • Voltage Std: {stability.get('voltage_std', 0):.4f} pu")
        print(f"  - Economic Impact:")
        print(f"    • Price Volatility: ${economic.get('price_volatility', 0):.2f}/MWh")
    
    return results

async def demo_wind_ramping():
    """Demonstrate wind ramping stress test"""
    print("\n🌪️ DEMO: Wind Ramping Stress Test")
    print("="*50)
    
    framework = RenewableStressTestFramework()
    
    config = StressTestConfig(
        test_name="demo_wind_ramping",
        description="Demonstrate extreme wind power ramping events",
        duration_hours=8,
        agents_config={
            "generators": [
                {
                    "agent_id": "wind_farm_alpha", 
                    "params": {
                        "max_capacity_mw": 250,
                        "fuel_cost_per_mwh": 0,
                        "emissions_rate_kg_co2_per_mwh": 0
                    }
                },
                {
                    "agent_id": "wind_farm_beta", 
                    "params": {
                        "max_capacity_mw": 200,
                        "fuel_cost_per_mwh": 0,
                        "emissions_rate_kg_co2_per_mwh": 0
                    }
                },
                {
                    "agent_id": "coal_baseload", 
                    "params": {
                        "max_capacity_mw": 180,
                        "fuel_cost_per_mwh": 55,
                        "emissions_rate_kg_co2_per_mwh": 900
                    }
                }
            ],
            "storage": [
                {
                    "agent_id": "pumped_hydro", 
                    "params": {
                        "max_capacity_mwh": 800,
                        "max_power_mw": 200,
                        "round_trip_efficiency": 0.75
                    }
                }
            ]
        },
        disturbance_patterns=[
            {
                "type": "wind_variability",
                "wind_agents": ["wind_farm_alpha", "wind_farm_beta"],
                "wind_speed_pattern": [
                    22,  # High wind
                    25,  # Very high wind
                    8,   # Sudden drop (wind shadow)
                    5,   # Low wind
                    3,   # Very low wind
                    18,  # Wind picks up
                    28,  # Storm winds
                    12   # Moderate wind
                ]
            }
        ],
        validation_metrics=[
            "grid_stability",
            "ramping_adequacy",
            "storage_response",
            "frequency_regulation"
        ],
        expected_challenges=[
            "extreme_wind_ramping",
            "grid_inertia_loss",
            "storage_coordination",
            "backup_generation"
        ]
    )
    
    print("Configuration:")
    print(f"  - Wind Farm A: {config.agents_config['generators'][0]['params']['max_capacity_mw']} MW")
    print(f"  - Wind Farm B: {config.agents_config['generators'][1]['params']['max_capacity_mw']} MW")
    print(f"  - Coal Baseload: {config.agents_config['generators'][2]['params']['max_capacity_mw']} MW")
    print(f"  - Pumped Hydro: {config.agents_config['storage'][0]['params']['max_capacity_mwh']} MWh")
    
    print("\n⚡ Starting simulation...")
    results = await framework.run_stress_test(config)
    
    print("\n📊 Results Summary:")
    print(f"  - Total Violations: {len(results['violations'])}")
    print(f"  - Critical Events: {len([v for v in results['violations'] if v.get('severity') == 'critical'])}")
    
    return results

async def demo_duck_curve():
    """Demonstrate duck curve challenge"""
    print("\n🦆 DEMO: Duck Curve Challenge")
    print("="*50)
    
    framework = RenewableStressTestFramework()
    
    config = StressTestConfig(
        test_name="demo_duck_curve",
        description="Demonstrate evening ramping challenge with high solar",
        duration_hours=24,
        agents_config={
            "generators": [
                {
                    "agent_id": "solar_array", 
                    "params": {
                        "max_capacity_mw": 500,
                        "fuel_cost_per_mwh": 0,
                        "emissions_rate_kg_co2_per_mwh": 0
                    }
                },
                {
                    "agent_id": "gas_peaker_1", 
                    "params": {
                        "max_capacity_mw": 200,
                        "fuel_cost_per_mwh": 130,
                        "emissions_rate_kg_co2_per_mwh": 350
                    }
                },
                {
                    "agent_id": "gas_peaker_2", 
                    "params": {
                        "max_capacity_mw": 180,
                        "fuel_cost_per_mwh": 125,
                        "emissions_rate_kg_co2_per_mwh": 350
                    }
                }
            ],
            "storage": [
                {
                    "agent_id": "battery_farm", 
                    "params": {
                        "max_capacity_mwh": 600,
                        "max_power_mw": 150,
                        "round_trip_efficiency": 0.90
                    }
                }
            ]
        },
        disturbance_patterns=[
            {
                "type": "solar_variability",
                "solar_agents": ["solar_array"],
                "irradiance_pattern": [
                    0, 0, 0, 0, 0, 0,        # Night (hours 0-5)
                    200, 500, 800, 950,     # Morning ramp (hours 6-9)
                    1000, 1000, 950, 800,   # Midday peak (hours 10-13)
                    600, 400, 200, 50,      # Evening decline (hours 14-17)
                    0, 0, 0, 0, 0, 0         # Night (hours 18-23)
                ]
            },
            {
                "type": "demand_spike",
                "demand_multipliers": [
                    0.6, 0.5, 0.5, 0.5, 0.6, 0.7,  # Night/early morning
                    0.8, 0.7, 0.6, 0.6, 0.7, 0.8,  # Morning
                    0.9, 1.0, 0.9, 0.8, 0.9, 1.0,  # Afternoon
                    1.1, 1.0, 0.9, 0.8, 0.7, 0.6   # Evening peak
                ]
            }
        ],
        validation_metrics=[
            "ramping_capability",
            "storage_arbitrage",
            "price_volatility",
            "renewable_curtailment"
        ],
        expected_challenges=[
            "evening_ramp_up",
            "storage_optimization",
            "peaker_coordination",
            "net_load_variability"
        ]
    )
    
    print("Configuration:")
    print(f"  - Solar Array: {config.agents_config['generators'][0]['params']['max_capacity_mw']} MW")
    print(f"  - Gas Peakers: {config.agents_config['generators'][1]['params']['max_capacity_mw']} + {config.agents_config['generators'][2]['params']['max_capacity_mw']} MW")
    print(f"  - Battery Farm: {config.agents_config['storage'][0]['params']['max_capacity_mwh']} MWh")
    print(f"  - Scenario: Classic California duck curve")
    
    print("\n⚡ Starting simulation...")
    results = await framework.run_stress_test(config)
    
    print("\n📊 Results Summary:")
    print(f"  - Total Violations: {len(results['violations'])}")
    
    analysis = results.get('performance_analysis', {})
    if analysis:
        renewable = analysis.get('renewable_metrics', {})
        economic = analysis.get('economic_metrics', {})
        
        print("  - Duck Curve Metrics:")
        print(f"    • Avg Renewable Penetration: {renewable.get('average_penetration', 0):.1%}")
        print(f"    • Storage Utilization: {renewable.get('storage_utilization', 0):.1%}")
        print(f"    • Price Volatility: ${economic.get('price_volatility', 0):.2f}/MWh")
    
    return results

async def main():
    """Run all demonstration tests"""
    print("🔬 Renewable Energy Integration Stress Test Demo")
    print("📊 Demonstrating key challenges in renewable integration")
    print("="*60)
    
    demos = [
        ("Solar Intermittency", demo_solar_intermittency),
        ("Wind Ramping", demo_wind_ramping),
        ("Duck Curve", demo_duck_curve)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n🚀 Running {demo_name} Demo...")
            result = await demo_func()
            results[demo_name] = result
            print(f"✅ {demo_name} Demo completed successfully!")
            
        except Exception as e:
            print(f"❌ {demo_name} Demo failed: {e}")
            results[demo_name] = {"error": str(e)}
    
    print("\n" + "="*60)
    print("📋 DEMO SUMMARY")
    print("="*60)
    
    for demo_name, result in results.items():
        if "error" in result:
            print(f"❌ {demo_name}: Failed - {result['error']}")
        else:
            violations = len(result.get('violations', []))
            print(f"✅ {demo_name}: {violations} violations detected")
    
    print("\n💡 Key Insights:")
    print("  - Solar intermittency requires fast-responding storage and backup")
    print("  - Wind ramping tests grid stability and inertia management")
    print("  - Duck curve challenges evening demand/generation balance")
    print("  - Each scenario reveals different system stress points")
    
    print("\n🎯 Next Steps:")
    print("  1. Analyze detailed results in renewable_stress_results/ directory")
    print("  2. Identify specific improvement areas for your agents")
    print("  3. Run custom scenarios targeting your system's weak points")
    print("  4. Iterate on agent strategies and test again")
    
    return results

if __name__ == "__main__":
    print("Starting Renewable Energy Integration Stress Test Demo...")
    
    try:
        results = asyncio.run(main())
        print("\n✅ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1) 