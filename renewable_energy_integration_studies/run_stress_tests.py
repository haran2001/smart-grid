#!/usr/bin/env python3
"""
Simple runner script for renewable energy integration stress tests
"""

import asyncio
import sys
import os
from renewable_stress_tests import RenewableStressTestFramework, run_individual_stress_test, run_all_stress_tests

def print_menu():
    """Print test selection menu"""
    print("\n" + "="*60)
    print("RENEWABLE ENERGY INTEGRATION STRESS TEST MENU")
    print("="*60)
    print("1. Solar Intermittency Stress Test")
    print("2. Wind Variability Stress Test") 
    print("3. Duck Curve Challenge Test")
    print("4. Extreme Weather Event Test")
    print("5. Run All Tests")
    print("6. Custom Test Setup")
    print("7. Exit")
    print("="*60)

async def run_custom_test():
    """Run a custom stress test configuration"""
    print("\n=== Custom Test Configuration ===")
    
    framework = RenewableStressTestFramework()
    
    # Get custom parameters from user
    test_name = input("Enter test name: ").strip()
    if not test_name:
        test_name = "custom_test"
    
    duration = input("Enter test duration (hours, default 24): ").strip()
    try:
        duration = int(duration) if duration else 24
    except ValueError:
        duration = 24
    
    print("\nAvailable disturbance types:")
    print("- solar_variability: Solar irradiance changes")
    print("- wind_variability: Wind speed changes")
    print("- demand_spike: Demand fluctuations")
    print("- frequency_event: Frequency disturbances")
    print("- voltage_event: Voltage disturbances")
    
    # Simple custom configuration
    from renewable_stress_tests import StressTestConfig
    
    config = StressTestConfig(
        test_name=test_name,
        description="Custom user-defined stress test",
        duration_hours=duration,
        agents_config={
            "generators": [
                {"agent_id": "solar_1", "params": {"max_capacity_mw": 200, "fuel_cost_per_mwh": 0}},
                {"agent_id": "wind_1", "params": {"max_capacity_mw": 150, "fuel_cost_per_mwh": 0}},
                {"agent_id": "gas_1", "params": {"max_capacity_mw": 100, "fuel_cost_per_mwh": 80}}
            ],
            "storage": [
                {"agent_id": "battery_1", "params": {"max_capacity_mwh": 200, "max_power_mw": 50}}
            ]
        },
        disturbance_patterns=[
            {
                "type": "solar_variability",
                "solar_agents": ["solar_1"],
                "irradiance_pattern": [800, 200, 900, 100, 850, 50] * 4  # Repeating pattern
            }
        ],
        validation_metrics=["frequency_stability", "voltage_stability", "grid_stability"],
        expected_challenges=["renewable_integration", "grid_stability"]
    )
    
    print(f"\nRunning custom test: {test_name}")
    results = await framework.run_stress_test(config)
    
    print(f"\n=== Custom Test Results ===")
    print(f"Total Violations: {len(results['violations'])}")
    print(f"Test Duration: {duration} hours")
    
    return results

async def main():
    """Main menu loop"""
    while True:
        print_menu()
        
        try:
            choice = input("Select an option (1-7): ").strip()
            
            if choice == "1":
                print("\n🌞 Running Solar Intermittency Stress Test...")
                await run_individual_stress_test("solar_intermittency_stress")
                
            elif choice == "2":
                print("\n🌪️ Running Wind Variability Stress Test...")
                await run_individual_stress_test("wind_variability_stress")
                
            elif choice == "3":
                print("\n🦆 Running Duck Curve Challenge Test...")
                await run_individual_stress_test("duck_curve_stress")
                
            elif choice == "4":
                print("\n⛈️ Running Extreme Weather Event Test...")
                await run_individual_stress_test("extreme_weather_stress")
                
            elif choice == "5":
                print("\n🔄 Running All Stress Tests...")
                await run_all_stress_tests()
                
            elif choice == "6":
                await run_custom_test()
                
            elif choice == "7":
                print("\n👋 Exiting stress test framework...")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-7.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Test interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error running test: {e}")
            
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    print("🔬 Renewable Energy Integration Stress Test Framework")
    print("📊 Testing smart grid resilience against renewable energy challenges")
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 