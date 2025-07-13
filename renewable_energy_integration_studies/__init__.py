"""
Renewable Energy Integration Studies Package

This package provides comprehensive stress testing frameworks and analysis tools
for renewable energy integration challenges in smart grids.

Key modules:
- renewable_stress_tests: Main stress testing framework
- run_stress_tests: Interactive test runner
- demo_stress_test: Demonstration scenarios

Usage:
    from renewable_energy_integration_studies import RenewableStressTestFramework
    
    framework = RenewableStressTestFramework()
    results = await framework.run_stress_test(config)
"""

from .renewable_stress_tests import (
    RenewableStressTestFramework,
    StressTestConfig,
    run_individual_stress_test,
    run_all_stress_tests
)

__version__ = "1.0.0"
__author__ = "Smart Grid Research Team"
__description__ = "Renewable Energy Integration Stress Testing Framework"

__all__ = [
    "RenewableStressTestFramework",
    "StressTestConfig", 
    "run_individual_stress_test",
    "run_all_stress_tests"
] 