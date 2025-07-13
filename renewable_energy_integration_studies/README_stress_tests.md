# Renewable Energy Integration Stress Test Framework

## Overview

This framework provides comprehensive stress testing capabilities for renewable energy integration challenges in smart grids. It tests your multi-agent system against the key challenges identified in renewable energy integration research.

## Key Features

- **Multi-scenario Testing**: Predefined scenarios covering major renewable integration challenges
- **Real-time Monitoring**: Continuous monitoring of grid stability, market efficiency, and system performance
- **Comprehensive Metrics**: Detailed analysis of frequency stability, voltage regulation, market dynamics, and more
- **Interactive Interface**: Simple menu-driven interface for running tests
- **Customizable Tests**: Framework for creating custom stress test scenarios

## Test Scenarios

### 1. Solar Intermittency Stress Test
**Challenge**: Rapid solar power fluctuations due to cloud cover
- **Duration**: 12 hours
- **Focus**: Tests storage response, backup generation, and grid stability
- **Key Metrics**: Frequency stability, voltage stability, storage utilization

### 2. Wind Variability Stress Test  
**Challenge**: Wind power variability and extreme ramping events
- **Duration**: 24 hours
- **Focus**: Tests ramping capability, grid inertia, and forecast accuracy
- **Key Metrics**: Grid stability, market efficiency, ramping adequacy

### 3. Duck Curve Challenge Test
**Challenge**: Evening ramping with high solar penetration
- **Duration**: 24 hours
- **Focus**: Tests storage arbitrage, peaker coordination, and demand management
- **Key Metrics**: Ramping capability, storage arbitrage, price volatility

### 4. Extreme Weather Event Test
**Challenge**: System resilience during compound weather events
- **Duration**: 48 hours
- **Focus**: Tests emergency response, resource adequacy, and system recovery
- **Key Metrics**: System resilience, emergency response, blackout prevention

## Installation & Setup

1. **Prerequisites**: Ensure you have the main smart grid simulation system installed
2. **Dependencies**: Install required Python packages:
   ```bash
   pip install numpy pandas matplotlib asyncio
   ```
3. **File Structure**: Place the stress test files in your `experiments/` directory

## Running Tests

### Option 1: Interactive Menu
```bash
cd experiments
python run_stress_tests.py
```

### Option 2: Individual Test
```bash
python -c "import asyncio; from renewable_stress_tests import run_individual_stress_test; asyncio.run(run_individual_stress_test('solar_intermittency_stress'))"
```

### Option 3: All Tests
```bash
python -c "import asyncio; from renewable_stress_tests import run_all_stress_tests; asyncio.run(run_all_stress_tests())"
```

## Test Results

Results are automatically saved to `renewable_stress_results/` directory:
- **JSON Files**: Detailed test results with metrics timeline
- **Log Files**: Comprehensive logging of test execution
- **Summary Reports**: Performance analysis and violation counts

## Understanding Results

### Violation Types
- **Frequency Violations**: Grid frequency outside 49.8-50.2 Hz range
- **Voltage Violations**: Voltage outside 0.95-1.05 per unit range
- **Reserve Shortages**: Available reserves below 50 MW threshold

### Performance Metrics
- **Stability Metrics**: Frequency/voltage standard deviation, violation counts
- **Economic Metrics**: Average prices, price volatility, system costs
- **Renewable Metrics**: Penetration levels, curtailment events, storage utilization
- **Reliability Metrics**: Total violations, recovery times, critical events

## Customization

### Adding New Test Scenarios
1. Create a new `StressTestConfig` in `create_stress_test_scenarios()`
2. Define agent configurations, disturbance patterns, and validation metrics
3. Add scenario to the test menu in `run_stress_tests.py`

### Custom Disturbance Patterns
```python
disturbance_patterns = [
    {
        "type": "solar_variability",
        "solar_agents": ["solar_farm_1"],
        "irradiance_pattern": [800, 200, 900, 100, 850, 50, 750]
    },
    {
        "type": "wind_variability", 
        "wind_agents": ["wind_farm_1"],
        "wind_speed_pattern": [15, 5, 20, 3, 18, 8, 2, 25]
    }
]
```

### Custom Validation Metrics
```python
validation_metrics = [
    "frequency_stability",
    "voltage_stability", 
    "storage_utilization",
    "market_efficiency",
    "renewable_penetration",
    "grid_stability_score"
]
```

## Interpreting Challenges

### Based on Smart Grid Research
The stress tests are designed around key challenges identified in renewable energy integration literature:

1. **Intermittency & Variability**: Tests system response to unpredictable renewable output
2. **Power Quality Issues**: Validates frequency/voltage control under renewable variability
3. **Storage Integration**: Tests storage coordination and arbitrage strategies
4. **Market Mechanisms**: Evaluates market efficiency with high renewable penetration
5. **Grid Stability**: Tests system resilience and emergency response capabilities

### Expected Outcomes
- **Successful Integration**: Low violation counts, stable grid metrics, efficient markets
- **Stress Points**: Identifying thresholds where system performance degrades
- **Improvement Areas**: Specific agents or mechanisms that need enhancement

## Advanced Features

### Compound Event Testing
Run multiple simultaneous disturbances:
```python
disturbance_patterns = [
    {"type": "solar_variability", "solar_agents": ["solar_1"], ...},
    {"type": "demand_spike", "demand_multipliers": [1.5, 1.2, 1.8], ...},
    {"type": "frequency_event", "frequency_pattern": [49.9, 50.1, 49.8], ...}
]
```

### Performance Benchmarking
Compare different grid configurations:
- High renewable penetration vs. traditional mix
- With/without storage systems
- Different market mechanisms
- Various agent strategies

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure src/ directory is in Python path
2. **Timeout Errors**: Reduce test duration or increase time step
3. **Memory Issues**: Limit number of concurrent agents
4. **Async Errors**: Use Python 3.7+ with proper asyncio support

### Performance Tips
- Start with shorter test durations (6-12 hours)
- Use fewer agents for initial testing
- Monitor system resources during execution
- Save intermediate results for long tests

## Contributing

To add new stress test scenarios:
1. Identify specific renewable integration challenge
2. Design appropriate disturbance patterns
3. Define relevant validation metrics
4. Test scenario thoroughly
5. Document expected outcomes and interpretation

## References

Based on renewable energy integration challenges from academic literature:
- Intermittency and stochastic behavior of renewables
- Power quality issues (voltage fluctuations, harmonics, EMI)
- Grid stability and synchronization challenges
- Energy storage coordination requirements
- Market mechanism adaptations for renewable integration 