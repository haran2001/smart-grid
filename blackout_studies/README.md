# Smart Grid Blackout Scenario Visualization System

This visualization system provides comprehensive analysis and interactive charts for smart grid blackout scenarios, based on real-world extreme weather events.

## 📊 Overview

The visualization system analyzes three major blackout scenarios:

1. **Texas Winter Storm Uri (February 2021)** - Extreme cold scenario
2. **California Heat Wave (August 2020)** - Extreme heat scenario  
3. **Winter Storm Elliott (December 2022)** - Cold weather with equipment failures

## 🎯 Key Findings

### Performance Summary
- **Average System Reliability**: 40.3% across all scenarios
- **Total Blackout Events**: 40 events across 3 scenarios
- **Total Blackout Duration**: 20 hours combined
- **Worst Performing**: Texas Winter Storm Uri (37.9% reliability)
- **Best Performing**: Winter Storm Elliott (42.2% reliability)

### Critical Insights
- **Cold weather vulnerability**: Cold scenarios performed worse than heat scenarios
- **Equipment failures**: Texas event saw 87% generation capacity loss
- **Frequency instability**: All scenarios experienced >1 Hz frequency deviations
- **Agent coordination**: System breaks down after multiple simultaneous failures
- **Renewable resilience**: Renewable penetration increased during blackouts (thermal plants failed first)

## 📁 Generated Files

### Visualizations
- `blackout_visualizations/scenario_comparison.png` - 4-panel dashboard comparing all scenarios
- `blackout_visualizations/vulnerability_heatmap.png` - Risk assessment across scenarios
- `blackout_visualizations/timeline_*.png` - Detailed timeline for each scenario

### Data & Reports
- `blackout_visualizations/scenario_data.csv` - Raw metrics for all scenarios
- `blackout_visualizations/analysis_summary.txt` - Detailed text analysis
- `blackout_dashboard.html` - Interactive HTML dashboard

### Source Code
- `blackout_visualization.py` - Visualization generation script
- `blackout_scenarios.py` - Original simulation code
- `blackout_interpretation.md` - Detailed analysis documentation

## 🌐 Interactive Dashboard

Open `blackout_dashboard.html` in your web browser to access the interactive dashboard featuring:

- **Executive Summary** with key metrics
- **Scenario Comparison Charts** showing reliability, blackout events, and frequency stability
- **Vulnerability Heatmap** highlighting risk categories
- **Detailed Timeline Analysis** for each scenario
- **Performance Details** with scenario-specific metrics
- **Insights & Recommendations** for grid improvements

## 📈 Chart Types

### 1. Scenario Comparison Dashboard
Four-panel comparison showing:
- System reliability scores by scenario
- Total blackout events
- Generation capacity vs peak demand
- Average frequency deviation

### 2. Vulnerability Heatmap
Risk assessment matrix showing:
- Frequency risk (stability issues)
- Blackout frequency (event occurrence)
- Load shed risk (unserved demand)
- Duration risk (outage length)
- Supply shortage (generation gaps)

### 3. Timeline Analysis
Six-panel timeline for each scenario:
- System frequency over time
- Generation vs load balance
- Active vs failed agents
- Unserved load (blackout severity)
- Renewable penetration changes
- Communication message volume

## 🔧 Technical Implementation

### Dependencies
- **matplotlib**: Chart generation
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **plotly** (optional): Interactive charts

### Usage
```bash
# Generate all visualizations
python blackout_visualization.py

# Open interactive dashboard
open blackout_dashboard.html
```

### Architecture
- **BlackoutVisualizationDashboard**: Main class handling all visualization generation
- **Mock data generation**: Creates realistic scenario data based on analysis results
- **Multi-format output**: PNG images, HTML dashboard, CSV data, text reports

## 📊 Key Metrics Tracked

### System Performance
- **Reliability Score**: Overall system performance (0-100%)
- **Blackout Events**: Number of grid instability events
- **Blackout Duration**: Total time in unstable state
- **Frequency Deviation**: Average deviation from 50 Hz
- **Unserved Load**: Peak load that couldn't be served

### Generation Analysis
- **Peak Demand**: Maximum load during scenario
- **Minimum Generation**: Lowest generation capacity
- **Generation Loss**: Percentage capacity lost
- **Renewable Penetration**: Share of renewable generation

### Agent Coordination
- **Active Agents**: Functional grid components
- **Failed Agents**: Non-responsive components
- **Message Volume**: Communication overhead
- **Coordination Breakdown**: Multi-agent system performance

## 🏆 Scenario Rankings

1. **Winter Storm Elliott (42.2%)** - Best performance, improved cold weather response
2. **California Heat Wave (40.8%)** - Good heat resilience, limited thermal stress
3. **Texas Winter Storm Uri (37.9%)** - Worst performance, extreme cold vulnerabilities

## ⚠️ Critical Vulnerabilities Identified

### Cold Weather Risks
- Thermal plant cascade failures (30-35% failure rates)
- Fuel supply disruption (gas pipeline freezing)
- Equipment winterization inadequacy
- Demand surge from heating (80% increase)

### System Design Issues
- No emergency frequency response below 49 Hz
- Communication overload during crisis (300+ messages/hour)
- Market mechanism breakdown under stress
- Insufficient agent redundancy

## 💡 Recommendations

### Immediate Actions (0-6 months)
- Implement comprehensive equipment winterization
- Deploy automatic emergency frequency response at 49.5 Hz
- Enhance communication protocols for crisis scenarios

### Medium-term (6-18 months)  
- Increase battery storage capacity by 40%
- Expand demand response automation
- Improve agent coordination algorithms

### Long-term (18+ months)
- Diversify fuel supply chains
- Deploy microgrids for resilience
- Implement predictive maintenance systems

## 🔍 Future Enhancements

### Proposed Improvements
- Real-time data integration
- Interactive parameter adjustment
- 3D visualization capabilities
- Machine learning failure prediction
- Integration with live grid data

### Additional Scenarios
- Hurricane impact analysis
- Cyber attack simulations
- Equipment aging effects
- Renewable intermittency stress tests

## 📞 Usage Instructions

1. **Run visualization generation**: `python blackout_visualization.py`
2. **Open dashboard**: Double-click `blackout_dashboard.html`
3. **Explore charts**: View PNG files in `blackout_visualizations/` folder
4. **Analyze data**: Open CSV files for detailed metrics
5. **Read analysis**: Review text summary in analysis_summary.txt

## 📚 References

- Original simulation: `blackout_scenarios.py`
- Analysis documentation: `blackout_interpretation.md`
- Visualization code: `blackout_visualization.py`
- Interactive dashboard: `blackout_dashboard.html`

This visualization system transforms complex simulation data into actionable insights for improving smart grid resilience against extreme weather events. 