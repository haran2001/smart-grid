# 🔌 Smart Grid Performance Analytics Dashboard

A comprehensive dashboard for analyzing renewable energy integration stress test results and blackout simulation data. This dashboard transforms raw JSON log files into interactive visualizations and actionable insights.

## 🎯 Overview

The Smart Grid Dashboard provides the same analytical depth as expert manual analysis but in an **interactive, visual format**. It automatically extracts insights from your JSON result files and presents them through intuitive charts, KPIs, and alerts.

### 🔍 **Key Capabilities**

- **📊 Multi-Source Data Integration**: Handles renewable stress tests + blackout simulations
- **🎛️ Interactive Analysis**: Click, zoom, and filter through your data
- **⚡ Real-time KPI Monitoring**: Critical metrics with threshold alerting
- **📈 Time-Series Visualization**: Track system performance over time
- **🔍 Comparative Analysis**: Side-by-side performance comparison
- **⚠️ Automated Insights**: Automatic detection of critical issues and violations

## 🎨 Dashboard Features

### **1. 📊 Executive Summary KPI Cards**
- **System Cost Overview**: Total costs, peak pricing, cost trends
- **Grid Stability Metrics**: Frequency deviations, voltage stability
- **Renewable Performance**: Penetration levels, utilization rates
- **Critical Alerts**: Immediate visibility into system failures

### **2. 📈 Time-Series Analysis Panel**
- **Multi-metric Timeline**: Frequency, voltage, costs, renewable penetration
- **Interactive Zooming**: Drill down into specific time periods
- **Reference Lines**: Compare against nominal/target values
- **Multi-scenario Overlay**: Compare different test scenarios

### **3. 📊 Comparative Performance Analysis**
- **Side-by-Side Comparison**: Solar vs Wind vs Duck Curve performance
- **Grid Stability Scatter Plot**: Frequency vs Voltage correlation analysis
- **Cost Efficiency Analysis**: System costs across different scenarios
- **Violations Summary**: Count and severity of system violations

### **4. ⚠️ Violations & Critical Events Tracker**
- **Violation Type Analysis**: Pie charts showing violation categories
- **Severity Classification**: Critical, high, medium, low severity tracking
- **Timeline Tracking**: When violations occurred during tests
- **Detailed Violation Table**: Complete violation details with timestamps

### **5. 🔋 Storage Systems Performance**
- **State of Charge (SOC) Analysis**: Charging/discharging patterns
- **Capacity Utilization**: How effectively storage is being used
- **Revenue Performance**: Economic value captured by storage systems
- **Efficiency Metrics**: Round-trip efficiency and cycle analysis

### **6. 🚨 Blackout Scenario Analysis**
- **System Degradation Timeline**: How systems fail during extreme events
- **Multi-Scenario Comparison**: Texas Uri vs California Heat Wave vs Winter Storm
- **Reliability Scoring**: Quantitative reliability assessment
- **Failure Pattern Analysis**: Understanding cascade failure patterns

## 🚀 Quick Start Guide

### **Step 1: Installation**
```bash
# Navigate to the renewable energy integration studies directory
cd renewable_energy_integration_studies

# Install required packages
pip install -r dashboard_requirements.txt
```

### **Step 2: Launch Dashboard**
```bash
# Option 1: Interactive Dashboard (recommended)
python run_dashboard.py

# Option 2: Demo Mode (with explanations)
python run_dashboard.py --demo

# Option 3: Generate Static Report
python run_dashboard.py --report

# Option 4: Check Dependencies
python run_dashboard.py --check
```

### **Step 3: Access Dashboard**
- **Interactive Dashboard**: Open browser to `http://127.0.0.1:8050`
- **Static Report**: Open generated HTML file in browser

## 📁 Data Sources

The dashboard automatically loads data from:

### **Renewable Energy Integration Stress Tests**
- **File Location**: `renewable_stress_results/*.json`
- **Supported Tests**: Solar intermittency, Wind ramping, Duck curve challenges
- **Data Extracted**: Grid stability, economic performance, storage behavior, violations

### **Blackout Simulation Results**
- **File Location**: `blackout_studies/*.json`
- **Supported Scenarios**: Texas Winter Uri, California Heat Wave, Winter Storm Elliott
- **Data Extracted**: System degradation, failure cascades, reliability metrics

### **Required JSON Structure**
Your JSON files should contain:
```json
{
  "test_name": "demo_solar_intermittency",
  "start_time": "2025-07-11T12:31:35.755846",
  "initial_state": { /* agent states */ },
  "metrics_timeline": [ /* time-series data */ ],
  "violations": [ /* violation events */ ],
  "final_state": { /* final agent states */ },
  "performance_analysis": { /* analysis results */ }
}
```

## 🎛️ Dashboard Usage Guide

### **Navigation**
- **Scroll Down**: View different analysis sections
- **Click Charts**: Interactive zooming and selection
- **Hover Elements**: Detailed tooltips and information
- **Filter Data**: Use dropdown menus to focus on specific tests

### **Key Metrics to Monitor**

#### **🔴 Critical Warning Indicators**
- **System Cost > $300/MWh**: Extreme pricing indicating system stress
- **Frequency Deviation > ±0.5 Hz**: Grid stability compromised
- **Renewable Penetration = 0%**: Complete renewable resource failure
- **Violations > 5 per test**: System reliability issues

#### **🟡 Performance Degradation Signs**
- **Rising System Costs**: Economic inefficiency trends
- **Voltage Fluctuations**: Power quality issues
- **Storage Revenue = $0**: Market mechanism failures
- **Agent Failures**: Communication/coordination breakdowns

#### **🟢 System Health Indicators**
- **Stable Frequency (50.0 ± 0.1 Hz)**: Good grid control
- **Positive Reserve Margins**: Adequate capacity planning
- **Storage Arbitrage Revenue**: Functional market mechanisms
- **High Renewable Utilization**: Successful integration

### **Interpreting Results**

#### **Grid Stability Analysis**
- **Frequency Chart**: Should stay within 49.8-50.2 Hz band
- **Voltage Chart**: Should remain within 0.95-1.05 pu range
- **Stability Scatter**: Points should cluster around (50.0, 1.0)

#### **Economic Performance**
- **Cost Trends**: Should be stable and predictable
- **Price Spikes**: Indicate capacity shortages or market failures
- **Storage Revenue**: Should be positive during price arbitrage

#### **Storage Performance**
- **SOC Changes**: Should follow logical charge/discharge patterns
- **Utilization**: Higher is better (indicates active grid participation)
- **Revenue**: Positive values indicate successful arbitrage

## 🔧 Customization Options

### **Advanced Dashboard Configuration**
```python
# Custom dashboard with specific data directory
dashboard = SmartGridDashboard(results_dir="custom_results")

# Custom host and port
dashboard.run_dashboard(host="0.0.0.0", port=8080)

# Generate custom report
generate_static_report(
    results_dir="custom_results",
    output_file="custom_report.html"
)
```

### **Adding New Data Sources**
1. Place JSON files in the `renewable_stress_results/` directory
2. Ensure JSON structure matches expected format
3. Restart dashboard to load new data

### **Custom Analysis**
The dashboard code is modular and extensible:
- **Add new charts**: Modify chart creation functions
- **Custom metrics**: Add new KPI calculations
- **Enhanced filtering**: Implement additional data filters

## 📊 Dashboard Sections Detailed

### **1. Executive Summary KPIs**
**Purpose**: Immediate system health overview
**Key Metrics**: 
- Tests conducted count
- Average system cost
- Peak clearing price
- Average renewable penetration
- Critical alerts panel

### **2. Time-Series Analysis**
**Purpose**: Understand system behavior over time
**Charts**: 
- Grid frequency timeline
- System cost progression
- Voltage stability tracking
- Renewable penetration evolution

### **3. Comparative Performance**
**Purpose**: Compare different test scenarios
**Analysis**: 
- Cost comparison across tests
- Grid stability correlation analysis
- Renewable performance comparison
- Violations count summary

### **4. Violations Tracker**
**Purpose**: Monitor system failures and violations
**Features**: 
- Violation type distribution (pie chart)
- Severity level analysis
- Detailed violations table
- Timeline correlation

### **5. Storage Performance**
**Purpose**: Analyze energy storage effectiveness
**Metrics**: 
- State of charge changes
- Capacity utilization rates
- Revenue generation
- Efficiency vs cycling analysis

### **6. Blackout Analysis**
**Purpose**: Understand system resilience during extreme events
**Scenarios**: 
- Texas Winter Storm Uri simulation
- California Heat Wave analysis
- Winter Storm Elliott impacts
- Comparative reliability scoring

## 🚨 Troubleshooting

### **Common Issues**

#### **Dashboard Won't Start**
```bash
# Check dependencies
python run_dashboard.py --check

# Install missing packages
pip install -r dashboard_requirements.txt
```

#### **No Data Visible**
- Ensure JSON files are in `renewable_stress_results/` directory
- Check JSON file format matches expected structure
- Verify file permissions allow reading

#### **Charts Not Loading**
- Check browser console for JavaScript errors
- Try different browser (Chrome/Firefox recommended)
- Ensure internet connection for external libraries

#### **Performance Issues**
- Reduce data file sizes if very large
- Close other browser tabs
- Restart dashboard if memory usage high

### **Error Messages**
- **"No data files found"**: Run stress tests first to generate data
- **"Missing required packages"**: Install dependencies with pip
- **"Dashboard failed to start"**: Check port 8050 isn't already in use

## 📈 Performance Benchmarks

### **Expected Dashboard Performance**
- **Load Time**: < 5 seconds for typical data sets
- **Interactivity**: Smooth zooming and filtering
- **Memory Usage**: < 500MB for standard data volumes
- **Supported Data**: Up to 1000 test result files

### **Optimization Tips**
- **Large Datasets**: Use data filtering to focus on specific time periods
- **Multiple Scenarios**: Consider separate analysis sessions
- **Historical Data**: Archive older test results to improve performance

## 🔮 Future Enhancements

### **Planned Features**
- **Real-time Data Streaming**: Live dashboard updates during test execution
- **Advanced Filtering**: Multi-dimensional data filtering capabilities
- **Custom Alert Rules**: User-defined thresholds and notifications
- **Export Functionality**: PDF reports and data export options
- **Mobile Responsiveness**: Optimized mobile and tablet viewing

### **Integration Possibilities**
- **API Integration**: REST API for programmatic access
- **Database Backend**: Store results in SQL/NoSQL databases
- **Cloud Deployment**: Deploy dashboard to cloud platforms
- **Multi-user Support**: Role-based access and collaborative analysis

## 📞 Support

### **Getting Help**
- Check this README for common solutions
- Review example JSON files for proper format
- Examine console output for error details
- Test with demo mode to verify installation

### **Contributing**
- Report bugs through issue tracking
- Suggest new features or improvements
- Submit pull requests for enhancements
- Share example datasets for testing

---

**🎯 The Smart Grid Dashboard transforms complex JSON logs into actionable insights, providing the same analytical depth as expert manual analysis but with the speed and interactivity of modern data visualization tools.** 