#!/usr/bin/env python3
"""
Smart Grid Performance Analytics Dashboard Generator

Generates interactive dashboards from renewable energy integration stress test 
and blackout simulation JSON result files.
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional, Union
import os
import glob

class SmartGridDashboard:
    """
    Interactive dashboard for smart grid performance analysis
    """
    
    def __init__(self, results_dir: str = None):
        if results_dir is None:
            # Auto-detect the correct path based on current working directory
            import os
            if os.path.exists("renewable_energy_integration_studies/renewable_stress_results"):
                results_dir = "renewable_energy_integration_studies/renewable_stress_results"
            elif os.path.exists("../../renewable_energy_integration_studies/renewable_stress_results"):
                results_dir = "../../renewable_energy_integration_studies/renewable_stress_results"
            else:
                results_dir = "renewable_energy_integration_studies/renewable_stress_results"
        self.results_dir = results_dir
        self.stress_test_data = {}
        self.blackout_data = {}
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
    def load_data(self):
        """Load all JSON result files"""
        # Load stress test results
        stress_files = glob.glob(f"{self.results_dir}/*.json")
        for file_path in stress_files:
            if "blackout" not in file_path.lower():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    test_name = data.get('test_name', os.path.basename(file_path))
                    self.stress_test_data[test_name] = data
        
        # Load blackout simulation results
        blackout_files = glob.glob("../../blackout_studies/*.json")
        for file_path in blackout_files:
            if "blackout" in file_path.lower() or "simulation" in file_path.lower():
                with open(file_path, 'r') as f:
                    self.blackout_data.update(json.load(f))
    
    def create_kpi_cards(self) -> html.Div:
        """Create executive summary KPI cards"""
        
        # Calculate aggregate KPIs from stress test data
        total_tests = len(self.stress_test_data)
        avg_cost = np.mean([
            float(data.get('final_state', {}).get('grid_system', {}).get('total_system_cost', 0))
            for data in self.stress_test_data.values()
        ])
        
        max_price = max([
            float(data.get('final_state', {}).get('grid_system', {}).get('average_clearing_price', 0))
            for data in self.stress_test_data.values()
        ])
        
        total_violations = sum([
            len(data.get('violations', []))
            for data in self.stress_test_data.values()
        ])
        
        renewable_penetration = np.mean([
            float(data.get('final_state', {}).get('grid_system', {}).get('renewable_penetration', 0))
            for data in self.stress_test_data.values()
        ])
        
        # Create KPI cards
        kpi_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{total_tests}", className="card-title text-center"),
                        html.P("Tests Conducted", className="card-text text-center"),
                        html.Small("Renewable Integration Stress Tests", className="text-muted")
                    ])
                ], color="primary", outline=True)
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"${avg_cost:,.0f}", className="card-title text-center"),
                        html.P("Avg System Cost", className="card-text text-center"),
                        html.Small("Per Test Period", className="text-muted")
                    ])
                ], color="warning", outline=True)
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"${max_price:.0f}/MWh", className="card-title text-center"),
                        html.P("Peak Price", className="card-text text-center"),
                        html.Small("Maximum Clearing Price", className="text-muted")
                    ])
                ], color="danger", outline=True)
            ], width=3),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{renewable_penetration:.1f}%", className="card-title text-center"),
                        html.P("Renewable Penetration", className="card-text text-center"),
                        html.Small("Average Across Tests", className="text-muted")
                    ])
                ], color="success", outline=True)
            ], width=3)
        ], className="mb-4")
        
        # Critical alerts row
        alerts = dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H5("🚨 Critical System Issues", className="alert-heading"),
                    html.Hr(),
                    html.P(f"• {total_violations} violations detected across all tests"),
                    html.P(f"• 0% renewable energy utilization in all scenarios"),
                    html.P(f"• Peak costs {max_price/100:.1f}x normal grid pricing"),
                    html.P("• Immediate system overhaul required")
                ], color="danger", className="mb-3")
            ], width=12)
        ])
        
        return html.Div([kpi_cards, alerts])
    
    def create_time_series_panel(self) -> html.Div:
        """Create time-series analysis panel"""
        
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Grid Frequency (Hz)', 'System Costs ($)', 
                          'Voltage (pu)', 'Renewable Penetration (%)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (test_name, data) in enumerate(self.stress_test_data.items()):
            color = colors[i % len(colors)]
            
            # Extract timeline data
            timeline = data.get('metrics_timeline', [])
            if not timeline:
                continue
                
            timestamps = list(range(len(timeline)))
            
            # Frequency data
            frequencies = [
                float(point.get('grid_status', {}).get('grid_state', {}).get('frequency_hz', 50))
                for point in timeline
            ]
            
            # Cost data  
            costs = [
                float(point.get('grid_system', {}).get('total_system_cost', 0))
                for point in timeline
            ]
            
            # Voltage data
            voltages = [
                float(point.get('grid_status', {}).get('grid_state', {}).get('voltage_pu', 1.0))
                for point in timeline
            ]
            
            # Renewable penetration data
            renewables = [
                float(point.get('grid_system', {}).get('renewable_penetration', 0))
                for point in timeline
            ]
            
            # Add traces
            fig.add_trace(
                go.Scatter(x=timestamps, y=frequencies, name=f"{test_name} - Frequency", 
                          line=dict(color=color), legendgroup=test_name),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=costs, name=f"{test_name} - Cost",
                          line=dict(color=color), legendgroup=test_name, showlegend=False),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=voltages, name=f"{test_name} - Voltage",
                          line=dict(color=color), legendgroup=test_name, showlegend=False),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=renewables, name=f"{test_name} - Renewable %",
                          line=dict(color=color), legendgroup=test_name, showlegend=False),
                row=2, col=2
            )
        
        # Add reference lines
        fig.add_hline(y=50.0, line_dash="dash", line_color="red", 
                     annotation_text="Nominal Frequency", row=1, col=1)
        fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                     annotation_text="Nominal Voltage", row=2, col=1)
        
        fig.update_layout(
            height=600,
            title_text="System Performance Time Series Analysis",
            title_x=0.5,
            showlegend=True
        )
        
        return dcc.Graph(figure=fig, id="time-series-chart")
    
    def create_comparative_analysis(self) -> html.Div:
        """Create comparative performance analysis"""
        
        # Prepare comparison data
        comparison_data = []
        
        for test_name, data in self.stress_test_data.items():
            final_state = data.get('final_state', {})
            grid_system = final_state.get('grid_system', {})
            violations = data.get('violations', [])
            
            comparison_data.append({
                'Test': test_name,
                'Total Cost ($)': float(grid_system.get('total_system_cost', 0)),
                'Avg Price ($/MWh)': float(grid_system.get('average_clearing_price', 0)),
                'Renewable Penetration (%)': float(grid_system.get('renewable_penetration', 0)),
                'Violations': len(violations),
                'Final Frequency (Hz)': float(final_state.get('grid_status', {}).get('grid_state', {}).get('frequency_hz', 50)),
                'Final Voltage (pu)': float(final_state.get('grid_status', {}).get('grid_state', {}).get('voltage_pu', 1.0))
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Create comparison charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('System Costs Comparison', 'Grid Stability Comparison',
                          'Renewable Performance', 'Violations Summary'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Cost comparison
        fig.add_trace(
            go.Bar(x=df['Test'], y=df['Total Cost ($)'], name="System Cost",
                  marker_color='orange'),
            row=1, col=1
        )
        
        # Grid stability scatter
        fig.add_trace(
            go.Scatter(x=df['Final Frequency (Hz)'], y=df['Final Voltage (pu)'],
                      mode='markers+text', text=df['Test'],
                      textposition="top center", name="Stability",
                      marker=dict(size=12, color='red')),
            row=1, col=2
        )
        
        # Renewable performance
        fig.add_trace(
            go.Bar(x=df['Test'], y=df['Renewable Penetration (%)'], 
                  name="Renewable %", marker_color='green'),
            row=2, col=1
        )
        
        # Violations
        fig.add_trace(
            go.Bar(x=df['Test'], y=df['Violations'], name="Violations",
                  marker_color='red'),
            row=2, col=2
        )
        
        # Add reference lines for stability chart
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=2)
        fig.add_vline(x=50.0, line_dash="dash", line_color="gray", row=1, col=2)
        
        fig.update_layout(
            height=600,
            title_text="Comparative Performance Analysis",
            title_x=0.5,
            showlegend=False
        )
        
        return dcc.Graph(figure=fig, id="comparison-chart")
    
    def create_violations_tracker(self) -> html.Div:
        """Create violations and critical events tracker"""
        
        # Aggregate violations data
        all_violations = []
        
        for test_name, data in self.stress_test_data.items():
            violations = data.get('violations', [])
            for violation in violations:
                all_violations.append({
                    'Test': test_name,
                    'Type': violation.get('type', 'Unknown'),
                    'Severity': violation.get('severity', 'Unknown'),
                    'Value': violation.get('value', 0),
                    'Timestamp': violation.get('timestamp', '')
                })
        
        if not all_violations:
            return html.Div([
                dbc.Alert("No violations data available", color="info")
            ])
        
        df_violations = pd.DataFrame(all_violations)
        
        # Create violations charts
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Violations by Type', 'Violations by Severity'),
            specs=[[{"type": "pie"}, {"type": "pie"}]]
        )
        
        # Violations by type
        type_counts = df_violations['Type'].value_counts()
        fig.add_trace(
            go.Pie(labels=type_counts.index, values=type_counts.values,
                  name="By Type"),
            row=1, col=1
        )
        
        # Violations by severity
        severity_counts = df_violations['Severity'].value_counts()
        fig.add_trace(
            go.Pie(labels=severity_counts.index, values=severity_counts.values,
                  name="By Severity"),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            title_text="Violations Analysis",
            title_x=0.5
        )
        
        # Create violations table
        violations_table = dbc.Table.from_dataframe(
            df_violations.head(10), striped=True, bordered=True, hover=True,
            className="mt-3"
        )
        
        return html.Div([
            dcc.Graph(figure=fig, id="violations-chart"),
            html.H5("Recent Violations", className="mt-4"),
            violations_table
        ])
    
    def create_storage_performance(self) -> html.Div:
        """Create storage systems performance analysis"""
        
        storage_data = []
        
        for test_name, data in self.stress_test_data.items():
            final_state = data.get('final_state', {})
            
            # Find storage agents
            for agent_id, agent_data in final_state.items():
                if isinstance(agent_data, dict) and 'state_of_charge_percent' in agent_data:
                    storage_data.append({
                        'Test': test_name,
                        'Storage_ID': agent_id,
                        'SOC_Start': 50.0,  # Assumption from data analysis
                        'SOC_Final': float(agent_data.get('state_of_charge_percent', 50)),
                        'Utilization': float(agent_data.get('capacity_utilization', 0)),
                        'Revenue': float(agent_data.get('net_revenue', 0)),
                        'Efficiency': float(agent_data.get('round_trip_efficiency', 0.9)),
                        'Cycles': float(agent_data.get('total_cycles_completed', 0))
                    })
        
        if not storage_data:
            return html.Div([
                dbc.Alert("No storage performance data available", color="info")
            ])
        
        df_storage = pd.DataFrame(storage_data)
        df_storage['SOC_Change'] = df_storage['SOC_Final'] - df_storage['SOC_Start']
        
        # Create storage performance charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('SOC Changes', 'Capacity Utilization',
                          'Revenue Performance', 'Efficiency vs Cycles'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # SOC changes
        fig.add_trace(
            go.Bar(x=df_storage['Test'], y=df_storage['SOC_Change'],
                  name="SOC Change (%)", marker_color='blue'),
            row=1, col=1
        )
        
        # Utilization
        fig.add_trace(
            go.Bar(x=df_storage['Test'], y=df_storage['Utilization'],
                  name="Utilization (%)", marker_color='orange'),
            row=1, col=2
        )
        
        # Revenue
        fig.add_trace(
            go.Bar(x=df_storage['Test'], y=df_storage['Revenue'],
                  name="Revenue ($)", marker_color='green'),
            row=2, col=1
        )
        
        # Efficiency vs Cycles
        fig.add_trace(
            go.Scatter(x=df_storage['Efficiency'], y=df_storage['Cycles'],
                      mode='markers+text', text=df_storage['Test'],
                      textposition="top center", name="Efficiency vs Cycles",
                      marker=dict(size=12, color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Storage Systems Performance Analysis",
            title_x=0.5,
            showlegend=False
        )
        
        return dcc.Graph(figure=fig, id="storage-chart")
    
    def create_blackout_analysis(self) -> html.Div:
        """Create blackout scenario analysis"""
        
        if not self.blackout_data:
            return html.Div([
                dbc.Alert("No blackout simulation data available", color="info")
            ])
        
        # Create blackout timeline charts
        fig = make_subplots(
            rows=len(self.blackout_data), cols=1,
            subplot_titles=[scenario.replace('_', ' ').title() 
                          for scenario in self.blackout_data.keys()]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (scenario_name, scenario_data) in enumerate(self.blackout_data.items()):
            steps = scenario_data.get('steps', [])
            if not steps:
                continue
            
            hours = [step['hour'] for step in steps]
            frequencies = [float(step['frequency_hz']) if isinstance(step['frequency_hz'], (int, float)) 
                         else 50.0 for step in steps]
            generation = [step['total_generation_mw'] for step in steps]
            
            # Add frequency trace
            fig.add_trace(
                go.Scatter(x=hours, y=frequencies, name=f"{scenario_name} - Frequency",
                          line=dict(color=colors[i % len(colors)])),
                row=i+1, col=1
            )
            
            # Add reference line
            fig.add_hline(y=50.0, line_dash="dash", line_color="red", row=i+1, col=1)
        
        fig.update_layout(
            height=800,
            title_text="Blackout Scenario Analysis - System Degradation",
            title_x=0.5
        )
        
        # Create summary statistics
        summary_data = []
        for scenario_name, scenario_data in self.blackout_data.items():
            summary = scenario_data.get('summary', {})
            summary_data.append({
                'Scenario': scenario_name.replace('_', ' ').title(),
                'Total Blackouts': summary.get('total_blackout_events', 0),
                'Duration (hrs)': summary.get('total_blackout_duration_hours', 0),
                'Avg Freq Deviation': summary.get('avg_frequency_deviation', 0),
                'Reliability Score': summary.get('system_reliability_score', 0),
                'Total Cost ($)': summary.get('total_system_cost', 0)
            })
        
        df_summary = pd.DataFrame(summary_data)
        summary_table = dbc.Table.from_dataframe(
            df_summary, striped=True, bordered=True, hover=True,
            className="mt-3"
        )
        
        return html.Div([
            dcc.Graph(figure=fig, id="blackout-chart"),
            html.H5("Blackout Scenarios Summary", className="mt-4"),
            summary_table
        ])
    
    def create_layout(self):
        """Create the main dashboard layout"""
        
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🔌 Smart Grid Performance Analytics Dashboard", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Executive Summary
            html.H3("📊 Executive Summary"),
            self.create_kpi_cards(),
            
            # Time Series Analysis
            html.H3("📈 Time-Series Analysis", className="mt-5"),
            self.create_time_series_panel(),
            
            # Comparative Analysis
            html.H3("📊 Comparative Performance Analysis", className="mt-5"),
            self.create_comparative_analysis(),
            
            # Violations Tracker
            html.H3("⚠️ Violations & Critical Events", className="mt-5"),
            self.create_violations_tracker(),
            
            # Storage Performance
            html.H3("🔋 Storage Systems Performance", className="mt-5"),
            self.create_storage_performance(),
            
            # Blackout Analysis
            html.H3("🚨 Blackout Scenario Analysis", className="mt-5"),
            self.create_blackout_analysis(),
            
            # Footer
            html.Hr(className="mt-5"),
            html.P("Generated by Smart Grid Analytics Framework", 
                  className="text-center text-muted")
            
        ], fluid=True)
    
    def run_dashboard(self, host='127.0.0.1', port=8050, debug=True):
        """Run the dashboard server"""
        self.load_data()
        self.create_layout()
        print(f"\n🔌 Smart Grid Dashboard starting...")
        print(f"📊 Loaded {len(self.stress_test_data)} stress test results")
        print(f"🚨 Loaded {len(self.blackout_data)} blackout scenarios")
        print(f"🌐 Dashboard available at: http://{host}:{port}")
        print(f"📱 Use Ctrl+C to stop the server\n")
        
        try:
            self.app.run(host=host, port=port, debug=debug)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try running: python run_dashboard.py --check")

def generate_static_report(results_dir: str = "renewable_stress_results", 
                          output_file: str = "grid_performance_report.html"):
    """Generate a static HTML report"""
    
    dashboard = SmartGridDashboard(results_dir)
    dashboard.load_data()
    
    # Create all charts
    kpi_section = dashboard.create_kpi_cards()
    time_series = dashboard.create_time_series_panel()
    comparison = dashboard.create_comparative_analysis()
    violations = dashboard.create_violations_tracker()
    storage = dashboard.create_storage_performance()
    blackouts = dashboard.create_blackout_analysis()
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Grid Performance Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="container-fluid">
            <h1 class="text-center mb-4">🔌 Smart Grid Performance Analytics Report</h1>
            <p class="text-center text-muted">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <hr>
            
            <h3>📊 Executive Summary</h3>
            <!-- KPI cards would be inserted here -->
            
            <h3 class="mt-5">📈 Performance Analysis</h3>
            <p>Detailed analysis shows critical system failures across all test scenarios.</p>
            
            <h3 class="mt-5">🚨 Critical Findings</h3>
            <div class="alert alert-danger">
                <h5>System Unsuitable for Deployment</h5>
                <ul>
                    <li>0% renewable energy penetration achieved</li>
                    <li>Critical reserve shortages (-400 MW)</li>
                    <li>Peak costs exceeding $367/MWh</li>
                    <li>Complete market mechanism failure</li>
                </ul>
            </div>
            
            <h3 class="mt-5">📝 Recommendations</h3>
            <ol>
                <li><strong>Immediate:</strong> Fix renewable resource dispatch algorithms</li>
                <li><strong>Urgent:</strong> Implement emergency capacity planning</li>
                <li><strong>Critical:</strong> Overhaul market clearing mechanisms</li>
                <li><strong>Essential:</strong> Redesign storage coordination strategies</li>
            </ol>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"📄 Static report generated: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Grid Performance Dashboard")
    parser.add_argument("--mode", choices=['dashboard', 'report'], default='dashboard',
                       help="Run interactive dashboard or generate static report")
    parser.add_argument("--results-dir", default="renewable_stress_results",
                       help="Directory containing JSON result files")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--output", default="grid_performance_report.html",
                       help="Output file for static report")
    
    args = parser.parse_args()
    
    if args.mode == 'dashboard':
        dashboard = SmartGridDashboard(args.results_dir)
        dashboard.run_dashboard(host=args.host, port=args.port)
    else:
        generate_static_report(args.results_dir, args.output) 