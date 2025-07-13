#!/usr/bin/env python3
"""
Smart Grid Dashboard Launcher

Quick access to the Smart Grid Performance Analytics Dashboard from the project root.

Usage:
    python dashboard.py                 # Launch interactive dashboard
    python dashboard.py --report        # Generate static report
    python dashboard.py --test          # Run dashboard tests
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

STRESS_RESULTS_DIR = 'renewable_energy_integration_studies/renewable_stress_results'
BLACKOUT_FILE = 'blackout_studies/blackout_simulation_results.json'

def load_json_file(filepath):
    """Load JSON data from file"""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_key_metrics(test_data):
    """Extract important metrics from a stress test JSON"""
    if not test_data:
        return {}
    final_state = test_data.get('final_state', {})
    grid_system = final_state.get('grid_system', {})
    grid_status = final_state.get('grid_status', {})
    grid_state = grid_status.get('grid_state', {})
    metrics = {
        'test_name': test_data.get('test_name', 'Unknown'),
        'total_system_cost': grid_system.get('total_system_cost', 0),
        'average_clearing_price': grid_system.get('average_clearing_price', 0),
        'renewable_penetration': grid_system.get('renewable_penetration', 0) * 100,
        'frequency_hz': grid_state.get('frequency_hz', 50.0),
        'voltage_pu': grid_state.get('voltage_pu', 1.0),
        'total_generation_mw': grid_state.get('total_generation_mw', 0),
        'renewable_generation_mw': grid_state.get('renewable_generation_mw', 0),
        'reserve_margin_mw': grid_system.get('reserve_margin_mw', 0),
        'frequency_violations': grid_system.get('frequency_violations', 0),
        'market_clearings': final_state.get('simulation', {}).get('market_clearings', 0),
    }
    storage_key = next((k for k in final_state if 'battery' in k or 'hydro' in k), None)
    if storage_key:
        storage = final_state[storage_key]
        metrics['storage_soc'] = storage.get('state_of_charge_percent', 50.0)
        metrics['storage_utilization'] = storage.get('capacity_utilization', 0) * 100
        metrics['storage_revenue'] = storage.get('total_revenue', 0)
    return metrics

def analyze_blackout_data(blackout_data):
    """Extract summary metrics from blackout scenarios"""
    summaries = {}
    for scenario, data in blackout_data.items():
        summary = data.get('summary', {})
        summaries[scenario] = {
            'total_events': summary.get('total_blackout_events', 0),
            'duration_hours': summary.get('total_blackout_duration_hours', 0),
            'avg_freq_dev': summary.get('avg_frequency_deviation', 0),
            'system_cost': summary.get('total_system_cost', 0),
            'reliability_score': summary.get('system_reliability_score', 0)
        }
    return summaries

def plot_time_series(test_name, timeline_data):
    """Generate plots from metrics_timeline and save as PNG"""
    if not timeline_data:
        return []
    
    df = pd.DataFrame(timeline_data)
    if 'timestamp' not in df.columns:
        df['hour'] = np.arange(len(df)) * (1/12)  # Assuming 5-min intervals
    else:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour + (pd.to_datetime(df['timestamp']).dt.minute / 60)
    
    plots = []
    os.makedirs('plots', exist_ok=True)
    
    # Frequency plot
    plt.figure(figsize=(10, 6))
    if 'grid_status' in df.columns:
        freq = df['grid_status'].apply(lambda x: x.get('grid_state', {}).get('frequency_hz', 50.0))
    else:
        freq = pd.Series([50.0] * len(df))  # Default
    plt.plot(df.get('hour', df.index), freq)
    plt.title(f'{test_name} - Frequency Over Time')
    plt.xlabel('Hour')
    plt.ylabel('Frequency (Hz)')
    plt.axhline(50.0, color='r', linestyle='--')
    freq_file = f'plots/{test_name}_frequency.png'
    plt.savefig(freq_file)
    plt.close()
    plots.append(freq_file)
    
    # Generation plot
    plt.figure(figsize=(10, 6))
    if 'grid_status' in df.columns:
        total_gen = df['grid_status'].apply(lambda x: x.get('grid_state', {}).get('total_generation_mw', 0))
        renew_gen = df['grid_status'].apply(lambda x: x.get('grid_state', {}).get('renewable_generation_mw', 0))
    else:
        total_gen = pd.Series([0] * len(df))
        renew_gen = pd.Series([0] * len(df))
    plt.plot(df.get('hour', df.index), total_gen, label='Total Generation')
    plt.plot(df.get('hour', df.index), renew_gen, label='Renewable Generation')
    plt.title(f'{test_name} - Generation Over Time')
    plt.xlabel('Hour')
    plt.ylabel('MW')
    plt.legend()
    gen_file = f'plots/{test_name}_generation.png'
    plt.savefig(gen_file)
    plt.close()
    plots.append(gen_file)
    
    # Add more plots as needed (costs, storage SOC, etc.)
    
    return plots

def generate_markdown_report(stress_metrics, blackout_summaries, all_plots):
    """Generate markdown report with embedded image links"""
    md = "# Renewable Energy Integration Stress Test Results\n\n"
    md += f"**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}\n"
    md += f"**Tests Analyzed:** {len(stress_metrics)}\n"
    md += f"**Blackout Scenarios:** {len(blackout_summaries)}\n\n"
    md += "## Executive Summary\n\n"
    avg_cost = np.mean([m['total_system_cost'] for m in stress_metrics.values()])
    avg_penetration = np.mean([m['renewable_penetration'] for m in stress_metrics.values()])
    avg_freq = np.mean([m['frequency_hz'] for m in stress_metrics.values()])
    md += f"Average System Cost: ${avg_cost:,.2f}\n"
    md += f"Average Renewable Penetration: {avg_penetration:.1f}%\n"
    md += f"Average Frequency: {avg_freq:.2f} Hz\n\n"
    md += "## Test Scenarios Overview\n\n"
    df = pd.DataFrame.from_dict(stress_metrics, orient='index')
    md += df.to_markdown() + "\n\n"
    md += "## Blackout Scenarios Summary\n\n"
    df_blackout = pd.DataFrame.from_dict(blackout_summaries, orient='index')
    md += df_blackout.to_markdown() + "\n\n"
    md += "## Detailed Analysis\n\n"
    for test_name, metrics in stress_metrics.items():
        md += f"### {test_name}\n"
        md += f"- Total Cost: ${metrics['total_system_cost']:,.2f}\n"
        md += f"- Renewable Penetration: {metrics['renewable_penetration']:.1f}%\n"
        md += f"- Frequency: {metrics['frequency_hz']:.2f} Hz\n"
        md += f"- Reserve Margin: {metrics['reserve_margin_mw']} MW\n\n"
    md += "## Conclusions\n\n"
    if avg_penetration < 10:
        md += "🚨 Critical: Low renewable penetration indicates dispatch issues.\n"
    if any(m['frequency_violations'] > 0 for m in stress_metrics.values()):
        md += "⚠️ Warning: Frequency violations detected.\n"
    md += "## Visualizations\n\n"
    for test_name, plots in all_plots.items():
        md += f"### {test_name}\n"
        for plot in plots:
            md += f"![{os.path.basename(plot)}]({plot})\n\n"
    return md

def main():
    """Main function to run the analysis"""
    print("🔌 Loading stress test results...")
    stress_files = [
        'demo_duck_curve_20250711_123435.json',
        'demo_wind_ramping_20250711_123335.json',
        'demo_solar_intermittency_20250711_123235.json'
    ]
    stress_metrics = {}
    all_plots = {}
    for file in stress_files:
        data = load_json_file(os.path.join(STRESS_RESULTS_DIR, file))
        if data:
            test_name = data.get('test_name', file)
            stress_metrics[test_name] = extract_key_metrics(data)
            timeline = data.get('metrics_timeline', [])
            all_plots[test_name] = plot_time_series(test_name, timeline)
    blackout_data = load_json_file(BLACKOUT_FILE)
    blackout_summaries = analyze_blackout_data(blackout_data) if blackout_data else {}
    report = generate_markdown_report(stress_metrics, blackout_summaries, all_plots)
    output_file = 'analysis_report.md'
    with open(output_file, 'w') as f:
        f.write(report)
    print(f"📊 Analysis complete! Report saved to {output_file}")
    print("📈 Plots saved in 'plots/' directory")

if __name__ == '__main__':
    main() 