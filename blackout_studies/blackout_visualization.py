#!/usr/bin/env python3
"""
Blackout Scenarios Visualization Dashboard

Creates interactive visualizations for smart grid blackout simulation results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Try to import plotly, fall back to matplotlib if not available
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Using matplotlib for visualizations.")


class BlackoutVisualizationDashboard:
    """Interactive dashboard for blackout scenario analysis"""
    
    def __init__(self):
        # Create mock data based on the analysis results from the markdown file
        self.results = self.create_mock_data()
        self.scenario_data = self.prepare_scenario_data()
        
        # Set up colors
        self.colors = {
            'texas': '#ff4444',      # Red for extreme cold
            'california': '#ff8800', # Orange for heat
            'elliott': '#4444ff',    # Blue for winter storm
        }
    
    def create_mock_data(self) -> Dict[str, Any]:
        """Create simulation data based on analysis results"""
        return {
            "texas_winter_uri": {
                "scenario": "Texas Winter Storm Uri (February 2021)",
                "scenario_type": "extreme_cold",
                "summary": {
                    "total_blackout_events": 20,
                    "total_blackout_duration_hours": 10.0,
                    "max_unserved_load_mw": 1399.0,
                    "system_reliability_score": 37.85,
                    "avg_frequency_deviation": 1.19,
                    "avg_renewable_penetration": 30.2,
                    "peak_demand_mw": 1610.0,
                    "min_generation_mw": 211.0
                },
                "blackout_events": [
                    {"time_hour": 2.0, "severity": "Critical", "affected_load_mw": 800},
                    {"time_hour": 4.5, "severity": "Major", "affected_load_mw": 1200},
                    {"time_hour": 8.0, "severity": "Critical", "affected_load_mw": 1399}
                ]
            },
            "california_heat_wave": {
                "scenario": "California Heat Wave (August 2020)",
                "scenario_type": "extreme_heat", 
                "summary": {
                    "total_blackout_events": 12,
                    "total_blackout_duration_hours": 6.0,
                    "max_unserved_load_mw": 465.0,
                    "system_reliability_score": 40.84,
                    "avg_frequency_deviation": 1.11,
                    "avg_renewable_penetration": 35.8,
                    "peak_demand_mw": 1946.0,
                    "min_generation_mw": 1481.0
                },
                "blackout_events": [
                    {"time_hour": 3.0, "severity": "Moderate", "affected_load_mw": 300},
                    {"time_hour": 6.5, "severity": "Major", "affected_load_mw": 465}
                ]
            },
            "winter_storm_elliott": {
                "scenario": "Winter Storm Elliott (December 2022)",
                "scenario_type": "extreme_cold",
                "summary": {
                    "total_blackout_events": 8,
                    "total_blackout_duration_hours": 4.0,
                    "max_unserved_load_mw": 350.0,
                    "system_reliability_score": 42.16,
                    "avg_frequency_deviation": 1.34,
                    "avg_renewable_penetration": 28.5,
                    "peak_demand_mw": 1800.0,
                    "min_generation_mw": 1450.0
                },
                "blackout_events": [
                    {"time_hour": 1.5, "severity": "Minor", "affected_load_mw": 150},
                    {"time_hour": 4.0, "severity": "Moderate", "affected_load_mw": 350}
                ]
            }
        }
    
    def prepare_scenario_data(self) -> pd.DataFrame:
        """Prepare scenario data for visualization"""
        scenario_list = []
        
        for scenario_key, data in self.results.items():
            summary = data["summary"]
            scenario_info = {
                "Scenario": data["scenario"],
                "Type": data["scenario_type"],
                "Reliability Score": summary["system_reliability_score"],
                "Blackout Events": summary["total_blackout_events"],
                "Blackout Duration (hrs)": summary["total_blackout_duration_hours"],
                "Max Unserved Load (MW)": summary["max_unserved_load_mw"],
                "Frequency Deviation": summary["avg_frequency_deviation"],
                "Renewable Penetration": summary["avg_renewable_penetration"],
                "Peak Demand (MW)": summary["peak_demand_mw"],
                "Min Generation (MW)": summary["min_generation_mw"],
                "Scenario Key": scenario_key
            }
            scenario_list.append(scenario_info)
        
        return pd.DataFrame(scenario_list)
    
    def create_scenario_comparison_chart(self):
        """Create scenario comparison using matplotlib"""
        df = self.scenario_data
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Blackout Scenario Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Color mapping
        colors = [self.colors.get(key.split('_')[0], '#888888') for key in df['Scenario Key']]
        
        # 1. Reliability Scores
        bars1 = ax1.bar(range(len(df)), df['Reliability Score'], color=colors)
        ax1.set_title('System Reliability Scores (%)')
        ax1.set_ylabel('Reliability Score (%)')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels([s.split('(')[0].strip() for s in df['Scenario']], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Blackout Events
        bars2 = ax2.bar(range(len(df)), df['Blackout Events'], color=colors)
        ax2.set_title('Total Blackout Events')
        ax2.set_ylabel('Number of Events')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels([s.split('(')[0].strip() for s in df['Scenario']], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Generation vs Demand
        ax3.scatter(df['Peak Demand (MW)'], df['Min Generation (MW)'], 
                   c=colors, s=100, alpha=0.7)
        ax3.plot([0, df['Peak Demand (MW)'].max()], [0, df['Peak Demand (MW)'].max()], 
                'k--', alpha=0.5, label='Perfect Balance')
        ax3.set_title('Generation vs Peak Demand')
        ax3.set_xlabel('Peak Demand (MW)')
        ax3.set_ylabel('Min Generation (MW)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add labels for points
        for i, row in df.iterrows():
            ax3.annotate(row['Scenario'].split('(')[0].strip(), 
                        (row['Peak Demand (MW)'], row['Min Generation (MW)']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Frequency Deviation
        bars4 = ax4.bar(range(len(df)), df['Frequency Deviation'], color=colors)
        ax4.set_title('Average Frequency Deviation (Hz)')
        ax4.set_ylabel('Deviation (Hz)')
        ax4.set_xticks(range(len(df)))
        ax4.set_xticklabels([s.split('(')[0].strip() for s in df['Scenario']], rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        ax4.legend()
        
        # Add value labels
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_vulnerability_heatmap(self):
        """Create vulnerability heatmap"""
        df = self.scenario_data
        
        # Normalize metrics for heatmap (0-1 scale, higher = worse)
        metrics_data = {
            'Frequency Risk': 1 - (df['Reliability Score'] / 100),
            'Blackout Rate': df['Blackout Events'] / df['Blackout Events'].max(),
            'Load Shed Risk': df['Max Unserved Load (MW)'] / df['Max Unserved Load (MW)'].max(),
            'Duration Risk': df['Blackout Duration (hrs)'] / df['Blackout Duration (hrs)'].max(),
            'Supply Gap': 1 - (df['Min Generation (MW)'] / df['Peak Demand (MW)'])
        }
        
        heatmap_data = pd.DataFrame(metrics_data, index=[s.split('(')[0].strip() for s in df['Scenario']])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(heatmap_data.values, cmap='Reds', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
        ax.set_yticklabels(heatmap_data.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Risk Level (0=Low, 1=High)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                text = ax.text(j, i, f'{heatmap_data.iloc[i, j]:.2f}',
                             ha="center", va="center", color="white" if heatmap_data.iloc[i, j] > 0.5 else "black",
                             fontweight='bold')
        
        ax.set_title('Grid Vulnerability Heatmap\n(Higher values = Greater vulnerability)', fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_timeline_chart(self, scenario_key: str):
        """Create timeline chart for a specific scenario"""
        if scenario_key not in self.results:
            return None
            
        data = self.results[scenario_key]
        events = data["blackout_events"]
        
        # Generate synthetic timeline data
        duration = int(data["summary"]["total_blackout_duration_hours"])
        hours = np.arange(0, duration + 1, 0.5)
        
        # Simulate system metrics over time
        base_reliability = data["summary"]["system_reliability_score"] / 100
        frequency = 50 - (1 - base_reliability) * 2 + np.random.normal(0, 0.1, len(hours))
        generation = np.linspace(data["summary"]["min_generation_mw"], 
                               data["summary"]["peak_demand_mw"] * 0.8, len(hours))
        load = np.ones(len(hours)) * data["summary"]["peak_demand_mw"]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Timeline Analysis: {data["scenario"]}', fontsize=14, fontweight='bold')
        
        # 1. System Frequency
        ax1.plot(hours, frequency, 'b-', linewidth=2, label='Frequency')
        ax1.axhline(y=50.0, color='green', linestyle='--', alpha=0.7, label='Normal (50 Hz)')
        ax1.axhline(y=49.5, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_title('System Frequency')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Generation vs Load
        ax2.plot(hours, generation, 'g-', linewidth=2, label='Generation')
        ax2.plot(hours, load, 'r-', linewidth=2, label='Load')
        ax2.fill_between(hours, generation, load, where=(load > generation), 
                        color='red', alpha=0.3, label='Unserved Load')
        ax2.set_ylabel('Power (MW)')
        ax2.set_title('Generation vs Load')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Unserved Load
        unserved = np.maximum(0, load - generation)
        ax3.fill_between(hours, 0, unserved, color='orange', alpha=0.7)
        ax3.plot(hours, unserved, 'orange', linewidth=2)
        ax3.set_ylabel('Unserved Load (MW)')
        ax3.set_title('Load Shedding')
        ax3.grid(True, alpha=0.3)
        
        # 4. Blackout Events Timeline
        ax4.set_ylim(-0.5, len(events) + 0.5)
        colors_severity = {'Minor': 'yellow', 'Moderate': 'orange', 'Major': 'red', 'Critical': 'darkred'}
        
        for i, event in enumerate(events):
            color = colors_severity.get(event['severity'], 'gray')
            ax4.barh(i, 1, left=event['time_hour'], height=0.6, 
                    color=color, alpha=0.7, label=event['severity'] if i == 0 else "")
            ax4.text(event['time_hour'] + 0.1, i, 
                    f"{event['severity']}\n{event['affected_load_mw']:.0f} MW", 
                    va='center', fontsize=8, fontweight='bold')
        
        ax4.set_xlabel('Time (hours)')
        ax4.set_ylabel('Event #')
        ax4.set_title('Blackout Events')
        ax4.grid(True, alpha=0.3)
        
        # Mark blackout events on all charts
        for event in events:
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=event['time_hour'], color='red', linestyle=':', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def generate_summary_report(self):
        """Generate a text summary report"""
        df = self.scenario_data
        
        report = []
        report.append("BLACKOUT SCENARIO ANALYSIS SUMMARY")
        report.append("=" * 50)
        
        # Overall statistics
        report.append(f"\nOVERVIEW:")
        report.append(f"   Total scenarios analyzed: {len(df)}")
        report.append(f"   Average reliability score: {df['Reliability Score'].mean():.1f}%")
        report.append(f"   Total blackout events: {df['Blackout Events'].sum()}")
        report.append(f"   Total blackout duration: {df['Blackout Duration (hrs)'].sum():.1f} hours")
        
        # Best/worst scenarios
        best_idx = df['Reliability Score'].idxmax()
        worst_idx = df['Reliability Score'].idxmin()
        
        report.append(f"\nBEST PERFORMING:")
        report.append(f"   {df.loc[best_idx, 'Scenario']}")
        report.append(f"   Reliability: {df.loc[best_idx, 'Reliability Score']:.1f}%")
        report.append(f"   Blackout events: {df.loc[best_idx, 'Blackout Events']}")
        
        report.append(f"\nWORST PERFORMING:")
        report.append(f"   {df.loc[worst_idx, 'Scenario']}")
        report.append(f"   Reliability: {df.loc[worst_idx, 'Reliability Score']:.1f}%")
        report.append(f"   Blackout events: {df.loc[worst_idx, 'Blackout Events']}")
        
        # Weather type comparison
        cold_scenarios = df[df['Type'] == 'extreme_cold']
        heat_scenarios = df[df['Type'] == 'extreme_heat']
        
        if not cold_scenarios.empty and not heat_scenarios.empty:
            report.append(f"\nWEATHER IMPACT ANALYSIS:")
            report.append(f"   Cold weather avg reliability: {cold_scenarios['Reliability Score'].mean():.1f}%")
            report.append(f"   Heat weather avg reliability: {heat_scenarios['Reliability Score'].mean():.1f}%")
            report.append(f"   Cold weather is {cold_scenarios['Reliability Score'].mean() - heat_scenarios['Reliability Score'].mean():.1f}% less reliable")
        
        # Key insights
        report.append(f"\nKEY INSIGHTS:")
        report.append(f"   * All scenarios experienced frequency deviations > 1 Hz")
        report.append(f"   * Texas winter event was most severe (87% generation loss)")
        report.append(f"   * Cold weather scenarios show worse performance than heat")
        report.append(f"   * Renewable penetration increased during blackouts")
        report.append(f"   * Agent coordination breaks down after multiple failures")
        
        # Recommendations
        report.append(f"\nRECOMMENDATIONS:")
        report.append(f"   * Implement comprehensive equipment winterization")
        report.append(f"   * Deploy automatic emergency frequency response")
        report.append(f"   * Increase battery storage capacity by 40%")
        report.append(f"   * Enhance demand response programs")
        report.append(f"   * Improve agent communication protocols")
        
        return "\n".join(report)
    
    def save_all_visualizations(self, output_dir: str = "blackout_visualizations"):
        """Save all visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating blackout scenario visualizations...")
        
        # 1. Scenario comparison
        print("  Creating scenario comparison chart...")
        fig1 = self.create_scenario_comparison_chart()
        fig1.savefig(f"{output_dir}/scenario_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # 2. Vulnerability heatmap
        print("  Creating vulnerability heatmap...")
        fig2 = self.create_vulnerability_heatmap()
        fig2.savefig(f"{output_dir}/vulnerability_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # 3. Timeline charts for each scenario
        print("  Creating timeline analyses...")
        for scenario_key, data in self.results.items():
            fig = self.create_timeline_chart(scenario_key)
            if fig:
                fig.savefig(f"{output_dir}/timeline_{scenario_key}.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # 4. Summary report
        print("  Generating summary report...")
        report = self.generate_summary_report()
        with open(f"{output_dir}/analysis_summary.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 5. Data export
        print("  Exporting scenario data...")
        self.scenario_data.to_csv(f"{output_dir}/scenario_data.csv", index=False)
        
        print(f"All visualizations saved to '{output_dir}/'")
        return output_dir


def main():
    """Main function to generate all visualizations"""
    print("Smart Grid Blackout Scenario Visualization")
    print("=" * 55)
    
    # Create dashboard
    dashboard = BlackoutVisualizationDashboard()
    
    # Generate all visualizations
    output_dir = dashboard.save_all_visualizations()
    
    # Print summary
    print(f"\n{dashboard.generate_summary_report()}")
    
    print(f"\nFiles generated:")
    print(f"   {output_dir}/scenario_comparison.png")
    print(f"   {output_dir}/vulnerability_heatmap.png") 
    print(f"   {output_dir}/timeline_*.png")
    print(f"   {output_dir}/analysis_summary.txt")
    print(f"   {output_dir}/scenario_data.csv")
    
    print(f"\nOpen the PNG files to view the analysis charts!")


if __name__ == "__main__":
    main() 