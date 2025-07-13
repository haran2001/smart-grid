#!/usr/bin/env python3
"""
Quick Start Script for Smart Grid Dashboard

Usage:
    python run_dashboard.py                    # Run interactive dashboard
    python run_dashboard.py --report           # Generate static report
    python run_dashboard.py --demo             # Run with demo data
"""

import os
import sys
import subprocess
from dashboard_generator import SmartGridDashboard, generate_static_report

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['dash', 'plotly', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install with: pip install -r dashboard_requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def run_interactive_dashboard():
    """Run the interactive dashboard"""
    print("🚀 Starting Smart Grid Performance Dashboard...")
    print("📊 Loading data from renewable stress test results...")
    
    dashboard = SmartGridDashboard()
    dashboard.run_dashboard()

def run_demo_dashboard():
    """Run dashboard with demo explanation"""
    print("🎮 DEMO MODE: Smart Grid Dashboard")
    print("=" * 50)
    print("📄 This dashboard analyzes JSON result files from:")
    print("   • Renewable energy integration stress tests")
    print("   • Blackout simulation scenarios")
    print("   • Duck curve challenge results")
    print("\n🔍 Key Features:")
    print("   • Real-time KPI monitoring")
    print("   • Interactive time-series charts")
    print("   • Comparative performance analysis")
    print("   • Violations tracking")
    print("   • Storage performance metrics")
    print("   • Blackout scenario analysis")
    print("\n🌐 Dashboard will open in your browser...")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    dashboard = SmartGridDashboard()
    dashboard.run_dashboard()

def generate_report():
    """Generate static HTML report"""
    print("📄 Generating Static Performance Report...")
    output_file = "smart_grid_performance_report.html"
    
    generate_static_report(output_file=output_file)
    
    print(f"✅ Report generated: {output_file}")
    print("🌐 Open the file in your browser to view the report")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smart Grid Performance Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dashboard.py              # Run interactive dashboard
  python run_dashboard.py --report     # Generate static HTML report  
  python run_dashboard.py --demo       # Run with demo explanations
  python run_dashboard.py --check      # Check dependencies only
        """
    )
    
    parser.add_argument("--report", action="store_true", 
                       help="Generate static HTML report instead of interactive dashboard")
    parser.add_argument("--demo", action="store_true",
                       help="Run in demo mode with explanations")
    parser.add_argument("--check", action="store_true",
                       help="Check dependencies and exit")
    
    args = parser.parse_args()
    
    print("🔌 Smart Grid Performance Analytics")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    if args.check:
        print("✅ All systems ready!")
        return
    
    # Check if data files exist
    data_exists = (
        os.path.exists("../../renewable_energy_integration_studies/renewable_stress_results") and 
        len([f for f in os.listdir("../../renewable_energy_integration_studies/renewable_stress_results") if f.endswith('.json')]) > 0
    )
    
    if not data_exists:
        print("⚠️  No data files found in renewable_energy_integration_studies/renewable_stress_results/")
        print("💡 Run some stress tests first to generate data")
        if not args.demo:
            response = input("🎮 Run in demo mode anyway? (y/N): ")
            if response.lower() != 'y':
                return
            args.demo = True
    
    # Run appropriate mode
    if args.report:
        generate_report()
    elif args.demo:
        run_demo_dashboard()
    else:
        run_interactive_dashboard()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Try running: python run_dashboard.py --check") 