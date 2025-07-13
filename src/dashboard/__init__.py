"""
Smart Grid Performance Analytics Dashboard Package

This package provides comprehensive dashboard and visualization tools
for analyzing renewable energy integration stress test results and
blackout simulation data.

Key modules:
- dashboard_generator: Main dashboard engine with interactive visualizations
- run_dashboard: User-friendly launcher with multiple modes
- test_dashboard: Validation and testing framework

Usage:
    from src.dashboard import SmartGridDashboard
    
    dashboard = SmartGridDashboard()
    dashboard.run_dashboard()
"""

from .dashboard_generator import SmartGridDashboard, generate_static_report

__version__ = "1.0.0"
__author__ = "Smart Grid Research Team"
__description__ = "Smart Grid Performance Analytics Dashboard"

__all__ = [
    "SmartGridDashboard",
    "generate_static_report"
] 