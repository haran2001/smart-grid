#!/usr/bin/env python3
"""
Test script for Smart Grid Dashboard

Tests dashboard functionality with existing JSON data files
"""

import json
import os
import sys
from dashboard_generator import SmartGridDashboard

def test_data_loading():
    """Test if dashboard can load existing data files"""
    print("🧪 Testing Dashboard Data Loading...")
    
    dashboard = SmartGridDashboard()
    dashboard.load_data()
    
    print(f"📊 Loaded {len(dashboard.stress_test_data)} stress test files:")
    for test_name in dashboard.stress_test_data.keys():
        print(f"   ✅ {test_name}")
    
    print(f"🚨 Loaded {len(dashboard.blackout_data)} blackout scenarios:")
    for scenario_name in dashboard.blackout_data.keys():
        print(f"   ✅ {scenario_name}")
    
    return len(dashboard.stress_test_data) > 0

def test_chart_generation():
    """Test if dashboard can generate charts without errors"""
    print("\n🎨 Testing Chart Generation...")
    
    dashboard = SmartGridDashboard()
    dashboard.load_data()
    
    try:
        # Test each chart generation function
        charts_to_test = [
            ("KPI Cards", dashboard.create_kpi_cards),
            ("Time Series", dashboard.create_time_series_panel),
            ("Comparative Analysis", dashboard.create_comparative_analysis),
            ("Violations Tracker", dashboard.create_violations_tracker),
            ("Storage Performance", dashboard.create_storage_performance),
            ("Blackout Analysis", dashboard.create_blackout_analysis)
        ]
        
        for chart_name, chart_function in charts_to_test:
            try:
                result = chart_function()
                print(f"   ✅ {chart_name}: Generated successfully")
            except Exception as e:
                print(f"   ❌ {chart_name}: Error - {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Chart generation failed: {e}")
        return False

def test_json_file_structure():
    """Test if existing JSON files have the expected structure"""
    print("\n📄 Testing JSON File Structure...")
    
    required_fields = ['test_name', 'final_state', 'violations']
    stress_files = []
    
    # Find stress test files
    results_dir = "../../renewable_energy_integration_studies/renewable_stress_results"
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.json'):
                stress_files.append(os.path.join(results_dir, file))
    
    if not stress_files:
        print("   ⚠️  No stress test JSON files found")
        return False
    
    for file_path in stress_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                print(f"   ⚠️  {os.path.basename(file_path)}: Missing fields {missing_fields}")
            else:
                print(f"   ✅ {os.path.basename(file_path)}: Structure OK")
                
        except Exception as e:
            print(f"   ❌ {os.path.basename(file_path)}: Error reading - {e}")
            return False
    
    return True

def test_dashboard_startup():
    """Test if dashboard can start without errors (dry run)"""
    print("\n🚀 Testing Dashboard Startup (Dry Run)...")
    
    try:
        dashboard = SmartGridDashboard()
        dashboard.load_data()
        dashboard.create_layout()
        print("   ✅ Dashboard layout created successfully")
        print("   ✅ Ready for launch")
        return True
        
    except Exception as e:
        print(f"   ❌ Dashboard startup failed: {e}")
        return False

def run_all_tests():
    """Run all dashboard tests"""
    print("🔌 Smart Grid Dashboard Test Suite")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Chart Generation", test_chart_generation),
        ("JSON Structure", test_json_file_structure),
        ("Dashboard Startup", test_dashboard_startup)
    ]
    
    results = {}
    
    for test_name, test_function in tests:
        try:
            results[test_name] = test_function()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 Test Results Summary")
    print("=" * 30)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard is ready to use.")
        print("\n🚀 To start the dashboard, run:")
        print("   python run_dashboard.py")
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        print("\n💡 Common solutions:")
        print("   • Install missing packages: pip install -r dashboard_requirements.txt")
        print("   • Ensure JSON files are in renewable_energy_integration_studies/renewable_stress_results/")
        print("   • Check JSON file structure matches expected format")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 