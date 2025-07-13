# Smart Grid Dashboard Module

## 📍 Location
This dashboard module is now part of the main smart grid system under `/src/dashboard/`.

## 🚀 Quick Start

### From Project Root
```bash
# Launch dashboard from main project directory
python dashboard.py

# Or with options
python dashboard.py --demo     # Demo mode
python dashboard.py --report   # Static report
python dashboard.py --test     # Run tests
```

### From Dashboard Directory
```bash
# Navigate to dashboard directory
cd src/dashboard

# Install dependencies
pip install -r dashboard_requirements.txt

# Run dashboard
python run_dashboard.py
```

## 📁 Module Structure
```
src/dashboard/
├── __init__.py                 # Package initialization
├── dashboard_generator.py      # Main dashboard engine (515+ lines)
├── run_dashboard.py           # User-friendly launcher
├── test_dashboard.py          # Testing framework
├── dashboard_requirements.txt # Dependencies
├── README_dashboard.md        # Comprehensive documentation
└── README.md                  # This file
```

## 🔗 Data Sources
The dashboard automatically loads data from:
- **Stress Tests**: `renewable_energy_integration_studies/renewable_stress_results/*.json`
- **Blackout Simulations**: `blackout_studies/*.json`

## 📖 Full Documentation
See `README_dashboard.md` for comprehensive documentation including:
- Detailed feature descriptions
- Usage guides and examples
- Customization options
- Troubleshooting help

## 🎯 Integration with Smart Grid System
This dashboard module integrates seamlessly with the main smart grid system:

```python
# Import from the smart grid system
from src.dashboard import SmartGridDashboard

# Create and run dashboard
dashboard = SmartGridDashboard()
dashboard.run_dashboard()
```

## 🧪 Testing
```bash
# Test dashboard functionality
python test_dashboard.py

# Or from project root
python dashboard.py --test
```

---
*Part of the Smart Grid Research Framework* 