"""
Curriculum Learning Package for Smart Grid Renewable Integration

This package contains all curriculum-based MARL training approaches for
improving renewable energy integration in smart grid systems.

## Components:

### Training Frameworks:
- curriculum_training.py: Production-ready curriculum learning framework  
- direct_curriculum_run.py: Simple demo script for quick testing
- run_curriculum_training.py: Alternative curriculum runner

### Testing & Debugging:
- simple_curriculum_test.py: Basic curriculum testing
- test_curriculum_debug.py: Debugging utilities

### Documentation:
- README.md: Comprehensive curriculum learning guide
- curriculum_rl_paper_ai_econ.txt: Reference paper ("The AI Economist")

### Results:
- curriculum_training_results_*.json: Training experiment results

## Quick Start:

```python
# Run simple curriculum training demo
python direct_curriculum_run.py

# Run production curriculum training  
python curriculum_training.py

# Import for custom experiments
from curriculum_learning.curriculum_training import RenewableCurriculumTrainer, CurriculumConfig
```

## Expected Results:
- 0% → 50-70% renewable penetration
- $367/MWh → $40-80/MWh efficient pricing  
- Critical frequency violations → ±0.05 Hz stability
- Counterproductive storage → Grid-stabilizing arbitrage
"""

import sys
import os

# Add parent directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import key components
try:
    from .curriculum_training import RenewableCurriculumTrainer, CurriculumConfig
    __all__ = ['RenewableCurriculumTrainer', 'CurriculumConfig']
except ImportError:
    # Handle import errors gracefully
    __all__ = []

__version__ = "1.0.0"
__author__ = "Smart Grid MARL Team" 