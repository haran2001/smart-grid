#!/usr/bin/env python3
"""
Quick Curriculum Test Runner

Standalone script to run a fast curriculum validation test (< 10 minutes)
to check if the curriculum approach shows marginal improvement before
committing to full training.

Usage:
    python test_quick_curriculum.py
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

async def main():
    """Run quick curriculum validation test"""
    
    print("🧪 Quick Curriculum Validation Test")
    print("=" * 45)
    print("⏱️ Estimated time: < 10 minutes")
    print("🎯 Goal: Verify curriculum approach works")
    print("📊 Measures: Before/after performance comparison")
    print("")
    
    try:
        # Import the quick test function
        from curriculum_training import run_quick_curriculum_test
        
        print("🚀 Starting quick curriculum test...")
        start_time = datetime.now()
        
        # Run the test
        result = await run_quick_curriculum_test()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n⏱️ Total test duration: {duration}")
        
        if result:
            assessment = result.get('assessment', 'unknown')
            improvements = result.get('improvements', {})
            
            print("\n" + "=" * 45)
            print("🎯 FINAL ASSESSMENT")
            print("=" * 45)
            
            if assessment == 'positive':
                print("✅ CURRICULUM APPROACH WORKS!")
                print("📈 Clear performance improvement detected")
                print("💡 Recommendation: Proceed with full training")
                
                print(f"\n📊 Key improvements:")
                for metric, improvement in improvements.items():
                    if improvement > 0:
                        print(f"   • {metric.replace('_', ' ').title()}: +{improvement:.1f}%")
                
            elif assessment == 'marginal':
                print("🟡 MARGINAL IMPROVEMENT DETECTED")
                print("📊 Small but positive changes observed")
                print("💡 Recommendation: Consider parameter tuning")
                
            else:
                print("❌ NO IMPROVEMENT DETECTED")
                print("🔍 Curriculum approach may need adjustment")
                print("💡 Recommendation: Debug or try different approach")
            
            # Show next steps
            print(f"\n🔄 Next steps:")
            if assessment == 'positive':
                print("   1. Run full curriculum training:")
                print("      python run_curriculum.py --mode full")
                print("   2. Monitor training progress")
                print("   3. Test with renewable stress scenarios")
            else:
                print("   1. Review configuration parameters")
                print("   2. Check agent neural network architectures")
                print("   3. Debug training loop if needed")
            
        else:
            print("\n❌ Quick test failed!")
            print("🔍 Check error messages above for debugging")
            return 1
            
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("🔧 Make sure you're in the curriculum_learning directory")
        print("📁 Current directory should contain curriculum_training.py")
        return 1
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n✅ Quick test completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 