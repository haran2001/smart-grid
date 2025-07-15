#!/usr/bin/env python3
"""
Run Curriculum-Based Training for Renewable Energy Integration

This script demonstrates how to apply curriculum learning to improve
renewable energy integration performance in the smart grid system.
"""

import asyncio
import logging
import sys
import os

# Add the renewable studies directory to the path
sys.path.insert(0, 'renewable_energy_integration_studies')

from renewable_energy_integration_studies.curriculum_integration import run_curriculum_enhanced_stress_tests


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('curriculum_training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


async def main():
    """Main execution function"""
    
    print("🚀 Starting Curriculum-Based Renewable Integration Training")
    print("=" * 60)
    
    setup_logging()
    logger = logging.getLogger("CurriculumMain")
    
    try:
        # Run curriculum-enhanced training and testing
        logger.info("Launching curriculum training...")
        results = await run_curriculum_enhanced_stress_tests()
        
        print("\n" + "=" * 60)
        print("📊 TRAINING COMPLETED - RESULTS SUMMARY")
        print("=" * 60)
        
        # Display curriculum training results
        training_results = results.get("curriculum_training", {})
        final_performance = training_results.get("final_performance", {})
        
        print("\n🎯 Final Performance on Challenge Scenarios:")
        print("-" * 40)
        
        for scenario, metrics in final_performance.items():
            status = "✅ PASS" if metrics.get("success", False) else "❌ FAIL"
            score = metrics.get("overall_score", 0.0)
            renewable_score = metrics.get("renewable_score", 0.0)
            stability_score = metrics.get("stability_score", 0.0)
            violations = metrics.get("violations", 0)
            
            print(f"{scenario:25} {status}")
            print(f"  Overall Score:     {score:.2f}/1.00")
            print(f"  Renewable Usage:   {renewable_score:.2f}/1.00")
            print(f"  Grid Stability:    {stability_score:.2f}/1.00")
            print(f"  Violations:        {violations}")
            print()
        
        # Display enhanced stress test results
        enhanced_results = results.get("enhanced_stress_tests", {})
        
        if enhanced_results:
            print("🔬 Enhanced Stress Test Results:")
            print("-" * 40)
            
            for test_name, test_result in enhanced_results.items():
                performance = test_result.get("performance_analysis", {})
                violations = test_result.get("violations", [])
                
                print(f"Test: {test_name}")
                print(f"  Duration: {performance.get('test_duration_minutes', 'N/A')} minutes")
                print(f"  Violations: {len(violations)}")
                print(f"  Status: {'✅ PASS' if len(violations) < 3 else '❌ FAIL'}")
                print()
        
        # Summary
        successful_scenarios = sum(1 for metrics in final_performance.values() if metrics.get("success", False))
        total_scenarios = len(final_performance)
        success_rate = (successful_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0
        
        print(f"📈 Overall Success Rate: {success_rate:.1f}% ({successful_scenarios}/{total_scenarios})")
        
        if success_rate >= 75:
            print("🎉 EXCELLENT: Curriculum training significantly improved renewable integration!")
        elif success_rate >= 50:
            print("👍 GOOD: Curriculum training showed improvement - consider further tuning")
        else:
            print("⚠️  NEEDS WORK: Additional curriculum refinement needed")
        
        print("\n💾 Detailed results saved to curriculum_training_results_*.json")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        print(f"\n❌ Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    print("🌱 Smart Grid Curriculum Learning for Renewable Integration")
    print("Based on 'The AI Economist' curriculum methodology")
    print()
    
    # Run the training
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 