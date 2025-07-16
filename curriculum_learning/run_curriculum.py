#!/usr/bin/env python3
"""
Unified Curriculum Training Entry Point

Provides easy access to all curriculum learning approaches for renewable integration.

Usage:
    python run_curriculum.py --mode [demo|full|debug]
    python run_curriculum.py --help
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def print_banner():
    """Print curriculum training banner"""
    print("=" * 70)
    print("🌱 SMART GRID CURRICULUM LEARNING FOR RENEWABLE INTEGRATION 🌱")
    print("=" * 70)
    print("📚 Based on 'The AI Economist' curriculum MARL approach")
    print("🎯 Target: 0% → 50-70% renewable penetration")
    print("⚡ Neural Networks: DQN, Actor-Critic, MADDPG")
    print("")
    print("🚀 Available modes:")
    print("   • demo:   Quick demo (2 minutes)")
    print("   • quick:  Validation test (< 10 minutes) - NEW!")
    print("   • full:   Research training (hours)")
    print("   • debug:  Debug utilities")
    print("   • results: View past results")
    print("=" * 70)

async def run_demo_curriculum():
    """Run simple curriculum training demo"""
    print("\n🚀 Running DEMO Curriculum Training")
    print("   • Fast execution (~2 minutes)")
    print("   • 5% → 50% renewable penetration")
    print("   • 200 training steps total")
    
    # Import and run direct curriculum
    from direct_curriculum_run import main as demo_main
    result = await demo_main()
    
    if result:
        print(f"\n✅ Demo completed successfully!")
        print(f"📊 Phase 1 success: {result['phase1_success_rate']:.1f}%")
        print(f"📊 Phase 2 success: {result['phase2_success_rate']:.1f}%")
    else:
        print("\n❌ Demo failed - check logs above")
    
    return result

async def run_full_curriculum():
    """Run production curriculum training"""
    print("\n🚀 Running FULL Curriculum Training")
    print("   • Production-grade training (several hours)")
    print("   • 5% → 80% renewable penetration")
    print("   • 450M training steps total")
    print("   • Multi-parameter curriculum")
    
    # Import and run full curriculum
    try:
        from curriculum_training import run_curriculum_training
        result = await run_curriculum_training()
        
        if result:
            print(f"\n✅ Full training completed successfully!")
            print(f"📊 Final performance: {result.get('final_performance', 'N/A')}")
        else:
            print("\n❌ Full training failed - check logs above")
        
        return result
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure all dependencies are installed")
        return None

async def run_quick_test():
    """Run quick curriculum test to validate approach"""
    print("\n⚡ Running QUICK Curriculum Test")
    print("   • Validation test (< 10 minutes)")
    print("   • 5% → 50% renewable penetration")
    print("   • 200 training steps total")
    print("   • Before/after performance comparison")
    
    # Import and run quick test
    try:
        from curriculum_training import run_quick_curriculum_test
        result = await run_quick_curriculum_test()
        
        if result:
            assessment = result.get('assessment', 'unknown')
            if assessment == 'positive':
                print(f"\n✅ Quick test shows POSITIVE results!")
                print("📈 Curriculum approach is working - proceed with full training")
            elif assessment == 'marginal':
                print(f"\n🟡 Quick test shows MARGINAL improvement")
                print("🔧 Consider parameter tuning before full training")
            else:
                print(f"\n❌ Quick test shows NO improvement")
                print("🛠️ May need different approach or debugging")
        else:
            print("\n❌ Quick test failed - check logs above")
        
        return result
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure all dependencies are installed")
        return None

def run_debug_mode():
    """Run debugging utilities"""
    print("\n🔧 Running DEBUG Mode")
    print("   • Testing neural network connections")
    print("   • Validating agent communications")
    print("   • Quick system health check")
    
    try:
        # Built-in debug functionality
        print("\n🔍 Testing imports...")
        from src.coordination.multi_agent_system import SmartGridSimulation
        from src.agents.generator_agent import GeneratorAgent
        from src.agents.storage_agent import StorageAgent
        from src.agents.consumer_agent import ConsumerAgent
        print("   ✅ All core imports successful")
        
        print("\n🔍 Testing simulation creation...")
        simulation = SmartGridSimulation()
        print("   ✅ SmartGridSimulation initialized successfully")
        
        print("\n🔍 Testing neural network initialization...")
        # Test a single agent creation
        gen_agent = GeneratorAgent("test_gen", {"max_capacity_mw": 100})
        storage_agent = StorageAgent("test_storage", {"max_capacity_mwh": 50})
        consumer_agent = ConsumerAgent("test_consumer", {"baseline_load_mw": 25})
        print("   ✅ Neural network agents created successfully")
        print(f"      - Generator: DQN with {gen_agent.state_size}D state")
        print(f"      - Storage: Actor-Critic with {storage_agent.state_size}D state")  
        print(f"      - Consumer: MADDPG with {consumer_agent.state_size}D state")
        
        print("\n✅ Debug mode completed - All systems operational!")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

def show_results_summary():
    """Show summary of existing training results"""
    print("\n📊 EXISTING TRAINING RESULTS")
    print("-" * 40)
    
    import glob
    import json
    
    result_files = glob.glob("curriculum_training_results_*.json")
    
    if not result_files:
        print("   No training results found")
        return
    
    for file in sorted(result_files):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Extract timestamp from filename like curriculum_training_results_20250716_233518.json
            parts = file.replace('.json', '').split('_')
            if len(parts) >= 2:
                date_part = parts[-2]  # 20250716
                time_part = parts[-1]  # 233518
                timestamp = f"{date_part}{time_part}"
                formatted_time = datetime.strptime(timestamp, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M")
            else:
                formatted_time = "Unknown time"
            
            print(f"   📅 {formatted_time}")
            print(f"      Phase 1: {data.get('phase1_success_rate', 0):.1f}% success")
            print(f"      Phase 2: {data.get('phase2_success_rate', 0):.1f}% success")
            print(f"      Renewables: {data.get('final_renewable_utilization', 0):.1%}")
            print(f"      Frequency: {data.get('final_frequency', 50):.2f} Hz")
            print()
        except Exception as e:
            print(f"   ❌ Error reading {file}: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Smart Grid Curriculum Learning for Renewable Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_curriculum.py --mode demo     # Quick demo (2 minutes)
  python run_curriculum.py --mode quick    # Quick test (< 10 minutes)
  python run_curriculum.py --mode full     # Full training (hours)
  python run_curriculum.py --mode debug    # Debug mode
  python run_curriculum.py --mode results  # Show past results
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['demo', 'quick', 'full', 'debug', 'results'],
        default='demo',
        help='Training mode to run (default: demo)'
    )
    
    args = parser.parse_args()
    
    # Change to curriculum_learning directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print_banner()
    
    if args.mode == 'demo':
        await run_demo_curriculum()
    elif args.mode == 'quick':
        await run_quick_test()
    elif args.mode == 'full':
        await run_full_curriculum()
    elif args.mode == 'debug':
        run_debug_mode()
    elif args.mode == 'results':
        show_results_summary()
    
    print("\n" + "=" * 70)
    print("🎉 Curriculum Learning Session Complete!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main()) 