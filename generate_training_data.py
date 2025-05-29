#!/usr/bin/env python3
"""
Generate Training Data for Smart Grid Agents

Standalone script to create historical training data for RL agents.
Run this before training agents if you want to generate data separately.
"""

import time
from pathlib import Path
from src.agents.training_data import create_training_data, TrainingDataManager

def main():
    print("="*60)
    print("📊 SMART GRID TRAINING DATA GENERATOR")
    print("="*60)
    
    # Check if training data already exists
    data_dir = Path("training_data")
    existing_files = [
        "generator_0_training.pkl",
        "generator_1_training.pkl", 
        "generator_2_training.pkl",
        "storage_0_training.pkl",
        "storage_1_training.pkl",
        "consumer_0_training.pkl",
        "consumer_1_training.pkl",
        "consumer_2_training.pkl"
    ]
    
    existing_count = sum(1 for f in existing_files if (data_dir / f).exists())
    
    if existing_count > 0:
        print(f"⚠️  Found {existing_count} existing training data files")
        response = input("Do you want to regenerate training data? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Cancelled. Using existing training data.")
            return
    
    print("🚀 Generating comprehensive training dataset...")
    print("   - 365 days of market conditions")
    print("   - 8,760 hourly scenarios")
    print("   - 8 agent configurations")
    print("   - Realistic price, weather, and grid patterns")
    
    start_time = time.time()
    
    try:
        # Generate the training datasets
        datasets = create_training_data()
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"\n✅ Training data generation completed in {generation_time:.2f} seconds")
        
        # Display statistics
        data_manager = TrainingDataManager()
        
        print("\n📈 Dataset Statistics:")
        print("-" * 40)
        
        total_samples = 0
        for name, data in datasets.items():
            stats = data_manager.get_statistics(data)
            samples = stats['total_samples']
            total_samples += samples
            
            agent_type = stats['agent_type']
            reward_mean = stats['reward_stats']['mean']
            reward_std = stats['reward_stats']['std']
            
            print(f"{name}:")
            print(f"  Agent Type: {agent_type}")
            print(f"  Samples: {samples:,}")
            print(f"  Reward: {reward_mean:.3f} ± {reward_std:.3f}")
            print()
        
        print(f"📊 Total Training Samples: {total_samples:,}")
        print(f"💾 Data Size: ~{total_samples * 64 * 4 / 1024 / 1024:.1f} MB")
        print(f"📁 Location: training_data/ directory")
        
        print(f"\n🎯 Next Steps:")
        print("1. Run pre-training: python -c \"from src.agents.pre_training import *; pretrain_all_agents(sim)\"")
        print("2. Or run complete demo: python demo_with_training.py")
        
    except Exception as e:
        print(f"\n❌ Error generating training data: {e}")
        raise

if __name__ == "__main__":
    main() 