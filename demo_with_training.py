#!/usr/bin/env python3
"""
Smart Grid Multi-Agent System with Pre-Training Demo

Complete workflow demonstration:
1. Generate historical training data
2. Pre-train RL agents 
3. Run simulation with trained agents
4. Compare performance vs untrained agents
"""

import asyncio
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.coordination.multi_agent_system import SmartGridSimulation
from src.agents.training_data import TrainingDataManager, create_training_data
from src.agents.pre_training import pretrain_all_agents, AgentPreTrainer

async def full_workflow_demo():
    """Complete workflow demonstration"""
    print("="*80)
    print("🌟 SMART GRID MULTI-AGENT SYSTEM - COMPLETE TRAINING WORKFLOW")
    print("="*80)
    
    # Step 1: Generate Training Data
    print("\n" + "="*60)
    print("📊 STEP 1: GENERATING TRAINING DATA")
    print("="*60)
    
    data_manager = TrainingDataManager()
    
    # Check if training data already exists
    training_data_exists = all([
        (Path("training_data") / f).exists() 
        for f in ["generator_0_training.pkl", "storage_0_training.pkl", "consumer_0_training.pkl"]
    ])
    
    if not training_data_exists:
        print("Creating 1 year of historical training data...")
        start_time = time.time()
        datasets = create_training_data()
        end_time = time.time()
        print(f"✅ Training data created in {end_time - start_time:.2f} seconds")
        
        # Print dataset statistics
        for name, data in datasets.items():
            stats = data_manager.get_statistics(data)
            print(f"\n{name}: {stats['total_samples']} samples")
            print(f"  Reward range: {stats['reward_stats']['min']:.3f} to {stats['reward_stats']['max']:.3f}")
    else:
        print("✅ Training data already exists")
    
    # Step 2: Create and Pre-train Agents
    print("\n" + "="*60)
    print("🧠 STEP 2: PRE-TRAINING AGENTS")
    print("="*60)
    
    # Create simulation
    print("Creating smart grid simulation...")
    simulation = SmartGridSimulation()
    simulation.create_sample_scenario()
    
    print(f"Created simulation with {len(simulation.agents)} agents")
    
    # Check if pre-trained models exist
    pretrained_models_exist = any(
        (Path("pretrained_models") / f).exists() 
        for f in ["coal_plant_1_generator.pth", "battery_1_storage.pth", "industrial_consumer_1_consumer.pth"]
    )
    
    if not pretrained_models_exist:
        print("\nPre-training agents with historical data...")
        start_time = time.time()
        training_results = pretrain_all_agents(simulation, epochs=30)  # Reduced for demo
        end_time = time.time()
        print(f"✅ Pre-training completed in {end_time - start_time:.2f} seconds")
        
        # Show training results summary
        print("\n📈 Training Results Summary:")
        for agent_id, metrics in training_results.items():
            if "losses" in metrics:
                final_loss = metrics["losses"][-1]
                print(f"  {agent_id}: Final Loss = {final_loss:.4f}")
            if "rewards" in metrics:
                final_reward = metrics["rewards"][-1]
                print(f"  {agent_id}: Final Reward = {final_reward:.4f}")
    else:
        print("✅ Pre-trained models already exist, loading them...")
        pretrainer = AgentPreTrainer()
        pretrainer.load_pretrained_models(simulation.agents)
    
    # Step 3: Run Simulation with Trained Agents
    print("\n" + "="*60)
    print("🚀 STEP 3: RUNNING SIMULATION WITH TRAINED AGENTS")
    print("="*60)
    
    print("Running 4-hour simulation with pre-trained agents...")
    start_time = time.time()
    await simulation.run_simulation(duration_hours=4)
    end_time = time.time()
    
    print(f"✅ Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Get results
    trained_summary = simulation.get_simulation_summary()
    
    # Step 4: Compare with Untrained Agents
    print("\n" + "="*60)
    print("📊 STEP 4: COMPARING WITH UNTRAINED AGENTS")
    print("="*60)
    
    print("Creating simulation with untrained agents...")
    untrained_simulation = SmartGridSimulation()
    untrained_simulation.create_sample_scenario()
    
    print("Running simulation with untrained agents...")
    start_time = time.time()
    await untrained_simulation.run_simulation(duration_hours=4)
    end_time = time.time()
    
    untrained_summary = untrained_simulation.get_simulation_summary()
    
    # Step 5: Performance Comparison
    print("\n" + "="*60)
    print("📈 STEP 5: PERFORMANCE COMPARISON")
    print("="*60)
    
    print("\n🎯 SIMULATION PERFORMANCE COMPARISON:")
    print("-" * 50)
    
    # Grid stability comparison
    if 'grid_status' in trained_summary and 'grid_status' in untrained_summary:
        trained_grid = trained_summary['grid_status']['grid_state']
        untrained_grid = untrained_summary['grid_status']['grid_state']
        
        print("GRID STABILITY:")
        print(f"  Trained   Frequency: {trained_grid['frequency_hz']:.4f} Hz")
        print(f"  Untrained Frequency: {untrained_grid['frequency_hz']:.4f} Hz")
        print(f"  Trained   Voltage:   {trained_grid['voltage_pu']:.4f} pu")
        print(f"  Untrained Voltage:   {untrained_grid['voltage_pu']:.4f} pu")
        
        # Calculate renewable penetration
        trained_renewable_pct = (trained_grid['renewable_generation_mw'] / 
                                max(trained_grid['total_generation_mw'], 1)) * 100
        untrained_renewable_pct = (untrained_grid['renewable_generation_mw'] / 
                                  max(untrained_grid['total_generation_mw'], 1)) * 100
        
        print(f"\nRENEWABLE INTEGRATION:")
        print(f"  Trained   Penetration: {trained_renewable_pct:.1f}%")
        print(f"  Untrained Penetration: {untrained_renewable_pct:.1f}%")
        
        print(f"\nCARBON INTENSITY:")
        print(f"  Trained   Carbon: {trained_grid['carbon_intensity']:.1f} kg CO2/MWh")
        print(f"  Untrained Carbon: {untrained_grid['carbon_intensity']:.1f} kg CO2/MWh")
    
    # Economic comparison
    print(f"\nECONOMIC METRICS:")
    trained_cost = trained_summary['performance_metrics'].get('total_cost', 0)
    untrained_cost = untrained_summary['performance_metrics'].get('total_cost', 0)
    print(f"  Trained   Total Cost: ${trained_cost:,.2f}")
    print(f"  Untrained Total Cost: ${untrained_cost:,.2f}")
    
    if untrained_cost > 0:
        cost_savings = ((untrained_cost - trained_cost) / untrained_cost) * 100
        print(f"  Cost Savings: {cost_savings:.1f}%")
    
    # Message efficiency
    trained_messages = trained_summary['simulation_info']['messages_sent']
    untrained_messages = untrained_summary['simulation_info']['messages_sent']
    print(f"\nCOMMUNICATION EFFICIENCY:")
    print(f"  Trained   Messages: {trained_messages}")
    print(f"  Untrained Messages: {untrained_messages}")
    
    # Step 6: Export Results
    print("\n" + "="*60)
    print("💾 STEP 6: EXPORTING RESULTS")
    print("="*60)
    
    # Export both results
    simulation.export_results("trained_agents_results.json")
    untrained_simulation.export_results("untrained_agents_results.json")
    
    print("✅ Results exported:")
    print("  - trained_agents_results.json")
    print("  - untrained_agents_results.json")
    
    # Step 7: Summary and Insights
    print("\n" + "="*60)
    print("💡 STEP 7: KEY INSIGHTS")
    print("="*60)
    
    print("\n🎉 TRAINING WORKFLOW COMPLETED!")
    print("\nKey Benefits of Pre-trained Agents:")
    
    # Calculate improvements
    improvements = []
    
    if 'grid_status' in trained_summary and 'grid_status' in untrained_summary:
        trained_freq_stability = abs(trained_grid['frequency_hz'] - 50.0)
        untrained_freq_stability = abs(untrained_grid['frequency_hz'] - 50.0)
        
        if untrained_freq_stability > trained_freq_stability:
            improvements.append(f"✅ Better frequency stability (±{trained_freq_stability:.4f} Hz vs ±{untrained_freq_stability:.4f} Hz)")
        
        if trained_renewable_pct > untrained_renewable_pct:
            improvements.append(f"✅ Higher renewable integration ({trained_renewable_pct:.1f}% vs {untrained_renewable_pct:.1f}%)")
        
        if trained_grid['carbon_intensity'] < untrained_grid['carbon_intensity']:
            carbon_reduction = ((untrained_grid['carbon_intensity'] - trained_grid['carbon_intensity']) / 
                              untrained_grid['carbon_intensity']) * 100
            improvements.append(f"✅ Lower carbon intensity ({carbon_reduction:.1f}% reduction)")
    
    if trained_cost < untrained_cost and untrained_cost > 0:
        improvements.append(f"✅ Reduced operational costs ({cost_savings:.1f}% savings)")
    
    if trained_messages < untrained_messages:
        msg_efficiency = ((untrained_messages - trained_messages) / untrained_messages) * 100
        improvements.append(f"✅ More efficient communication ({msg_efficiency:.1f}% fewer messages)")
    
    if improvements:
        for improvement in improvements:
            print(improvement)
    else:
        print("⚠️  Training benefits may require longer simulation periods to observe")
    
    print(f"\n📊 Training Data Generated: {8760} hours (1 year)")  # 365 * 24
    print(f"🧠 Agents Pre-trained: {len([a for a in simulation.agents.values() if hasattr(a, 'q_network') or hasattr(a, 'actor')])}")
    print(f"⚡ Models Saved: pretrained_models/ directory")
    print(f"📈 Training Metrics: Available for analysis")
    
    print("\n" + "="*80)
    print("🌍 SMART GRID AI SYSTEM READY FOR DEPLOYMENT!")
    print("="*80)

async def quick_demo():
    """Quick demo for testing"""
    print("="*60)
    print("⚡ QUICK SMART GRID DEMO")
    print("="*60)
    
    # Create simulation
    simulation = SmartGridSimulation()
    simulation.create_sample_scenario()
    
    # Check if we have pre-trained models
    pretrained_models_exist = (Path("pretrained_models") / "coal_plant_1_generator.pth").exists()
    
    if pretrained_models_exist:
        print("Loading pre-trained models...")
        pretrainer = AgentPreTrainer()
        pretrainer.load_pretrained_models(simulation.agents)
        print("✅ Pre-trained models loaded")
    else:
        print("⚠️  No pre-trained models found. Agents will learn during simulation.")
    
    # Run short simulation
    print("Running 1-hour simulation...")
    await simulation.run_simulation(duration_hours=1)
    
    # Show results
    summary = simulation.get_simulation_summary()
    print(f"\n📊 Results Summary:")
    print(f"  Messages Sent: {summary['simulation_info']['messages_sent']}")
    print(f"  Total Steps: {summary['simulation_info']['total_steps']}")
    
    if 'grid_status' in summary:
        grid_state = summary['grid_status']['grid_state']
        print(f"  Final Frequency: {grid_state['frequency_hz']:.3f} Hz")
        print(f"  Renewable Generation: {grid_state['renewable_generation_mw']:.1f} MW")
    
    print("\n✅ Quick demo completed!")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        asyncio.run(quick_demo())
    else:
        asyncio.run(full_workflow_demo())

if __name__ == "__main__":
    main() 