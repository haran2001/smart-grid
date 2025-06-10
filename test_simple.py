#!/usr/bin/env python3
"""
Simple test to check market clearing with proper demand
"""

import asyncio
import logging
from src.coordination.multi_agent_system import SmartGridSimulation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def simple_test():
    """Simple test focusing on market clearing"""
    print("🔍 Simple Market Test")
    
    # Create simulation
    simulation = SmartGridSimulation()
    await simulation.add_grid_operator()
    
    # Add one generator
    generator = simulation.add_generator_agent("gen1", {
        "max_capacity_mw": 50.0,
        "fuel_cost_per_mwh": 60.0
    })
    
    # Add one consumer  
    consumer = simulation.add_consumer_agent("load1", {
        "baseline_load_mw": 30.0,
        "flexible_load_mw": 5.0
    })
    
    # Manually set some demand on the grid state to test
    simulation.grid_operator.grid_state.total_load_mw = 30.0
    
    print(f"✅ Created simulation with {len(simulation.agents)} agents")
    print(f"✅ Grid demand set to: {simulation.grid_operator.grid_state.total_load_mw} MW")
    
    # Run one step to request bids
    print("\n🚀 Step 1: Request bids")
    await simulation.run_simulation_step()
    
    print(f"📊 Generation bids: {len(simulation.grid_operator.generation_bids)}")
    
    # Run second step to clear market
    print("\n🚀 Step 2: Clear market")
    await simulation.run_simulation_step()
    
    print(f"📊 Market clearings: {len(simulation.grid_operator.market_efficiency_history)}")
    
    # Show results
    if simulation.grid_operator.market_efficiency_history:
        clearing = simulation.grid_operator.market_efficiency_history[-1]
        print(f"✅ Market cleared: {clearing['cleared_quantity']:.1f} MW @ ${clearing['clearing_price']:.2f}/MWh")
    else:
        print("❌ No market clearing occurred")

if __name__ == "__main__":
    asyncio.run(simple_test()) 