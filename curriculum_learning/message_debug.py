#!/usr/bin/env python3
"""
Message Debug Test - Trace GENERATION_BID message flow
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.coordination.multi_agent_system import SmartGridSimulation
from src.agents.grid_operator_agent import GridOperatorAgent
from src.agents.generator_agent import GeneratorAgent

async def debug_message_flow():
    """Debug the complete message flow"""
    
    print("🔍 MESSAGE FLOW DEBUG TEST")
    print("=" * 50)
    
    # Create simulation
    simulation = SmartGridSimulation()
    await simulation.create_sample_scenario()
    
    # Get agents
    grid_operator = None
    coal_plant = None
    
    for agent_id, agent in simulation.agents.items():
        if isinstance(agent, GridOperatorAgent):
            grid_operator = agent
        elif agent_id == "coal_plant_1":
            coal_plant = agent
    
    if not grid_operator or not coal_plant:
        print("❌ Could not find required agents")
        return
    
    print(f"✅ Found grid operator: {grid_operator.agent_id}")
    print(f"✅ Found coal plant: {coal_plant.agent_id}")
    
    # Step 1: Check initial state
    print(f"\n📊 INITIAL STATE:")
    print(f"Grid operator generation bids: {len(grid_operator.generation_bids)}")
    
    # Step 2: Manually trigger generator decision
    print(f"\n🧠 MANUAL GENERATOR DECISION:")
    # Ensure state is properly formatted
    if isinstance(coal_plant.state, dict):
        from src.agents.base_agent import AgentState, AgentType
        state = AgentState(
            agent_id=coal_plant.agent_id,
            agent_type=AgentType.GENERATOR,
            market_data=coal_plant.state.get('market_data', {}),
            operational_status=coal_plant.state.get('operational_status', {}),
            messages=coal_plant.state.get('messages', []),
            decisions=coal_plant.state.get('decisions', []),
            performance_metrics=coal_plant.state.get('performance_metrics', {})
        )
    else:
        state = coal_plant.state
    
    decision = await coal_plant.make_strategic_decision(state)
    print(f"Decision: {decision}")
    
    # Step 3: Execute decision (this should send GENERATION_BID)
    print(f"\n📤 EXECUTING DECISION (should send GENERATION_BID):")
    await coal_plant.execute_decision(decision)
    
    # Step 4: Check if message was sent
    print(f"\n📨 MESSAGE HISTORY:")
    print(f"Total messages in history: {len(simulation.message_router.message_history)}")
    for i, msg in enumerate(simulation.message_router.message_history[-3:]):
        print(f"  {i}: {msg.sender_id} → {msg.receiver_id} ({msg.message_type.value})")
    
    # Step 5: Manually process grid operator messages
    print(f"\n⚙️ MANUAL MESSAGE PROCESSING:")
    
    # Check grid operator's message queue
    grid_operator_queue_size = 0
    try:
        while not grid_operator.message_queue.empty():
            grid_operator_queue_size += 1
            message = await asyncio.wait_for(grid_operator.message_queue.get(), timeout=0.1)
            print(f"  Processing message: {message.sender_id} → {message.message_type.value}")
            await grid_operator._handle_message(message)
    except:
        pass
    
    print(f"Processed {grid_operator_queue_size} messages from grid operator queue")
    
    # Step 6: Check final state
    print(f"\n📊 FINAL STATE:")
    print(f"Grid operator generation bids: {len(grid_operator.generation_bids)}")
    
    for i, bid in enumerate(grid_operator.generation_bids):
        print(f"  Bid {i}: {bid.agent_id} - {bid.quantity_mw} MW @ ${bid.price_per_mwh}/MWh")
    
    # Step 7: Test full simulation step
    print(f"\n🏃 FULL SIMULATION STEP:")
    initial_bids = len(grid_operator.generation_bids)
    
    await simulation.run_simulation_step()
    
    final_bids = len(grid_operator.generation_bids)
    print(f"Bids before simulation step: {initial_bids}")
    print(f"Bids after simulation step: {final_bids}")
    print(f"Bids added during step: {final_bids - initial_bids}")
    
    return final_bids > initial_bids

if __name__ == "__main__":
    asyncio.run(debug_message_flow()) 