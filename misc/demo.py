#!/usr/bin/env python3
"""
Smart Grid Multi-Agent Energy Management System Demo

This demo showcases the AI-powered smart grid system with:
- DQN-based Generator Agents
- Actor-Critic Storage Agents  
- MADDPG Consumer Agents
- Grid Operator coordination
- Real-time market clearing
"""

import asyncio
import logging
from src.coordination.multi_agent_system import SmartGridSimulation, create_renewable_heavy_scenario

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def run_basic_demo():
    """Run a basic demonstration of the smart grid system"""
    print("=" * 60)
    print("🌟 Smart Grid Multi-Agent Energy Management System Demo")
    print("=" * 60)
    
    # Create and initialize the simulation
    print("\n📋 Creating simulation with sample scenario...")
    simulation = SmartGridSimulation()
    await simulation.create_sample_scenario()
    
    print(f"✅ Simulation initialized with {len(simulation.agents)} agents:")
    for agent_id, agent in simulation.agents.items():
        print(f"   - {agent_id} ({agent.agent_type.value})")
    
    # Run a short simulation
    print("\n🚀 Running 2-hour simulation (24 time steps)...")
    await simulation.run_simulation(duration_hours=2)
    
    # Display results
    print("\n📊 Simulation Results Summary:")
    summary = simulation.get_simulation_summary()
    
    print(f"   Total Steps: {summary['simulation_info']['total_steps']}")
    print(f"   Messages Sent: {summary['simulation_info']['messages_sent']}")
    print(f"   Duration: {summary['simulation_info']['duration_hours']:.1f} hours")
    
    if 'grid_status' in summary:
        grid_state = summary['grid_status']['grid_state']
        print(f"   Final Frequency: {grid_state['frequency_hz']:.3f} Hz")
        print(f"   Final Voltage: {grid_state['voltage_pu']:.3f} pu")
        print(f"   Total Generation: {grid_state['total_generation_mw']:.1f} MW")
        print(f"   Total Load: {grid_state['total_load_mw']:.1f} MW")
        print(f"   Renewable Generation: {grid_state['renewable_generation_mw']:.1f} MW")
        renewable_pct = (grid_state['renewable_generation_mw'] / 
                        max(grid_state['total_generation_mw'], 1)) * 100
        print(f"   Renewable Penetration: {renewable_pct:.1f}%")
    
    # Export results
    print("\n💾 Exporting results...")
    simulation.export_results("demo_results.json")
    print("   Results saved to demo_results.json")
    
    print("\n✨ Demo completed successfully!")

async def run_renewable_scenario_demo():
    """Run a demo with high renewable penetration"""
    print("\n" + "=" * 60)
    print("🌱 Renewable Heavy Scenario Demo")
    print("=" * 60)
    
    # Create renewable-heavy scenario
    print("\n📋 Creating renewable-heavy scenario...")
    simulation = await create_renewable_heavy_scenario()
    
    print(f"✅ Renewable scenario initialized with {len(simulation.agents)} agents:")
    for agent_id, agent in simulation.agents.items():
        agent_type = agent.agent_type.value
        if hasattr(agent, 'generator_state'):
            emissions = getattr(agent.generator_state, 'emissions_rate_kg_co2_per_mwh', 'N/A')
            print(f"   - {agent_id} ({agent_type}) - Emissions: {emissions} kg CO2/MWh")
        else:
            print(f"   - {agent_id} ({agent_type})")
    
    # Run simulation
    print("\n🚀 Running 4-hour renewable scenario...")
    await simulation.run_simulation(duration_hours=4)
    
    # Display environmental results
    print("\n🌍 Environmental Impact Summary:")
    summary = simulation.get_simulation_summary()
    
    if 'grid_status' in summary:
        grid_state = summary['grid_status']['grid_state']
        renewable_pct = (grid_state['renewable_generation_mw'] / 
                        max(grid_state['total_generation_mw'], 1)) * 100
        
        print(f"   Renewable Penetration: {renewable_pct:.1f}%")
        print(f"   Carbon Intensity: {grid_state['carbon_intensity']:.1f} kg CO2/MWh")
        print(f"   Total Clean Generation: {grid_state['renewable_generation_mw']:.1f} MW")
    
    # Export renewable scenario results
    simulation.export_results("renewable_demo_results.json")
    print("   Renewable scenario results saved to renewable_demo_results.json")

async def demonstrate_agent_interactions():
    """Demonstrate specific agent interactions and decision-making"""
    print("\n" + "=" * 60)
    print("🤖 Agent Decision-Making Demonstration")
    print("=" * 60)
    
    simulation = SmartGridSimulation()
    await simulation.create_sample_scenario()
    
    # Get specific agents for demonstration
    generator_agent = None
    storage_agent = None
    consumer_agent = None
    
    for agent_id, agent in simulation.agents.items():
        if agent.agent_type.value == 'generator' and generator_agent is None:
            generator_agent = agent
        elif agent.agent_type.value == 'storage' and storage_agent is None:
            storage_agent = agent
        elif agent.agent_type.value == 'consumer' and consumer_agent is None:
            consumer_agent = agent
    
    print("\n🔍 Running one decision cycle to show agent reasoning...")
    
    # Update market data for all agents
    for agent in simulation.agents.values():
        # Handle different state structures
        if hasattr(agent.state, 'market_data'):
            agent.state.market_data.update(simulation.market_data)
        elif isinstance(agent.state, dict):
            if 'market_data' not in agent.state:
                agent.state['market_data'] = {}
            agent.state['market_data'].update(simulation.market_data)
    
    # Show generator decision-making
    if generator_agent:
        print(f"\n⚡ Generator Agent ({generator_agent.agent_id}) Decision:")
        decision = await generator_agent.make_strategic_decision(generator_agent.state)
        print(f"   Action: {decision['reasoning']}")
        print(f"   Bid Price: ${decision['bid_price_mwh']:.2f}/MWh")
        print(f"   Bid Quantity: {decision['bid_quantity_mw']:.1f} MW")
    
    # Show storage decision-making
    if storage_agent:
        print(f"\n🔋 Storage Agent ({storage_agent.agent_id}) Decision:")
        decision = await storage_agent.make_strategic_decision(storage_agent.state)
        print(f"   Action: {decision['reasoning']}")
        print(f"   Action Type: {decision['action_type']}")
        print(f"   Power: {decision['power_mw']:.1f} MW")
        print(f"   Current SoC: {decision['current_soc']:.1f}%")
    
    # Show consumer decision-making
    if consumer_agent:
        print(f"\n🏭 Consumer Agent ({consumer_agent.agent_id}) Decision:")
        decision = await consumer_agent.make_strategic_decision(consumer_agent.state)
        print(f"   Action: {decision['reasoning']}")
        print(f"   DR Participation: {decision['dr_participation']:.2f}")
        print(f"   Predicted Load: {decision['predicted_load_mw']:.1f} MW")
    
    print("\n✨ Agent interactions demonstrated!")

def run_dashboard_info():
    """Display information about running the dashboard"""
    print("\n" + "=" * 60)
    print("📊 Dashboard Information")
    print("=" * 60)
    print("\nTo run the interactive dashboard:")
    print("1. Install Streamlit: pip install streamlit")
    print("2. Run the dashboard: streamlit run src/visualization/dashboard.py")
    print("3. Open your browser to the displayed URL (usually http://localhost:8501)")
    print("\nThe dashboard provides:")
    print("   - Real-time grid monitoring")
    print("   - Agent performance visualization")
    print("   - Market information displays")
    print("   - Environmental impact tracking")
    print("   - Interactive simulation controls")

async def main():
    """Main demo function"""
    try:
        # Run basic demo
        await run_basic_demo()
        
        # Run renewable scenario
        await run_renewable_scenario_demo()
        
        # Demonstrate agent interactions
        await demonstrate_agent_interactions()
        
        # Show dashboard info
        run_dashboard_info()
        
        print("\n" + "=" * 60)
        print("🎉 All demos completed successfully!")
        print("🔧 Check the generated JSON files for detailed results")
        print("📈 Run the Streamlit dashboard for interactive visualization")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        logging.exception("Demo failed")

if __name__ == "__main__":
    asyncio.run(main()) 