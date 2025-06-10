import asyncio
import logging
from typing import Dict, Any, List, Optional, Type
from datetime import datetime, timedelta
import json
import numpy as np
from collections import defaultdict

from ..agents.base_agent import BaseAgent, AgentType, AgentMessage, MessageType
from ..agents.generator_agent import GeneratorAgent
from ..agents.storage_agent import StorageAgent
from ..agents.consumer_agent import ConsumerAgent
from ..agents.grid_operator_agent import GridOperatorAgent


class MessageRouter:
    """Routes messages between agents"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self._message_queue = None  # Lazy-loaded
        self.message_history: List[AgentMessage] = []
        
    @property
    def message_queue(self):
        """Lazy-loaded message queue"""
        if self._message_queue is None:
            try:
                self._message_queue = asyncio.Queue()
            except RuntimeError:
                # No event loop running, create a simple queue alternative
                import queue
                self._message_queue = queue.Queue()
        return self._message_queue
        
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the message router"""
        self.agents[agent.agent_id] = agent
        # Set the message router on the agent so it can send messages
        agent.message_router = self
        
    async def route_message(self, message: AgentMessage) -> None:
        """Route a message to its destination"""
        if message.receiver_id in self.agents:
            await self.agents[message.receiver_id].receive_message(message)
            self.message_history.append(message)
        else:
            logging.warning(f"Agent {message.receiver_id} not found for message from {message.sender_id}")
    
    async def broadcast_message(self, sender_id: str, message_type: MessageType, content: Dict[str, Any]) -> None:
        """Broadcast a message to all agents except sender"""
        for agent_id, agent in self.agents.items():
            if agent_id != sender_id:
                message = AgentMessage(
                    sender_id=sender_id,
                    receiver_id=agent_id,
                    message_type=message_type,
                    content=content
                )
                await self.route_message(message)


class SmartGridSimulation:
    """Main smart grid multi-agent simulation system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.message_router = MessageRouter()
        self.agents: Dict[str, BaseAgent] = {}
        self.grid_operator: Optional[GridOperatorAgent] = None
        
        # Simulation state
        self.simulation_time = datetime.now()
        self.time_step_minutes = 5
        self.is_running = False
        
        # Market data simulation
        self.market_data = {
            "current_price": 50.0,
            "price_forecast": [50.0] * 24,
            "demand_forecast": {"expected_peak": 1000.0},
            "generation_forecast": {"total_available": 1500.0, "renewable_total": 500.0},
            "weather": {"temperature": 20.0, "wind_speed": 10.0, "solar_irradiance": 500.0},
            "carbon_price": 25.0,
            "dr_price": 100.0,
            "frequency_hz": 50.0,
            "voltage_pu": 1.0
        }
        
        # Performance tracking
        self.simulation_metrics = {
            "total_steps": 0,
            "messages_sent": 0,
            "market_clearings": 0,
            "total_cost": 0.0,
            "renewable_penetration": 0.0,
            "grid_stability_score": 100.0
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SmartGridSimulation")
        
    def add_generator_agent(self, agent_id: str, config: Dict[str, Any] = None) -> GeneratorAgent:
        """Add a generator agent to the simulation"""
        agent = GeneratorAgent(agent_id, config)
        self.agents[agent_id] = agent
        self.message_router.register_agent(agent)
        
        if self.grid_operator:
            asyncio.create_task(self.grid_operator.register_agent(agent_id, AgentType.GENERATOR))
            
        self.logger.info(f"Added generator agent: {agent_id}")
        return agent
    
    def add_storage_agent(self, agent_id: str, config: Dict[str, Any] = None) -> StorageAgent:
        """Add a storage agent to the simulation"""
        agent = StorageAgent(agent_id, config)
        self.agents[agent_id] = agent
        self.message_router.register_agent(agent)
        
        if self.grid_operator:
            asyncio.create_task(self.grid_operator.register_agent(agent_id, AgentType.STORAGE))
            
        self.logger.info(f"Added storage agent: {agent_id}")
        return agent
    
    def add_consumer_agent(self, agent_id: str, config: Dict[str, Any] = None) -> ConsumerAgent:
        """Add a consumer agent to the simulation"""
        agent = ConsumerAgent(agent_id, config)
        self.agents[agent_id] = agent
        self.message_router.register_agent(agent)
        
        if self.grid_operator:
            asyncio.create_task(self.grid_operator.register_agent(agent_id, AgentType.CONSUMER))
            
        self.logger.info(f"Added consumer agent: {agent_id}")
        return agent
    
    async def add_grid_operator(self, agent_id: str = "grid_operator", config: Dict[str, Any] = None) -> GridOperatorAgent:
        """Add the grid operator agent to the simulation"""
        self.grid_operator = GridOperatorAgent(agent_id, config)
        self.agents[agent_id] = self.grid_operator
        self.message_router.register_agent(self.grid_operator)
        
        # Register all existing agents with the grid operator
        for existing_agent_id, agent in self.agents.items():
            if existing_agent_id != agent_id:
                await self.grid_operator.register_agent(existing_agent_id, agent.agent_type)
        
        self.logger.info(f"Added grid operator: {agent_id}")
        return self.grid_operator
    
    async def create_sample_scenario(self) -> None:
        """Create a sample scenario with multiple agents"""
        # Add grid operator
        await self.add_grid_operator()
        
        # Add generators
        self.add_generator_agent("coal_plant_1", {
            "max_capacity_mw": 200.0,
            "fuel_cost_per_mwh": 60.0,
            "emissions_rate_kg_co2_per_mwh": 800.0,
            "efficiency": 0.35
        })
        
        self.add_generator_agent("gas_plant_1", {
            "max_capacity_mw": 150.0,
            "fuel_cost_per_mwh": 80.0,
            "emissions_rate_kg_co2_per_mwh": 400.0,
            "efficiency": 0.50
        })
        
        self.add_generator_agent("solar_farm_1", {
            "max_capacity_mw": 100.0,
            "fuel_cost_per_mwh": 0.0,
            "emissions_rate_kg_co2_per_mwh": 0.0,
            "efficiency": 1.0
        })
        
        self.add_generator_agent("wind_farm_1", {
            "max_capacity_mw": 80.0,
            "fuel_cost_per_mwh": 0.0,
            "emissions_rate_kg_co2_per_mwh": 0.0,
            "efficiency": 1.0
        })
        
        # Add storage systems
        self.add_storage_agent("battery_1", {
            "max_capacity_mwh": 200.0,
            "max_power_mw": 50.0,
            "round_trip_efficiency": 0.90
        })
        
        self.add_storage_agent("battery_2", {
            "max_capacity_mwh": 100.0,
            "max_power_mw": 25.0,
            "round_trip_efficiency": 0.88
        })
        
        # Add consumers
        self.add_consumer_agent("industrial_consumer_1", {
            "baseline_load_mw": 100.0,
            "flexible_load_mw": 30.0,
            "comfort_preference": 75.0
        })
        
        self.add_consumer_agent("commercial_consumer_1", {
            "baseline_load_mw": 50.0,
            "flexible_load_mw": 15.0,
            "comfort_preference": 80.0
        })
        
        self.add_consumer_agent("residential_cluster_1", {
            "baseline_load_mw": 25.0,
            "flexible_load_mw": 8.0,
            "comfort_preference": 85.0
        })
        
        self.logger.info("Sample scenario created with 9 agents")
    
    async def update_market_data(self) -> None:
        """Update market data based on time and conditions"""
        # Simple market price simulation
        hour = self.simulation_time.hour
        base_price = 50.0
        
        # Peak pricing pattern
        if 6 <= hour <= 10 or 17 <= hour <= 22:  # Peak hours
            price_multiplier = 1.5 + np.random.normal(0, 0.1)
        elif 0 <= hour <= 6:  # Off-peak hours
            price_multiplier = 0.7 + np.random.normal(0, 0.05)
        else:  # Mid-peak hours
            price_multiplier = 1.0 + np.random.normal(0, 0.1)
        
        self.market_data["current_price"] = max(10.0, base_price * price_multiplier)
        
        # Weather simulation
        self.market_data["weather"]["temperature"] = 20 + 10 * np.sin(hour * np.pi / 12) + np.random.normal(0, 2)
        self.market_data["weather"]["wind_speed"] = max(0, 10 + 5 * np.random.normal(0, 1))
        self.market_data["weather"]["solar_irradiance"] = max(0, 800 * np.sin(max(0, hour - 6) * np.pi / 12))
        
        # Demand forecast
        base_demand = 800.0
        demand_multiplier = 1.0 + 0.3 * np.sin((hour - 12) * np.pi / 12)
        self.market_data["demand_forecast"]["expected_peak"] = base_demand * demand_multiplier
        
        # Update price forecast (next 24 hours)
        self.market_data["price_forecast"] = [
            self.market_data["current_price"] * (1 + 0.1 * np.random.normal()) 
            for _ in range(24)
        ]
        
        # Broadcast updated market data
        await self.message_router.broadcast_message(
            sender_id="market_data_service",
            message_type=MessageType.MARKET_PRICE_UPDATE,
            content=self.market_data
        )
    
    async def run_simulation_step(self) -> None:
        """Run one simulation time step"""
        self.logger.info(f"Running simulation step at {self.simulation_time}")
        
        # Update market data
        await self.update_market_data()
        
        # Run decision cycles for all agents in parallel
        agent_tasks = []
        for agent in self.agents.values():
            # Update agent's market data - handle both AgentState and dict cases
            if hasattr(agent.state, 'market_data'):
                agent.state.market_data.update(self.market_data)
            elif isinstance(agent.state, dict):
                if 'market_data' not in agent.state:
                    agent.state['market_data'] = {}
                agent.state['market_data'].update(self.market_data)
            
            # Run agent decision cycle
            agent_tasks.append(agent.run_decision_cycle())
        
        # Wait for all agents to complete their decision cycles
        await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Process messages
        await self._process_message_queue()
        
        # Update simulation metrics
        await self._update_simulation_metrics()
        
        # Advance simulation time
        self.simulation_time += timedelta(minutes=self.time_step_minutes)
        self.simulation_metrics["total_steps"] += 1
        
    async def _process_message_queue(self) -> None:
        """Process any pending messages in the system"""
        # This is handled by the message router during agent execution
        # Could add additional message processing logic here
        pass
    
    async def _update_simulation_metrics(self) -> None:
        """Update overall simulation performance metrics"""
        if self.grid_operator:
            grid_metrics = await self.grid_operator.calculate_performance_metrics()
            
            self.simulation_metrics.update({
                "market_clearings": grid_metrics.get("market_clearing_frequency", 0),
                "total_cost": grid_metrics.get("total_system_cost", 0.0),
                "renewable_penetration": grid_metrics.get("renewable_penetration", 0.0),
                "grid_stability_score": min(
                    grid_metrics.get("frequency_stability_index", 100.0),
                    grid_metrics.get("voltage_stability_index", 100.0)
                )
            })
        
        self.simulation_metrics["messages_sent"] = len(self.message_router.message_history)
    
    async def run_simulation(self, duration_hours: int = 24, real_time: bool = False) -> None:
        """Run the full simulation"""
        self.is_running = True
        total_steps = duration_hours * 60 // self.time_step_minutes
        
        self.logger.info(f"Starting simulation for {duration_hours} hours ({total_steps} steps)")
        
        try:
            for step in range(total_steps):
                if not self.is_running:
                    break
                    
                await self.run_simulation_step()
                
                # Progress logging
                if step % 12 == 0:  # Every hour
                    progress = (step / total_steps) * 100
                    self.logger.info(f"Simulation progress: {progress:.1f}% (Step {step}/{total_steps})")
                
                # Real-time simulation delay
                if real_time:
                    await asyncio.sleep(1)  # 1 second per time step
                    
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            raise
        finally:
            self.is_running = False
        
        self.logger.info("Simulation completed")
    
    def stop_simulation(self) -> None:
        """Stop the running simulation"""
        self.is_running = False
        self.logger.info("Simulation stop requested")
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get a summary of simulation results"""
        agent_summaries = {}
        
        for agent_id, agent in self.agents.items():
            agent_summaries[agent_id] = agent.get_current_state()
        
        summary = {
            "simulation_info": {
                "total_steps": self.simulation_metrics["total_steps"],
                "simulation_time": self.simulation_time.isoformat(),
                "duration_hours": self.simulation_metrics["total_steps"] * self.time_step_minutes / 60,
                "messages_sent": self.simulation_metrics["messages_sent"]
            },
            "performance_metrics": self.simulation_metrics,
            "final_market_data": self.market_data,
            "agent_states": agent_summaries
        }
        
        if self.grid_operator:
            summary["grid_status"] = self.grid_operator.get_system_status()
        
        return summary
    
    def export_results(self, filename: str) -> None:
        """Export simulation results to a JSON file"""
        summary = self.get_simulation_summary()
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Simulation results exported to {filename}")
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time system metrics for monitoring"""
        metrics = {}
        
        # Individual agent metrics
        for agent_id, agent in self.agents.items():
            metrics[agent_id] = await agent.calculate_performance_metrics()
        
        # System-wide metrics
        if self.grid_operator:
            metrics["grid_system"] = await self.grid_operator.calculate_performance_metrics()
            metrics["grid_status"] = self.grid_operator.get_system_status()
        
        metrics["simulation"] = self.simulation_metrics
        metrics["market_data"] = self.market_data
        
        return metrics


# Utility functions for creating specific scenarios

async def create_renewable_heavy_scenario() -> SmartGridSimulation:
    """Create a scenario with high renewable penetration"""
    sim = SmartGridSimulation()
    await sim.add_grid_operator()
    
    # High renewable generation
    sim.add_generator_agent("solar_farm_1", {"max_capacity_mw": 300.0, "emissions_rate_kg_co2_per_mwh": 0.0})
    sim.add_generator_agent("wind_farm_1", {"max_capacity_mw": 250.0, "emissions_rate_kg_co2_per_mwh": 0.0})
    sim.add_generator_agent("wind_farm_2", {"max_capacity_mw": 200.0, "emissions_rate_kg_co2_per_mwh": 0.0})
    
    # Backup conventional generation
    sim.add_generator_agent("gas_plant_1", {"max_capacity_mw": 150.0, "emissions_rate_kg_co2_per_mwh": 400.0})
    
    # Large storage capacity
    sim.add_storage_agent("battery_1", {"max_capacity_mwh": 500.0, "max_power_mw": 100.0})
    sim.add_storage_agent("battery_2", {"max_capacity_mwh": 300.0, "max_power_mw": 75.0})
    
    # Flexible consumers
    sim.add_consumer_agent("smart_city_1", {"baseline_load_mw": 200.0, "flexible_load_mw": 80.0})
    sim.add_consumer_agent("industrial_1", {"baseline_load_mw": 150.0, "flexible_load_mw": 60.0})
    
    return sim


async def create_traditional_grid_scenario() -> SmartGridSimulation:
    """Create a scenario with traditional generation mix"""
    sim = SmartGridSimulation()
    await sim.add_grid_operator()
    
    # Traditional generation
    sim.add_generator_agent("coal_plant_1", {"max_capacity_mw": 400.0, "emissions_rate_kg_co2_per_mwh": 800.0})
    sim.add_generator_agent("gas_plant_1", {"max_capacity_mw": 300.0, "emissions_rate_kg_co2_per_mwh": 400.0})
    sim.add_generator_agent("nuclear_plant_1", {"max_capacity_mw": 500.0, "emissions_rate_kg_co2_per_mwh": 50.0})
    
    # Limited renewables
    sim.add_generator_agent("solar_farm_1", {"max_capacity_mw": 50.0, "emissions_rate_kg_co2_per_mwh": 0.0})
    
    # Minimal storage
    sim.add_storage_agent("pumped_hydro_1", {"max_capacity_mwh": 200.0, "max_power_mw": 50.0})
    
    # Less flexible consumers
    sim.add_consumer_agent("industrial_1", {"baseline_load_mw": 300.0, "flexible_load_mw": 30.0})
    sim.add_consumer_agent("residential_1", {"baseline_load_mw": 200.0, "flexible_load_mw": 20.0})
    
    return sim 