import numpy as np
import asyncio
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging

from .base_agent import BaseAgent, AgentType, AgentState, MessageType, AgentMessage


@dataclass
class GridState:
    """Grid operational state"""
    frequency_hz: float = 50.0  # Grid frequency
    voltage_pu: float = 1.0  # Voltage in per unit
    total_generation_mw: float = 0.0  # Total generation
    total_load_mw: float = 0.0  # Total load
    renewable_generation_mw: float = 0.0  # Renewable generation
    storage_charge_mw: float = 0.0  # Net storage charging/discharging
    reserve_margin_mw: float = 0.0  # Available reserves
    transmission_loading: Dict[str, float] = field(default_factory=dict)  # Line loadings
    carbon_intensity_kg_per_mwh: float = 400.0  # Grid carbon intensity
    system_cost_per_hour: float = 0.0  # Total system cost


@dataclass
class MarketBid:
    """Standard bid format"""
    agent_id: str
    bid_type: str  # "generation", "demand_response", "storage"
    price_per_mwh: float
    quantity_mw: float
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketResult:
    """Market clearing results"""
    clearing_price_mwh: float
    total_cleared_mw: float
    cleared_bids: List[Tuple[str, float, float]]  # (agent_id, quantity, price)
    system_cost: float
    frequency_hz: float
    voltage_pu: float
    renewable_penetration: float
    carbon_intensity: float


class GridOperatorAgent(BaseAgent):
    """Grid operator agent responsible for market clearing and grid coordination"""
    
    def __init__(self, agent_id: str = "grid_operator", config: Dict[str, Any] = None):
        super().__init__(agent_id, AgentType.GRID_OPERATOR, config)
        
        # Grid state
        self.grid_state = GridState()
        
        # Market management
        self.generation_bids: List[MarketBid] = []
        self.demand_response_offers: List[MarketBid] = []
        self.storage_bids: List[MarketBid] = []
        
        # Agent tracking
        self.registered_agents: Dict[str, AgentType] = {}
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        
        # Market timing
        self.market_clearing_interval_minutes = 5  # Clear market every 5 minutes
        self.last_market_clearing = datetime.now()
        
        # Grid stability monitoring
        self.frequency_history = deque(maxlen=100)
        self.voltage_history = deque(maxlen=100)
        self.load_forecast = deque(maxlen=48)  # 48-hour load forecast
        
        # Performance metrics
        self.market_efficiency_history = deque(maxlen=24)
        self.reliability_metrics = {
            "saidi_minutes": 0.0,  # System Average Interruption Duration Index
            "frequency_violations": 0,
            "voltage_violations": 0,
            "reserve_shortfalls": 0
        }
        
        # Economic tracking
        self.total_system_cost = 0.0
        self.consumer_surplus = 0.0
        self.producer_surplus = 0.0
        
        # Environmental tracking
        self.total_emissions = 0.0
        self.renewable_curtailment = 0.0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"GridOperator-{agent_id}")
    
    async def register_agent(self, agent_id: str, agent_type: AgentType) -> None:
        """Register an agent with the grid operator"""
        self.registered_agents[agent_id] = agent_type
        self.agent_states[agent_id] = {}
        self.logger.info(f"Registered agent {agent_id} of type {agent_type.value}")
    
    async def analyze_market_data(self) -> Dict[str, Any]:
        """Analyze overall grid and market conditions"""
        # Calculate load-generation balance
        generation_total = sum(
            bid.quantity_mw for bid in self.generation_bids
        )
        load_total = self.grid_state.total_load_mw
        
        # Calculate reserve margin
        reserve_margin = generation_total - load_total
        
        # Analyze renewable penetration
        renewable_penetration = (
            self.grid_state.renewable_generation_mw / generation_total
            if generation_total > 0 else 0.0
        )
        
        # Price volatility analysis
        if len(self.market_efficiency_history) > 1:
            price_volatility = np.std([result['clearing_price'] for result in self.market_efficiency_history])
        else:
            price_volatility = 0.0
        
        return {
            "load_generation_balance": reserve_margin,
            "renewable_penetration": renewable_penetration,
            "price_volatility": price_volatility,
            "frequency_stability": np.std(list(self.frequency_history)) if self.frequency_history else 0.0,
            "voltage_stability": np.std(list(self.voltage_history)) if self.voltage_history else 0.0,
            "system_cost_trend": self.total_system_cost
        }
    
    async def make_strategic_decision(self, state: AgentState) -> Dict[str, Any]:
        """Make grid coordination and market clearing decisions"""
        # Check if it's time for market clearing
        time_since_clearing = datetime.now() - self.last_market_clearing
        should_clear_market = time_since_clearing.total_seconds() >= (self.market_clearing_interval_minutes * 60)
        
        decision = {
            "action_type": "grid_coordination",
            "clear_market": should_clear_market,
            "market_interval_minutes": self.market_clearing_interval_minutes,
            "reasoning": f"Grid coordination - Market clearing: {should_clear_market}"
        }
        
        # Additional grid stability actions
        if self.grid_state.frequency_hz < 49.9 or self.grid_state.frequency_hz > 50.1:
            decision["frequency_regulation_needed"] = True
            decision["target_frequency"] = 50.0
        
        if self.grid_state.voltage_pu < 0.95 or self.grid_state.voltage_pu > 1.05:
            decision["voltage_regulation_needed"] = True
            decision["target_voltage"] = 1.0
        
        # Reserve margin management
        reserve_margin = self._calculate_reserve_margin()
        if reserve_margin < 100.0:  # MW
            decision["insufficient_reserves"] = True
            decision["required_reserves"] = 100.0 - reserve_margin
        
        return decision
    
    async def execute_decision(self, decision: Dict[str, Any]) -> None:
        """Execute grid coordination decisions"""
        if decision.get("clear_market", False):
            await self._clear_market()
        
        if decision.get("frequency_regulation_needed", False):
            await self._request_frequency_regulation()
        
        if decision.get("voltage_regulation_needed", False):
            await self._request_voltage_regulation()
        
        if decision.get("insufficient_reserves", False):
            await self._request_additional_reserves(decision["required_reserves"])
    
    async def _clear_market(self) -> MarketResult:
        """Clear the energy market using economic dispatch"""
        self.logger.info("Starting market clearing process")
        
        # Combine all bids
        all_supply_bids = []
        all_demand_bids = []
        
        # Generation bids (supply)
        for bid in self.generation_bids:
            all_supply_bids.append((bid.price_per_mwh, bid.quantity_mw, bid.agent_id, "generation"))
        
        # Storage discharge bids (supply) and charge bids (demand)
        for bid in self.storage_bids:
            if bid.additional_params.get("action_type") == "discharge":
                all_supply_bids.append((bid.price_per_mwh, bid.quantity_mw, bid.agent_id, "storage_discharge"))
            elif bid.additional_params.get("action_type") == "charge":
                all_demand_bids.append((bid.price_per_mwh, bid.quantity_mw, bid.agent_id, "storage_charge"))
        
        # Demand response offers (demand reduction = negative demand)
        baseline_demand = self.grid_state.total_load_mw
        
        # Sort supply bids by price (ascending)
        all_supply_bids.sort(key=lambda x: x[0])
        
        # Find market clearing point
        clearing_price, cleared_supply, cleared_bids = self._find_market_equilibrium(
            all_supply_bids, baseline_demand, all_demand_bids
        )
        
        # Calculate market result
        market_result = MarketResult(
            clearing_price_mwh=clearing_price,
            total_cleared_mw=cleared_supply,
            cleared_bids=cleared_bids,
            system_cost=clearing_price * cleared_supply,
            frequency_hz=self.grid_state.frequency_hz,
            voltage_pu=self.grid_state.voltage_pu,
            renewable_penetration=self.grid_state.renewable_generation_mw / cleared_supply if cleared_supply > 0 else 0,
            carbon_intensity=self._calculate_carbon_intensity(cleared_bids)
        )
        
        # Update grid state
        self.grid_state.total_generation_mw = cleared_supply
        self.grid_state.system_cost_per_hour = market_result.system_cost
        self.total_system_cost += market_result.system_cost
        
        # Send dispatch instructions to cleared agents
        await self._send_dispatch_instructions(market_result)
        
        # Update performance metrics
        self.market_efficiency_history.append({
            'timestamp': datetime.now(),
            'clearing_price': clearing_price,
            'cleared_quantity': cleared_supply,
            'system_cost': market_result.system_cost
        })
        
        self.last_market_clearing = datetime.now()
        
        # Broadcast market results
        await self._broadcast_market_results(market_result)
        
        self.logger.info(f"Market cleared: {cleared_supply:.2f} MW at ${clearing_price:.2f}/MWh")
        
        return market_result
    
    def _find_market_equilibrium(self, supply_bids: List[Tuple], baseline_demand: float, 
                                demand_bids: List[Tuple]) -> Tuple[float, float, List]:
        """Find market equilibrium using supply-demand intersection"""
        
        # Build supply curve
        cumulative_supply = 0.0
        supply_curve = []
        
        for price, quantity, agent_id, bid_type in supply_bids:
            supply_curve.append((price, cumulative_supply, cumulative_supply + quantity, agent_id, bid_type))
            cumulative_supply += quantity
        
        # Simple approach: clear at marginal cost of last dispatched unit
        target_supply = baseline_demand
        
        cleared_bids = []
        total_cleared = 0.0
        clearing_price = 0.0
        
        for price, start_qty, end_qty, agent_id, bid_type in supply_curve:
            if total_cleared >= target_supply:
                break
                
            quantity_needed = min(end_qty - start_qty, target_supply - total_cleared)
            if quantity_needed > 0:
                cleared_bids.append((agent_id, quantity_needed, price))
                total_cleared += quantity_needed
                clearing_price = price  # Marginal price
        
        return clearing_price, total_cleared, cleared_bids
    
    def _calculate_carbon_intensity(self, cleared_bids: List[Tuple]) -> float:
        """Calculate grid carbon intensity based on cleared generation"""
        total_generation = 0.0
        total_emissions = 0.0
        
        for agent_id, quantity, price in cleared_bids:
            # Look up emissions rate for this agent
            agent_state = self.agent_states.get(agent_id, {})
            emissions_rate = agent_state.get("emissions_rate_kg_co2_per_mwh", 400.0)  # Default
            
            total_generation += quantity
            total_emissions += quantity * emissions_rate
        
        return total_emissions / total_generation if total_generation > 0 else 0.0
    
    async def _send_dispatch_instructions(self, market_result: MarketResult) -> None:
        """Send dispatch instructions to cleared agents"""
        
        for agent_id, quantity, price in market_result.cleared_bids:
            await self.send_message(
                receiver_id=agent_id,
                message_type=MessageType.DISPATCH_INSTRUCTION,
                content={
                    "cleared_quantity_mw": quantity,
                    "clearing_price_mwh": price,
                    "dispatch_time": datetime.now().isoformat(),
                    "market_result": {
                        "frequency_hz": market_result.frequency_hz,
                        "voltage_pu": market_result.voltage_pu,
                        "renewable_penetration": market_result.renewable_penetration,
                        "carbon_intensity": market_result.carbon_intensity
                    }
                }
            )
    
    async def _broadcast_market_results(self, market_result: MarketResult) -> None:
        """Broadcast market results to all agents"""
        market_update = {
            "clearing_price_mwh": market_result.clearing_price_mwh,
            "total_cleared_mw": market_result.total_cleared_mw,
            "frequency_hz": market_result.frequency_hz,
            "voltage_pu": market_result.voltage_pu,
            "renewable_penetration": market_result.renewable_penetration,
            "carbon_intensity": market_result.carbon_intensity,
            "system_cost": market_result.system_cost,
            "timestamp": datetime.now().isoformat()
        }
        
        for agent_id in self.registered_agents.keys():
            await self.send_message(
                receiver_id=agent_id,
                message_type=MessageType.MARKET_PRICE_UPDATE,
                content=market_update
            )
    
    async def _request_frequency_regulation(self) -> None:
        """Request frequency regulation from capable agents"""
        frequency_error = self.grid_state.frequency_hz - 50.0
        
        # Request response from storage and generators
        for agent_id, agent_type in self.registered_agents.items():
            if agent_type in [AgentType.STORAGE, AgentType.GENERATOR]:
                await self.send_message(
                    receiver_id=agent_id,
                    message_type=MessageType.STATUS_UPDATE,
                    content={
                        "service_request": "frequency_regulation",
                        "frequency_error_hz": frequency_error,
                        "response_needed_mw": abs(frequency_error) * 100,  # Simplified
                        "priority": 5
                    }
                )
    
    async def _request_voltage_regulation(self) -> None:
        """Request voltage regulation from capable agents"""
        voltage_error = self.grid_state.voltage_pu - 1.0
        
        for agent_id, agent_type in self.registered_agents.items():
            if agent_type == AgentType.GENERATOR:
                await self.send_message(
                    receiver_id=agent_id,
                    message_type=MessageType.STATUS_UPDATE,
                    content={
                        "service_request": "voltage_regulation",
                        "voltage_error_pu": voltage_error,
                        "reactive_power_needed_mvar": abs(voltage_error) * 50,  # Simplified
                        "priority": 4
                    }
                )
    
    async def _request_additional_reserves(self, required_mw: float) -> None:
        """Request additional reserves from agents"""
        for agent_id, agent_type in self.registered_agents.items():
            if agent_type in [AgentType.GENERATOR, AgentType.STORAGE]:
                await self.send_message(
                    receiver_id=agent_id,
                    message_type=MessageType.STATUS_UPDATE,
                    content={
                        "service_request": "additional_reserves",
                        "reserve_needed_mw": required_mw,
                        "priority": 3
                    }
                )
    
    def _calculate_reserve_margin(self) -> float:
        """Calculate current reserve margin"""
        total_capacity = sum(bid.quantity_mw for bid in self.generation_bids)
        current_load = self.grid_state.total_load_mw
        return total_capacity - current_load
    
    async def _handle_generation_bid(self, message: AgentMessage) -> None:
        """Handle generation bid from generator agents"""
        content = message.content
        bid = MarketBid(
            agent_id=message.sender_id,
            bid_type="generation",
            price_per_mwh=content["bid_price"],
            quantity_mw=content["capacity_available"],
            additional_params={
                "ramp_rate": content.get("ramp_rate", 10.0),
                "startup_bid": content.get("startup_bid", False),
                "emissions_rate": content.get("emissions_rate", 400.0),
                "min_output": content.get("min_output", 0.0)
            }
        )
        self.generation_bids.append(bid)
        
        # Update agent state
        self.agent_states[message.sender_id] = {
            "capacity_mw": content["capacity_available"],
            "bid_price": content["bid_price"],
            "emissions_rate_kg_co2_per_mwh": content.get("emissions_rate", 400.0),
            "last_update": datetime.now()
        }
    
    async def _handle_demand_response_offer(self, message: AgentMessage) -> None:
        """Handle demand response offer from consumer agents"""
        content = message.content
        bid = MarketBid(
            agent_id=message.sender_id,
            bid_type="demand_response",
            price_per_mwh=content["price_required_per_mwh"],
            quantity_mw=content["flexible_load_mw"],
            additional_params={
                "duration_hours": content.get("duration_hours", 1.0),
                "notice_period_minutes": content.get("notice_period_minutes", 15),
                "comfort_constraints": content.get("comfort_constraints", {})
            }
        )
        self.demand_response_offers.append(bid)
        
        # Update agent state
        self.agent_states[message.sender_id] = {
            "flexible_load_mw": content["flexible_load_mw"],
            "dr_price": content["price_required_per_mwh"],
            "last_update": datetime.now()
        }
    
    async def _handle_status_update(self, message: AgentMessage) -> None:
        """Handle status updates from all agents"""
        content = message.content
        
        # Update agent state
        if message.sender_id not in self.agent_states:
            self.agent_states[message.sender_id] = {}
        
        self.agent_states[message.sender_id].update(content)
        self.agent_states[message.sender_id]["last_update"] = datetime.now()
        
        # Update grid-level aggregations
        self._update_grid_aggregates()
    
    def _update_grid_aggregates(self) -> None:
        """Update grid-level aggregated values"""
        total_load = 0.0
        total_renewable = 0.0
        total_generation = 0.0
        storage_net = 0.0
        
        for agent_id, state in self.agent_states.items():
            agent_type = self.registered_agents.get(agent_id)
            
            if agent_type == AgentType.CONSUMER:
                total_load += state.get("current_load_mw", 0.0)
                total_renewable += state.get("solar_generation_kw", 0.0) / 1000.0
                
            elif agent_type == AgentType.GENERATOR:
                total_generation += state.get("output_mw", 0.0)
                if "renewable" in agent_id.lower() or "solar" in agent_id.lower() or "wind" in agent_id.lower():
                    total_renewable += state.get("output_mw", 0.0)
                    
            elif agent_type == AgentType.STORAGE:
                storage_net += state.get("charge_rate_mw", 0.0)  # Positive = charging
        
        # Update grid state
        self.grid_state.total_load_mw = total_load
        self.grid_state.total_generation_mw = total_generation
        self.grid_state.renewable_generation_mw = total_renewable
        self.grid_state.storage_charge_mw = storage_net
        
        # Simple frequency model based on load-generation balance
        balance_error = total_generation - total_load - storage_net
        frequency_deviation = balance_error * 0.01  # Simplified model
        self.grid_state.frequency_hz = 50.0 + frequency_deviation
        
        # Voltage stability (simplified)
        self.grid_state.voltage_pu = 1.0 + np.random.normal(0, 0.01)  # Add small variations
        
        # Update history
        self.frequency_history.append(self.grid_state.frequency_hz)
        self.voltage_history.append(self.grid_state.voltage_pu)
        
        # Check for violations
        if abs(self.grid_state.frequency_hz - 50.0) > 0.1:
            self.reliability_metrics["frequency_violations"] += 1
        
        if abs(self.grid_state.voltage_pu - 1.0) > 0.05:
            self.reliability_metrics["voltage_violations"] += 1
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """Override base message handling for grid operator specific messages"""
        await super()._handle_message(message)
        
        # Grid operator specific message handling
        if message.message_type == MessageType.GENERATION_BID:
            await self._handle_generation_bid(message)
        elif message.message_type == MessageType.DEMAND_RESPONSE_OFFER:
            await self._handle_demand_response_offer(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
            await self._handle_status_update(message)
    
    def clear_market_bids(self) -> None:
        """Clear all market bids after market clearing"""
        self.generation_bids.clear()
        self.demand_response_offers.clear()
        self.storage_bids.clear()
    
    async def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate grid operator specific performance metrics"""
        base_metrics = await super().calculate_performance_metrics()
        
        # Market efficiency metrics
        if self.market_efficiency_history:
            recent_prices = [result['clearing_price'] for result in self.market_efficiency_history]
            price_volatility = np.std(recent_prices)
            avg_price = np.mean(recent_prices)
        else:
            price_volatility = 0.0
            avg_price = 0.0
        
        # Grid stability metrics
        frequency_stability = 1.0 / (1.0 + np.std(list(self.frequency_history))) if self.frequency_history else 1.0
        voltage_stability = 1.0 / (1.0 + np.std(list(self.voltage_history))) if self.voltage_history else 1.0
        
        # Economic metrics
        total_agents = len(self.registered_agents)
        
        grid_metrics = {
            "market_clearing_frequency": len(self.market_efficiency_history),
            "average_clearing_price": avg_price,
            "price_volatility": price_volatility,
            "frequency_stability_index": frequency_stability * 100,
            "voltage_stability_index": voltage_stability * 100,
            "frequency_violations": self.reliability_metrics["frequency_violations"],
            "voltage_violations": self.reliability_metrics["voltage_violations"],
            "total_system_cost": self.total_system_cost,
            "renewable_penetration": (self.grid_state.renewable_generation_mw / 
                                    self.grid_state.total_generation_mw * 100 
                                    if self.grid_state.total_generation_mw > 0 else 0),
            "load_generation_balance": self.grid_state.total_generation_mw - self.grid_state.total_load_mw,
            "registered_agents": total_agents,
            "reserve_margin_mw": self._calculate_reserve_margin(),
            "carbon_intensity": self.grid_state.carbon_intensity_kg_per_mwh
        }
        
        base_metrics.update(grid_metrics)
        return base_metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for monitoring"""
        return {
            "grid_state": {
                "frequency_hz": self.grid_state.frequency_hz,
                "voltage_pu": self.grid_state.voltage_pu,
                "total_generation_mw": self.grid_state.total_generation_mw,
                "total_load_mw": self.grid_state.total_load_mw,
                "renewable_generation_mw": self.grid_state.renewable_generation_mw,
                "storage_charge_mw": self.grid_state.storage_charge_mw,
                "carbon_intensity": self.grid_state.carbon_intensity_kg_per_mwh
            },
            "market_status": {
                "last_clearing_time": self.last_market_clearing.isoformat(),
                "active_generation_bids": len(self.generation_bids),
                "active_dr_offers": len(self.demand_response_offers),
                "active_storage_bids": len(self.storage_bids)
            },
            "agent_registry": {
                "total_agents": len(self.registered_agents),
                "agents_by_type": {
                    agent_type.value: sum(1 for t in self.registered_agents.values() if t == agent_type)
                    for agent_type in AgentType
                }
            },
            "reliability_metrics": self.reliability_metrics,
            "economic_metrics": {
                "total_system_cost": self.total_system_cost,
                "consumer_surplus": self.consumer_surplus,
                "producer_surplus": self.producer_surplus
            }
        } 