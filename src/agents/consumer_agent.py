import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import random

from .base_agent import BaseAgent, AgentType, AgentState, MessageType


@dataclass
class ConsumerState:
    """Consumer-specific operational state"""
    current_load_mw: float = 50.0  # Current power consumption
    baseline_load_mw: float = 50.0  # Baseline consumption without DR
    flexible_load_mw: float = 20.0  # Amount of load that can be shifted
    comfort_level: float = 75.0  # Current comfort level (0-100)
    comfort_preference: float = 80.0  # Preferred comfort level
    temperature_setpoint: float = 22.0  # HVAC setpoint (°C)
    ev_battery_soc: float = 80.0  # Electric vehicle state of charge (%)
    ev_charging_rate_kw: float = 0.0  # Current EV charging rate
    solar_generation_kw: float = 0.0  # Local solar generation
    battery_soc: float = 50.0  # Home battery state of charge (%)
    last_dr_participation: float = 0.0  # Last demand response participation level (0-1)
    total_energy_cost: float = 0.0  # Cumulative energy costs
    dr_payments_received: float = 0.0  # Demand response payments received


class MADDPGActor(nn.Module):
    """Actor network for MADDPG"""
    
    def __init__(self, state_size: int = 40, action_size: int = 4):
        super(MADDPGActor, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # Output actions in range [0, 1] using sigmoid
        return torch.sigmoid(self.fc3(x))


class MADDPGCritic(nn.Module):
    """Critic network for MADDPG - takes actions from all agents"""
    
    def __init__(self, state_size: int = 40, action_size: int = 4, num_agents: int = 3):
        super(MADDPGCritic, self).__init__()
        # Critic sees states and actions from all agents
        total_input_size = state_size + action_size * num_agents
        self.fc1 = nn.Linear(total_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, states, actions):
        # Concatenate states and actions from all agents
        x = torch.cat([states, actions], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ConsumerAgent(BaseAgent):
    """Consumer agent using MADDPG for continuous control"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, AgentType.CONSUMER, config)
        
        # Consumer-specific state
        self.consumer_state = ConsumerState()
        if config:
            for key, value in config.items():
                if hasattr(self.consumer_state, key):
                    setattr(self.consumer_state, key, value)
        
        # MADDPG components
        self.state_size = 40
        self.action_size = 4  # [DR participation, EV charging, HVAC adjustment, battery dispatch]
        self.num_agents = 3  # Assume we interact with 3 other agent types
        
        self.actor = MADDPGActor(self.state_size, self.action_size)
        self.actor_target = MADDPGActor(self.state_size, self.action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        
        self.critic = MADDPGCritic(self.state_size, self.action_size, self.num_agents)
        self.critic_target = MADDPGCritic(self.state_size, self.action_size, self.num_agents)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Replay buffer for multi-agent experiences
        self.replay_buffer = deque(maxlen=10000)
        
        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.001  # Soft update rate
        self.batch_size = 64
        self.noise_scale = 0.1
        
        # Historical data
        self.price_history = deque(maxlen=48)
        self.load_history = deque(maxlen=24)
        self.comfort_history = deque(maxlen=24)
        
        # Performance tracking
        self.total_inconvenience = 0.0
        self.total_savings = 0.0
        
        # Scheduling and comfort parameters
        self.critical_load_periods = []  # Time periods when load cannot be reduced
        self.appliance_schedules = {}  # Flexible appliance schedules
        
    def _encode_state_vector(self) -> np.ndarray:
        """Encode current state into neural network input vector (40 dimensions)"""
        state_vector = np.zeros(self.state_size)
        
        # Current operational state (indices 0-9)
        state_vector[0] = self.consumer_state.current_load_mw / 100.0  # Normalize load
        state_vector[1] = self.consumer_state.flexible_load_mw / 50.0  # Normalize flexible load
        state_vector[2] = self.consumer_state.comfort_level / 100.0
        state_vector[3] = self.consumer_state.comfort_preference / 100.0
        state_vector[4] = self.consumer_state.temperature_setpoint / 30.0  # Normalize temperature
        state_vector[5] = self.consumer_state.ev_battery_soc / 100.0
        state_vector[6] = self.consumer_state.ev_charging_rate_kw / 50.0  # Normalize EV charging
        state_vector[7] = self.consumer_state.solar_generation_kw / 20.0  # Normalize solar
        state_vector[8] = self.consumer_state.battery_soc / 100.0
        state_vector[9] = self.consumer_state.last_dr_participation
        
        # Market price information (indices 10-19)
        current_price = self.state.market_data.get("current_price", 50.0)
        state_vector[10] = current_price / 100.0  # Normalize price
        
        # Price history (last 6 hours)
        if len(self.price_history) > 0:
            recent_prices = list(self.price_history)[-6:]
            for i, price in enumerate(recent_prices):
                if i < 6:
                    state_vector[11 + i] = price / 100.0
        
        # Price forecasts (next 3 hours)
        price_forecast = self.state.market_data.get("price_forecast", [current_price] * 3)
        for i, price in enumerate(price_forecast[:3]):
            state_vector[17 + i] = price / 100.0
        
        # Weather conditions (indices 20-25)
        weather = self.state.market_data.get("weather", {})
        outdoor_temp = weather.get("temperature", 20)
        state_vector[20] = outdoor_temp / 40.0  # Normalize outdoor temperature
        state_vector[21] = weather.get("humidity", 50) / 100.0
        state_vector[22] = weather.get("solar_irradiance", 500) / 1000.0
        state_vector[23] = weather.get("wind_speed", 5) / 20.0
        
        # Time-based features (indices 24-27)
        now = datetime.now()
        state_vector[24] = now.hour / 24.0
        state_vector[25] = now.weekday() / 7.0
        state_vector[26] = now.month / 12.0
        is_weekend = 1.0 if now.weekday() >= 5 else 0.0
        state_vector[27] = is_weekend
        
        # Grid conditions and DR opportunities (indices 28-33)
        demand_forecast = self.state.market_data.get("demand_forecast", {})
        expected_peak = demand_forecast.get("expected_peak", 1000)
        state_vector[28] = expected_peak / 2000.0  # Normalize
        
        dr_price = self.state.market_data.get("dr_price", 100.0)
        state_vector[29] = dr_price / 200.0  # Normalize DR payment
        
        # Comfort and convenience factors (indices 30-35)
        comfort_deviation = abs(self.consumer_state.comfort_level - self.consumer_state.comfort_preference)
        state_vector[30] = comfort_deviation / 50.0  # Normalize comfort deviation
        
        # Load patterns and historical performance (indices 36-39)
        if len(self.load_history) > 0:
            avg_load = np.mean(list(self.load_history))
            state_vector[36] = avg_load / 100.0
        
        state_vector[37] = self.consumer_state.total_energy_cost / 10000.0  # Normalize cumulative cost
        state_vector[38] = self.consumer_state.dr_payments_received / 1000.0  # Normalize DR payments
        state_vector[39] = self.total_inconvenience / 100.0  # Normalize inconvenience
        
        return state_vector.astype(np.float32)
    
    def _decode_actions(self, action_tensor: torch.Tensor) -> Dict[str, float]:
        """Decode neural network actions into specific control commands"""
        actions = action_tensor.detach().numpy()
        
        # Action 0: Demand response participation level (0-1)
        dr_participation = actions[0]
        
        # Action 1: EV charging rate adjustment (0-1, where 0.5 is baseline)
        ev_charging_adjustment = (actions[1] - 0.5) * 2  # Map to [-1, 1]
        
        # Action 2: HVAC temperature setpoint adjustment (0-1, where 0.5 is no change)
        hvac_adjustment = (actions[2] - 0.5) * 6  # ±3°C adjustment
        
        # Action 3: Home battery dispatch (0-1, where 0.5 is no action)
        battery_dispatch = (actions[3] - 0.5) * 2  # Map to [-1, 1]
        
        return {
            "dr_participation": dr_participation,
            "ev_charging_adjustment": ev_charging_adjustment,
            "hvac_adjustment": hvac_adjustment,
            "battery_dispatch": battery_dispatch,
            "raw_actions": actions
        }
    
    def _calculate_utility(self, actions: Dict[str, float], market_result: Dict[str, Any]) -> float:
        """Calculate utility: Comfort_Level - Energy_Costs + DR_Payments - Inconvenience_Penalty"""
        
        # Current energy price
        current_price = market_result.get("clearing_price_mwh", 50.0) / 1000.0  # Convert to $/kWh
        
        # Calculate actual load after demand response
        dr_load_reduction = actions["dr_participation"] * self.consumer_state.flexible_load_mw * 1000  # Convert to kW
        actual_load_kw = self.consumer_state.baseline_load_mw * 1000 - dr_load_reduction
        
        # EV charging adjustment
        ev_baseline_charging = 7.0  # kW baseline
        ev_actual_charging = max(0, ev_baseline_charging + actions["ev_charging_adjustment"] * 10)
        actual_load_kw += ev_actual_charging
        
        # Battery dispatch (positive = discharge, negative = charge)
        battery_power = actions["battery_dispatch"] * 5.0  # ±5 kW
        if battery_power > 0:  # Discharging (reducing net load)
            actual_load_kw -= battery_power
        else:  # Charging (increasing load)
            actual_load_kw -= battery_power  # battery_power is negative
        
        # Solar generation offset
        actual_load_kw -= self.consumer_state.solar_generation_kw
        actual_load_kw = max(0, actual_load_kw)  # Can't have negative net load
        
        # Energy costs
        energy_cost = actual_load_kw * current_price  # $/hour
        
        # Demand response payments
        dr_payment_rate = market_result.get("dr_price", 100.0) / 1000.0  # $/kWh
        dr_payments = dr_load_reduction * dr_payment_rate
        
        # Comfort level calculation
        # HVAC adjustment affects comfort
        temp_deviation = abs(actions["hvac_adjustment"])
        comfort_impact = max(0, temp_deviation * 5)  # Comfort reduction per degree
        current_comfort = max(0, self.consumer_state.comfort_preference - comfort_impact)
        
        # Inconvenience penalty
        inconvenience = 0.0
        
        # DR participation inconvenience
        inconvenience += actions["dr_participation"] * 20  # Base inconvenience for DR
        
        # EV charging inconvenience (if reducing below baseline)
        if actions["ev_charging_adjustment"] < 0:
            inconvenience += abs(actions["ev_charging_adjustment"]) * 10
        
        # HVAC inconvenience
        inconvenience += temp_deviation * 15
        
        # Calculate utility
        utility = current_comfort - energy_cost + dr_payments - inconvenience
        
        # Update state tracking
        self.consumer_state.total_energy_cost += energy_cost
        self.consumer_state.dr_payments_received += dr_payments
        self.consumer_state.comfort_level = current_comfort
        self.consumer_state.last_dr_participation = actions["dr_participation"]
        self.total_inconvenience += inconvenience
        
        return utility / 100.0  # Scale for numerical stability
    
    def _update_consumer_state(self, actions: Dict[str, float]) -> None:
        """Update consumer state based on actions taken"""
        
        # Update load based on DR participation
        dr_reduction = actions["dr_participation"] * self.consumer_state.flexible_load_mw
        self.consumer_state.current_load_mw = self.consumer_state.baseline_load_mw - dr_reduction
        
        # Update EV charging
        ev_baseline = 7.0  # kW
        ev_adjustment = actions["ev_charging_adjustment"] * 10  # ±10 kW
        self.consumer_state.ev_charging_rate_kw = max(0, ev_baseline + ev_adjustment)
        
        # Update EV battery SoC (simplified model)
        if self.consumer_state.ev_charging_rate_kw > 0:
            # Charging increases SoC
            soc_increase = self.consumer_state.ev_charging_rate_kw / 100.0  # Assume 100 kWh battery
            self.consumer_state.ev_battery_soc = min(100.0, self.consumer_state.ev_battery_soc + soc_increase)
        
        # Update home battery SoC
        battery_power = actions["battery_dispatch"] * 5.0  # ±5 kW
        if battery_power > 0:  # Discharging
            soc_decrease = battery_power / 20.0  # Assume 20 kWh battery
            self.consumer_state.battery_soc = max(0.0, self.consumer_state.battery_soc - soc_decrease)
        elif battery_power < 0:  # Charging
            soc_increase = abs(battery_power) / 20.0
            self.consumer_state.battery_soc = min(100.0, self.consumer_state.battery_soc + soc_increase)
        
        # Update temperature setpoint
        self.consumer_state.temperature_setpoint += actions["hvac_adjustment"]
        self.consumer_state.temperature_setpoint = max(18.0, min(28.0, self.consumer_state.temperature_setpoint))
    
    async def analyze_market_data(self) -> Dict[str, Any]:
        """Analyze market conditions for consumer decision making"""
        current_price = self.state.market_data.get("current_price", 50.0)
        self.price_history.append(current_price)
        self.load_history.append(self.consumer_state.current_load_mw)
        self.comfort_history.append(self.consumer_state.comfort_level)
        
        # Price analysis
        if len(self.price_history) >= 24:
            daily_prices = list(self.price_history)[-24:]
            avg_price = np.mean(daily_prices)
            peak_price = max(daily_prices)
            off_peak_price = min(daily_prices)
        else:
            avg_price = current_price
            peak_price = current_price * 1.5
            off_peak_price = current_price * 0.7
        
        # DR opportunity analysis
        dr_price = self.state.market_data.get("dr_price", 100.0)
        dr_value = dr_price * self.consumer_state.flexible_load_mw / 1000.0  # Value in $/hour
        
        # Load flexibility assessment
        weather = self.state.market_data.get("weather", {})
        outdoor_temp = weather.get("temperature", 20)
        heating_cooling_load = abs(outdoor_temp - self.consumer_state.temperature_setpoint) * 2  # Simplified model
        
        return {
            "price_relative_to_average": current_price / avg_price,
            "price_percentile": len([p for p in daily_prices if p <= current_price]) / len(daily_prices) if len(daily_prices) > 0 else 0.5,
            "dr_opportunity_value": dr_value,
            "heating_cooling_demand": heating_cooling_load,
            "ev_charging_flexibility": max(0, 100 - self.consumer_state.ev_battery_soc),
            "battery_flexibility": self.consumer_state.battery_soc if self.consumer_state.battery_soc > 20 else 0,
            "comfort_margin": self.consumer_state.comfort_level - 60,  # Minimum acceptable comfort
            "solar_forecast": weather.get("solar_irradiance", 500) / 1000.0
        }
    
    async def make_strategic_decision(self, state: AgentState) -> Dict[str, Any]:
        """Make strategic decisions using MADDPG"""
        state_vector = self._encode_state_vector()
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        
        # Get action from actor network
        with torch.no_grad():
            action_tensor = self.actor(state_tensor)
            
        # Add exploration noise
        noise = torch.randn_like(action_tensor) * self.noise_scale
        action_tensor_noisy = torch.clamp(action_tensor + noise, 0.0, 1.0)
        
        actions = self._decode_actions(action_tensor_noisy.squeeze(0))
        
        # Store current state and action for learning
        self.current_state = state_vector
        self.current_action = action_tensor_noisy.squeeze(0).numpy()
        
        return {
            "action_type": "demand_response_and_control",
            "dr_participation": actions["dr_participation"],
            "ev_charging_adjustment": actions["ev_charging_adjustment"],
            "hvac_adjustment": actions["hvac_adjustment"],
            "battery_dispatch": actions["battery_dispatch"],
            "reasoning": f"MADDPG decision - DR: {actions['dr_participation']:.2f}, EV: {actions['ev_charging_adjustment']:.2f}",
            "state_vector": state_vector.tolist(),
            "predicted_load_mw": self.consumer_state.baseline_load_mw * (1 - actions["dr_participation"] * 0.4)
        }
    
    async def execute_decision(self, decision: Dict[str, Any]) -> None:
        """Execute the strategic decision"""
        # Update consumer state
        self._update_consumer_state(decision)
        
        # Send demand response offer if participating
        if decision["dr_participation"] > 0.1:  # Minimum participation threshold
            await self.send_message(
                receiver_id="grid_operator",
                message_type=MessageType.DEMAND_RESPONSE_OFFER,
                content={
                    "consumer_id": self.agent_id,
                    "flexible_load_mw": self.consumer_state.flexible_load_mw * decision["dr_participation"],
                    "duration_hours": 2,
                    "price_required_per_mwh": 80.0,  # Minimum price for participation
                    "notice_period_minutes": 15,
                    "comfort_constraints": {
                        "min_temperature": 20.0,
                        "max_temperature": 26.0,
                        "critical_periods": self.critical_load_periods
                    }
                }
            )
        
        # Send status update to grid operator
        await self.send_message(
            receiver_id="grid_operator",
            message_type=MessageType.STATUS_UPDATE,
            content={
                "consumer_id": self.agent_id,
                "current_load_mw": self.consumer_state.current_load_mw,
                "flexible_load_available_mw": self.consumer_state.flexible_load_mw * (1 - decision["dr_participation"]),
                "ev_charging_kw": self.consumer_state.ev_charging_rate_kw,
                "solar_generation_kw": self.consumer_state.solar_generation_kw,
                "battery_soc_percent": self.consumer_state.battery_soc,
                "comfort_level": self.consumer_state.comfort_level
            }
        )
    
    def learn_from_market_result(self, market_result: Dict[str, Any], other_agent_actions: List[np.ndarray] = None) -> None:
        """Learn from market results using MADDPG"""
        if not hasattr(self, 'current_state') or not hasattr(self, 'current_action'):
            return
        
        # Calculate utility (reward)
        actions_dict = {
            "dr_participation": self.consumer_state.last_dr_participation,
            "ev_charging_adjustment": 0.0,  # Simplified for now
            "hvac_adjustment": 0.0,
            "battery_dispatch": 0.0
        }
        utility = self._calculate_utility(actions_dict, market_result)
        
        # Get next state
        next_state = self._encode_state_vector()
        
        # For MADDPG, we need actions from other agents
        # In practice, this would come from the multi-agent environment
        if other_agent_actions is None:
            # Create dummy actions for other agents
            other_agent_actions = [np.random.rand(4) for _ in range(self.num_agents - 1)]
        
        # Combine all actions for critic
        all_actions = np.concatenate([self.current_action] + other_agent_actions)
        
        # Store experience
        self.replay_buffer.append((
            self.current_state,
            self.current_action,
            all_actions,  # All agent actions for critic
            utility,
            next_state,
            False  # Not terminal
        ))
        
        # Train networks
        if len(self.replay_buffer) > self.batch_size:
            self._train_maddpg()
        
        # Update operational status
        self.state.operational_status.update({
            "current_load_mw": self.consumer_state.current_load_mw,
            "dr_participation": self.consumer_state.last_dr_participation,
            "comfort_level": self.consumer_state.comfort_level,
            "ev_soc": self.consumer_state.ev_battery_soc,
            "battery_soc": self.consumer_state.battery_soc,
            "last_utility": utility,
            "total_energy_cost": self.consumer_state.total_energy_cost,
            "dr_payments": self.consumer_state.dr_payments_received
        })
    
    def _train_maddpg(self) -> None:
        """Train MADDPG networks"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(list(self.replay_buffer), self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.FloatTensor([e[1] for e in batch])
        all_actions = torch.FloatTensor([e[2] for e in batch])
        rewards = torch.FloatTensor([e[3] for e in batch])
        next_states = torch.FloatTensor([e[4] for e in batch])
        
        # Train Critic
        with torch.no_grad():
            # Get next actions for all agents (simplified - using own actor for all)
            next_actions_own = self.actor_target(next_states)
            # For other agents, use random actions (in practice, would use their actual actors)
            next_actions_others = torch.rand(self.batch_size, self.action_size * (self.num_agents - 1))
            next_actions_all = torch.cat([next_actions_own, next_actions_others], dim=1)
            
            target_q_values = self.critic_target(next_states, next_actions_all)
            target_q_values = rewards.unsqueeze(1) + self.gamma * target_q_values
        
        current_q_values = self.critic(states, all_actions)
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Train Actor
        predicted_actions_own = self.actor(states)
        predicted_actions_others = torch.rand(self.batch_size, self.action_size * (self.num_agents - 1))
        predicted_actions_all = torch.cat([predicted_actions_own, predicted_actions_others], dim=1)
        
        actor_loss = -self.critic(states, predicted_actions_all).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    async def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate consumer-specific performance metrics"""
        base_metrics = await super().calculate_performance_metrics()
        
        # Calculate consumer-specific metrics
        net_savings = self.consumer_state.dr_payments_received - self.consumer_state.total_energy_cost
        
        # Comfort and convenience metrics
        if len(self.comfort_history) > 0:
            avg_comfort = np.mean(list(self.comfort_history))
            comfort_variance = np.var(list(self.comfort_history))
        else:
            avg_comfort = self.consumer_state.comfort_level
            comfort_variance = 0.0
        
        consumer_metrics = {
            "average_comfort_level": avg_comfort,
            "comfort_stability": 100.0 - comfort_variance,  # Higher is better
            "total_energy_cost": self.consumer_state.total_energy_cost,
            "dr_payments_received": self.consumer_state.dr_payments_received,
            "net_savings": net_savings,
            "dr_participation_rate": self.consumer_state.last_dr_participation * 100,
            "ev_battery_soc": self.consumer_state.ev_battery_soc,
            "home_battery_soc": self.consumer_state.battery_soc,
            "load_factor": self.consumer_state.current_load_mw / self.consumer_state.baseline_load_mw,
            "inconvenience_score": self.total_inconvenience,
            "utility_score": avg_comfort - self.consumer_state.total_energy_cost + self.consumer_state.dr_payments_received - self.total_inconvenience
        }
        
        base_metrics.update(consumer_metrics)
        return base_metrics 