import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

from .base_agent import BaseAgent, AgentType, AgentState, MessageType


@dataclass
class StorageState:
    """Storage-specific operational state"""
    state_of_charge_percent: float = 50.0  # Current SoC
    max_capacity_mwh: float = 100.0  # Maximum energy capacity
    max_power_mw: float = 25.0  # Maximum charge/discharge rate
    round_trip_efficiency: float = 0.90  # Round-trip efficiency
    degradation_cost_per_cycle: float = 50.0  # Cost per full cycle
    cycles_completed: float = 0.0  # Total cycles completed
    temperature_celsius: float = 25.0  # Operating temperature
    charge_rate_mw: float = 0.0  # Current charge rate (+ for charging, - for discharging)
    last_action: str = "idle"  # Last action taken
    last_price_arbitrage: float = 0.0  # Last arbitrage profit


class ActorNetwork(nn.Module):
    """Actor network for continuous action selection"""
    
    def __init__(self, state_size: int = 32, action_size: int = 1):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # Output in range [-1, 1] for charge/discharge rate
        return torch.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    """Critic network for value function estimation"""
    
    def __init__(self, state_size: int = 32, action_size: int = 1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process for exploration noise"""
    
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)
        
    def reset(self):
        self.state = np.copy(self.mu)
        
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state = self.state + dx
        return self.state


class StorageAgent(BaseAgent):
    """Storage agent using Actor-Critic reinforcement learning"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, AgentType.STORAGE, config)
        
        # Storage-specific state
        self.storage_state = StorageState()
        if config:
            for key, value in config.items():
                if hasattr(self.storage_state, key):
                    setattr(self.storage_state, key, value)
        
        # Actor-Critic components
        self.state_size = 32
        self.action_size = 1  # Continuous action: charge/discharge rate
        
        self.actor = ActorNetwork(self.state_size, self.action_size)
        self.actor_target = ActorNetwork(self.state_size, self.action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        
        self.critic = CriticNetwork(self.state_size, self.action_size)
        self.critic_target = CriticNetwork(self.state_size, self.action_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.001  # Soft update rate
        self.batch_size = 64
        
        # Exploration noise
        self.noise = OrnsteinUhlenbeckNoise(self.action_size)
        
        # Historical data for state representation
        self.price_history = deque(maxlen=48)  # 48-hour price history
        self.arbitrage_opportunities = deque(maxlen=24)  # Recent arbitrage analysis
        
        # Performance tracking
        self.total_revenue = 0.0
        self.total_degradation_cost = 0.0
        
    def _encode_state_vector(self) -> np.ndarray:
        """Encode current state into neural network input vector (32 dimensions)"""
        state_vector = np.zeros(self.state_size)
        
        # Storage operational state (indices 0-7)
        state_vector[0] = self.storage_state.state_of_charge_percent / 100.0
        state_vector[1] = self.storage_state.charge_rate_mw / self.storage_state.max_power_mw
        state_vector[2] = self.storage_state.cycles_completed / 1000.0  # Normalize
        state_vector[3] = self.storage_state.temperature_celsius / 50.0  # Normalize
        state_vector[4] = self.storage_state.round_trip_efficiency
        
        # Market price information (indices 5-15)
        current_price = self.state.market_data.get("current_price", 50.0)
        state_vector[5] = current_price / 100.0  # Normalize price
        
        # Price history and volatility (last 6 hours)
        if len(self.price_history) > 0:
            recent_prices = list(self.price_history)[-6:]
            for i, price in enumerate(recent_prices):
                if i < 6:
                    state_vector[6 + i] = price / 100.0
            
            # Price volatility
            if len(recent_prices) > 1:
                state_vector[12] = np.std(recent_prices) / 100.0
        
        # Price forecasts (next 3 hours)
        price_forecast = self.state.market_data.get("price_forecast", [current_price] * 3)
        for i, price in enumerate(price_forecast[:3]):
            state_vector[13 + i] = price / 100.0
        
        # Grid stability requirements (indices 16-19)
        frequency = self.state.market_data.get("frequency_hz", 50.0)
        state_vector[16] = (frequency - 50.0) / 0.5  # Normalize frequency deviation
        
        voltage = self.state.market_data.get("voltage_pu", 1.0)
        state_vector[17] = voltage - 1.0  # Voltage deviation from nominal
        
        # Market demand and supply (indices 18-23)
        demand_forecast = self.state.market_data.get("demand_forecast", {})
        expected_peak = demand_forecast.get("expected_peak", 1000)
        state_vector[18] = expected_peak / 2000.0  # Normalize
        
        generation_forecast = self.state.market_data.get("generation_forecast", {})
        renewable_output = generation_forecast.get("renewable_total", 500)
        state_vector[19] = renewable_output / 1000.0
        
        # Time-based features (indices 20-23)
        now = datetime.now()
        state_vector[20] = now.hour / 24.0
        state_vector[21] = now.weekday() / 7.0
        state_vector[22] = now.month / 12.0
        
        # Economic indicators (indices 24-31)
        # Price spread analysis
        if len(self.price_history) > 0:
            max_price = max(list(self.price_history)[-24:]) if len(self.price_history) >= 24 else current_price
            min_price = min(list(self.price_history)[-24:]) if len(self.price_history) >= 24 else current_price
            state_vector[24] = (max_price - min_price) / 100.0  # Daily price spread
            state_vector[25] = (current_price - min_price) / (max_price - min_price + 1e-6)  # Relative position
        
        # Degradation cost consideration
        state_vector[26] = self.storage_state.degradation_cost_per_cycle / 1000.0
        
        # Recent performance
        state_vector[27] = self.storage_state.last_price_arbitrage / 100.0
        
        return state_vector.astype(np.float32)
    
    def _decode_action(self, action_tensor: torch.Tensor) -> Dict[str, float]:
        """Decode neural network action into charge/discharge command"""
        action_value = action_tensor.item()  # Convert to scalar
        
        # Map action from [-1, 1] to actual charge/discharge rate
        max_charge_rate = self._calculate_max_charge_rate()
        max_discharge_rate = self._calculate_max_discharge_rate()
        
        if action_value > 0:
            # Charging
            charge_rate = action_value * max_charge_rate
            action_type = "charge"
        elif action_value < 0:
            # Discharging
            charge_rate = action_value * max_discharge_rate  # Negative value
            action_type = "discharge"
        else:
            # Idle
            charge_rate = 0.0
            action_type = "idle"
        
        return {
            "action_type": action_type,
            "charge_rate_mw": charge_rate,
            "power_mw": abs(charge_rate),
            "action_value": action_value
        }
    
    def _calculate_max_charge_rate(self) -> float:
        """Calculate maximum charging rate based on current SoC and constraints"""
        remaining_capacity = (100.0 - self.storage_state.state_of_charge_percent) / 100.0 * self.storage_state.max_capacity_mwh
        max_power_limited = min(self.storage_state.max_power_mw, remaining_capacity)
        
        # Temperature derating
        temp_factor = 1.0 - max(0.0, (self.storage_state.temperature_celsius - 35.0) / 50.0) * 0.2
        
        return max_power_limited * temp_factor
    
    def _calculate_max_discharge_rate(self) -> float:
        """Calculate maximum discharge rate based on current SoC and constraints"""
        available_energy = self.storage_state.state_of_charge_percent / 100.0 * self.storage_state.max_capacity_mwh
        max_power_limited = min(self.storage_state.max_power_mw, available_energy)
        
        # Temperature derating
        temp_factor = 1.0 - max(0.0, (self.storage_state.temperature_celsius - 35.0) / 50.0) * 0.2
        
        return max_power_limited * temp_factor
    
    def _calculate_value_function(self, immediate_profit: float, expected_future_value: float) -> float:
        """Calculate value function: Immediate_Profit + γ × Expected_Future_Value - Degradation_Cost"""
        
        # Calculate degradation cost based on current action
        if abs(self.storage_state.charge_rate_mw) > 0:
            cycle_fraction = abs(self.storage_state.charge_rate_mw) / self.storage_state.max_capacity_mwh
            degradation_cost = cycle_fraction * self.storage_state.degradation_cost_per_cycle
        else:
            degradation_cost = 0.0
        
        value = immediate_profit + self.gamma * expected_future_value - degradation_cost
        return value
    
    def _calculate_reward(self, action: Dict[str, float], market_result: Dict[str, Any]) -> float:
        """Calculate immediate reward for the action taken"""
        current_price = market_result.get("clearing_price_mwh", 50.0)
        power_mw = action["power_mw"]
        
        # Revenue/cost calculation
        if action["action_type"] == "charge":
            # Negative revenue (cost) when charging
            immediate_profit = -power_mw * current_price
        elif action["action_type"] == "discharge":
            # Positive revenue when discharging
            immediate_profit = power_mw * current_price * self.storage_state.round_trip_efficiency
        else:
            immediate_profit = 0.0
        
        # Degradation cost
        if power_mw > 0:
            cycle_fraction = power_mw / self.storage_state.max_capacity_mwh
            degradation_cost = cycle_fraction * self.storage_state.degradation_cost_per_cycle
        else:
            degradation_cost = 0.0
        
        # Grid services bonus (frequency regulation)
        frequency_deviation = abs(market_result.get("frequency_hz", 50.0) - 50.0)
        if frequency_deviation > 0.05 and action["action_type"] != "idle":
            grid_service_bonus = power_mw * 10.0  # Bonus for grid stability services
        else:
            grid_service_bonus = 0.0
        
        reward = immediate_profit - degradation_cost + grid_service_bonus
        
        # Track revenue and costs
        self.total_revenue += immediate_profit
        self.total_degradation_cost += degradation_cost
        self.storage_state.last_price_arbitrage = immediate_profit
        
        return reward / 100.0  # Scale for numerical stability
    
    def _update_storage_state(self, action: Dict[str, float], timestep_hours: float = 1.0) -> None:
        """Update storage state based on action taken"""
        power_mw = action["power_mw"]
        
        if action["action_type"] == "charge":
            # Charging: increase SoC
            energy_charged = power_mw * timestep_hours * self.storage_state.round_trip_efficiency
            soc_increase = (energy_charged / self.storage_state.max_capacity_mwh) * 100.0
            self.storage_state.state_of_charge_percent = min(100.0, 
                self.storage_state.state_of_charge_percent + soc_increase)
            self.storage_state.charge_rate_mw = power_mw
            
        elif action["action_type"] == "discharge":
            # Discharging: decrease SoC
            energy_discharged = power_mw * timestep_hours
            soc_decrease = (energy_discharged / self.storage_state.max_capacity_mwh) * 100.0
            self.storage_state.state_of_charge_percent = max(0.0, 
                self.storage_state.state_of_charge_percent - soc_decrease)
            self.storage_state.charge_rate_mw = -power_mw
            
        else:  # idle
            self.storage_state.charge_rate_mw = 0.0
        
        # Update cycle count
        if power_mw > 0:
            cycle_fraction = power_mw * timestep_hours / self.storage_state.max_capacity_mwh
            self.storage_state.cycles_completed += cycle_fraction
        
        self.storage_state.last_action = action["action_type"]
    
    async def analyze_market_data(self) -> Dict[str, Any]:
        """Analyze market conditions for storage decision making"""
        current_price = self.state.market_data.get("current_price", 50.0)
        self.price_history.append(current_price)
        
        # Price spread analysis
        if len(self.price_history) >= 24:
            recent_prices = list(self.price_history)[-24:]
            max_price = max(recent_prices)
            min_price = min(recent_prices)
            price_spread = max_price - min_price
            
            # Identify arbitrage opportunities
            arbitrage_potential = price_spread * self.storage_state.round_trip_efficiency - \
                                (self.storage_state.degradation_cost_per_cycle / self.storage_state.max_capacity_mwh)
        else:
            price_spread = 0.0
            arbitrage_potential = 0.0
        
        self.arbitrage_opportunities.append(arbitrage_potential)
        
        # Market trend analysis
        if len(self.price_history) > 3:
            recent_trend = np.polyfit(range(3), list(self.price_history)[-3:], 1)[0]
        else:
            recent_trend = 0.0
        
        return {
            "price_spread_24h": price_spread,
            "arbitrage_potential": arbitrage_potential,
            "price_trend": recent_trend,
            "current_soc": self.storage_state.state_of_charge_percent,
            "charge_capacity_available": self._calculate_max_charge_rate(),
            "discharge_capacity_available": self._calculate_max_discharge_rate(),
            "cycles_remaining": max(0, 8000 - self.storage_state.cycles_completed)  # Typical battery lifetime
        }
    
    async def make_strategic_decision(self, state: AgentState) -> Dict[str, Any]:
        """Make strategic decisions using Actor-Critic"""
        state_vector = self._encode_state_vector()
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        
        # Get action from actor network
        with torch.no_grad():
            action_tensor = self.actor(state_tensor)
            
        # Add exploration noise
        noise = self.noise.sample()
        action_tensor_noisy = action_tensor + torch.FloatTensor(noise).unsqueeze(0)
        action_tensor_noisy = torch.clamp(action_tensor_noisy, -1.0, 1.0)
        
        action = self._decode_action(action_tensor_noisy)
        
        # Store current state and action for learning
        self.current_state = state_vector
        self.current_action = action_tensor_noisy.squeeze(0).numpy()
        
        return {
            "action_type": action["action_type"],
            "charge_rate_mw": action["charge_rate_mw"],
            "power_mw": action["power_mw"],
            "action_value": action["action_value"],
            "reasoning": f"Actor-Critic decision: {action['action_type']} at {action['power_mw']:.2f} MW",
            "state_vector": state_vector.tolist(),
            "current_soc": self.storage_state.state_of_charge_percent
        }
    
    async def execute_decision(self, decision: Dict[str, Any]) -> None:
        """Execute the strategic decision"""
        # Update storage state
        self._update_storage_state(decision)
        
        # Send status update to grid operator
        await self.send_message(
            receiver_id="grid_operator",
            message_type=MessageType.STATUS_UPDATE,
            content={
                "storage_id": self.agent_id,
                "state_of_charge_percent": self.storage_state.state_of_charge_percent,
                "charge_rate_mw": self.storage_state.charge_rate_mw,
                "available_capacity_mw": self._calculate_max_discharge_rate(),
                "available_charging_mw": self._calculate_max_charge_rate(),
                "action_type": decision["action_type"],
                "power_mw": decision["power_mw"]
            }
        )
        
        # If participating in frequency regulation
        frequency = self.state.market_data.get("frequency_hz", 50.0)
        if abs(frequency - 50.0) > 0.05:
            await self.send_message(
                receiver_id="grid_operator",
                message_type=MessageType.STATUS_UPDATE,
                content={
                    "service_type": "frequency_regulation",
                    "available_power_mw": self.storage_state.max_power_mw,
                    "response_time_seconds": 1.0,
                    "duration_minutes": 15
                }
            )
    
    def learn_from_market_result(self, market_result: Dict[str, Any]) -> None:
        """Learn from market results using Actor-Critic temporal difference learning"""
        if not hasattr(self, 'current_state') or not hasattr(self, 'current_action'):
            return
        
        # Calculate reward
        action_dict = {
            "action_type": self.storage_state.last_action,
            "power_mw": abs(self.storage_state.charge_rate_mw)
        }
        reward = self._calculate_reward(action_dict, market_result)
        
        # Get next state
        next_state = self._encode_state_vector()
        
        # Store experience
        self.replay_buffer.append((
            self.current_state,
            self.current_action,
            reward,
            next_state,
            False  # Not terminal
        ))
        
        # Train networks
        if len(self.replay_buffer) > self.batch_size:
            self._train_actor_critic()
        
        # Update operational status
        self.state.operational_status.update({
            "state_of_charge": self.storage_state.state_of_charge_percent,
            "charge_rate_mw": self.storage_state.charge_rate_mw,
            "last_reward": reward,
            "total_revenue": self.total_revenue,
            "total_degradation_cost": self.total_degradation_cost,
            "cycles_completed": self.storage_state.cycles_completed
        })
    
    def _train_actor_critic(self) -> None:
        """Train Actor-Critic networks using temporal difference learning"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = list(self.replay_buffer)[-self.batch_size:]
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.FloatTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        
        # Train Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards.unsqueeze(1) + self.gamma * target_q_values
        
        current_q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Train Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    async def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate storage-specific performance metrics"""
        base_metrics = await super().calculate_performance_metrics()
        
        # Calculate storage-specific metrics
        net_revenue = self.total_revenue - self.total_degradation_cost
        
        # Utilization metrics
        if len(self.price_history) > 0:
            price_volatility = np.std(list(self.price_history))
        else:
            price_volatility = 0.0
        
        storage_metrics = {
            "state_of_charge_percent": self.storage_state.state_of_charge_percent,
            "total_cycles_completed": self.storage_state.cycles_completed,
            "net_revenue": net_revenue,
            "total_revenue": self.total_revenue,
            "total_degradation_cost": self.total_degradation_cost,
            "round_trip_efficiency": self.storage_state.round_trip_efficiency,
            "capacity_utilization": abs(self.storage_state.charge_rate_mw) / self.storage_state.max_power_mw,
            "arbitrage_efficiency": net_revenue / (self.storage_state.cycles_completed * self.storage_state.max_capacity_mwh + 1e-6),
            "price_volatility_captured": price_volatility
        }
        
        base_metrics.update(storage_metrics)
        return base_metrics 