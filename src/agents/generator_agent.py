import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
from collections import deque

from .base_agent import BaseAgent, AgentType, AgentState, MessageType


@dataclass
class GeneratorState:
    """Generator-specific operational state"""
    online_status: bool = False
    current_output_mw: float = 0.0
    max_capacity_mw: float = 100.0
    min_output_mw: float = 10.0
    ramp_rate_mw_per_min: float = 10.0
    fuel_cost_per_mwh: float = 50.0
    startup_cost: float = 5000.0
    shutdown_cost: float = 1000.0
    efficiency: float = 0.40
    emissions_rate_kg_co2_per_mwh: float = 350.0
    maintenance_hours_remaining: int = 0
    last_bid_price: float = 0.0
    last_bid_quantity: float = 0.0


class DQNNetwork(nn.Module):
    """Deep Q-Network for generator decision making"""
    
    def __init__(self, state_size: int = 64, action_size: int = 20):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return torch.softmax(self.fc4(x), dim=-1)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio


class GeneratorAgent(BaseAgent):
    """Generator agent using DQN for decision making"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        super().__init__(agent_id, AgentType.GENERATOR, config)
        
        # Generator-specific state
        self.generator_state = GeneratorState()
        if config:
            for key, value in config.items():
                if hasattr(self.generator_state, key):
                    setattr(self.generator_state, key, value)
        
        # DQN components
        self.state_size = 64
        self.action_size = 20  # Discrete action space for bid prices/quantities
        self.q_network = DQNNetwork(self.state_size, self.action_size)
        self.target_network = DQNNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer()
        
        # Training parameters
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.95  # Discount factor
        self.tau = 0.001  # Soft update rate
        self.update_frequency = 4
        self.step_count = 0
        
        # Reward function weights (as specified in requirements)
        self.alpha_1 = 1.0  # Revenue weight
        self.alpha_2 = 1.0  # Cost weight
        self.alpha_3 = 0.5  # Grid stability penalty weight
        self.alpha_4 = 0.3  # Environmental bonus weight
        
        # Historical data for state representation
        self.price_history = deque(maxlen=48)  # 48-hour price history
        self.demand_history = deque(maxlen=24)  # 24-hour demand history
        
    def _encode_state_vector(self) -> np.ndarray:
        """Encode current state into neural network input vector (64 dimensions)"""
        state_vector = np.zeros(self.state_size)
        
        # Market price information (indices 0-11)
        current_price = self.state.market_data.get("current_price", 50.0)
        state_vector[0] = current_price / 100.0  # Normalize price
        
        # Price history (last 8 hours)
        for i, price in enumerate(list(self.price_history)[-8:]):
            if i < 8:
                state_vector[1 + i] = price / 100.0
        
        # Price forecast (next 3 hours)
        price_forecast = self.state.market_data.get("price_forecast", [current_price] * 3)
        for i, price in enumerate(price_forecast[:3]):
            state_vector[9 + i] = price / 100.0
        
        # Generator operational status (indices 12-23)
        state_vector[12] = float(self.generator_state.online_status)
        state_vector[13] = self.generator_state.current_output_mw / self.generator_state.max_capacity_mw
        state_vector[14] = self.generator_state.fuel_cost_per_mwh / 100.0
        state_vector[15] = self.generator_state.efficiency
        state_vector[16] = self.generator_state.ramp_rate_mw_per_min / 50.0  # Normalize
        state_vector[17] = min(1.0, self.generator_state.maintenance_hours_remaining / 100.0)
        
        # Grid conditions (indices 18-29)
        demand_forecast = self.state.market_data.get("demand_forecast", {})
        expected_peak = demand_forecast.get("expected_peak", 1000)
        state_vector[18] = expected_peak / 2000.0  # Normalize to typical grid size
        
        # Weather impact (indices 24-31)
        weather = self.state.market_data.get("weather", {})
        state_vector[24] = weather.get("temperature", 20) / 40.0  # Normalize temperature
        state_vector[25] = weather.get("wind_speed", 5) / 20.0
        state_vector[26] = weather.get("solar_irradiance", 500) / 1000.0
        
        # Time-based features (indices 32-39)
        now = datetime.now()
        state_vector[32] = now.hour / 24.0
        state_vector[33] = now.weekday() / 7.0
        state_vector[34] = now.month / 12.0
        
        # Market competition (indices 40-47)
        generation_forecast = self.state.market_data.get("generation_forecast", {})
        renewable_output = generation_forecast.get("renewable_total", 500)
        state_vector[40] = renewable_output / 1000.0
        
        # Economic indicators (indices 48-55)
        state_vector[48] = self.generator_state.last_bid_price / 100.0
        state_vector[49] = self.generator_state.last_bid_quantity / self.generator_state.max_capacity_mw
        
        # Environmental factors (indices 56-63)
        state_vector[56] = self.generator_state.emissions_rate_kg_co2_per_mwh / 1000.0
        carbon_price = self.state.market_data.get("carbon_price", 25.0)
        state_vector[57] = carbon_price / 100.0
        
        return state_vector.astype(np.float32)
    
    def _decode_action(self, action_index: int) -> Dict[str, float]:
        """Decode neural network action into bid price and quantity"""
        # Map action index to bid price and quantity
        # Action space: combinations of 4 price levels and 5 quantity levels
        price_levels = [0.8, 0.9, 1.0, 1.1, 1.2]  # Multipliers of marginal cost
        quantity_levels = [0.0, 0.25, 0.5, 0.75, 1.0]  # Fractions of max capacity
        
        price_idx = action_index // 5
        quantity_idx = action_index % 5
        
        # Calculate marginal cost
        marginal_cost = self.generator_state.fuel_cost_per_mwh / self.generator_state.efficiency
        
        # Add carbon cost if applicable
        carbon_price = self.state.market_data.get("carbon_price", 25.0)
        carbon_cost = carbon_price * self.generator_state.emissions_rate_kg_co2_per_mwh / 1000.0
        marginal_cost += carbon_cost
        
        bid_price = marginal_cost * price_levels[price_idx % len(price_levels)]
        bid_quantity = self.generator_state.max_capacity_mw * quantity_levels[quantity_idx % len(quantity_levels)]
        
        return {
            "bid_price_mwh": bid_price,
            "bid_quantity_mw": bid_quantity,
            "startup_bid": not self.generator_state.online_status and bid_quantity > 0,
            "shutdown_bid": self.generator_state.online_status and bid_quantity == 0
        }
    
    def _calculate_reward(self, action: Dict[str, float], market_result: Dict[str, Any]) -> float:
        """Calculate reward using the specified multi-objective function"""
        # Revenue component
        cleared_quantity = market_result.get("cleared_quantity_mw", 0)
        clearing_price = market_result.get("clearing_price_mwh", 0)
        revenue = cleared_quantity * clearing_price
        
        # Operating costs
        fuel_cost = cleared_quantity * self.generator_state.fuel_cost_per_mwh / self.generator_state.efficiency
        startup_cost = self.generator_state.startup_cost if action.get("startup_bid", False) else 0
        shutdown_cost = self.generator_state.shutdown_cost if action.get("shutdown_bid", False) else 0
        operating_costs = fuel_cost + startup_cost + shutdown_cost
        
        # Grid stability penalty
        frequency_deviation = abs(market_result.get("frequency_hz", 50.0) - 50.0)
        voltage_deviation = abs(market_result.get("voltage_pu", 1.0) - 1.0)
        stability_penalty = (frequency_deviation * 1000 + voltage_deviation * 100) * cleared_quantity
        
        # Environmental bonus
        renewable_penetration = market_result.get("renewable_penetration", 0.3)
        environmental_bonus = renewable_penetration * 10 * cleared_quantity
        
        # Combined reward function
        reward = (self.alpha_1 * revenue - 
                 self.alpha_2 * operating_costs - 
                 self.alpha_3 * stability_penalty + 
                 self.alpha_4 * environmental_bonus)
        
        return reward / 1000.0  # Scale down for numerical stability
    
    async def analyze_market_data(self) -> Dict[str, Any]:
        """Analyze market conditions for generator decision making"""
        current_price = self.state.market_data.get("current_price", 50.0)
        self.price_history.append(current_price)
        
        # Calculate price volatility
        if len(self.price_history) > 1:
            price_volatility = np.std(list(self.price_history))
        else:
            price_volatility = 0.0
        
        # Analyze demand patterns
        demand_forecast = self.state.market_data.get("demand_forecast", {})
        expected_peak = demand_forecast.get("expected_peak", 1000)
        
        # Calculate competitive position
        generation_forecast = self.state.market_data.get("generation_forecast", {})
        total_available = generation_forecast.get("total_available", 1500)
        renewable_output = generation_forecast.get("renewable_total", 500)
        
        return {
            "price_volatility": price_volatility,
            "demand_trend": "increasing" if expected_peak > 1000 else "stable",
            "competitive_pressure": renewable_output / total_available,
            "marginal_cost": self.generator_state.fuel_cost_per_mwh / self.generator_state.efficiency,
            "capacity_utilization": self.generator_state.current_output_mw / self.generator_state.max_capacity_mw
        }
    
    async def make_strategic_decision(self, state: AgentState) -> Dict[str, Any]:
        """Make strategic decisions using DQN"""
        state_vector = self._encode_state_vector()
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action_index = random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action_index = q_values.argmax().item()
        
        action = self._decode_action(action_index)
        
        # Store current state for learning
        self.current_state = state_vector
        self.current_action = action_index
        
        return {
            "action_type": "market_bid",
            "bid_price_mwh": action["bid_price_mwh"],
            "bid_quantity_mw": action["bid_quantity_mw"],
            "startup_bid": action["startup_bid"],
            "shutdown_bid": action["shutdown_bid"],
            "reasoning": f"DQN action {action_index}, Q-value based decision",
            "state_vector": state_vector.tolist(),
            "action_index": action_index
        }
    
    async def execute_decision(self, decision: Dict[str, Any]) -> None:
        """Execute the strategic decision"""
        if decision["action_type"] == "market_bid":
            # Update generator state based on decision
            self.generator_state.last_bid_price = decision["bid_price_mwh"]
            self.generator_state.last_bid_quantity = decision["bid_quantity_mw"]
            
            # Send bid to market operator
            await self.send_message(
                receiver_id="grid_operator",
                message_type=MessageType.GENERATION_BID,
                content={
                    "capacity_available": decision["bid_quantity_mw"],
                    "bid_price": decision["bid_price_mwh"],
                    "ramp_rate": self.generator_state.ramp_rate_mw_per_min,
                    "startup_bid": decision["startup_bid"],
                    "shutdown_bid": decision["shutdown_bid"],
                    "emissions_rate": self.generator_state.emissions_rate_kg_co2_per_mwh,
                    "min_output": self.generator_state.min_output_mw if decision["startup_bid"] else 0
                }
            )
    
    def learn_from_market_result(self, market_result: Dict[str, Any]) -> None:
        """Learn from market clearing results using DQN"""
        if not hasattr(self, 'current_state') or not hasattr(self, 'current_action'):
            return
        
        # Calculate reward
        reward = self._calculate_reward(
            {"bid_price_mwh": self.generator_state.last_bid_price,
             "bid_quantity_mw": self.generator_state.last_bid_quantity},
            market_result
        )
        
        # Get next state
        next_state = self._encode_state_vector()
        
        # Store experience in replay buffer
        self.replay_buffer.add(
            self.current_state,
            self.current_action,
            reward,
            next_state,
            False  # Not terminal
        )
        
        # Train the network
        if len(self.replay_buffer.buffer) > 32:
            self._train_dqn()
        
        # Update generator operational state based on market results
        cleared_quantity = market_result.get("cleared_quantity_mw", 0)
        if cleared_quantity > 0:
            self.generator_state.online_status = True
            self.generator_state.current_output_mw = cleared_quantity
        else:
            self.generator_state.online_status = False
            self.generator_state.current_output_mw = 0.0
        
        # Update operational status in agent state
        self.state.operational_status.update({
            "online": self.generator_state.online_status,
            "output_mw": self.generator_state.current_output_mw,
            "last_reward": reward,
            "capacity_factor": self.generator_state.current_output_mw / self.generator_state.max_capacity_mw
        })
    
    def _train_dqn(self) -> None:
        """Train the DQN using prioritized experience replay"""
        batch_size = 32
        samples, indices, weights = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor([s[0] for s in samples])
        actions = torch.LongTensor([s[1] for s in samples])
        rewards = torch.FloatTensor([s[2] for s in samples])
        next_states = torch.FloatTensor([s[3] for s in samples])
        dones = torch.BoolTensor([s[4] for s in samples])
        weights = torch.FloatTensor(weights)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        td_errors = target_q_values.unsqueeze(1) - current_q_values
        loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update priorities
        priorities = abs(td_errors.detach().numpy().flatten()) + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)
        
        # Soft update target network
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        
        self.step_count += 1
    
    async def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate generator-specific performance metrics"""
        base_metrics = await super().calculate_performance_metrics()
        
        # Calculate generator-specific metrics
        revenue = self.generator_state.current_output_mw * self.generator_state.last_bid_price
        fuel_cost = self.generator_state.current_output_mw * self.generator_state.fuel_cost_per_mwh / self.generator_state.efficiency
        
        generator_metrics = {
            "capacity_factor": self.generator_state.current_output_mw / self.generator_state.max_capacity_mw,
            "revenue_per_hour": revenue,
            "operating_cost_per_hour": fuel_cost,
            "profit_per_hour": revenue - fuel_cost,
            "emissions_per_hour": self.generator_state.current_output_mw * self.generator_state.emissions_rate_kg_co2_per_mwh,
            "online_time_percentage": float(self.generator_state.online_status) * 100,
            "bid_success_rate": 1.0 if self.generator_state.current_output_mw > 0 else 0.0
        }
        
        base_metrics.update(generator_metrics)
        return base_metrics 