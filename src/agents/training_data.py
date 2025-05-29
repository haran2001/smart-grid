"""
Training Data Management System for Smart Grid RL Agents

Provides historical data generation, pre-training capabilities, and data management
for DQN, Actor-Critic, and MADDPG agents.
"""

import numpy as np
import pandas as pd
import torch
import pickle
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import random
from pathlib import Path

@dataclass
class TrainingDataPoint:
    """Single training data point for RL agents"""
    timestamp: datetime
    state_vector: np.ndarray
    action: Any  # Could be int (DQN) or np.ndarray (continuous)
    reward: float
    next_state_vector: np.ndarray
    done: bool
    agent_type: str
    metadata: Dict[str, Any]

@dataclass
class MarketCondition:
    """Market conditions for a given time"""
    timestamp: datetime
    price: float
    demand: float
    renewable_generation: float
    frequency: float
    voltage: float
    weather: Dict[str, float]

class HistoricalDataGenerator:
    """Generate realistic historical data for training"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_market_conditions(self, start_date: datetime, days: int = 365) -> List[MarketCondition]:
        """Generate realistic market conditions for training"""
        conditions = []
        current_time = start_date
        
        for day in range(days):
            for hour in range(24):
                # Base price pattern (peak/off-peak)
                if 6 <= hour <= 10 or 17 <= hour <= 22:  # Peak hours
                    base_price = 80 + np.random.normal(0, 10)
                elif 0 <= hour <= 6:  # Off-peak
                    base_price = 30 + np.random.normal(0, 5)
                else:  # Mid-peak
                    base_price = 50 + np.random.normal(0, 8)
                
                # Seasonal variations
                season_factor = 1.0 + 0.3 * np.sin(2 * np.pi * day / 365)
                price = max(10.0, base_price * season_factor)
                
                # Demand pattern
                daily_pattern = 0.8 + 0.4 * np.sin((hour - 6) * np.pi / 12)
                seasonal_demand = 1000 + 200 * np.sin(2 * np.pi * day / 365)
                demand = seasonal_demand * daily_pattern + np.random.normal(0, 50)
                
                # Renewable generation (weather dependent)
                temperature = 20 + 15 * np.sin(2 * np.pi * day / 365) + np.random.normal(0, 3)
                wind_speed = max(0, 8 + 5 * np.random.normal())
                
                # Solar generation (time and weather dependent)
                if 6 <= hour <= 18:
                    solar_potential = np.sin((hour - 6) * np.pi / 12)
                    cloud_factor = 0.7 + 0.3 * np.random.random()
                    solar_irradiance = 800 * solar_potential * cloud_factor
                else:
                    solar_irradiance = 0
                
                renewable_gen = min(600, solar_irradiance * 0.3 + wind_speed * 15)
                
                # Grid stability (normally stable with occasional disturbances)
                frequency = 50.0 + np.random.normal(0, 0.02)
                if np.random.random() < 0.01:  # 1% chance of disturbance
                    frequency += np.random.normal(0, 0.1)
                
                voltage = 1.0 + np.random.normal(0, 0.01)
                
                conditions.append(MarketCondition(
                    timestamp=current_time,
                    price=price,
                    demand=demand,
                    renewable_generation=renewable_gen,
                    frequency=frequency,
                    voltage=voltage,
                    weather={
                        'temperature': temperature,
                        'wind_speed': wind_speed,
                        'solar_irradiance': solar_irradiance,
                        'humidity': 50 + np.random.normal(0, 10)
                    }
                ))
                
                current_time += timedelta(hours=1)
        
        return conditions
    
    def generate_generator_scenarios(self, market_conditions: List[MarketCondition], 
                                   generator_config: Dict[str, Any]) -> List[TrainingDataPoint]:
        """Generate training scenarios for generator agents"""
        training_data = []
        
        # Generator parameters
        max_capacity = generator_config.get('max_capacity_mw', 100.0)
        fuel_cost = generator_config.get('fuel_cost_per_mwh', 50.0)
        emissions_rate = generator_config.get('emissions_rate_kg_co2_per_mwh', 400.0)
        efficiency = generator_config.get('efficiency', 0.4)
        
        price_history = []
        
        for i, condition in enumerate(market_conditions):
            price_history.append(condition.price)
            if len(price_history) > 48:
                price_history.pop(0)
            
            # Create state vector (similar to actual agent)
            state_vector = self._create_generator_state_vector(
                condition, price_history, generator_config
            )
            
            # Simulate optimal actions based on market conditions
            marginal_cost = fuel_cost / efficiency
            if condition.price > marginal_cost * 1.2:
                # High price - bid aggressively
                action = self._encode_generator_action(condition.price * 0.95, max_capacity * 0.9)
                cleared_quantity = max_capacity * 0.8
            elif condition.price > marginal_cost:
                # Moderate price - conservative bid
                action = self._encode_generator_action(marginal_cost * 1.1, max_capacity * 0.6)
                cleared_quantity = max_capacity * 0.5
            else:
                # Low price - minimal or no bid
                action = self._encode_generator_action(marginal_cost * 1.5, max_capacity * 0.2)
                cleared_quantity = 0
            
            # Calculate reward
            revenue = cleared_quantity * condition.price
            costs = cleared_quantity * fuel_cost / efficiency
            stability_penalty = abs(condition.frequency - 50.0) * 100 * cleared_quantity
            reward = (revenue - costs - stability_penalty) / 1000.0
            
            # Next state (next hour)
            if i < len(market_conditions) - 1:
                next_condition = market_conditions[i + 1]
                next_price_history = price_history + [next_condition.price]
                if len(next_price_history) > 48:
                    next_price_history.pop(0)
                next_state_vector = self._create_generator_state_vector(
                    next_condition, next_price_history, generator_config
                )
            else:
                next_state_vector = state_vector
            
            training_data.append(TrainingDataPoint(
                timestamp=condition.timestamp,
                state_vector=state_vector,
                action=action,
                reward=reward,
                next_state_vector=next_state_vector,
                done=False,
                agent_type='generator',
                metadata={
                    'price': condition.price,
                    'cleared_quantity': cleared_quantity,
                    'marginal_cost': marginal_cost
                }
            ))
        
        return training_data
    
    def generate_storage_scenarios(self, market_conditions: List[MarketCondition],
                                 storage_config: Dict[str, Any]) -> List[TrainingDataPoint]:
        """Generate training scenarios for storage agents"""
        training_data = []
        
        # Storage parameters
        max_capacity_mwh = storage_config.get('max_capacity_mwh', 100.0)
        max_power_mw = storage_config.get('max_power_mw', 25.0)
        efficiency = storage_config.get('round_trip_efficiency', 0.9)
        degradation_cost = storage_config.get('degradation_cost_per_cycle', 50.0)
        
        soc = 50.0  # Start at 50% SoC
        price_history = []
        
        for i, condition in enumerate(market_conditions):
            price_history.append(condition.price)
            if len(price_history) > 48:
                price_history.pop(0)
            
            # Create state vector
            state_vector = self._create_storage_state_vector(
                condition, price_history, soc, storage_config
            )
            
            # Simulate optimal storage strategy (price arbitrage)
            if len(price_history) >= 24:
                recent_prices = price_history[-24:]
                avg_price = np.mean(recent_prices)
                price_percentile = len([p for p in recent_prices if p <= condition.price]) / len(recent_prices)
            else:
                avg_price = condition.price
                price_percentile = 0.5
            
            # Storage decision logic
            if condition.price < avg_price * 0.8 and soc < 90:
                # Low price - charge
                charge_power = min(max_power_mw, (90 - soc) / 100 * max_capacity_mwh)
                action = np.array([charge_power / max_power_mw])  # Normalized
                soc = min(90, soc + (charge_power * efficiency / max_capacity_mwh) * 100)
                reward = -charge_power * condition.price / 100.0
            elif condition.price > avg_price * 1.2 and soc > 20:
                # High price - discharge
                discharge_power = min(max_power_mw, (soc - 20) / 100 * max_capacity_mwh)
                action = np.array([-discharge_power / max_power_mw])  # Negative for discharge
                soc = max(20, soc - (discharge_power / max_capacity_mwh) * 100)
                reward = discharge_power * condition.price * efficiency / 100.0
            else:
                # Hold
                action = np.array([0.0])
                reward = 0.0
            
            # Add degradation cost
            if abs(action[0]) > 0:
                cycle_fraction = abs(action[0])
                reward -= cycle_fraction * degradation_cost / 100.0
            
            # Next state
            if i < len(market_conditions) - 1:
                next_condition = market_conditions[i + 1]
                next_price_history = price_history + [next_condition.price]
                if len(next_price_history) > 48:
                    next_price_history.pop(0)
                next_state_vector = self._create_storage_state_vector(
                    next_condition, next_price_history, soc, storage_config
                )
            else:
                next_state_vector = state_vector
            
            training_data.append(TrainingDataPoint(
                timestamp=condition.timestamp,
                state_vector=state_vector,
                action=action,
                reward=reward,
                next_state_vector=next_state_vector,
                done=False,
                agent_type='storage',
                metadata={
                    'soc': soc,
                    'price_percentile': price_percentile,
                    'action_type': 'charge' if action[0] > 0 else 'discharge' if action[0] < 0 else 'hold'
                }
            ))
        
        return training_data
    
    def generate_consumer_scenarios(self, market_conditions: List[MarketCondition],
                                  consumer_config: Dict[str, Any]) -> List[TrainingDataPoint]:
        """Generate training scenarios for consumer agents"""
        training_data = []
        
        # Consumer parameters
        baseline_load = consumer_config.get('baseline_load_mw', 50.0)
        flexible_load = consumer_config.get('flexible_load_mw', 15.0)
        comfort_preference = consumer_config.get('comfort_preference', 80.0)
        
        price_history = []
        comfort_level = comfort_preference
        
        for i, condition in enumerate(market_conditions):
            price_history.append(condition.price)
            if len(price_history) > 48:
                price_history.pop(0)
            
            # Create state vector
            state_vector = self._create_consumer_state_vector(
                condition, price_history, comfort_level, consumer_config
            )
            
            # Simulate optimal consumer response
            if len(price_history) >= 24:
                recent_prices = price_history[-24:]
                avg_price = np.mean(recent_prices)
                high_price_threshold = avg_price * 1.3
                dr_price = max(100.0, avg_price * 2.0)
            else:
                avg_price = condition.price
                high_price_threshold = condition.price * 1.3
                dr_price = 100.0
            
            # Consumer decision logic
            if condition.price > high_price_threshold:
                # High price - participate in DR
                dr_participation = min(0.8, (condition.price - avg_price) / avg_price)
                comfort_impact = dr_participation * 10
                comfort_level = max(60, comfort_preference - comfort_impact)
            else:
                # Normal price - minimal DR
                dr_participation = 0.1
                comfort_level = min(comfort_preference, comfort_level + 2)
            
            # Multi-dimensional action [DR, EV charging, HVAC, battery]
            action = np.array([
                dr_participation,
                0.5 - dr_participation * 0.3,  # Reduce EV charging when DR active
                0.5 + dr_participation * 0.2,  # Adjust HVAC when DR active
                dr_participation * 0.5  # Use battery during DR
            ])
            
            # Calculate utility (reward)
            energy_cost = baseline_load * (1 - dr_participation * 0.4) * condition.price / 1000
            dr_payment = dr_participation * flexible_load * dr_price / 1000
            comfort_score = comfort_level
            inconvenience = dr_participation * 20
            
            utility = comfort_score - energy_cost + dr_payment - inconvenience
            reward = utility / 100.0
            
            # Update comfort for next iteration
            comfort_level = max(60, min(100, comfort_level))
            
            # Next state
            if i < len(market_conditions) - 1:
                next_condition = market_conditions[i + 1]
                next_price_history = price_history + [next_condition.price]
                if len(next_price_history) > 48:
                    next_price_history.pop(0)
                next_state_vector = self._create_consumer_state_vector(
                    next_condition, next_price_history, comfort_level, consumer_config
                )
            else:
                next_state_vector = state_vector
            
            training_data.append(TrainingDataPoint(
                timestamp=condition.timestamp,
                state_vector=state_vector,
                action=action,
                reward=reward,
                next_state_vector=next_state_vector,
                done=False,
                agent_type='consumer',
                metadata={
                    'dr_participation': dr_participation,
                    'comfort_level': comfort_level,
                    'energy_cost': energy_cost,
                    'dr_payment': dr_payment
                }
            ))
        
        return training_data
    
    def _create_generator_state_vector(self, condition: MarketCondition, 
                                     price_history: List[float], 
                                     config: Dict[str, Any]) -> np.ndarray:
        """Create state vector for generator agent"""
        state_vector = np.zeros(64)
        
        # Current price and history
        state_vector[0] = condition.price / 100.0
        for i, price in enumerate(price_history[-8:]):
            if i < 8:
                state_vector[1 + i] = price / 100.0
        
        # Generator parameters
        state_vector[12] = 1.0  # Assume online
        state_vector[13] = 0.5  # Assume 50% current output
        state_vector[14] = config.get('fuel_cost_per_mwh', 50.0) / 100.0
        state_vector[15] = config.get('efficiency', 0.4)
        
        # Grid conditions
        state_vector[18] = condition.demand / 2000.0
        state_vector[24] = condition.weather['temperature'] / 40.0
        state_vector[25] = condition.weather['wind_speed'] / 20.0
        state_vector[26] = condition.weather['solar_irradiance'] / 1000.0
        
        # Time features
        hour = condition.timestamp.hour
        state_vector[32] = hour / 24.0
        state_vector[33] = condition.timestamp.weekday() / 7.0
        state_vector[34] = condition.timestamp.month / 12.0
        
        # Environmental
        state_vector[56] = config.get('emissions_rate_kg_co2_per_mwh', 400.0) / 1000.0
        
        return state_vector.astype(np.float32)
    
    def _create_storage_state_vector(self, condition: MarketCondition,
                                   price_history: List[float],
                                   soc: float,
                                   config: Dict[str, Any]) -> np.ndarray:
        """Create state vector for storage agent"""
        state_vector = np.zeros(32)
        
        # Storage state
        state_vector[0] = soc / 100.0
        state_vector[1] = 0.0  # No current charge rate initially
        
        # Price information
        state_vector[5] = condition.price / 100.0
        for i, price in enumerate(price_history[-6:]):
            if i < 6:
                state_vector[6 + i] = price / 100.0
        
        # Grid conditions
        state_vector[16] = (condition.frequency - 50.0) / 0.5
        state_vector[17] = condition.voltage - 1.0
        state_vector[18] = condition.demand / 2000.0
        state_vector[19] = condition.renewable_generation / 1000.0
        
        # Time features
        hour = condition.timestamp.hour
        state_vector[20] = hour / 24.0
        state_vector[21] = condition.timestamp.weekday() / 7.0
        state_vector[22] = condition.timestamp.month / 12.0
        
        return state_vector.astype(np.float32)
    
    def _create_consumer_state_vector(self, condition: MarketCondition,
                                    price_history: List[float],
                                    comfort_level: float,
                                    config: Dict[str, Any]) -> np.ndarray:
        """Create state vector for consumer agent"""
        state_vector = np.zeros(40)
        
        # Consumer state
        state_vector[0] = config.get('baseline_load_mw', 50.0) / 100.0
        state_vector[1] = config.get('flexible_load_mw', 15.0) / 50.0
        state_vector[2] = comfort_level / 100.0
        state_vector[3] = config.get('comfort_preference', 80.0) / 100.0
        
        # Price information
        state_vector[10] = condition.price / 100.0
        for i, price in enumerate(price_history[-6:]):
            if i < 6:
                state_vector[11 + i] = price / 100.0
        
        # Weather
        state_vector[20] = condition.weather['temperature'] / 40.0
        state_vector[21] = condition.weather['humidity'] / 100.0
        
        # Time features
        hour = condition.timestamp.hour
        state_vector[24] = hour / 24.0
        state_vector[25] = condition.timestamp.weekday() / 7.0
        state_vector[26] = condition.timestamp.month / 12.0
        state_vector[27] = 1.0 if condition.timestamp.weekday() >= 5 else 0.0
        
        return state_vector.astype(np.float32)
    
    def _encode_generator_action(self, bid_price: float, bid_quantity: float) -> int:
        """Encode generator bid into discrete action index"""
        # Simple encoding for 20 actions (4x5 grid)
        price_levels = 4  # 0, 1, 2, 3
        quantity_levels = 5  # 0, 1, 2, 3, 4
        
        # Normalize and discretize price (relative to marginal cost)
        price_ratio = max(0.5, min(2.0, bid_price / 50.0))  # Clamp to reasonable range
        price_idx = min(3, max(0, int((price_ratio - 0.5) / 0.375 * 3)))  # Map to 0-3
        
        # Normalize and discretize quantity
        quantity_ratio = max(0.0, min(1.0, bid_quantity / 100.0))  # Normalize to 0-1
        quantity_idx = min(4, max(0, int(quantity_ratio * 4)))  # Map to 0-4
        
        # Combine into single action index (max 19)
        action = price_idx * 5 + quantity_idx
        return min(19, max(0, action))  # Ensure within valid range


class TrainingDataManager:
    """Manage training data for RL agents"""
    
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def save_training_data(self, training_data: List[TrainingDataPoint], 
                          filename: str) -> None:
        """Save training data to file"""
        filepath = self.data_dir / filename
        
        # Convert to pandas DataFrame for easier handling
        data_dict = {
            'timestamp': [dp.timestamp for dp in training_data],
            'state_vector': [dp.state_vector for dp in training_data],
            'action': [dp.action for dp in training_data],
            'reward': [dp.reward for dp in training_data],
            'next_state_vector': [dp.next_state_vector for dp in training_data],
            'done': [dp.done for dp in training_data],
            'agent_type': [dp.agent_type for dp in training_data],
            'metadata': [dp.metadata for dp in training_data]
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Saved {len(training_data)} training samples to {filepath}")
    
    def load_training_data(self, filename: str) -> List[TrainingDataPoint]:
        """Load training data from file"""
        filepath = self.data_dir / filename
        
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        
        training_data = []
        for i in range(len(data_dict['timestamp'])):
            training_data.append(TrainingDataPoint(
                timestamp=data_dict['timestamp'][i],
                state_vector=data_dict['state_vector'][i],
                action=data_dict['action'][i],
                reward=data_dict['reward'][i],
                next_state_vector=data_dict['next_state_vector'][i],
                done=data_dict['done'][i],
                agent_type=data_dict['agent_type'][i],
                metadata=data_dict['metadata'][i]
            ))
        
        print(f"Loaded {len(training_data)} training samples from {filepath}")
        return training_data
    
    def create_training_dataset(self, days: int = 365, save: bool = True) -> Dict[str, List[TrainingDataPoint]]:
        """Create a complete training dataset"""
        print(f"Generating {days} days of training data...")
        
        # Generate market conditions
        generator = HistoricalDataGenerator()
        start_date = datetime(2023, 1, 1)
        market_conditions = generator.generate_market_conditions(start_date, days)
        
        # Generate training data for each agent type
        datasets = {}
        
        # Generator scenarios
        generator_configs = [
            {'max_capacity_mw': 200, 'fuel_cost_per_mwh': 60, 'emissions_rate_kg_co2_per_mwh': 800, 'efficiency': 0.35},  # Coal
            {'max_capacity_mw': 150, 'fuel_cost_per_mwh': 80, 'emissions_rate_kg_co2_per_mwh': 400, 'efficiency': 0.5},   # Gas
            {'max_capacity_mw': 100, 'fuel_cost_per_mwh': 0, 'emissions_rate_kg_co2_per_mwh': 0, 'efficiency': 1.0},      # Solar
        ]
        
        for i, config in enumerate(generator_configs):
            gen_data = generator.generate_generator_scenarios(market_conditions, config)
            datasets[f'generator_{i}'] = gen_data
            if save:
                self.save_training_data(gen_data, f'generator_{i}_training.pkl')
        
        # Storage scenarios
        storage_configs = [
            {'max_capacity_mwh': 200, 'max_power_mw': 50, 'round_trip_efficiency': 0.9},
            {'max_capacity_mwh': 100, 'max_power_mw': 25, 'round_trip_efficiency': 0.88}
        ]
        
        for i, config in enumerate(storage_configs):
            storage_data = generator.generate_storage_scenarios(market_conditions, config)
            datasets[f'storage_{i}'] = storage_data
            if save:
                self.save_training_data(storage_data, f'storage_{i}_training.pkl')
        
        # Consumer scenarios
        consumer_configs = [
            {'baseline_load_mw': 100, 'flexible_load_mw': 30, 'comfort_preference': 75},  # Industrial
            {'baseline_load_mw': 50, 'flexible_load_mw': 15, 'comfort_preference': 80},   # Commercial
            {'baseline_load_mw': 25, 'flexible_load_mw': 8, 'comfort_preference': 85}     # Residential
        ]
        
        for i, config in enumerate(consumer_configs):
            consumer_data = generator.generate_consumer_scenarios(market_conditions, config)
            datasets[f'consumer_{i}'] = consumer_data
            if save:
                self.save_training_data(consumer_data, f'consumer_{i}_training.pkl')
        
        print(f"Generated training datasets for {len(datasets)} agent configurations")
        return datasets
    
    def get_statistics(self, training_data: List[TrainingDataPoint]) -> Dict[str, Any]:
        """Get statistics about training data"""
        if not training_data:
            return {}
        
        rewards = [dp.reward for dp in training_data]
        
        stats = {
            'total_samples': len(training_data),
            'agent_type': training_data[0].agent_type,
            'time_span': {
                'start': min(dp.timestamp for dp in training_data),
                'end': max(dp.timestamp for dp in training_data)
            },
            'reward_stats': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'percentiles': {
                    '25': np.percentile(rewards, 25),
                    '50': np.percentile(rewards, 50),
                    '75': np.percentile(rewards, 75)
                }
            },
            'state_vector_size': len(training_data[0].state_vector),
            'action_info': {
                'type': type(training_data[0].action).__name__,
                'shape': getattr(training_data[0].action, 'shape', 'scalar')
            }
        }
        
        return stats


def create_training_data():
    """Main function to create training data"""
    manager = TrainingDataManager()
    datasets = manager.create_training_dataset(days=365, save=True)
    
    # Print statistics for each dataset
    for name, data in datasets.items():
        print(f"\n{name.upper()} Dataset Statistics:")
        stats = manager.get_statistics(data)
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    return datasets


if __name__ == "__main__":
    create_training_data() 