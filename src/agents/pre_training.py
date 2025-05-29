"""
Pre-training Module for Smart Grid RL Agents

Provides pre-training capabilities for DQN, Actor-Critic, and MADDPG agents
using historical data before deployment in live simulation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from .training_data import TrainingDataManager, TrainingDataPoint
from .generator_agent import GeneratorAgent, DQNNetwork
from .storage_agent import StorageAgent, ActorNetwork, CriticNetwork
from .consumer_agent import ConsumerAgent, MADDPGActor, MADDPGCritic


class AgentPreTrainer:
    """Pre-train RL agents using historical data"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.data_manager = TrainingDataManager()
        
    def pretrain_generator_agent(self, agent: GeneratorAgent, 
                               training_data: List[TrainingDataPoint],
                               epochs: int = 100,
                               batch_size: int = 64,
                               learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """Pre-train a generator agent using DQN"""
        
        print(f"Pre-training generator agent with {len(training_data)} samples...")
        
        # Filter for generator data
        gen_data = [dp for dp in training_data if dp.agent_type == 'generator']
        if not gen_data:
            raise ValueError("No generator training data found")
        
        # Setup optimizer
        optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=learning_rate)
        
        # Training metrics
        losses = []
        rewards = []
        
        # Training loop
        for epoch in tqdm(range(epochs), desc="Training Generator"):
            epoch_losses = []
            epoch_rewards = []
            
            # Shuffle data
            random.shuffle(gen_data)
            
            # Batch training
            for i in range(0, len(gen_data) - batch_size, batch_size):
                batch = gen_data[i:i + batch_size]
                
                # Prepare batch tensors
                states = torch.FloatTensor([dp.state_vector for dp in batch]).to(self.device)
                actions = torch.LongTensor([dp.action for dp in batch]).to(self.device)
                rewards_batch = torch.FloatTensor([dp.reward for dp in batch]).to(self.device)
                next_states = torch.FloatTensor([dp.next_state_vector for dp in batch]).to(self.device)
                dones = torch.BoolTensor([dp.done for dp in batch]).to(self.device)
                
                # Forward pass
                current_q_values = agent.q_network(states).gather(1, actions.unsqueeze(1))
                next_q_values = agent.target_network(next_states).max(1)[0].detach()
                target_q_values = rewards_batch + (agent.gamma * next_q_values * ~dones)
                
                # Compute loss
                loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                epoch_rewards.append(rewards_batch.mean().item())
            
            # Update target network
            if epoch % 10 == 0:
                agent.target_network.load_state_dict(agent.q_network.state_dict())
            
            avg_loss = np.mean(epoch_losses)
            avg_reward = np.mean(epoch_rewards)
            losses.append(avg_loss)
            rewards.append(avg_reward)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Avg Reward = {avg_reward:.4f}")
        
        print("Generator pre-training completed!")
        return {"losses": losses, "rewards": rewards}
    
    def pretrain_storage_agent(self, agent: StorageAgent,
                             training_data: List[TrainingDataPoint],
                             epochs: int = 100,
                             batch_size: int = 64,
                             learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """Pre-train a storage agent using Actor-Critic"""
        
        print(f"Pre-training storage agent with {len(training_data)} samples...")
        
        # Filter for storage data
        storage_data = [dp for dp in training_data if dp.agent_type == 'storage']
        if not storage_data:
            raise ValueError("No storage training data found")
        
        # Setup optimizers
        actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=learning_rate)
        critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=learning_rate * 2)
        
        # Training metrics
        actor_losses = []
        critic_losses = []
        rewards = []
        
        # Training loop
        for epoch in tqdm(range(epochs), desc="Training Storage"):
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_rewards = []
            
            # Shuffle data
            random.shuffle(storage_data)
            
            # Batch training
            for i in range(0, len(storage_data) - batch_size, batch_size):
                batch = storage_data[i:i + batch_size]
                
                # Prepare batch tensors
                states = torch.FloatTensor([dp.state_vector for dp in batch]).to(self.device)
                actions = torch.FloatTensor([dp.action for dp in batch]).to(self.device)
                rewards_batch = torch.FloatTensor([dp.reward for dp in batch]).to(self.device)
                next_states = torch.FloatTensor([dp.next_state_vector for dp in batch]).to(self.device)
                
                # Train Critic
                with torch.no_grad():
                    next_actions = agent.actor_target(next_states)
                    target_q_values = agent.critic_target(next_states, next_actions)
                    target_q_values = rewards_batch.unsqueeze(1) + agent.gamma * target_q_values
                
                current_q_values = agent.critic(states, actions)
                critic_loss = F.mse_loss(current_q_values, target_q_values)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # Train Actor
                predicted_actions = agent.actor(states)
                actor_loss = -agent.critic(states, predicted_actions).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # Soft update target networks
                for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)
                
                for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)
                
                epoch_actor_losses.append(actor_loss.item())
                epoch_critic_losses.append(critic_loss.item())
                epoch_rewards.append(rewards_batch.mean().item())
            
            avg_actor_loss = np.mean(epoch_actor_losses)
            avg_critic_loss = np.mean(epoch_critic_losses)
            avg_reward = np.mean(epoch_rewards)
            
            actor_losses.append(avg_actor_loss)
            critic_losses.append(avg_critic_loss)
            rewards.append(avg_reward)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Actor Loss = {avg_actor_loss:.4f}, "
                      f"Critic Loss = {avg_critic_loss:.4f}, Avg Reward = {avg_reward:.4f}")
        
        print("Storage pre-training completed!")
        return {"actor_losses": actor_losses, "critic_losses": critic_losses, "rewards": rewards}
    
    def pretrain_consumer_agent(self, agent: ConsumerAgent,
                              training_data: List[TrainingDataPoint],
                              epochs: int = 100,
                              batch_size: int = 64,
                              learning_rate: float = 0.001) -> Dict[str, List[float]]:
        """Pre-train a consumer agent using MADDPG"""
        
        print(f"Pre-training consumer agent with {len(training_data)} samples...")
        
        # Filter for consumer data
        consumer_data = [dp for dp in training_data if dp.agent_type == 'consumer']
        if not consumer_data:
            raise ValueError("No consumer training data found")
        
        # Setup optimizers
        actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=learning_rate)
        critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=learning_rate * 2)
        
        # Training metrics
        actor_losses = []
        critic_losses = []
        rewards = []
        
        # Training loop
        for epoch in tqdm(range(epochs), desc="Training Consumer"):
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_rewards = []
            
            # Shuffle data
            random.shuffle(consumer_data)
            
            # Batch training
            for i in range(0, len(consumer_data) - batch_size, batch_size):
                batch = consumer_data[i:i + batch_size]
                
                # Prepare batch tensors
                states = torch.FloatTensor([dp.state_vector for dp in batch]).to(self.device)
                actions = torch.FloatTensor([dp.action for dp in batch]).to(self.device)
                rewards_batch = torch.FloatTensor([dp.reward for dp in batch]).to(self.device)
                next_states = torch.FloatTensor([dp.next_state_vector for dp in batch]).to(self.device)
                
                # For MADDPG, create dummy actions for other agents
                other_actions = torch.rand(batch_size, agent.action_size * (agent.num_agents - 1)).to(self.device)
                all_actions = torch.cat([actions, other_actions], dim=1)
                
                # Train Critic
                with torch.no_grad():
                    next_actions_own = agent.actor_target(next_states)
                    next_actions_others = torch.rand(batch_size, agent.action_size * (agent.num_agents - 1)).to(self.device)
                    next_actions_all = torch.cat([next_actions_own, next_actions_others], dim=1)
                    
                    target_q_values = agent.critic_target(next_states, next_actions_all)
                    target_q_values = rewards_batch.unsqueeze(1) + agent.gamma * target_q_values
                
                current_q_values = agent.critic(states, all_actions)
                critic_loss = F.mse_loss(current_q_values, target_q_values)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # Train Actor
                predicted_actions_own = agent.actor(states)
                predicted_actions_others = torch.rand(batch_size, agent.action_size * (agent.num_agents - 1)).to(self.device)
                predicted_actions_all = torch.cat([predicted_actions_own, predicted_actions_others], dim=1)
                
                actor_loss = -agent.critic(states, predicted_actions_all).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # Soft update target networks
                for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)
                
                for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)
                
                epoch_actor_losses.append(actor_loss.item())
                epoch_critic_losses.append(critic_loss.item())
                epoch_rewards.append(rewards_batch.mean().item())
            
            avg_actor_loss = np.mean(epoch_actor_losses)
            avg_critic_loss = np.mean(epoch_critic_losses)
            avg_reward = np.mean(epoch_rewards)
            
            actor_losses.append(avg_actor_loss)
            critic_losses.append(avg_critic_loss)
            rewards.append(avg_reward)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Actor Loss = {avg_actor_loss:.4f}, "
                      f"Critic Loss = {avg_critic_loss:.4f}, Avg Reward = {avg_reward:.4f}")
        
        print("Consumer pre-training completed!")
        return {"actor_losses": actor_losses, "critic_losses": critic_losses, "rewards": rewards}
    
    def save_pretrained_models(self, agents: Dict[str, Any], model_dir: str = "pretrained_models") -> None:
        """Save pre-trained agent models"""
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        for agent_id, agent in agents.items():
            if isinstance(agent, GeneratorAgent):
                torch.save({
                    'q_network': agent.q_network.state_dict(),
                    'target_network': agent.target_network.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                    'config': agent.config
                }, model_path / f"{agent_id}_generator.pth")
                
            elif isinstance(agent, StorageAgent):
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'actor_target': agent.actor_target.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'critic_target': agent.critic_target.state_dict(),
                    'actor_optimizer': agent.actor_optimizer.state_dict(),
                    'critic_optimizer': agent.critic_optimizer.state_dict(),
                    'config': agent.config
                }, model_path / f"{agent_id}_storage.pth")
                
            elif isinstance(agent, ConsumerAgent):
                torch.save({
                    'actor': agent.actor.state_dict(),
                    'actor_target': agent.actor_target.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'critic_target': agent.critic_target.state_dict(),
                    'actor_optimizer': agent.actor_optimizer.state_dict(),
                    'critic_optimizer': agent.critic_optimizer.state_dict(),
                    'config': agent.config
                }, model_path / f"{agent_id}_consumer.pth")
        
        print(f"Saved {len(agents)} pre-trained models to {model_path}")
    
    def load_pretrained_models(self, agents: Dict[str, Any], model_dir: str = "pretrained_models") -> None:
        """Load pre-trained agent models"""
        model_path = Path(model_dir)
        
        for agent_id, agent in agents.items():
            if isinstance(agent, GeneratorAgent):
                model_file = model_path / f"{agent_id}_generator.pth"
                if model_file.exists():
                    checkpoint = torch.load(model_file, map_location=self.device)
                    agent.q_network.load_state_dict(checkpoint['q_network'])
                    agent.target_network.load_state_dict(checkpoint['target_network'])
                    agent.optimizer.load_state_dict(checkpoint['optimizer'])
                    print(f"Loaded pre-trained generator model for {agent_id}")
                
            elif isinstance(agent, StorageAgent):
                model_file = model_path / f"{agent_id}_storage.pth"
                if model_file.exists():
                    checkpoint = torch.load(model_file, map_location=self.device)
                    agent.actor.load_state_dict(checkpoint['actor'])
                    agent.actor_target.load_state_dict(checkpoint['actor_target'])
                    agent.critic.load_state_dict(checkpoint['critic'])
                    agent.critic_target.load_state_dict(checkpoint['critic_target'])
                    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                    print(f"Loaded pre-trained storage model for {agent_id}")
                
            elif isinstance(agent, ConsumerAgent):
                model_file = model_path / f"{agent_id}_consumer.pth"
                if model_file.exists():
                    checkpoint = torch.load(model_file, map_location=self.device)
                    agent.actor.load_state_dict(checkpoint['actor'])
                    agent.actor_target.load_state_dict(checkpoint['actor_target'])
                    agent.critic.load_state_dict(checkpoint['critic'])
                    agent.critic_target.load_state_dict(checkpoint['critic_target'])
                    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                    print(f"Loaded pre-trained consumer model for {agent_id}")
    
    def plot_training_metrics(self, metrics: Dict[str, List[float]], 
                            title: str, save_path: Optional[str] = None) -> None:
        """Plot training metrics"""
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            axes[i].plot(values)
            axes[i].set_title(f"{title} - {metric_name}")
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel(metric_name)
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


def pretrain_all_agents(simulation, epochs: int = 50) -> None:
    """Pre-train all agents in a simulation using historical data"""
    
    # Create training data manager
    data_manager = TrainingDataManager()
    
    # Check if training data exists, if not create it
    try:
        print("Loading existing training data...")
        datasets = {}
        datasets['generator_0'] = data_manager.load_training_data('generator_0_training.pkl')
        datasets['storage_0'] = data_manager.load_training_data('storage_0_training.pkl')
        datasets['consumer_0'] = data_manager.load_training_data('consumer_0_training.pkl')
    except FileNotFoundError:
        print("Training data not found. Creating new training data...")
        datasets = data_manager.create_training_dataset(days=365, save=True)
    
    # Initialize pre-trainer
    pretrainer = AgentPreTrainer()
    
    # Pre-train each agent type
    training_results = {}
    
    for agent_id, agent in simulation.agents.items():
        if isinstance(agent, GeneratorAgent):
            print(f"\nPre-training generator agent: {agent_id}")
            metrics = pretrainer.pretrain_generator_agent(
                agent, datasets['generator_0'], epochs=epochs
            )
            training_results[agent_id] = metrics
            pretrainer.plot_training_metrics(
                metrics, f"Generator {agent_id}", f"generator_{agent_id}_training.png"
            )
            
        elif isinstance(agent, StorageAgent):
            print(f"\nPre-training storage agent: {agent_id}")
            metrics = pretrainer.pretrain_storage_agent(
                agent, datasets['storage_0'], epochs=epochs
            )
            training_results[agent_id] = metrics
            pretrainer.plot_training_metrics(
                metrics, f"Storage {agent_id}", f"storage_{agent_id}_training.png"
            )
            
        elif isinstance(agent, ConsumerAgent):
            print(f"\nPre-training consumer agent: {agent_id}")
            metrics = pretrainer.pretrain_consumer_agent(
                agent, datasets['consumer_0'], epochs=epochs
            )
            training_results[agent_id] = metrics
            pretrainer.plot_training_metrics(
                metrics, f"Consumer {agent_id}", f"consumer_{agent_id}_training.png"
            )
    
    # Save pre-trained models
    pretrainer.save_pretrained_models(simulation.agents)
    
    print("\n" + "="*60)
    print("🎉 PRE-TRAINING COMPLETED!")
    print("All agents have been pre-trained with historical data")
    print("Models saved to 'pretrained_models/' directory")
    print("="*60)
    
    return training_results


if __name__ == "__main__":
    # Example usage
    from ..coordination.multi_agent_system import SmartGridSimulation
    
    # Create simulation
    sim = SmartGridSimulation()
    sim.create_sample_scenario()
    
    # Pre-train all agents
    results = pretrain_all_agents(sim, epochs=50) 