#!/usr/bin/env python3
"""
Smart Grid Blackout Animation System

Creates animated visualizations showing:
- Agent network topology and status
- Real-time communication flows
- Energy transfers between components
- Failure cascades and recovery dynamics
- System metrics evolution over time
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import json
from typing import Dict, List, Tuple, Any
import random
from datetime import datetime, timedelta
import os


class SmartGridAnimator:
    """Animated visualization of smart grid blackout scenarios"""
    
    def __init__(self, scenario_name: str = "texas_winter_uri"):
        self.scenario_name = scenario_name
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle('Smart Grid Blackout Simulation - Real-time Dynamics', 
                         fontsize=16, fontweight='bold')
        
        # Network topology axis
        self.network_ax = self.axes[0, 0]
        self.network_ax.set_title('Agent Network & Communication')
        self.network_ax.set_xlim(-1.2, 1.2)
        self.network_ax.set_ylim(-1.2, 1.2)
        self.network_ax.axis('off')
        
        # Energy flow axis
        self.energy_ax = self.axes[0, 1]
        self.energy_ax.set_title('Energy Generation & Consumption')
        self.energy_ax.set_xlabel('Time (hours)')
        self.energy_ax.set_ylabel('Power (MW)')
        
        # System metrics axis
        self.metrics_ax = self.axes[1, 0]
        self.metrics_ax.set_title('System Performance Metrics')
        self.metrics_ax.set_xlabel('Time (hours)')
        self.metrics_ax.set_ylabel('Frequency (Hz) / Agents Active')
        
        # Communication volume axis
        self.comm_ax = self.axes[1, 1]
        self.comm_ax.set_title('Communication & Coordination')
        self.comm_ax.set_xlabel('Time (hours)')
        self.comm_ax.set_ylabel('Messages/Hour')
        
        # Initialize data structures
        self.agents = self.create_agent_network()
        self.graph = self.create_network_graph()
        self.pos = nx.spring_layout(self.graph, k=0.8, iterations=50)
        
        # Animation data
        self.time_data = []
        self.energy_data = {'generation': [], 'load': [], 'unserved': []}
        self.metrics_data = {'frequency': [], 'agents_active': [], 'agents_failed': []}
        self.comm_data = {'messages': [], 'blackout_events': []}
        
        # Visual elements for animation
        self.node_colors = []
        self.edge_colors = []
        self.node_sizes = []
        self.communication_pulses = []
        
        # Generate simulation data
        self.simulation_data = self.generate_simulation_timeline()
        
        # Color schemes
        self.agent_colors = {
            'grid_operator': '#2E86AB',    # Blue
            'generator_gas': '#A23B72',    # Purple
            'generator_coal': '#F18F01',   # Orange
            'generator_wind': '#C73E1D',   # Red
            'generator_solar': '#FFBC42', # Yellow
            'storage': '#00A896',          # Teal
            'consumer': '#8B5A2B'          # Brown
        }
        
    def create_agent_network(self) -> Dict[str, Dict]:
        """Create smart grid agent network"""
        agents = {
            'grid_operator': {
                'type': 'grid_operator',
                'position': (0, 0),
                'capacity': 0,
                'status': 'active',
                'connections': ['gas_1', 'coal_1', 'wind_1', 'solar_1', 'storage_1', 'consumer_1']
            },
            'gas_1': {
                'type': 'generator_gas',
                'position': (0.8, 0.6),
                'capacity': 300,
                'status': 'active',
                'connections': ['grid_operator', 'storage_1']
            },
            'gas_2': {
                'type': 'generator_gas', 
                'position': (0.8, -0.6),
                'capacity': 300,
                'status': 'active',
                'connections': ['grid_operator', 'storage_2']
            },
            'coal_1': {
                'type': 'generator_coal',
                'position': (-0.8, 0.6),
                'capacity': 400,
                'status': 'active',
                'connections': ['grid_operator', 'consumer_2']
            },
            'wind_1': {
                'type': 'generator_wind',
                'position': (0.6, 0.8),
                'capacity': 200,
                'status': 'active',
                'connections': ['grid_operator', 'consumer_1']
            },
            'wind_2': {
                'type': 'generator_wind',
                'position': (-0.6, 0.8),
                'capacity': 200,
                'status': 'active', 
                'connections': ['grid_operator', 'consumer_3']
            },
            'solar_1': {
                'type': 'generator_solar',
                'position': (0, 0.9),
                'capacity': 150,
                'status': 'active',
                'connections': ['grid_operator', 'storage_1']
            },
            'storage_1': {
                'type': 'storage',
                'position': (0.9, 0),
                'capacity': 400,
                'status': 'active',
                'soc': 80,
                'connections': ['grid_operator', 'gas_1', 'solar_1']
            },
            'storage_2': {
                'type': 'storage',
                'position': (-0.9, 0),
                'capacity': 400,
                'status': 'active',
                'soc': 70,
                'connections': ['grid_operator', 'gas_2']
            },
            'consumer_1': {
                'type': 'consumer',
                'position': (0, -0.9),
                'capacity': 200,
                'status': 'active',
                'connections': ['grid_operator', 'wind_1']
            },
            'consumer_2': {
                'type': 'consumer',
                'position': (-0.6, -0.8),
                'capacity': 250,
                'status': 'active',
                'connections': ['grid_operator', 'coal_1']
            },
            'consumer_3': {
                'type': 'consumer',
                'position': (0.6, -0.8),
                'capacity': 180,
                'status': 'active',
                'connections': ['grid_operator', 'wind_2']
            }
        }
        return agents
    
    def create_network_graph(self) -> nx.Graph:
        """Create networkx graph from agent network"""
        G = nx.Graph()
        
        # Add nodes
        for agent_id, agent_data in self.agents.items():
            G.add_node(agent_id, **agent_data)
        
        # Add edges based on connections
        for agent_id, agent_data in self.agents.items():
            for connection in agent_data['connections']:
                if connection in self.agents:
                    G.add_edge(agent_id, connection)
        
        return G
    
    def generate_simulation_timeline(self) -> List[Dict]:
        """Generate realistic simulation timeline data"""
        duration_hours = 12
        steps_per_hour = 12  # 5-minute intervals
        total_steps = duration_hours * steps_per_hour
        
        timeline = []
        base_failure_prob = 0.02  # 2% chance per step
        
        # Initial system state
        current_state = {
            'frequency': 50.0,
            'total_generation': sum(a['capacity'] for a in self.agents.values() 
                                  if a['type'].startswith('generator')),
            'total_load': sum(a['capacity'] for a in self.agents.values() 
                            if a['type'] == 'consumer'),
            'agents_status': {k: 'active' for k in self.agents.keys()},
            'communications': 50,
            'blackout_active': False
        }
        
        for step in range(total_steps):
            hour = step / steps_per_hour
            
            # Apply scenario-specific effects
            if self.scenario_name == "texas_winter_uri":
                # Cold weather effects worsen over time
                cold_factor = min(1.0, hour / 6)  # Reaches max at 6 hours
                failure_prob = base_failure_prob * (1 + cold_factor * 10)
                demand_surge = 1.0 + cold_factor * 0.8  # 80% increase
            elif self.scenario_name == "california_heat_wave":
                # Heat effects peak midday
                heat_factor = abs(np.sin(hour * np.pi / 12))  # Sine wave over 12 hours
                failure_prob = base_failure_prob * (1 + heat_factor * 5)
                demand_surge = 1.0 + heat_factor * 0.6  # 60% increase
            else:  # winter_storm_elliott
                # Moderate cold with some recovery
                cold_factor = max(0, 1 - hour / 8)  # Improves over time
                failure_prob = base_failure_prob * (1 + cold_factor * 6)
                demand_surge = 1.0 + cold_factor * 0.5  # 50% increase
            
            # Agent failures
            for agent_id in list(current_state['agents_status'].keys()):
                if (current_state['agents_status'][agent_id] == 'active' and 
                    random.random() < failure_prob and
                    agent_id != 'grid_operator'):  # Grid operator never fails
                    current_state['agents_status'][agent_id] = 'failed'
            
            # Recovery attempts (lower probability)
            for agent_id in list(current_state['agents_status'].keys()):
                if (current_state['agents_status'][agent_id] == 'failed' and 
                    random.random() < 0.01):  # 1% recovery chance
                    current_state['agents_status'][agent_id] = 'active'
            
            # Calculate system metrics
            active_generators = [aid for aid, status in current_state['agents_status'].items() 
                               if status == 'active' and self.agents[aid]['type'].startswith('generator')]
            failed_agents = sum(1 for status in current_state['agents_status'].values() 
                              if status == 'failed')
            
            current_generation = sum(self.agents[aid]['capacity'] for aid in active_generators)
            current_load = current_state['total_load'] * demand_surge
            
            # Frequency calculation based on generation-load balance
            imbalance = (current_generation - current_load) / current_load
            frequency_deviation = imbalance * 2  # Simplified frequency response
            current_state['frequency'] = 50.0 + frequency_deviation
            
            # Communication overhead increases with failures
            base_comm = 50
            stress_comm = failed_agents * 50 + max(0, -imbalance * 1000)
            current_state['communications'] = base_comm + stress_comm
            
            # Blackout detection
            blackout_conditions = (
                current_state['frequency'] < 49.5 or 
                current_state['frequency'] > 50.5 or
                current_generation < current_load * 0.9
            )
            current_state['blackout_active'] = blackout_conditions
            
            # Store step data
            step_data = {
                'hour': hour,
                'frequency': current_state['frequency'],
                'generation': current_generation,
                'load': current_load,
                'unserved_load': max(0, current_load - current_generation),
                'agents_active': len(current_state['agents_status']) - failed_agents,
                'agents_failed': failed_agents,
                'communications': current_state['communications'],
                'blackout_active': current_state['blackout_active'],
                'agents_status': current_state['agents_status'].copy()
            }
            timeline.append(step_data)
        
        return timeline
    
    def get_node_color(self, agent_id: str, status: str, agent_type: str) -> str:
        """Get node color based on agent status"""
        if status == 'failed':
            return '#FF4444'  # Red for failed
        elif status == 'degraded':
            return '#FFAA44'  # Orange for degraded
        else:
            return self.agent_colors.get(agent_type, '#888888')
    
    def get_node_size(self, agent_id: str, agent_type: str) -> float:
        """Get node size based on agent type and importance"""
        size_map = {
            'grid_operator': 1000,
            'generator_gas': 800,
            'generator_coal': 800,
            'generator_wind': 600,
            'generator_solar': 600,
            'storage': 700,
            'consumer': 500
        }
        return size_map.get(agent_type, 500)
    
    def animate_network(self, frame: int):
        """Animate the network topology view"""
        self.network_ax.clear()
        self.network_ax.set_title('Agent Network & Communication')
        self.network_ax.set_xlim(-1.2, 1.2)
        self.network_ax.set_ylim(-1.2, 1.2)
        self.network_ax.axis('off')
        
        if frame >= len(self.simulation_data):
            return
        
        step_data = self.simulation_data[frame]
        hour = step_data['hour']
        
        # Draw edges (connections)
        edge_colors = []
        edge_widths = []
        
        for edge in self.graph.edges():
            agent1, agent2 = edge
            status1 = step_data['agents_status'][agent1]
            status2 = step_data['agents_status'][agent2]
            
            if status1 == 'active' and status2 == 'active':
                edge_colors.append('#00AA00')  # Green for active connections
                edge_widths.append(2.0)
            elif status1 == 'failed' or status2 == 'failed':
                edge_colors.append('#FF4444')  # Red for failed connections
                edge_widths.append(1.0)
            else:
                edge_colors.append('#FFAA44')  # Orange for degraded connections
                edge_widths.append(1.5)
        
        # Draw network
        nx.draw_networkx_edges(self.graph, self.pos, ax=self.network_ax,
                              edge_color=edge_colors, width=edge_widths, alpha=0.7)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        
        for agent_id in self.graph.nodes():
            agent = self.agents[agent_id]
            status = step_data['agents_status'][agent_id]
            
            color = self.get_node_color(agent_id, status, agent['type'])
            size = self.get_node_size(agent_id, agent['type'])
            
            node_colors.append(color)
            node_sizes.append(size)
        
        nx.draw_networkx_nodes(self.graph, self.pos, ax=self.network_ax,
                              node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        # Add labels
        labels = {agent_id: agent_id.replace('_', '\n') for agent_id in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, self.pos, labels, ax=self.network_ax,
                               font_size=8, font_weight='bold')
        
        # Add communication pulses (animated circles)
        if step_data['communications'] > 100:  # High communication activity
            for i in range(3):  # Draw 3 pulse rings
                pulse_radius = 0.1 + (frame % 10) * 0.05 + i * 0.05
                pulse_alpha = 1.0 - (frame % 10) * 0.1
                circle = Circle((0, 0), pulse_radius, fill=False, 
                              color='cyan', linewidth=2, alpha=pulse_alpha)
                self.network_ax.add_patch(circle)
        
        # Add blackout warning
        if step_data['blackout_active']:
            warning_box = FancyBboxPatch((-1.1, 1.0), 0.8, 0.15,
                                       boxstyle="round,pad=0.02",
                                       facecolor='red', alpha=0.8)
            self.network_ax.add_patch(warning_box)
            self.network_ax.text(-0.7, 1.075, 'BLACKOUT', fontsize=12, 
                                fontweight='bold', color='white', ha='center')
        
        # Add timestamp
        self.network_ax.text(-1.1, -1.1, f'Time: {hour:.1f} hours', 
                           fontsize=10, fontweight='bold')
    
    def animate_energy_flows(self, frame: int):
        """Animate energy generation and consumption"""
        if frame >= len(self.simulation_data):
            return
        
        # Collect data up to current frame
        hours = [self.simulation_data[i]['hour'] for i in range(frame + 1)]
        generation = [self.simulation_data[i]['generation'] for i in range(frame + 1)]
        load = [self.simulation_data[i]['load'] for i in range(frame + 1)]
        unserved = [self.simulation_data[i]['unserved_load'] for i in range(frame + 1)]
        
        self.energy_ax.clear()
        self.energy_ax.set_title('Energy Generation & Consumption')
        self.energy_ax.set_xlabel('Time (hours)')
        self.energy_ax.set_ylabel('Power (MW)')
        
        # Plot lines
        if len(hours) > 1:
            self.energy_ax.plot(hours, generation, 'g-', linewidth=2, label='Generation')
            self.energy_ax.plot(hours, load, 'b-', linewidth=2, label='Load Demand')
            
            # Fill unserved load area
            if any(u > 0 for u in unserved):
                self.energy_ax.fill_between(hours, load, 
                                          [l - u for l, u in zip(load, unserved)],
                                          color='red', alpha=0.3, label='Unserved Load')
        
        # Mark current point
        if frame < len(self.simulation_data):
            current_data = self.simulation_data[frame]
            self.energy_ax.plot(current_data['hour'], current_data['generation'], 
                              'go', markersize=8)
            self.energy_ax.plot(current_data['hour'], current_data['load'], 
                              'bo', markersize=8)
        
        self.energy_ax.grid(True, alpha=0.3)
        self.energy_ax.legend()
        self.energy_ax.set_xlim(0, 12)
        self.energy_ax.set_ylim(0, 2000)
    
    def animate_system_metrics(self, frame: int):
        """Animate system performance metrics"""
        if frame >= len(self.simulation_data):
            return
        
        # Collect data up to current frame
        hours = [self.simulation_data[i]['hour'] for i in range(frame + 1)]
        frequency = [self.simulation_data[i]['frequency'] for i in range(frame + 1)]
        agents_active = [self.simulation_data[i]['agents_active'] for i in range(frame + 1)]
        
        self.metrics_ax.clear()
        self.metrics_ax.set_title('System Performance Metrics')
        self.metrics_ax.set_xlabel('Time (hours)')
        
        if len(hours) > 1:
            # Frequency on primary axis
            color = 'tab:red'
            self.metrics_ax.set_ylabel('Frequency (Hz)', color=color)
            line1 = self.metrics_ax.plot(hours, frequency, color=color, linewidth=2, 
                                       label='Grid Frequency')
            self.metrics_ax.tick_params(axis='y', labelcolor=color)
            
            # Add frequency thresholds
            self.metrics_ax.axhline(y=50.0, color='green', linestyle='--', alpha=0.7, 
                                  label='Normal Frequency')
            self.metrics_ax.axhline(y=49.5, color='red', linestyle='--', alpha=0.7,
                                  label='Critical Threshold')
            
            # Agents on secondary axis
            ax2 = self.metrics_ax.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Active Agents', color=color)
            line2 = ax2.plot(hours, agents_active, color=color, linewidth=2,
                           label='Active Agents')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0, len(self.agents))
        
        # Mark current point
        if frame < len(self.simulation_data):
            current_data = self.simulation_data[frame]
            self.metrics_ax.plot(current_data['hour'], current_data['frequency'], 
                               'ro', markersize=8)
        
        self.metrics_ax.grid(True, alpha=0.3)
        self.metrics_ax.set_xlim(0, 12)
        self.metrics_ax.set_ylim(48, 52)
    
    def animate_communications(self, frame: int):
        """Animate communication and coordination metrics"""
        if frame >= len(self.simulation_data):
            return
        
        # Collect data up to current frame
        hours = [self.simulation_data[i]['hour'] for i in range(frame + 1)]
        communications = [self.simulation_data[i]['communications'] for i in range(frame + 1)]
        blackouts = [self.simulation_data[i]['blackout_active'] for i in range(frame + 1)]
        
        self.comm_ax.clear()
        self.comm_ax.set_title('Communication & Coordination')
        self.comm_ax.set_xlabel('Time (hours)')
        self.comm_ax.set_ylabel('Messages/Hour')
        
        if len(hours) > 1:
            # Communication volume
            self.comm_ax.plot(hours, communications, 'purple', linewidth=2, 
                            label='Message Volume')
            
            # Mark blackout periods
            for i, (h, blackout) in enumerate(zip(hours, blackouts)):
                if blackout:
                    self.comm_ax.axvspan(h - 0.05, h + 0.05, alpha=0.3, color='red')
        
        # Mark current point
        if frame < len(self.simulation_data):
            current_data = self.simulation_data[frame]
            self.comm_ax.plot(current_data['hour'], current_data['communications'], 
                            'mo', markersize=8)
        
        self.comm_ax.grid(True, alpha=0.3)
        self.comm_ax.legend()
        self.comm_ax.set_xlim(0, 12)
        self.comm_ax.set_ylim(0, max(500, max(communications) if communications else 100))
    
    def animate_frame(self, frame: int):
        """Animate all subplots for current frame"""
        self.animate_network(frame)
        self.animate_energy_flows(frame)
        self.animate_system_metrics(frame)
        self.animate_communications(frame)
        
        plt.tight_layout()
        return []
    
    def create_animation(self, interval: int = 200, save_path: str = None) -> animation.FuncAnimation:
        """Create the complete animation"""
        frames = len(self.simulation_data)
        
        anim = animation.FuncAnimation(
            self.fig, self.animate_frame, frames=frames,
            interval=interval, blit=False, repeat=True
        )
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=5)
            print("Animation saved!")
        
        return anim
    
    def export_frame_sequence(self, output_dir: str = "animation_frames"):
        """Export individual frames for external video creation"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Exporting {len(self.simulation_data)} frames to {output_dir}/...")
        
        for frame in range(len(self.simulation_data)):
            self.animate_frame(frame)
            self.fig.savefig(f"{output_dir}/frame_{frame:04d}.png", 
                           dpi=150, bbox_inches='tight')
            
            if frame % 20 == 0:
                print(f"  Exported frame {frame}/{len(self.simulation_data)}")
        
        print("Frame sequence export complete!")
        print(f"To create video, use: ffmpeg -r 5 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p blackout_animation.mp4")


def main():
    """Main function to create blackout animations"""
    print("Smart Grid Blackout Animation System")
    print("=" * 50)
    
    # Create animation for Texas Winter Storm (most dramatic)
    print("Creating Texas Winter Storm Uri animation...")
    animator = SmartGridAnimator("texas_winter_uri")
    
    # Show live animation
    print("Starting live animation preview...")
    anim = animator.create_animation(interval=100)
    
    # Save as GIF
    print("Saving as GIF...")
    anim.save("blackout_simulation.gif", writer='pillow', fps=10)
    
    # Export frames for high-quality video
    print("Exporting frames for video creation...")
    animator.export_frame_sequence("blackout_frames")
    
    print("\nAnimation complete!")
    print("Files created:")
    print("  blackout_simulation.gif - Animated GIF")
    print("  blackout_frames/ - Individual frames for video")
    print("\nTo create MP4 video:")
    print("  ffmpeg -r 10 -i blackout_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p blackout_video.mp4")
    
    plt.show()


if __name__ == "__main__":
    main() 