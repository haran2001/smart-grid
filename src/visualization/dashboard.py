import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List
import sys
import os

# Add the project root to Python path to enable absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now use absolute imports
from src.coordination.multi_agent_system import SmartGridSimulation, create_renewable_heavy_scenario, create_traditional_grid_scenario


class SmartGridDashboard:
    """Interactive dashboard for smart grid monitoring"""
    
    def __init__(self):
        self.simulation = None
        self.metrics_history = []
        self.max_history_points = 100
        
    def setup_page(self):
        """Setup the Streamlit page configuration"""
        st.set_page_config(
            page_title="Smart Grid Multi-Agent System Dashboard",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("⚡ Smart Grid Multi-Agent Energy Management System")
        st.markdown("Real-time monitoring and control dashboard for the AI-powered smart grid")
    
    def create_sidebar(self):
        """Create the sidebar with controls"""
        st.sidebar.header("Simulation Control")
        
        # Scenario selection
        scenario_type = st.sidebar.selectbox(
            "Select Scenario",
            ["Sample Scenario", "Renewable Heavy", "Traditional Grid", "Custom"]
        )
        
        if st.sidebar.button("Initialize Simulation"):
            try:
                if scenario_type == "Sample Scenario":
                    self.simulation = SmartGridSimulation()
                    self.simulation.create_sample_scenario()
                elif scenario_type == "Renewable Heavy":
                    self.simulation = create_renewable_heavy_scenario()
                elif scenario_type == "Traditional Grid":
                    self.simulation = create_traditional_grid_scenario()
                else:
                    self.simulation = SmartGridSimulation()
                
                st.sidebar.success(f"Initialized {scenario_type}")
            except Exception as e:
                st.sidebar.error(f"Error initializing simulation: {e}")
        
        # Simulation controls
        if self.simulation:
            if st.sidebar.button("Start Simulation"):
                self._start_simulation_background()
            
            if st.sidebar.button("Stop Simulation"):
                try:
                    if hasattr(self.simulation, 'is_running') and self.simulation.is_running:
                        self.simulation.stop_simulation()
                        st.sidebar.info("Simulation stopped")
                except Exception as e:
                    st.sidebar.error(f"Error stopping simulation: {e}")
            
            # Export controls
            if st.sidebar.button("Export Results"):
                try:
                    filename = f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self.simulation.export_results(filename)
                    st.sidebar.success(f"Results exported to {filename}")
                except Exception as e:
                    st.sidebar.error(f"Error exporting results: {e}")
        
        # Display simulation status
        if self.simulation:
            try:
                status = "Running" if hasattr(self.simulation, 'is_running') and self.simulation.is_running else "Stopped"
                st.sidebar.metric("Simulation Status", status)
                if hasattr(self.simulation, 'simulation_metrics'):
                    st.sidebar.metric("Total Steps", self.simulation.simulation_metrics.get("total_steps", 0))
            except Exception as e:
                st.sidebar.text(f"Status unavailable: {e}")
    
    def _start_simulation_background(self):
        """Start simulation in background (simplified for demo)"""
        if self.simulation and not self.simulation.is_running:
            # For demo purposes, just run a few steps
            try:
                asyncio.run(self._run_simulation_steps(5))
                st.sidebar.success("Simulation steps completed")
            except Exception as e:
                st.sidebar.error(f"Simulation error: {e}")
    
    async def _run_simulation_steps(self, steps: int):
        """Run a few simulation steps"""
        for _ in range(steps):
            await self.simulation.run_simulation_step()
            metrics = await self.simulation.get_real_time_metrics()
            self.metrics_history.append({
                'timestamp': datetime.now(),
                'metrics': metrics
            })
            # Keep only recent history
            if len(self.metrics_history) > self.max_history_points:
                self.metrics_history.pop(0)
    
    def display_grid_overview(self):
        """Display grid overview metrics"""
        if not self.simulation or not self.simulation.grid_operator:
            st.info("Please initialize a simulation to view grid metrics")
            return
        
        st.header("Grid System Overview")
        
        # Get current grid status
        grid_status = self.simulation.grid_operator.get_system_status()
        grid_state = grid_status["grid_state"]
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Frequency",
                f"{grid_state['frequency_hz']:.3f} Hz",
                delta=f"{grid_state['frequency_hz'] - 50.0:.3f}"
            )
            
        with col2:
            st.metric(
                "Voltage",
                f"{grid_state['voltage_pu']:.3f} pu",
                delta=f"{grid_state['voltage_pu'] - 1.0:.3f}"
            )
            
        with col3:
            renewable_pct = (grid_state['renewable_generation_mw'] / 
                           max(grid_state['total_generation_mw'], 1) * 100)
            st.metric(
                "Renewable Penetration",
                f"{renewable_pct:.1f}%"
            )
            
        with col4:
            balance = grid_state['total_generation_mw'] - grid_state['total_load_mw']
            st.metric(
                "Load-Generation Balance",
                f"{balance:.1f} MW",
                delta=f"{balance:.1f}"
            )
    
    def display_market_information(self):
        """Display market and pricing information"""
        if not self.simulation:
            return
        
        st.header("Market Information")
        
        market_data = self.simulation.market_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Market Prices")
            st.metric("Energy Price", f"${market_data['current_price']:.2f}/MWh")
            st.metric("Carbon Price", f"${market_data['carbon_price']:.2f}/tonne")
            st.metric("DR Price", f"${market_data['dr_price']:.2f}/MWh")
            
        with col2:
            st.subheader("Market Forecast")
            # Price forecast chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=market_data['price_forecast'],
                mode='lines+markers',
                name='Price Forecast',
                line=dict(color='blue')
            ))
            fig.update_layout(
                title="24-Hour Price Forecast",
                xaxis_title="Hours Ahead",
                yaxis_title="Price ($/MWh)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_agent_performance(self):
        """Display individual agent performance metrics"""
        if not self.simulation:
            return
        
        st.header("Agent Performance")
        
        # Agent selector
        agent_ids = list(self.simulation.agents.keys())
        selected_agent = st.selectbox("Select Agent", agent_ids)
        
        if selected_agent and selected_agent in self.simulation.agents:
            agent = self.simulation.agents[selected_agent]
            agent_state = agent.get_current_state()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Agent: {selected_agent}")
                st.json(agent_state["operational_status"])
                
            with col2:
                st.subheader("Performance Metrics")
                st.json(agent_state["performance_metrics"])
    
    def display_system_charts(self):
        """Display system-wide charts and visualizations"""
        if not self.metrics_history:
            st.info("No historical data available. Run some simulation steps to see charts.")
            return
        
        st.header("System Performance Charts")
        
        # Prepare data for charts
        timestamps = [entry['timestamp'] for entry in self.metrics_history]
        
        # Grid stability chart
        if self.simulation and self.simulation.grid_operator:
            grid_metrics = []
            for entry in self.metrics_history:
                if 'grid_system' in entry['metrics']:
                    grid_metrics.append(entry['metrics']['grid_system'])
            
            if grid_metrics:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Frequency Stability', 'Voltage Stability', 'Price Evolution', 'Renewable Penetration'],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Frequency stability
                frequencies = [self.simulation.grid_operator.grid_state.frequency_hz] * len(timestamps)
                fig.add_trace(
                    go.Scatter(x=timestamps, y=frequencies, name="Frequency", line=dict(color='red')),
                    row=1, col=1
                )
                
                # Voltage stability
                voltages = [self.simulation.grid_operator.grid_state.voltage_pu] * len(timestamps)
                fig.add_trace(
                    go.Scatter(x=timestamps, y=voltages, name="Voltage", line=dict(color='blue')),
                    row=1, col=2
                )
                
                # Price evolution
                prices = [self.simulation.market_data['current_price']] * len(timestamps)
                fig.add_trace(
                    go.Scatter(x=timestamps, y=prices, name="Price", line=dict(color='green')),
                    row=2, col=1
                )
                
                # Renewable penetration
                renewable_pct = [metric.get('renewable_penetration', 0) for metric in grid_metrics]
                fig.add_trace(
                    go.Scatter(x=timestamps, y=renewable_pct, name="Renewable %", line=dict(color='orange')),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    def display_communication_network(self):
        """Display agent communication network"""
        if not self.simulation:
            return
        
        st.header("Agent Communication Network")
        
        # Create network visualization
        agents = list(self.simulation.agents.keys())
        
        # Simple network graph showing agent connections
        fig = go.Figure()
        
        # Add nodes for each agent
        for i, agent_id in enumerate(agents):
            angle = 2 * np.pi * i / len(agents)
            x = np.cos(angle)
            y = np.sin(angle)
            
            agent_type = self.simulation.agents[agent_id].agent_type.value
            color_map = {
                'generator': 'red',
                'storage': 'blue',
                'consumer': 'green',
                'grid_operator': 'purple'
            }
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[agent_id],
                textposition="middle center",
                marker=dict(size=50, color=color_map.get(agent_type, 'gray')),
                name=agent_type,
                showlegend=True if i == 0 or agent_type not in [self.simulation.agents[agents[j]].agent_type.value for j in range(i)] else False
            ))
        
        # Add edges (connections) - simplified as all agents connect to grid operator
        if 'grid_operator' in agents:
            grid_idx = agents.index('grid_operator')
            grid_angle = 2 * np.pi * grid_idx / len(agents)
            grid_x, grid_y = np.cos(grid_angle), np.sin(grid_angle)
            
            for i, agent_id in enumerate(agents):
                if agent_id != 'grid_operator':
                    angle = 2 * np.pi * i / len(agents)
                    x, y = np.cos(angle), np.sin(angle)
                    
                    fig.add_trace(go.Scatter(
                        x=[x, grid_x, None],
                        y=[y, grid_y, None],
                        mode='lines',
                        line=dict(color='gray', width=1),
                        showlegend=False
                    ))
        
        fig.update_layout(
            title="Smart Grid Agent Network",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_environmental_metrics(self):
        """Display environmental impact metrics"""
        if not self.simulation or not self.simulation.grid_operator:
            return
        
        st.header("Environmental Impact")
        
        grid_status = self.simulation.grid_operator.get_system_status()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            carbon_intensity = grid_status["grid_state"]["carbon_intensity"]
            st.metric("Carbon Intensity", f"{carbon_intensity:.1f} kg CO2/MWh")
            
        with col2:
            renewable_mw = grid_status["grid_state"]["renewable_generation_mw"]
            st.metric("Renewable Generation", f"{renewable_mw:.1f} MW")
            
        with col3:
            total_gen = grid_status["grid_state"]["total_generation_mw"]
            renewable_pct = (renewable_mw / max(total_gen, 1)) * 100
            st.metric("Green Energy Share", f"{renewable_pct:.1f}%")
        
        # Environmental impact chart
        fig = go.Figure(data=[
            go.Bar(
                x=['Current Grid', 'Coal Average', 'Gas Average', 'Renewable Target'],
                y=[carbon_intensity, 800, 400, 50],
                marker_color=['blue', 'red', 'orange', 'green']
            )
        ])
        
        fig.update_layout(
            title="Carbon Intensity Comparison",
            yaxis_title="kg CO2/MWh",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run_dashboard(self):
        """Main dashboard execution"""
        self.setup_page()
        self.create_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Grid Overview", 
            "Market Info", 
            "Agent Performance", 
            "System Charts",
            "Environmental"
        ])
        
        with tab1:
            self.display_grid_overview()
            self.display_communication_network()
            
        with tab2:
            self.display_market_information()
            
        with tab3:
            self.display_agent_performance()
            
        with tab4:
            self.display_system_charts()
            
        with tab5:
            self.display_environmental_metrics()
        
        # Auto-refresh option
        if st.checkbox("Auto-refresh (every 10 seconds)", value=False):
            import time
            time.sleep(10)
            st.experimental_rerun()


def main():
    """Main function to run the dashboard"""
    dashboard = SmartGridDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main() 