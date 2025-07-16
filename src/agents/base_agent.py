from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import asyncio
import queue
import json
from enum import Enum
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

class AgentType(Enum):
    GENERATOR = "generator"
    STORAGE = "storage"
    CONSUMER = "consumer"
    GRID_OPERATOR = "grid_operator"

class MessageType(Enum):
    DEMAND_FORECAST_UPDATE = "demand_forecast_update"
    GENERATION_FORECAST = "generation_forecast"
    GENERATION_BID = "generation_bid"
    DEMAND_RESPONSE_OFFER = "demand_response_offer"
    PARTNERSHIP_ACCEPTANCE = "partnership_acceptance"
    DISPATCH_INSTRUCTION = "dispatch_instruction"
    STATUS_UPDATE = "status_update"
    MARKET_PRICE_UPDATE = "market_price_update"
    WEATHER_UPDATE = "weather_update"

@dataclass
class AgentMessage:
    """Standard message format for agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.STATUS_UPDATE
    content: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=low, 5=high

@dataclass
class AgentState:
    """State maintained by each agent"""
    agent_id: str
    agent_type: AgentType
    current_timestamp: datetime = field(default_factory=datetime.now)
    market_data: Dict[str, Any] = field(default_factory=dict)
    operational_status: Dict[str, Any] = field(default_factory=dict)
    messages: List[AgentMessage] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class BaseAgent(ABC):
    """Base class for all smart grid agents"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.state = AgentState(agent_id=agent_id, agent_type=agent_type)
        self._message_queue = asyncio.Queue() if hasattr(asyncio, 'Queue') else queue.Queue()
        self.graph = self._build_decision_graph()
        
        # Message router will be set by the simulation system
        self.message_router = None
        
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
    
    def _build_decision_graph(self):
        """Build the LangGraph decision-making workflow"""
        graph = StateGraph(AgentState)
        
        # Add nodes for different decision-making steps
        graph.add_node("process_messages", self._process_messages)
        graph.add_node("analyze_market", self._analyze_market_conditions)
        graph.add_node("make_decision", self._make_decision)
        graph.add_node("execute_action", self._execute_action)
        graph.add_node("update_state", self._update_internal_state)
        
        # Define the workflow
        graph.add_edge("process_messages", "analyze_market")
        graph.add_edge("analyze_market", "make_decision")
        graph.add_edge("make_decision", "execute_action")
        graph.add_edge("execute_action", "update_state")
        graph.add_edge("update_state", END)
        
        graph.set_entry_point("process_messages")
        
        return graph.compile()
    
    async def _process_messages(self, state: AgentState) -> AgentState:
        """Process incoming messages from other agents"""
        new_messages = []
        
        # Process all messages in queue
        try:
            # Try asyncio queue first
            if hasattr(self._message_queue, 'empty'):
                while not self.message_queue.empty():
                    try:
                        message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                        new_messages.append(message)
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        break
            else:
                # Handle regular queue
                import queue
                while True:
                    try:
                        message = self.message_queue.get_nowait()
                        new_messages.append(message)
                        await self._handle_message(message)
                    except queue.Empty:
                        break
        except Exception as e:
            # If there's any issue with message processing, continue
            pass
        
        state.messages.extend(new_messages)
        return state
    
    async def _analyze_market_conditions(self, state: AgentState) -> AgentState:
        """Analyze current market conditions and grid state"""
        market_analysis = await self.analyze_market_data()
        state.market_data.update(market_analysis)
        return state
    
    async def _make_decision(self, state: AgentState) -> AgentState:
        """Make strategic decisions based on current state"""
        decision = await self.make_strategic_decision(state)
        state.decisions.append({
            "timestamp": datetime.now(),
            "decision": decision,
            "reasoning": decision.get("reasoning", "")
        })
        return state
    
    async def _execute_action(self, state: AgentState) -> AgentState:
        """Execute the decided action"""
        if state.decisions:
            latest_decision = state.decisions[-1]["decision"]
            await self.execute_decision(latest_decision)
        return state
    
    async def _update_internal_state(self, state: AgentState) -> AgentState:
        """Update internal state and performance metrics"""
        state.current_timestamp = datetime.now()
        metrics = await self.calculate_performance_metrics()
        state.performance_metrics.update(metrics)
        return state
    
    @abstractmethod
    async def analyze_market_data(self) -> Dict[str, Any]:
        """Analyze market conditions specific to agent type"""
        pass
    
    @abstractmethod
    async def make_strategic_decision(self, state: AgentState) -> Dict[str, Any]:
        """Make strategic decisions based on agent type and current state"""
        pass
    
    @abstractmethod
    async def execute_decision(self, decision: Dict[str, Any]) -> None:
        """Execute the strategic decision"""
        pass
    
    async def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for this agent"""
        # Handle both AgentState and AddableValuesDict cases
        messages_count = 0
        decisions_count = 0
        
        if hasattr(self.state, 'messages'):
            messages_count = len(self.state.messages)
        elif isinstance(self.state, dict) and 'messages' in self.state:
            messages_count = len(self.state['messages'])
            
        if hasattr(self.state, 'decisions'):
            decisions_count = len(self.state.decisions)
        elif isinstance(self.state, dict) and 'decisions' in self.state:
            decisions_count = len(self.state['decisions'])
        
        return {
            "messages_processed": messages_count,
            "decisions_made": decisions_count,
            "last_update": datetime.now().timestamp()
        }
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """Handle specific message types"""
        handler_map = {
            MessageType.DEMAND_FORECAST_UPDATE: self._handle_demand_forecast,
            MessageType.GENERATION_FORECAST: self._handle_generation_forecast,
            MessageType.MARKET_PRICE_UPDATE: self._handle_market_price_update,
            MessageType.WEATHER_UPDATE: self._handle_weather_update,
            MessageType.STATUS_UPDATE: self._handle_status_update,
        }
        
        handler = handler_map.get(message.message_type)
        if handler:
            await handler(message)
    
    async def _handle_demand_forecast(self, message: AgentMessage) -> None:
        """Handle demand forecast updates"""
        self.state.market_data["demand_forecast"] = message.content
    
    async def _handle_generation_forecast(self, message: AgentMessage) -> None:
        """Handle generation forecast updates"""
        self.state.market_data["generation_forecast"] = message.content
    
    async def _handle_market_price_update(self, message: AgentMessage) -> None:
        """Handle market price updates"""
        self.state.market_data["current_prices"] = message.content
    
    async def _handle_weather_update(self, message: AgentMessage) -> None:
        """Handle weather forecast updates"""
        self.state.market_data["weather"] = message.content

    async def _handle_status_update(self, message: AgentMessage) -> None:
        """Handle status update messages - default implementation"""
        # Update any general status information
        content = message.content
        
        # Check if this is a bid request
        request_type = content.get("request_type")
        if request_type:
            # This is likely a bid request - subclasses should override to handle specifically
            print(f"[{self.agent_id}] Received {request_type} request from {message.sender_id}")
            
            # CRITICAL FIX: Ensure state is properly formatted for neural network
            try:
                if isinstance(self.state, dict):
                    # Convert dict back to AgentState (LangGraph converts AgentState -> dict)
                    agent_state = AgentState(
                        agent_id=self.agent_id,
                        agent_type=self.agent_type,
                        current_timestamp=datetime.now(),
                        market_data=self.state.get('market_data', {}),
                        operational_status=self.state.get('operational_status', {}),
                        messages=self.state.get('messages', []),
                        decisions=self.state.get('decisions', []),
                        performance_metrics=self.state.get('performance_metrics', {})
                    )
                    self.state = agent_state
                
                # Now use the proper neural network decision making
                decision = await self.make_strategic_decision(self.state)
                await self.execute_decision(decision)
                print(f"[{self.agent_id}] Made neural network decision for {request_type}")
            except Exception as e:
                print(f"[{self.agent_id}] Neural network decision failed for {request_type}: {e}")
                # Only in case of failure, log the issue but don't create fallback bids
                print(f"[{self.agent_id}] Agent {self.agent_type} cannot participate in {request_type}")
        
        # Store the status update in market data - handle both AgentState and dict cases
        if hasattr(self.state, 'market_data'):
            # AgentState object
            if "status_updates" not in self.state.market_data:
                self.state.market_data["status_updates"] = []
            self.state.market_data["status_updates"].append({
                "timestamp": datetime.now().isoformat(),
                "sender": message.sender_id,
                "content": content
            })
        elif isinstance(self.state, dict):
            # Dict state
            if 'market_data' not in self.state:
                self.state['market_data'] = {}
            if "status_updates" not in self.state['market_data']:
                self.state['market_data']["status_updates"] = []
            self.state['market_data']["status_updates"].append({
                "timestamp": datetime.now().isoformat(),
                "sender": message.sender_id,
                "content": content
            }) 
    
    async def send_message(self, receiver_id: str, message_type: MessageType, 
                          content: Dict[str, Any], priority: int = 1) -> None:
        """Send a message to another agent"""
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority
        )
        print(f"[{self.agent_id}] Sending {message_type.value} to {receiver_id}")
        
        # Route the message through the message router if available
        if self.message_router:
            await self.message_router.route_message(message)
    
    async def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent"""
        try:
            if hasattr(self._message_queue, 'put_nowait'):
                # Regular queue
                self.message_queue.put_nowait(message)
            else:
                # Asyncio queue
                await self.message_queue.put(message)
        except Exception:
            # If queue operations fail, just continue
            pass
    
    async def run_decision_cycle(self) -> AgentState:
        """Run one complete decision-making cycle"""
        config = RunnableConfig()
        result = await self.graph.ainvoke(self.state, config)
        self.state = result
        return result
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current agent state for monitoring"""
        # Handle both AgentState and AddableValuesDict cases
        timestamp = datetime.now().isoformat()
        if hasattr(self.state, 'current_timestamp'):
            timestamp = self.state.current_timestamp.isoformat()
        elif isinstance(self.state, dict) and 'current_timestamp' in self.state:
            timestamp = self.state['current_timestamp'].isoformat()
            
        operational_status = {}
        if hasattr(self.state, 'operational_status'):
            operational_status = self.state.operational_status
        elif isinstance(self.state, dict) and 'operational_status' in self.state:
            operational_status = self.state['operational_status']
            
        performance_metrics = {}
        if hasattr(self.state, 'performance_metrics'):
            performance_metrics = self.state.performance_metrics
        elif isinstance(self.state, dict) and 'performance_metrics' in self.state:
            performance_metrics = self.state['performance_metrics']
            
        decisions = []
        if hasattr(self.state, 'decisions'):
            decisions = self.state.decisions[-5:] if self.state.decisions else []
        elif isinstance(self.state, dict) and 'decisions' in self.state:
            decisions = self.state['decisions'][-5:] if self.state['decisions'] else []
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "timestamp": timestamp,
            "operational_status": operational_status,
            "performance_metrics": performance_metrics,
            "recent_decisions": decisions
        } 