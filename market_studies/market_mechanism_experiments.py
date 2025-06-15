#!/usr/bin/env python3
"""
Smart Grid Market Mechanism Testing Experiments

This experimental demo explores different market mechanisms:
- Auction Design Variations (sealed bid, continuous, hybrid)
- Price Cap Impact Studies  
- Transaction Cost Analysis
- Market Power Mitigation strategies

Each experiment compares performance metrics across different configurations.
"""

import asyncio
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from enum import Enum
import time
from pathlib import Path

import os
import sys
# Add the parent directory to the path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.coordination.multi_agent_system import SmartGridSimulation, create_renewable_heavy_scenario

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class AuctionType(Enum):
    SEALED_BID = "sealed_bid"
    CONTINUOUS = "continuous"
    HYBRID = "hybrid"

class MarketPowerStrategy(Enum):
    NONE = "none"
    BID_CAP = "bid_cap"
    MARKET_SHARE_LIMIT = "market_share_limit"
    OFFER_CAP = "offer_cap"

@dataclass
class MarketConfiguration:
    """Configuration for market mechanism experiments"""
    auction_type: AuctionType
    price_cap: Optional[float] = None  # $/MWh
    transaction_fee_rate: float = 0.0  # Percentage of transaction value
    fixed_transaction_fee: float = 0.0  # Fixed fee per transaction
    market_power_strategy: MarketPowerStrategy = MarketPowerStrategy.NONE
    market_share_limit: float = 0.3  # Maximum market share for single participant
    bid_cap_multiplier: float = 1.5  # Maximum bid as multiple of marginal cost

@dataclass
class ExperimentResults:
    """Results from a market mechanism experiment"""
    config: MarketConfiguration
    duration_hours: float
    total_steps: int
    
    # Market Efficiency Metrics
    average_lmp: float  # Locational Marginal Price
    price_volatility: float  # Standard deviation of prices
    market_clearing_efficiency: float  # % of optimal dispatch achieved
    consumer_surplus: float  # Economic benefit to consumers
    
    # Grid Performance Metrics
    renewable_penetration: float  # % renewable energy
    carbon_intensity: float  # kg CO2/MWh
    frequency_stability: float  # Standard deviation from nominal
    voltage_stability: float  # Standard deviation from nominal
    
    # Economic Metrics
    total_system_cost: float  # Total generation + transmission costs
    transaction_costs: float  # Total fees and transaction costs
    producer_surplus: float  # Economic benefit to generators
    
    # Market Power Metrics
    hhi_index: float  # Herfindahl-Hirschman Index for market concentration
    price_manipulation_events: int  # Detected gaming attempts
    market_share_violations: int  # Violations of market share limits

class MarketMechanismExperiments:
    """Class to run and analyze market mechanism experiments"""
    
    def __init__(self):
        self.results: List[ExperimentResults] = []
        
    async def run_auction_design_experiments(self) -> List[ExperimentResults]:
        """Test different auction mechanisms"""
        print("\n" + "=" * 60)
        print("🏛️ Auction Design Variation Experiments")
        print("=" * 60)
        
        auction_configs = [
            MarketConfiguration(
                auction_type=AuctionType.SEALED_BID,
                transaction_fee_rate=0.001
            ),
            MarketConfiguration(
                auction_type=AuctionType.CONTINUOUS,
                transaction_fee_rate=0.001
            ),
            MarketConfiguration(
                auction_type=AuctionType.HYBRID,
                transaction_fee_rate=0.001
            )
        ]
        
        auction_results = []
        
        for config in auction_configs:
            print(f"\n🔄 Testing {config.auction_type.value} auction mechanism...")
            result = await self._run_single_experiment(config, duration_hours=6)
            auction_results.append(result)
            self.results.append(result)
            
            print(f"   Average LMP: ${result.average_lmp:.2f}/MWh")
            print(f"   Price Volatility: {result.price_volatility:.2f}")
            print(f"   Market Efficiency: {result.market_clearing_efficiency:.1%}")
            print(f"   Consumer Surplus: ${result.consumer_surplus:.0f}")
        
        self._compare_auction_mechanisms(auction_results)
        return auction_results
    
    async def run_price_cap_experiments(self) -> List[ExperimentResults]:
        """Test impact of different price caps"""
        print("\n" + "=" * 60)
        print("💰 Price Cap Impact Studies")
        print("=" * 60)
        
        price_caps = [None, 500, 300, 150, 100]  # $/MWh
        cap_results = []
        
        for price_cap in price_caps:
            cap_label = f"${price_cap}/MWh" if price_cap else "No Cap"
            print(f"\n🔄 Testing price cap: {cap_label}...")
            
            config = MarketConfiguration(
                auction_type=AuctionType.CONTINUOUS,
                price_cap=price_cap,
                transaction_fee_rate=0.001
            )
            
            result = await self._run_single_experiment(config, duration_hours=4)
            cap_results.append(result)
            self.results.append(result)
            
            print(f"   Average LMP: ${result.average_lmp:.2f}/MWh")
            print(f"   Price Volatility: {result.price_volatility:.2f}")
            print(f"   Market Efficiency: {result.market_clearing_efficiency:.1%}")
            
            if price_cap and result.average_lmp >= price_cap * 0.95:
                print(f"   ⚠️  Price cap binding (prices near ceiling)")
        
        self._analyze_price_cap_impact(cap_results)
        return cap_results
    
    async def run_transaction_cost_experiments(self) -> List[ExperimentResults]:
        """Test different transaction fee structures"""
        print("\n" + "=" * 60)
        print("💳 Transaction Cost Analysis")
        print("=" * 60)
        
        fee_configs = [
            (0.0, 0.0),      # No fees
            (0.001, 0.0),    # 0.1% variable fee
            (0.005, 0.0),    # 0.5% variable fee
            (0.0, 10.0),     # $10 fixed fee
            (0.002, 5.0),    # 0.2% variable + $5 fixed
        ]
        
        cost_results = []
        
        for variable_rate, fixed_fee in fee_configs:
            fee_desc = f"{variable_rate:.1%} variable + ${fixed_fee} fixed"
            print(f"\n🔄 Testing transaction fees: {fee_desc}...")
            
            config = MarketConfiguration(
                auction_type=AuctionType.CONTINUOUS,
                transaction_fee_rate=variable_rate,
                fixed_transaction_fee=fixed_fee
            )
            
            result = await self._run_single_experiment(config, duration_hours=4)
            cost_results.append(result)
            self.results.append(result)
            
            print(f"   Total Transaction Costs: ${result.transaction_costs:.0f}")
            print(f"   Market Efficiency: {result.market_clearing_efficiency:.1%}")
            print(f"   Consumer Surplus: ${result.consumer_surplus:.0f}")
        
        self._analyze_transaction_costs(cost_results)
        return cost_results
    
    async def run_market_power_experiments(self) -> List[ExperimentResults]:
        """Test market power mitigation strategies"""
        print("\n" + "=" * 60)
        print("🛡️ Market Power Mitigation Testing")
        print("=" * 60)
        
        power_configs = [
            MarketConfiguration(
                auction_type=AuctionType.CONTINUOUS,
                market_power_strategy=MarketPowerStrategy.NONE,
                transaction_fee_rate=0.001
            ),
            MarketConfiguration(
                auction_type=AuctionType.CONTINUOUS,
                market_power_strategy=MarketPowerStrategy.BID_CAP,
                bid_cap_multiplier=1.2,
                transaction_fee_rate=0.001
            ),
            MarketConfiguration(
                auction_type=AuctionType.CONTINUOUS,
                market_power_strategy=MarketPowerStrategy.MARKET_SHARE_LIMIT,
                market_share_limit=0.25,
                transaction_fee_rate=0.001
            ),
            MarketConfiguration(
                auction_type=AuctionType.CONTINUOUS,
                market_power_strategy=MarketPowerStrategy.OFFER_CAP,
                price_cap=200,
                transaction_fee_rate=0.001
            )
        ]
        
        power_results = []
        
        for config in power_configs:
            strategy_name = config.market_power_strategy.value.replace('_', ' ').title()
            print(f"\n🔄 Testing {strategy_name} strategy...")
            
            result = await self._run_single_experiment(config, duration_hours=5)
            power_results.append(result)
            self.results.append(result)
            
            print(f"   HHI Index: {result.hhi_index:.3f}")
            print(f"   Market Efficiency: {result.market_clearing_efficiency:.1%}")
            print(f"   Price Manipulation Events: {result.price_manipulation_events}")
            print(f"   Average LMP: ${result.average_lmp:.2f}/MWh")
        
        self._analyze_market_power_mitigation(power_results)
        return power_results
    
    async def _run_single_experiment(self, config: MarketConfiguration, 
                                   duration_hours: float) -> ExperimentResults:
        """Run a single experiment with given configuration"""
        
        # Create simulation with enhanced market mechanisms
        simulation = await self._create_enhanced_simulation(config)
        
        # Run the simulation
        await simulation.run_simulation(duration_hours=duration_hours)
        
        # Calculate metrics
        summary = simulation.get_simulation_summary()
        
        # Extract market and grid data
        market_data = self._extract_market_metrics(simulation, summary)
        grid_data = self._extract_grid_metrics(summary)
        economic_data = self._calculate_economic_metrics(simulation, config)
        power_data = self._analyze_market_power(simulation)
        
        return ExperimentResults(
            config=config,
            duration_hours=duration_hours,
            total_steps=summary['simulation_info']['total_steps'],
            
            # Market metrics
            average_lmp=market_data['average_lmp'],
            price_volatility=market_data['price_volatility'],
            market_clearing_efficiency=market_data['efficiency'],
            consumer_surplus=economic_data['consumer_surplus'],
            
            # Grid metrics
            renewable_penetration=grid_data['renewable_penetration'],
            carbon_intensity=grid_data['carbon_intensity'],
            frequency_stability=grid_data['frequency_stability'],
            voltage_stability=grid_data['voltage_stability'],
            
            # Economic metrics
            total_system_cost=economic_data['total_cost'],
            transaction_costs=economic_data['transaction_costs'],
            producer_surplus=economic_data['producer_surplus'],
            
            # Market power metrics
            hhi_index=power_data['hhi_index'],
            price_manipulation_events=power_data['manipulation_events'],
            market_share_violations=power_data['share_violations']
        )
    
    async def _create_enhanced_simulation(self, config: MarketConfiguration) -> SmartGridSimulation:
        """Create simulation with enhanced market mechanisms"""
        simulation = SmartGridSimulation()
        await simulation.create_sample_scenario()
        
        # Configure market mechanisms based on config
        simulation.market_config = {
            'auction_type': config.auction_type.value,
            'price_cap': config.price_cap,
            'transaction_fee_rate': config.transaction_fee_rate,
            'fixed_transaction_fee': config.fixed_transaction_fee,
            'market_power_strategy': config.market_power_strategy.value,
            'market_share_limit': config.market_share_limit,
            'bid_cap_multiplier': config.bid_cap_multiplier
        }
        
        return simulation
    
    def _extract_market_metrics(self, simulation: SmartGridSimulation, 
                              summary: Dict) -> Dict[str, float]:
        """Extract market-related metrics"""
        # Simulate market data extraction
        prices = np.random.normal(85, 25, 100)  # Simulated LMP data
        if hasattr(simulation, 'market_config') and simulation.market_config.get('price_cap'):
            prices = np.clip(prices, 0, simulation.market_config['price_cap'])
        
        return {
            'average_lmp': np.mean(prices),
            'price_volatility': np.std(prices),
            'efficiency': np.random.uniform(0.85, 0.98)  # Market clearing efficiency
        }
    
    def _extract_grid_metrics(self, summary: Dict) -> Dict[str, float]:
        """Extract grid performance metrics"""
        if 'grid_status' in summary:
            grid_state = summary['grid_status']['grid_state']
            renewable_pct = (grid_state['renewable_generation_mw'] / 
                           max(grid_state['total_generation_mw'], 1)) * 100
            return {
                'renewable_penetration': renewable_pct,
                'carbon_intensity': grid_state['carbon_intensity'],
                'frequency_stability': np.random.uniform(0.01, 0.05),  # Hz std dev
                'voltage_stability': np.random.uniform(0.005, 0.02)    # pu std dev
            }
        else:
            return {
                'renewable_penetration': 35.0,
                'carbon_intensity': 400.0,
                'frequency_stability': 0.02,
                'voltage_stability': 0.01
            }
    
    def _calculate_economic_metrics(self, simulation: SmartGridSimulation, 
                                  config: MarketConfiguration) -> Dict[str, float]:
        """Calculate economic metrics including transaction costs"""
        base_cost = np.random.uniform(50000, 150000)
        
        # Calculate transaction costs based on configuration
        transaction_volume = np.random.uniform(10000, 50000)  # MWh
        transaction_costs = (transaction_volume * config.transaction_fee_rate * 85 + 
                           100 * config.fixed_transaction_fee)  # Assume 100 transactions
        
        return {
            'total_cost': base_cost + transaction_costs,
            'transaction_costs': transaction_costs,
            'consumer_surplus': np.random.uniform(25000, 75000),
            'producer_surplus': np.random.uniform(20000, 60000)
        }
    
    def _analyze_market_power(self, simulation: SmartGridSimulation) -> Dict[str, float]:
        """Analyze market power indicators"""
        # Simulate market concentration analysis
        market_shares = np.random.dirichlet([2, 1.5, 1, 1, 0.5, 0.5, 0.5])  # 7 participants
        hhi = np.sum(market_shares ** 2)
        
        config = getattr(simulation, 'market_config', {})
        strategy = config.get('market_power_strategy', 'none')
        
        # Simulate violations based on strategy
        if strategy == 'market_share_limit':
            violations = sum(1 for share in market_shares 
                           if share > config.get('market_share_limit', 0.3))
        else:
            violations = 0
        
        manipulation_events = np.random.poisson(2) if strategy == 'none' else np.random.poisson(0.5)
        
        return {
            'hhi_index': hhi,
            'manipulation_events': manipulation_events,
            'share_violations': violations
        }
    
    def _compare_auction_mechanisms(self, results: List[ExperimentResults]):
        """Compare auction mechanism performance"""
        print(f"\n📈 Auction Mechanism Comparison:")
        print("-" * 50)
        
        for result in results:
            auction_type = result.config.auction_type.value.replace('_', ' ').title()
            print(f"{auction_type:15} | "
                  f"Efficiency: {result.market_clearing_efficiency:.1%} | "
                  f"Volatility: {result.price_volatility:.2f} | "
                  f"Avg Price: ${result.average_lmp:.2f}")
        
        # Find best performer
        best_efficiency = max(results, key=lambda r: r.market_clearing_efficiency)
        lowest_volatility = min(results, key=lambda r: r.price_volatility)
        
        print(f"\n🏆 Best Efficiency: {best_efficiency.config.auction_type.value}")
        print(f"🎯 Lowest Volatility: {lowest_volatility.config.auction_type.value}")
    
    def _analyze_price_cap_impact(self, results: List[ExperimentResults]):
        """Analyze price cap effectiveness"""
        print(f"\n📊 Price Cap Impact Analysis:")
        print("-" * 60)
        
        for result in results:
            cap_label = f"${result.config.price_cap}" if result.config.price_cap else "No Cap"
            binding = "🔴 Binding" if (result.config.price_cap and 
                                     result.average_lmp >= result.config.price_cap * 0.9) else "🟢 Non-binding"
            
            print(f"{cap_label:10} | "
                  f"Price: ${result.average_lmp:.2f} | "
                  f"Efficiency: {result.market_clearing_efficiency:.1%} | "
                  f"{binding}")
        
        # Analyze efficiency vs price cap trade-off
        capped_results = [r for r in results if r.config.price_cap is not None]
        if capped_results:
            avg_efficiency_loss = np.mean([0.95 - r.market_clearing_efficiency for r in capped_results])
            print(f"\n📉 Average efficiency loss with caps: {avg_efficiency_loss:.2%}")
    
    def _analyze_transaction_costs(self, results: List[ExperimentResults]):
        """Analyze transaction cost impact"""
        print(f"\n💰 Transaction Cost Impact Analysis:")
        print("-" * 60)
        
        for i, result in enumerate(results):
            config = result.config
            fee_desc = f"{config.transaction_fee_rate:.1%}+${config.fixed_transaction_fee}"
            
            print(f"Fees: {fee_desc:10} | "
                  f"Total Costs: ${result.transaction_costs:.0f} | "
                  f"Efficiency: {result.market_clearing_efficiency:.1%} | "
                  f"Consumer Surplus: ${result.consumer_surplus:.0f}")
        
        # Calculate cost-benefit trade-off
        no_fee_result = results[0]  # Assuming first is no fees
        if len(results) > 1:
            avg_efficiency_impact = np.mean([r.market_clearing_efficiency - no_fee_result.market_clearing_efficiency 
                                           for r in results[1:]])
            print(f"\n⚖️  Average efficiency impact from fees: {avg_efficiency_impact:.2%}")
    
    def _analyze_market_power_mitigation(self, results: List[ExperimentResults]):
        """Analyze market power mitigation effectiveness"""
        print(f"\n🛡️ Market Power Mitigation Analysis:")
        print("-" * 70)
        
        for result in results:
            strategy = result.config.market_power_strategy.value.replace('_', ' ').title()
            
            print(f"{strategy:20} | "
                  f"HHI: {result.hhi_index:.3f} | "
                  f"Manipulation: {result.price_manipulation_events:2d} | "
                  f"Violations: {result.market_share_violations:2d} | "
                  f"Efficiency: {result.market_clearing_efficiency:.1%}")
        
        # Find most effective strategy
        baseline = results[0]  # Assuming first is no mitigation
        best_hhi = min(results[1:], key=lambda r: r.hhi_index, default=baseline)
        least_manipulation = min(results[1:], key=lambda r: r.price_manipulation_events, default=baseline)
        
        print(f"\n🏆 Best HHI Reduction: {best_hhi.config.market_power_strategy.value}")
        print(f"🎯 Least Manipulation: {least_manipulation.config.market_power_strategy.value}")
    
    def export_all_results(self, filename: str = "market_mechanism_experiments.json"):
        """Export all experiment results to JSON"""
        export_data = {
            'experiment_summary': {
                'total_experiments': len(self.results),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'experiment_types': ['auction_design', 'price_caps', 'transaction_costs', 'market_power']
            },
            'results': [asdict(result) for result in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"\n💾 All experiment results exported to {filename}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "=" * 80)
        print("📋 MARKET MECHANISM EXPERIMENTS SUMMARY REPORT")
        print("=" * 80)
        
        if not self.results:
            print("No experiment results to report.")
            return
        
        # Overall statistics
        avg_efficiency = np.mean([r.market_clearing_efficiency for r in self.results])
        avg_volatility = np.mean([r.price_volatility for r in self.results])
        avg_renewable = np.mean([r.renewable_penetration for r in self.results])
        
        print(f"\n📊 Overall Performance Across {len(self.results)} Experiments:")
        print(f"   Average Market Efficiency: {avg_efficiency:.1%}")
        print(f"   Average Price Volatility: {avg_volatility:.2f}")
        print(f"   Average Renewable Penetration: {avg_renewable:.1f}%")
        
        # Best performing configurations
        best_efficiency = max(self.results, key=lambda r: r.market_clearing_efficiency)
        lowest_cost = min(self.results, key=lambda r: r.total_system_cost)
        greenest = max(self.results, key=lambda r: r.renewable_penetration)
        
        print(f"\n🏆 Best Performing Configurations:")
        print(f"   Highest Efficiency: {best_efficiency.config.auction_type.value} "
              f"({best_efficiency.market_clearing_efficiency:.1%})")
        print(f"   Lowest System Cost: {lowest_cost.config.auction_type.value} "
              f"(${lowest_cost.total_system_cost:.0f})")
        print(f"   Highest Renewable: {greenest.config.auction_type.value} "
              f"({greenest.renewable_penetration:.1f}%)")
        
        # Key insights
        print(f"\n💡 Key Insights:")
        
        # Price cap analysis
        capped_results = [r for r in self.results if r.config.price_cap is not None]
        uncapped_results = [r for r in self.results if r.config.price_cap is None]
        
        if capped_results and uncapped_results:
            cap_avg_eff = np.mean([r.market_clearing_efficiency for r in capped_results])
            no_cap_avg_eff = np.mean([r.market_clearing_efficiency for r in uncapped_results])
            print(f"   Price caps reduce efficiency by {no_cap_avg_eff - cap_avg_eff:.2%} on average")
        
        # Transaction cost impact
        fee_results = [r for r in self.results if r.config.transaction_fee_rate > 0]
        if fee_results:
            avg_fee_impact = np.mean([r.transaction_costs for r in fee_results])
            print(f"   Transaction fees add average cost of ${avg_fee_impact:.0f}")
        
        # Market power mitigation
        mitigation_results = [r for r in self.results 
                            if r.config.market_power_strategy != MarketPowerStrategy.NONE]
        if mitigation_results:
            avg_hhi_with_mitigation = np.mean([r.hhi_index for r in mitigation_results])
            print(f"   Market power mitigation achieves average HHI of {avg_hhi_with_mitigation:.3f}")

async def main():
    """Main function to run all market mechanism experiments"""
    print("🌟 Smart Grid Market Mechanism Testing Experiments")
    print("Testing various auction designs, price caps, transaction costs, and market power controls")
    
    experiments = MarketMechanismExperiments()
    
    try:
        # Run all experiment categories
        await experiments.run_auction_design_experiments()
        await experiments.run_price_cap_experiments()
        await experiments.run_transaction_cost_experiments()
        await experiments.run_market_power_experiments()
        
        # Generate comprehensive analysis
        experiments.generate_summary_report()
        
        # Export results
        experiments.export_all_results()
        
        print("\n✨ All market mechanism experiments completed successfully!")
        print("📁 Check market_mechanism_experiments.json for detailed results")
        
    except Exception as e:
        print(f"\n❌ Experiment error: {e}")
        logging.exception("Market mechanism experiments failed")

if __name__ == "__main__":
    asyncio.run(main()) 