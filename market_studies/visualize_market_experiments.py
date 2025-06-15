#!/usr/bin/env python3
"""
Market Mechanism Experiments Visualization
Comprehensive analysis of different market configurations
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load and process the market experiments data"""
    with open('market_mechanism_experiments.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data['results'])
    
    # Extract config information into separate columns
    config_df = pd.json_normalize(df['config'])
    df = pd.concat([df.drop('config', axis=1), config_df], axis=1)
    
    # Clean up column names
    df.columns = [col.replace('config.', '') for col in df.columns]
    
    return df

def create_efficiency_vs_price_scatter(df):
    """Create scatter plot showing efficiency vs price trade-offs"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by auction type
    auction_types = df['auction_type'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, auction_type in enumerate(auction_types):
        subset = df[df['auction_type'] == auction_type]
        ax.scatter(subset['average_lmp'], subset['market_clearing_efficiency'], 
                  c=colors[i], s=subset['price_volatility']*10, alpha=0.7,
                  label=auction_type.replace('AuctionType.', ''))
    
    ax.set_xlabel('Average LMP ($/MWh)', fontsize=12)
    ax.set_ylabel('Market Clearing Efficiency', fontsize=12)
    ax.set_title('Market Efficiency vs Price Trade-off\n(Bubble size = Price Volatility)', 
                fontsize=14, fontweight='bold')
    ax.legend(title='Auction Type')
    ax.grid(True, alpha=0.3)
    
    # Add annotations for optimal points
    best_efficiency = df.loc[df['market_clearing_efficiency'].idxmax()]
    ax.annotate(f'Best Efficiency\n{best_efficiency["market_clearing_efficiency"]:.1%}', 
                xy=(best_efficiency['average_lmp'], best_efficiency['market_clearing_efficiency']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_price_cap_analysis(df):
    """Analyze the impact of different price caps"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Filter for price cap experiments
    price_cap_df = df[df['auction_type'] == 'AuctionType.CONTINUOUS'].copy()
    price_cap_df['price_cap_label'] = price_cap_df['price_cap'].fillna('No Cap').astype(str)
    
    # 1. Efficiency by price cap
    caps = price_cap_df.groupby('price_cap_label')['market_clearing_efficiency'].mean()
    ax1.bar(caps.index, caps.values, color='steelblue', alpha=0.8)
    ax1.set_title('Market Efficiency by Price Cap', fontweight='bold')
    ax1.set_ylabel('Market Clearing Efficiency')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Average price by cap
    prices = price_cap_df.groupby('price_cap_label')['average_lmp'].mean()
    ax2.bar(prices.index, prices.values, color='coral', alpha=0.8)
    ax2.set_title('Average LMP by Price Cap', fontweight='bold')
    ax2.set_ylabel('Average LMP ($/MWh)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Price volatility by cap
    volatility = price_cap_df.groupby('price_cap_label')['price_volatility'].mean()
    ax3.bar(volatility.index, volatility.values, color='lightgreen', alpha=0.8)
    ax3.set_title('Price Volatility by Price Cap', fontweight='bold')
    ax3.set_ylabel('Price Volatility')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Consumer surplus by cap
    surplus = price_cap_df.groupby('price_cap_label')['consumer_surplus'].mean()
    ax4.bar(surplus.index, surplus.values, color='purple', alpha=0.8)
    ax4.set_title('Consumer Surplus by Price Cap', fontweight='bold')
    ax4.set_ylabel('Consumer Surplus ($)')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def create_transaction_cost_impact(df):
    """Analyze transaction cost impacts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter for transaction cost experiments
    tx_df = df[(df['auction_type'] == 'AuctionType.CONTINUOUS') & 
               (df['market_power_strategy'] == 'MarketPowerStrategy.NONE')].copy()
    
    # 1. Consumer vs Producer surplus by transaction costs
    ax1.scatter(tx_df['transaction_costs'], tx_df['consumer_surplus'], 
               color='blue', alpha=0.7, s=100, label='Consumer Surplus')
    ax1.scatter(tx_df['transaction_costs'], tx_df['producer_surplus'], 
               color='red', alpha=0.7, s=100, label='Producer Surplus')
    ax1.set_xlabel('Transaction Costs ($)')
    ax1.set_ylabel('Surplus ($)')
    ax1.set_title('Welfare Distribution vs Transaction Costs', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Market efficiency vs transaction fee rate
    fee_rates = tx_df['transaction_fee_rate'] * 100  # Convert to percentage
    ax2.scatter(fee_rates, tx_df['market_clearing_efficiency'], 
               c=tx_df['average_lmp'], s=100, alpha=0.7, cmap='viridis')
    ax2.set_xlabel('Transaction Fee Rate (%)')
    ax2.set_ylabel('Market Clearing Efficiency')
    ax2.set_title('Efficiency vs Transaction Fee Rate\n(Color = Average LMP)', fontweight='bold')
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Average LMP ($/MWh)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_market_power_heatmap(df):
    """Create heatmap showing market power control effectiveness"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Prepare data for market power analysis
    mp_df = df[df['market_power_strategy'] != 'MarketPowerStrategy.NONE'].copy()
    mp_df['strategy'] = mp_df['market_power_strategy'].str.replace('MarketPowerStrategy.', '')
    
    if len(mp_df) > 0:
        # 1. Market concentration (HHI) by strategy
        hhi_data = mp_df.pivot_table(values='hhi_index', index='strategy', 
                                    columns='market_share_limit', aggfunc='mean')
        sns.heatmap(hhi_data, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax1)
        ax1.set_title('Market Concentration (HHI Index)', fontweight='bold')
        
        # 2. Price manipulation events
        manip_data = mp_df.pivot_table(values='price_manipulation_events', index='strategy',
                                      columns='market_share_limit', aggfunc='mean')
        sns.heatmap(manip_data, annot=True, fmt='.1f', cmap='Reds', ax=ax2)
        ax2.set_title('Price Manipulation Events', fontweight='bold')
        
        # 3. Market efficiency
        eff_data = mp_df.pivot_table(values='market_clearing_efficiency', index='strategy',
                                    columns='market_share_limit', aggfunc='mean')
        sns.heatmap(eff_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax3)
        ax3.set_title('Market Clearing Efficiency', fontweight='bold')
        
        # 4. Average LMP
        price_data = mp_df.pivot_table(values='average_lmp', index='strategy',
                                      columns='market_share_limit', aggfunc='mean')
        sns.heatmap(price_data, annot=True, fmt='.1f', cmap='viridis', ax=ax4)
        ax4.set_title('Average LMP ($/MWh)', fontweight='bold')
    else:
        # If no market power data, show message
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, 'No Market Power\nExperiments Found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def create_performance_radar(df):
    """Create radar chart comparing top configurations"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Select top 3 configurations by efficiency
    top_configs = df.nlargest(3, 'market_clearing_efficiency')
    
    # Normalize metrics to 0-1 scale for radar chart
    metrics = ['market_clearing_efficiency', 'consumer_surplus', 'producer_surplus']
    normalized_data = {}
    
    for metric in metrics:
        max_val = df[metric].max()
        min_val = df[metric].min()
        normalized_data[metric] = (df[metric] - min_val) / (max_val - min_val)
    
    # Add inverse of volatility (stability)
    max_vol = df['price_volatility'].max()
    min_vol = df['price_volatility'].min()
    normalized_data['price_stability'] = 1 - (df['price_volatility'] - min_vol) / (max_vol - min_vol)
    
    # Radar chart angles
    angles = np.linspace(0, 2 * np.pi, len(metrics) + 1, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    labels = ['Market Efficiency', 'Consumer Surplus', 'Producer Surplus', 'Price Stability']
    
    colors = ['red', 'blue', 'green']
    for i, (idx, config) in enumerate(top_configs.iterrows()):
        values = [normalized_data['market_clearing_efficiency'][idx],
                 normalized_data['consumer_surplus'][idx],
                 normalized_data['producer_surplus'][idx],
                 normalized_data['price_stability'][idx]]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Config {i+1}', color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title('Top 3 Configurations Performance Comparison', 
                fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def create_summary_metrics_table(df):
    """Create a summary table of key metrics"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate summary statistics
    summary_data = {
        'Metric': [
            'Experiments Run',
            'Best Market Efficiency',
            'Average LMP Range',
            'Lowest Price Volatility',
            'Highest Consumer Surplus',
            'Most Manipulation Events',
            'Average HHI Index'
        ],
        'Value': [
            len(df),
            f"{df['market_clearing_efficiency'].max():.1%}",
            f"${df['average_lmp'].min():.1f} - ${df['average_lmp'].max():.1f}/MWh",
            f"{df['price_volatility'].min():.1f}",
            f"${df['consumer_surplus'].max():,.0f}",
            int(df['price_manipulation_events'].max()),
            f"{df['hhi_index'].mean():.3f}"
        ],
        'Configuration': [
            'Total across all experiments',
            f"Price cap: {df.loc[df['market_clearing_efficiency'].idxmax(), 'price_cap']}",
            'Range across all experiments',
            f"Price cap: ${df.loc[df['price_volatility'].idxmin(), 'price_cap']}",
            f"Price cap: ${df.loc[df['consumer_surplus'].idxmax(), 'price_cap']}",
            'Transaction cost experiments',
            'Moderate market concentration'
        ]
    }
    
    table = ax.table(cellText=[[summary_data['Metric'][i], 
                               summary_data['Value'][i], 
                               summary_data['Configuration'][i]] 
                              for i in range(len(summary_data['Metric']))],
                    colLabels=['Performance Metric', 'Best Value', 'Notes'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.3, 0.2, 0.5])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(summary_data['Metric']) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#4ECDC4')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F8F9FA' if i % 2 == 0 else 'white')
    
    ax.set_title('Market Mechanism Experiments - Key Performance Summary', 
                fontweight='bold', fontsize=16, pad=20)
    
    plt.tight_layout()
    return fig

def main():
    """Generate all visualizations"""
    print("📊 Loading market experiments data...")
    df = load_data()
    print(f"✅ Loaded {len(df)} experiments")
    
    # Create all visualizations
    print("🎨 Creating visualizations...")
    
    figs = {
        'efficiency_vs_price': create_efficiency_vs_price_scatter(df),
        'price_cap_analysis': create_price_cap_analysis(df),
        'transaction_costs': create_transaction_cost_impact(df),
        'market_power': create_market_power_heatmap(df),
        'performance_radar': create_performance_radar(df),
        'summary_table': create_summary_metrics_table(df)
    }
    
    # Save all figures
    print("💾 Saving visualizations...")
    for name, fig in figs.items():
        fig.savefig(f'market_analysis_{name}.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: market_analysis_{name}.png")
    
    # Show all plots
    plt.show()
    
    print("🎉 All visualizations complete!")
    print("\n📈 Generated Charts:")
    print("   1. Efficiency vs Price Trade-off (Scatter)")
    print("   2. Price Cap Impact Analysis (Bar Charts)")
    print("   3. Transaction Cost Impact (Scatter)")
    print("   4. Market Power Controls (Heatmaps)")
    print("   5. Top Configurations Comparison (Radar)")
    print("   6. Summary Metrics Table")

if __name__ == "__main__":
    main() 