#!/usr/bin/env python3
"""
Market Mechanism Experiments Visualization
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data():
    with open('market_mechanism_experiments.json', 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['results'])
    config_df = pd.json_normalize(df['config'])
    df = pd.concat([df.drop('config', axis=1), config_df], axis=1)
    return df

def create_comprehensive_analysis():
    df = load_data()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Efficiency vs Price Scatter
    ax1 = plt.subplot(3, 3, 1)
    auction_types = df['auction_type'].unique()
    colors = ['red', 'blue', 'green', 'orange']
    for i, atype in enumerate(auction_types):
        subset = df[df['auction_type'] == atype]
        ax1.scatter(subset['average_lmp'], subset['market_clearing_efficiency'], 
                   c=colors[i], s=60, alpha=0.7, label=atype.split('.')[-1])
    ax1.set_xlabel('Average LMP ($/MWh)')
    ax1.set_ylabel('Market Efficiency')
    ax1.set_title('Efficiency vs Price by Auction Type')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Price Cap Impact
    ax2 = plt.subplot(3, 3, 2)
    price_cap_df = df[df['auction_type'] == 'AuctionType.CONTINUOUS']
    price_cap_df['cap_label'] = price_cap_df['price_cap'].fillna('No Cap').astype(str)
    cap_efficiency = price_cap_df.groupby('cap_label')['market_clearing_efficiency'].mean()
    ax2.bar(range(len(cap_efficiency)), cap_efficiency.values, color='steelblue')
    ax2.set_xticks(range(len(cap_efficiency)))
    ax2.set_xticklabels(cap_efficiency.index, rotation=45)
    ax2.set_title('Market Efficiency by Price Cap')
    ax2.set_ylabel('Efficiency')
    
    # 3. Volatility Comparison
    ax3 = plt.subplot(3, 3, 3)
    volatility_data = [df[df['auction_type'] == atype]['price_volatility'].values 
                      for atype in auction_types]
    ax3.boxplot(volatility_data, labels=[t.split('.')[-1] for t in auction_types])
    ax3.set_title('Price Volatility by Auction Type')
    ax3.set_ylabel('Price Volatility')
    plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # 4. Consumer vs Producer Surplus
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(df['consumer_surplus'], df['producer_surplus'], 
               c=df['market_clearing_efficiency'], s=60, alpha=0.7, cmap='viridis')
    ax4.set_xlabel('Consumer Surplus ($)')
    ax4.set_ylabel('Producer Surplus ($)')
    ax4.set_title('Consumer vs Producer Surplus\\n(Color = Efficiency)')
    
    # 5. Transaction Cost Impact
    ax5 = plt.subplot(3, 3, 5)
    tx_df = df[df['auction_type'] == 'AuctionType.CONTINUOUS']
    ax5.scatter(tx_df['transaction_costs'], tx_df['market_clearing_efficiency'], 
               c='red', s=60, alpha=0.7)
    ax5.set_xlabel('Transaction Costs ($)')
    ax5.set_ylabel('Market Efficiency')
    ax5.set_title('Efficiency vs Transaction Costs')
    ax5.grid(True, alpha=0.3)
    
    # 6. Market Concentration (HHI)
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(df['hhi_index'], bins=10, color='lightcoral', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('HHI Index')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Market Concentration Distribution')
    ax6.axvline(df['hhi_index'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["hhi_index"].mean():.3f}')
    ax6.legend()
    
    # 7. Price Manipulation Events
    ax7 = plt.subplot(3, 3, 7)
    manip_counts = df['price_manipulation_events'].value_counts().sort_index()
    ax7.bar(manip_counts.index, manip_counts.values, color='orange', alpha=0.7)
    ax7.set_xlabel('Manipulation Events')
    ax7.set_ylabel('Number of Experiments')
    ax7.set_title('Price Manipulation Events Distribution')
    
    # 8. Efficiency Distribution
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(df['market_clearing_efficiency'], bins=15, color='lightgreen', 
             alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Market Clearing Efficiency')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Market Efficiency Distribution')
    ax8.axvline(df['market_clearing_efficiency'].mean(), color='green', 
                linestyle='--', label=f'Mean: {df["market_clearing_efficiency"].mean():.3f}')
    ax8.legend()
    
    # 9. Performance Summary Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create summary statistics
    summary_text = f"""
    EXPERIMENT SUMMARY
    
    Total Experiments: {len(df)}
    
    EFFICIENCY METRICS:
    Best Efficiency: {df['market_clearing_efficiency'].max():.1%}
    Average Efficiency: {df['market_clearing_efficiency'].mean():.1%}
    
    PRICE METRICS:
    Price Range: ${df['average_lmp'].min():.1f} - ${df['average_lmp'].max():.1f}/MWh
    Avg Volatility: {df['price_volatility'].mean():.1f}
    
    WELFARE METRICS:
    Max Consumer Surplus: ${df['consumer_surplus'].max():,.0f}
    Max Producer Surplus: ${df['producer_surplus'].max():,.0f}
    
    MARKET STRUCTURE:
    Avg HHI Index: {df['hhi_index'].mean():.3f}
    Max Manipulation Events: {df['price_manipulation_events'].max():.0f}
    """
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('market_experiments_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("📊 Creating comprehensive market analysis...")
    create_comprehensive_analysis()
    print("✅ Analysis complete! Saved as 'market_experiments_comprehensive_analysis.png'") 