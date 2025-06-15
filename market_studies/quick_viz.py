import json
import pandas as pd
import matplotlib.pyplot as plt

# Load data
with open('market_mechanism_experiments.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['results'])
config_df = pd.json_normalize(df['config'])
df = pd.concat([df.drop('config', axis=1), config_df], axis=1)

print(f'Loaded {len(df)} experiments')

# Create simple analysis
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. Efficiency vs Price
ax1.scatter(df['average_lmp'], df['market_clearing_efficiency'], alpha=0.7)
ax1.set_xlabel('Average LMP ($/MWh)')
ax1.set_ylabel('Market Efficiency')
ax1.set_title('Market Efficiency vs Average Price')
ax1.grid(True, alpha=0.3)

# 2. Price Volatility Distribution
ax2.hist(df['price_volatility'], bins=10, alpha=0.7, color='orange')
ax2.set_xlabel('Price Volatility')
ax2.set_ylabel('Frequency')
ax2.set_title('Price Volatility Distribution')

# 3. Consumer vs Producer Surplus
ax3.scatter(df['consumer_surplus'], df['producer_surplus'], alpha=0.7, color='green')
ax3.set_xlabel('Consumer Surplus ($)')
ax3.set_ylabel('Producer Surplus ($)')
ax3.set_title('Consumer vs Producer Surplus')

# 4. Market Concentration
ax4.hist(df['hhi_index'], bins=10, alpha=0.7, color='red')
ax4.set_xlabel('HHI Index')
ax4.set_ylabel('Frequency')
ax4.set_title('Market Concentration (HHI)')

plt.tight_layout()
plt.savefig('market_analysis_quick.png', dpi=300, bbox_inches='tight')
plt.show()
print('Charts saved as market_analysis_quick.png') 