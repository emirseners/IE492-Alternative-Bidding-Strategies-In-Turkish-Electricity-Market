import math
import matplotlib.pyplot as plt
from itertools import accumulate

quantity = 1000
price = 2000
sigma = 250

prices = list(range(0, 3001))
quantities = [math.exp(-((p - price)**2) / (2 * sigma**2)) for p in prices]

total_quantity = sum(quantities)
normalized_quantities = [q / total_quantity for q in quantities]
quantities = [q * quantity for q in normalized_quantities]

cumulative_quantities = list(accumulate(quantities))

fig, ax = plt.subplots(figsize=(10, 6))

line1, = ax.plot(prices, quantities, label='Quantity Distribution', color='blue')
ax.set_xlabel('Price')
ax.set_ylabel('Quantity')
ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

ax2 = ax.twinx()
line2, = ax2.plot(prices, cumulative_quantities, label='Cumulative Quantities', color='red')
ax2.set_ylabel('Cumulative Quantity')

plt.title('Bidding Quantity vs Bidding Price')

lines = [line1, line2]
labels = [l.get_label() for l in lines]
legend = ax.legend(lines, labels, loc='upper left')

legend.get_frame().set_facecolor('none')
legend.get_frame().set_edgecolor('none')

plt.savefig('bidding_quantity_vs_price.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()