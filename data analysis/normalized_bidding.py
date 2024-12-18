import math

quantity = 1000
price = 2000
sigma = 2

lower_bound = price - int(3*sigma)
upper_bound = price + int(3*sigma)
prices = list(range(lower_bound, upper_bound + 1))
quantities = [math.exp(-((i - price)**2) / (2 * sigma**2)) for i in prices]
total_quantitiy = sum(quantities)
normalized_quantities = [q / total_quantitiy for q in quantities]
quantities = [q * quantity for q in normalized_quantities]

print(prices)
print(quantities)
print(sum(quantities))