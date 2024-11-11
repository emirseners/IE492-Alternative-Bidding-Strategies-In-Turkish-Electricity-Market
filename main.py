import pandas as pd
import ast
import random

class Simulation:
    def __init__(self, historical_prices, historical_demand, historical_supply, agent_attributes):
        self.historical_prices = historical_prices
        self.historical_demand = historical_demand
        self.historical_supply = historical_supply
        self.agent_attributes = agent_attributes
        self.agents = []
        self.market = DayAheadMarket()
        self.market_results = []
        self.agent_trades = {}

    def create_and_add_agents(self):
        for attributes in self.agent_attributes:
            strategy_class_name = attributes['bidding_strategy']
            strategy_params = attributes.get('strategy_params', {})
            if isinstance(strategy_params, str):
                strategy_params = ast.literal_eval(strategy_params)
            if strategy_class_name == 'NaiveBiddingStrategy':
                strategy = NaiveBiddingStrategy(self.historical_prices, **strategy_params)
            elif strategy_class_name == 'MovingAverageBiddingStrategy':
                strategy = MovingAverageBiddingStrategy(self.historical_prices, **strategy_params)
            else:
                raise ValueError(f"Bidding strategy error: {strategy_class_name}")

            schedule = [float(attributes[col]) for col in [f'production_demand_{i}' for i in range(1, 25)]]
            if attributes['type'].lower() == 'producer':
                agent = Producer(name=attributes['name'], generation_schedule=schedule, bidding_strategy=strategy, startup_cost=float(attributes.get('startup_cost', 0)), variable_cost=float(attributes.get('variable_cost', 0)))
            elif attributes['type'].lower() == 'consumer':
                agent = Consumer(name=attributes['name'], demand_schedule=schedule, bidding_strategy=strategy)
            else:
                raise ValueError(f"Agent type error: {attributes['type']}")
            self.agents.append(agent)

        self.market.agents.extend(self.agents)
        self.agent_trades = {agent.name: [0]*24 for agent in self.agents}

    def run_day_ahead_market(self):
        for hour in range(24):
            self.market.collect_bids(hour)
            market_result = self.market.clear_market(hour)
            self.market_results.append(market_result)
            for agent_name, quantity in market_result['agent_trades'].items():
                self.agent_trades[agent_name][hour] = quantity
            self.market.bids = []

    def run_simulation(self):
        self.create_and_add_agents()
        self.run_day_ahead_market()

class Market:
    def __init__(self):
        self.agents = []
        self.bids = []

class DayAheadMarket(Market):
    def __init__(self):
        super().__init__()

    def collect_bids(self, hour):
        for agent in self.agents:
            bid = agent.submit_bid(hour)
            if bid:
                self.bids.append(bid)

    def clear_market(self, hour):
        supply_bids = [bid for bid in self.bids if isinstance(bid.agent, Producer)]
        demand_bids = [bid for bid in self.bids if isinstance(bid.agent, Consumer)]
        supply_bids.sort(key=lambda x: x.price)
        demand_bids.sort(key=lambda x: -x.price)
        s_i = 0
        d_i = 0
        total_traded_quantity = 0
        market_clearing_price = None
        agent_hourly_trades = {agent.name: 0 for agent in self.agents}
        while s_i < len(supply_bids) and d_i < len(demand_bids):
            if supply_bids[s_i].price <= demand_bids[d_i].price:
                traded_quantity = min(supply_bids[s_i].quantity, demand_bids[d_i].quantity)
                market_clearing_price = (supply_bids[s_i].price + demand_bids[d_i].price) / 2
                total_traded_quantity += traded_quantity
                agent_hourly_trades[supply_bids[s_i].agent.name] += traded_quantity
                agent_hourly_trades[demand_bids[d_i].agent.name] -= traded_quantity
                supply_bids[s_i].quantity -= traded_quantity
                demand_bids[d_i].quantity -= traded_quantity
                if supply_bids[s_i].quantity == 0:
                    s_i += 1
                if demand_bids[d_i].quantity == 0:
                    d_i += 1
            else:
                break
        if market_clearing_price is None:
            raise ValueError(f"Market did not clear for hour {hour}")

        market_result = {'hour': hour, 'market_clearing_price': market_clearing_price, 'total_traded_quantity': total_traded_quantity, 'agent_trades': agent_hourly_trades}
        return market_result

class Agent:
    def __init__(self, name, bidding_strategy):
        self.name = name
        self.bidding_strategy = bidding_strategy

class Producer(Agent):
    def __init__(self, name, generation_schedule, bidding_strategy, startup_cost=0, variable_cost=0):
        super().__init__(name, bidding_strategy)
        self.generation_schedule = generation_schedule
        self.startup_cost = startup_cost
        self.variable_cost = variable_cost
        self.is_running = True

    def calculate_marginal_cost(self, quantity):                                #Adjust later
        return self.variable_cost

    def submit_bid(self, hour):                                                 #Adjust marginal cost also here later
        self.bidding_strategy.determine_price(hour, self)
        price = self.bidding_strategy.bidding_price
        quantity = self.generation_schedule[hour]
        marginal_cost = self.calculate_marginal_cost(quantity)
        price = max(price, marginal_cost)
        if quantity > 0:
            bid = Bid(agent=self, quantity=quantity, price=price)
            return bid
        else:
            return None

class Consumer(Agent):
    def __init__(self, name, demand_schedule, bidding_strategy):
        super().__init__(name, bidding_strategy)
        self.demand_schedule = demand_schedule

    def submit_bid(self, hour):
        self.bidding_strategy.determine_price(hour, self)
        price = self.bidding_strategy.bidding_price
        quantity = self.demand_schedule[hour]
        if quantity > 0:
            bid = Bid(agent=self, quantity=quantity, price=price)
            return bid
        else:
            return None

class BiddingStrategy:
    def __init__(self, historical_prices):
        self.historical_prices = historical_prices
        self.bidding_price = None

class NaiveBiddingStrategy(BiddingStrategy):
    def __init__(self, historical_prices, **kwargs):
        super().__init__(historical_prices)
        self.period = int(kwargs.get('period', 1))

    def determine_price(self, hour, agent):
        index = len(self.historical_prices) - self.period + hour
        if 0 <= index < len(self.historical_prices):
            self.bidding_price = self.historical_prices[index]
        else:
            raise ValueError(f"Naive strategy error: {agent.name}")

class MovingAverageBiddingStrategy(BiddingStrategy):
    def __init__(self, historical_prices, **kwargs):
        super().__init__(historical_prices)
        self.window_size = int(kwargs['window_size'])
        self.period = int(kwargs.get('period', 1))

    def determine_price(self, hour, agent):
        prices = []
        for i in range(1, self.window_size + 1):
            index = len(self.historical_prices) + hour - i * self.period
            if 0 <= index < len(self.historical_prices):
                prices.append(self.historical_prices[index])
            else:
                raise ValueError(f"Moving average error: {agent.name}")
        if prices:
            self.bidding_price = sum(prices) / len(prices)
        else:
            raise ValueError(f"Moving average error: {agent.name}")

class ZeroBiddingStrategy(BiddingStrategy):
    def __init__(self, historical_prices):
        super().__init__(historical_prices)

    def determine_price(self, hour, agent):
        self.bidding_price = 0

class RandomBiddingStrategy(BiddingStrategy):
    def __init__(self, historical_prices):
        super().__init__(historical_prices)

    def determine_price(self, hour, agent):
        pass

class HighRiskBiddingStrategy(BiddingStrategy):
    def __init__(self, historical_prices):
        super().__init__(historical_prices)

    def determine_price(self, hour, agent):
        pass

class LowRiskBiddingStrategy(BiddingStrategy):
    def __init__(self, historical_prices):
        super().__init__(historical_prices)

    def determine_price(self, hour, agent):
        pass

class Bid:
    def __init__(self, agent, quantity, price, bid_type='hourly', hours=None):
        self.agent = agent
        self.quantity = quantity
        self.price = price
        self.bid_type = bid_type
        self.hours = hours if hours else []

if __name__ == "__main__":
    historical_data = pd.read_excel('MarketData.xlsx')
    historical_prices = historical_data['Prices'].tolist()
    historical_demand = historical_data['Demand'].tolist()
    historical_supply = historical_data['Supply'].tolist()

    simulations = pd.ExcelFile('Agents.xlsx').sheet_names
    with pd.ExcelWriter('SimulationResults.xlsx') as writer:
        for simulation_name in simulations:
            print(f"Running Simulation: {simulation_name}")
            agent_df = pd.read_excel('Agents.xlsx', sheet_name=simulation_name)
            agent_df.fillna({'startup_cost': 0, 'variable_cost': 0, 'strategy_params': '{}'}, inplace=True)
            agent_attributes = agent_df.to_dict(orient='records')

            simulation = Simulation(historical_prices.copy(), historical_demand.copy(), historical_supply.copy(), agent_attributes)
            simulation.run_simulation()
            simulation_results = []
            for result in simulation.market_results:
                agent_trades = result['agent_trades']
                for agent_name, quantity in agent_trades.items():
                    simulation_results.append({
                        'Hour': result['hour'],
                        'Agent': agent_name,
                        'Quantity': quantity,
                        'MarketClearingPrice': result['market_clearing_price'],
                        'TotalTradedQuantity': result['total_traded_quantity']
                    })

            df_results = pd.DataFrame(simulation_results)
            df_results.to_excel(writer, sheet_name=simulation_name, index=False)

            for result in simulation.market_results:
                print(f"Hour {result['hour']}: Clearing Price = {result['market_clearing_price']}, "f"Total Traded Quantity = {result['total_traded_quantity']}")
                print("Agent Trades:")
                for agent_name, quantity in result['agent_trades'].items():
                    print(f"  {agent_name}: {quantity}")
                print()