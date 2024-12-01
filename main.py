import pandas as pd
import ast
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

class Simulation:
    def __init__(self, agent_attributes, training_data, validation_data, exogenous_data):
        self.agent_attributes = agent_attributes
        self.training_data = training_data
        self.validation_data = validation_data
        self.exogenous_data = exogenous_data
        self.agents = []
        self.market = DayAheadMarket()
        self.market_results = []
        self.agent_trades = {}
        self.dates = validation_data['Date'].tolist()
        self.num_periods = len(self.dates)
    
    def create_and_add_agents(self):
        for attributes in self.agent_attributes:
            strategy_class_name = attributes['bidding_strategy']
            strategy_params = attributes.get('strategy_params', {})
            if isinstance(strategy_params, str):
                strategy_params = ast.literal_eval(strategy_params)

            agent_name = attributes['name']
            if attributes['type'].lower() == 'consumer':
                if agent_name in self.validation_data.columns:
                    schedule = self.validation_data[agent_name].tolist()
                    price_column = f"{agent_name}_Price"
                    if price_column in self.validation_data.columns:
                        prices = self.validation_data[price_column].tolist()
                    else:
                        raise ValueError(f"Consumer error: {agent_name}")
                else:
                    raise ValueError(f"Consumer error: {agent_name}")
                agent = Consumer(name=agent_name, demand_schedule=schedule, bidding_prices=prices)
            else:
                if strategy_class_name == 'NaiveBiddingStrategy':
                    strategy = NaiveBiddingStrategy(**strategy_params)
                elif strategy_class_name == 'MovingAverageBiddingStrategy':
                    strategy = MovingAverageBiddingStrategy(**strategy_params)
                elif strategy_class_name == 'ArimaBiddingStrategy':
                    strategy = ArimaBiddingStrategy(**strategy_params)
                elif strategy_class_name == 'NaturalGasBiddingStrategy':
                    strategy = NaturalGasBiddingStrategy(exogenous_data=self.exogenous_data, training_data=self.training_data, **strategy_params)
                else:
                    raise ValueError(f"Bidding strategy error: {strategy_class_name}")
                strategy.train(self.training_data)

                if agent_name in self.validation_data.columns:
                    schedule = self.validation_data[agent_name].tolist()
                else:
                    raise ValueError(f"Agent error: {agent_name}")
                
                agent = Producer(name=agent_name, generation_schedule=schedule, bidding_strategy=strategy)

            self.agents.append(agent)
        self.market.agents.extend(self.agents)
        self.agent_trades = {agent.name: [0]*self.num_periods for agent in self.agents}
    
    def run_day_ahead_market(self):
        for index, date in enumerate(self.dates):
            self.market.collect_bids(index, date)
            market_result = self.market.clear_market(index)
            self.market_results.append(market_result)
            for agent_name, quantity in market_result['agent_trades'].items():
                self.agent_trades[agent_name][index] = quantity
            self.market.bids = []

class DayAheadMarket:
    def __init__(self):
        self.agents = []
        self.bids = []
        self.market_prices = []

    def collect_bids(self, index, date):
        for agent in self.agents:
            bids = agent.submit_bid(index, date)
            if bids:
                self.bids.extend(bids)

    def clear_market(self, index):
        supply_bids = [bid for bid in self.bids if isinstance(bid.agent, Producer)]
        demand_bids = [bid for bid in self.bids if isinstance(bid.agent, Consumer)]
        supply_bids.sort(key=lambda x: x.price)
        demand_bids.sort(key=lambda x: -x.price)
        s_i = 0
        d_i = 0
        total_traded_quantity = 0
        market_clearing_price = None
        agent_trades = {agent.name: 0 for agent in self.agents}

        while s_i < len(supply_bids) and d_i < len(demand_bids):
            if supply_bids[s_i].price <= demand_bids[d_i].price:
                traded_quantity = min(supply_bids[s_i].quantity, demand_bids[d_i].quantity)
                market_clearing_price = (supply_bids[s_i].price + demand_bids[d_i].price) / 2
                total_traded_quantity += traded_quantity
                agent_trades[supply_bids[s_i].agent.name] += traded_quantity
                agent_trades[demand_bids[d_i].agent.name] -= traded_quantity
                supply_bids[s_i].quantity -= traded_quantity
                demand_bids[d_i].quantity -= traded_quantity
                if supply_bids[s_i].quantity == 0:
                    s_i += 1
                if demand_bids[d_i].quantity == 0:
                    d_i += 1
            else:
                break

        if market_clearing_price is None:
            raise ValueError(f"Market clear error: {index}")

        self.market_prices.append(market_clearing_price)

        market_result = {'Date': self.agents[0].dates[index], 'market_clearing_price': market_clearing_price, 'total_traded_quantity': total_traded_quantity, 'agent_trades': agent_trades}
        return market_result

class Agent:
    def __init__(self, name):
        self.name = name
        self.dates = []

class Producer(Agent):
    def __init__(self, name, generation_schedule, bidding_strategy):
        super().__init__(name)
        self.generation_schedule = generation_schedule
        self.bidding_strategy = bidding_strategy

    def submit_bid(self, index, date):
        bids = []
        self.bidding_strategy.create_bid(index, date, self)
        for bid_info in self.bidding_strategy.bidding_prices_quantities:
            price = bid_info['price']
            quantity = bid_info['quantity']
            if quantity > 0:
                bid = Bid(agent=self, quantity=quantity, price=price)
                bids.append(bid)
        if bids:
            self.dates.append(date)
            return bids
        else:
            return []

class Consumer(Agent):
    def __init__(self, name, demand_schedule, bidding_prices):
        super().__init__(name)
        self.demand_schedule = demand_schedule
        self.bidding_prices = bidding_prices

    def submit_bid(self, index, date):
        if index < len(self.demand_schedule) and index < len(self.bidding_prices):
            quantity = self.demand_schedule[index]
            price = self.bidding_prices[index]
        else:
            quantity = 0
            price = 0

        if quantity > 0:
            bid = Bid(agent=self, quantity=quantity, price=price)
            self.dates.append(date)
            return [bid]
        else:
            return []

class BiddingStrategy:
    def __init__(self):
        self.bidding_prices_quantities = []

    def train(self, training_data):
        pass

    def create_bid(self, index, date, agent):
        self.bidding_prices_quantities = []

class NaiveBiddingStrategy(BiddingStrategy):
    def __init__(self, **kwargs):
        super().__init__()
        self.period = int(kwargs.get('period', 1))
        self.historical_prices = []

    def train(self, training_data):
        self.historical_prices = training_data['Prices'].tolist()

    def create_bid(self, index, date, agent):
        self.bidding_prices_quantities = []
        price_index = len(self.historical_prices) - self.period + index
        if 0 <= price_index < len(self.historical_prices):
            price = self.historical_prices[price_index]
        else:
            price = 50
        quantity = agent.generation_schedule[index] if index < len(agent.generation_schedule) else 0
        if quantity > 0:
            self.bidding_prices_quantities.append({'price': price, 'quantity': quantity})

class MovingAverageBiddingStrategy(BiddingStrategy):
    def __init__(self, **kwargs):
        super().__init__()
        self.window_size = int(kwargs.get('window_size', 3))
        self.historical_prices = []

    def train(self, training_data):
        self.historical_prices = training_data['Prices'].tolist()

    def create_bid(self, index, date, agent):
        self.bidding_prices_quantities = []
        if len(self.historical_prices) >= self.window_size:
            prices = self.historical_prices[-self.window_size:]
            price = sum(prices) / len(prices)
        else:
            price = 50
        quantity = agent.generation_schedule[index] if index < len(agent.generation_schedule) else 0
        if quantity > 0:
            self.bidding_prices_quantities.append({'price': price, 'quantity': quantity})

class ArimaBiddingStrategy(BiddingStrategy):
    def __init__(self, **kwargs):
        super().__init__()

    def train(self, training_data):
        pass

    def create_bid(self, index, date, agent):
        pass

class NaturalGasBiddingStrategy(BiddingStrategy):
    def __init__(self, exogenous_data, training_data, **kwargs):
        super().__init__()
        self.exogenous_data = exogenous_data
        self.training_data = training_data
        self.num_bids = 1000

    def train(self, training_data):
        pass

    def create_bid(self, index, date, agent):
        self.bidding_prices_quantities = []
        natural_gas_kgup = self.exogenous_data.loc[self.exogenous_data['Date'] == date, 'NaturalgasKgup']

        lag_1_time = (datetime.strptime(date, "%d.%m.%Y %H:%M:%S") - timedelta(days=1)).strftime("%d.%m.%Y %H:%M:%S")
        lag_7_time = (datetime.strptime(date, "%d.%m.%Y %H:%M:%S") - timedelta(days=7)).strftime("%d.%m.%Y %H:%M:%S")

        lag_1_price = self.training_data.loc[self.training_data['Date'] == lag_1_time, 'Prices']
        lag_7_price = self.training_data.loc[self.training_data['Date'] == lag_7_time, 'Prices']

        natural_gas_forecast = 604.57 * math.log(natural_gas_kgup) + 0.14 * lag_1_price + 0.22 * lag_7_price - 3802.96

        P_cost = 1778   # Marginal Cost
        P_estimated = natural_gas_forecast
        P_max = 3000       # Maksimum teklif fiyatı
        Q_max = 15330        # Maksimum üretim miktarı (Mwh)

        mu = P_estimated

        p1 = 0.01  # P_cost için
        p2 = 0.99  # P_max için

        z1 = norm.ppf(p1) 
        z2 = norm.ppf(p2)

        sigma1 = (P_cost - mu) / z1
        sigma2 = (P_max - mu) / z2
        sigma = (abs(sigma1) + abs(sigma2)) / 2

        price_range = np.linspace(P_cost, P_max, 1000)
        cdf_values = norm.cdf(price_range, loc=mu, scale=sigma)
        production = Q_max * cdf_values

        for i in range(self.num_bids):
            price = price_range[i]
            quantity = production[i]
            self.bidding_prices_quantities.append({'price': price, 'quantity': quantity})

class Bid:
    def __init__(self, agent, quantity, price):
        self.agent = agent
        self.quantity = quantity
        self.price = price

if __name__ == "__main__":
    historical_data_df = pd.read_excel('MarketData.xlsx', parse_dates=['Date'])
    historical_data_df.sort_values(by='Date', inplace=True)
    
    exogenous_data_df = pd.read_excel('ExogenousVariables.xlsx', parse_dates=['Date'])
    exogenous_data_df.sort_values(by='Date', inplace=True)

    simulations = pd.ExcelFile('Agents.xlsx').sheet_names
    validation_window_size = 24

    with pd.ExcelWriter('SimulationResults.xlsx') as writer:
        for simulation_name in simulations:
            print(f"Running Simulation: {simulation_name}")
            agent_df = pd.read_excel('Agents.xlsx', sheet_name=simulation_name)
            agent_df.fillna({'strategy_params': '{}'}, inplace=True)
            agent_attributes = agent_df.to_dict(orient='records')

            bidding_quantities_df = pd.read_excel('BiddingQuantities.xlsx', sheet_name=simulation_name, parse_dates=['Date'])
            bidding_quantities_df.fillna(0, inplace=True)
            bidding_quantities_df.sort_values(by='Date', inplace=True)

            all_simulation_results = []

            num_periods = len(bidding_quantities_df)
            for t_start in range(0, num_periods, validation_window_size):
                t_end = t_start + validation_window_size
                if t_end > num_periods:
                    break
                validation_data = bidding_quantities_df.iloc[t_start:t_end].reset_index(drop=True)
                validation_dates = validation_data['Date'].tolist()

                training_data = historical_data_df[historical_data_df['Date'] < validation_dates[0]]

                exogenous_data = exogenous_data_df[exogenous_data_df['Date'].isin(validation_dates)].reset_index(drop=True)

                simulation = Simulation(agent_attributes, training_data, validation_data, exogenous_data)
                simulation.create_and_add_agents()
                simulation.run_day_ahead_market()

                for index, result in enumerate(simulation.market_results):
                    agent_trades = result['agent_trades']
                    date = result['Date']
                    actual_price_row = historical_data_df[historical_data_df['Date'] == date]
                    if not actual_price_row.empty:
                        actual_price = actual_price_row['Prices'].values[0]
                    else:
                        actual_price = None
                    for agent_name, quantity in agent_trades.items():
                        all_simulation_results.append({
                            'Date': date,
                            'Agent': agent_name,
                            'Quantity': quantity,
                            'SimulatedPrice': result['market_clearing_price'],
                            'ActualPrice': actual_price,
                            'PriceError': result['market_clearing_price'] - actual_price if actual_price is not None else None,
                            'TotalTradedQuantity': result['total_traded_quantity']
                        })

            df_results = pd.DataFrame(all_simulation_results)
            df_results.to_excel(writer, sheet_name=simulation_name, index=False)
            print(f"Completed Simulation: {simulation_name}")
