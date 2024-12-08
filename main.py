import pandas as pd
import ast
import numpy as np
import math
from datetime import timedelta
from scipy.stats import norm

class Simulation:
    def __init__(self, agent_attributes, training_data, validation_data, exogenous_data, consumer_bid_data):
        self.agent_attributes = agent_attributes
        self.training_data = training_data
        self.validation_data = validation_data
        self.exogenous_data = exogenous_data
        self.consumer_bid_data = consumer_bid_data
        self.agents = []
        self.market = DayAheadMarket()
        self.market_results = []
        self.agent_trades = {}
        self.dates = self.validation_data['Date'].tolist()
        self.num_periods = len(self.dates)
        self.actual_prices_map = {}
        self.simulated_prices_map = {}

    def create_and_add_agents(self):
        date_value_map = {}
        for col in self.validation_data.columns:
            if col != 'Date':
                date_value_map[col] = dict(zip(self.validation_data['Date'], self.validation_data[col]))

        for attributes in self.agent_attributes:
            agent_type = attributes['type'].lower()
            agent_name = attributes['name']
            strategy_class_name = attributes.get('bidding_strategy', '')
            strategy_params = attributes.get('strategy_params', {})
            if isinstance(strategy_params, str):
                strategy_params = ast.literal_eval(strategy_params)
            if agent_type == 'consumer':
                strategy = ConsumerBiddingStrategy(exogenous_data=self.consumer_bid_data, training_data=self.training_data, **strategy_params)
                agent = Consumer(name=agent_name, bidding_strategy=strategy)
            else:
                if strategy_class_name == 'NaiveBiddingStrategy':
                    strategy = NaiveBiddingStrategy(training_data=self.training_data, **strategy_params)
                elif strategy_class_name == 'MovingAverageBiddingStrategy':
                    strategy = MovingAverageBiddingStrategy(training_data=self.training_data, **strategy_params)
                elif strategy_class_name == 'NaturalGasBiddingStrategy':
                    strategy = NaturalGasBiddingStrategy(exogenous_data=self.exogenous_data, training_data=self.training_data, **strategy_params)
                elif strategy_class_name == 'CoalBiddingStrategy':
                    strategy = CoalBiddingStrategy(exogenous_data=self.exogenous_data, training_data=self.training_data, **strategy_params)
                elif strategy_class_name == 'DammedHydroBiddingStrategy':
                    strategy = DammedHydroBiddingStrategy(exogenous_data=self.exogenous_data, training_data=self.training_data, **strategy_params)
                elif strategy_class_name == 'ZeroBiddingStrategy':
                    strategy = ZeroBiddingStrategy(exogenous_data=self.exogenous_data, training_data=self.training_data, **strategy_params)

                generation_schedule = date_value_map[agent_name]
                agent = Producer(name=agent_name, generation_schedule=generation_schedule, bidding_strategy=strategy)

            self.agents.append(agent)

        self.market.agents.extend(self.agents)
        self.agent_trades = {agent.name: {date:0 for date in self.dates} for agent in self.agents}

    def run_day_ahead_market(self):
        for date in self.dates:
            self.market.collect_bids(date)
            market_result = self.market.clear_market(date)
            self.market_results.append(market_result)

            actual_price_row = self.training_data[self.training_data['Date'] == date]
            actual_price = actual_price_row['Prices'].values[0] if not actual_price_row.empty else None
            simulated_price = market_result['market_clearing_price']
            if actual_price is not None:
                self.actual_prices_map[date] = actual_price
                self.simulated_prices_map[date] = simulated_price

            agent_trades = market_result['agent_trades']
            for agent_name, quantity in agent_trades.items():
                self.agent_trades[agent_name][date] = quantity

            self.market.bids = []

class DayAheadMarket:
    def __init__(self):
        self.agents = []
        self.bids = []
        self.market_prices = []

    def collect_bids(self, date):
        self.bids.clear()
        for agent in self.agents:
            new_bids = agent.submit_bid(date)
            if new_bids:
                self.bids.extend(new_bids)

    def clear_market(self, date):
        supply_bids = [bid for bid in self.bids if isinstance(bid.agent, Producer)]
        demand_bids = [bid for bid in self.bids if isinstance(bid.agent, Consumer)]
        supply_bids.sort(key=lambda x: x.price)
        demand_bids.sort(key=lambda x: -x.price)
        s_i, d_i = 0, 0
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

        self.market_prices.append(market_clearing_price)
        market_result = {'Date': date, 'market_clearing_price': market_clearing_price, 'total_traded_quantity': total_traded_quantity, 'agent_trades': agent_trades}
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

    def submit_bid(self, date):
        self.bidding_strategy.create_bid(date, self)
        bids = []
        for bid_info in self.bidding_strategy.bidding_prices_quantities:
            if bid_info['quantity'] > 0:
                bids.append(Bid(agent=self, quantity=bid_info['quantity'], price=bid_info['price']))
        if bids:
            self.dates.append(date)
            return bids
        else:
            return []

class Consumer(Agent):
    def __init__(self, name, bidding_strategy):
        super().__init__(name)
        self.bidding_strategy = bidding_strategy

    def submit_bid(self, date):
        self.bidding_strategy.create_bid(date, self)
        bids = []
        for bid_info in self.bidding_strategy.bidding_prices_quantities:
            if bid_info['quantity'] > 0:
                bids.append(Bid(agent=self, quantity=bid_info['quantity'], price=bid_info['price']))
        if bids:
            self.dates.append(date)
            return bids
        else:
            return []

class BiddingStrategy:
    def __init__(self):
        self.bidding_prices_quantities = []

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []

class ConsumerBiddingStrategy(BiddingStrategy):
    def __init__(self, exogenous_data, training_data, **kwargs):
        super().__init__()
        self.consumer_bid_data = exogenous_data
        self.training_data = training_data

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        date_row = self.consumer_bid_data[self.consumer_bid_data['Date'] == date]

        date_row_sorted = date_row.sort_values('Prices')
        prices = date_row_sorted['Prices'].values
        quantities = date_row_sorted['Quantity'].values

        previous_quantity = 0
        for i in range(len(prices)):
            price = prices[i]
            q = quantities[i]
            bidding_quantity = q - previous_quantity
            previous_quantity = q
            if bidding_quantity > 0:
                self.bidding_prices_quantities.append({'price': price, 'quantity': bidding_quantity})

class NaiveBiddingStrategy(BiddingStrategy):
    def __init__(self, training_data, **kwargs):
        super().__init__()
        self.period = int(kwargs.get('period', 1))
        self.historical = training_data[['Date','Prices']].sort_values('Date')
        self.date_price_map = dict(zip(self.historical['Date'], self.historical['Prices']))

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        lag_date = date - timedelta(hours=self.period)
        price = self.date_price_map[lag_date]
        quantity = agent.generation_schedule.get(date, 0)
        self.bidding_prices_quantities.append({'price': price, 'quantity': quantity})

class MovingAverageBiddingStrategy(BiddingStrategy):
    def __init__(self, training_data, **kwargs):
        super().__init__()
        self.window_size = int(kwargs.get('window_size', 3))
        self.historical = training_data[['Date','Prices']].sort_values('Date')
        self.date_price_map = dict(zip(self.historical['Date'], self.historical['Prices']))

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        relevant_dates = [d for d in self.date_price_map.keys() if d < date]
        relevant_dates.sort()
        last_dates = relevant_dates[-self.window_size:]
        price = sum(self.date_price_map[d] for d in last_dates)/len(last_dates)
        quantity = agent.generation_schedule.get(date, 0)
        self.bidding_prices_quantities.append({'price': price, 'quantity': quantity})

class NaturalGasBiddingStrategy(BiddingStrategy):
    def __init__(self, exogenous_data, training_data, **kwargs):
        super().__init__()
        self.exogenous_data = exogenous_data
        self.training_data = training_data
        self.num_bids = 1000

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        natural_gas_row = self.exogenous_data[self.exogenous_data['Date'] == date]

        natural_gas_kgup = natural_gas_row['NaturalgasKgup'].values[0]

        lag_1_price_row = self.training_data[self.training_data['Date'] == date - timedelta(days=1)]
        lag_7_price_row = self.training_data[self.training_data['Date'] == date - timedelta(days=7)]

        lag_1_price = lag_1_price_row['Prices'].values[0]
        lag_7_price = lag_7_price_row['Prices'].values[0]

        ##df['price_day_before'] = df['price'].shift(24)
        ##df['price_week_before'] = df['price'].shift(168)
        ##df = df.dropna().reset_index(drop=True)

        ##X = df[['production_forecast', 'price_day_before', 'price_week_before']].copy()
        ##X['production_forecast'] = np.log(X['production_forecast'])
        ##y = df['price']

        model = LinearRegression()
        model.fit(X, y)

        X_pred = np.array([[natural_gas_kgup, lag_1_price, lag_7_price]])
        point_estimate = model.predict(X_pred)[0]

        std_dev = point_estimate * 0.05

        cdf_at_mean = norm.cdf(point_estimate, loc=point_estimate, scale=std_dev)
        cdf_at_0 = norm.cdf(0, loc=point_estimate, scale=std_dev)

        cdf_diff = cdf_at_mean - cdf_at_0

        scale_factor = target_production / cdf_diff

        price_range = np.linspace(0, 3000, 3001)
        pdf_values = norm.pdf(price_range, loc=point_estimate, scale=std_dev)
        quantities = pdf_values * scale_factor

        for i in range(price_range):
            price = price_range[i]
            quantity = quantities[i]
            if quantity > 0:
                self.bidding_prices_quantities.append({'price': price, 'quantity': quantity})

class CoalBiddingStrategy(BiddingStrategy):
    def __init__(self, exogenous_data, training_data, **kwargs):
        super().__init__()
        self.exogenous_data = exogenous_data
        self.training_data = training_data

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        coal_price_row = self.exogenous_data[self.exogenous_data['Date'] == date]
        coal_price = coal_price_row['CoalPrice'].values[0]

        efficiencies = [0.31,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49]
        productions = [309.26,505.16,933.19,1347.35,1522.42,1567.96,1293.54,1170.84,1270.37,1085.99,1039.55,793.80,597.04,562.35,386.15,341.74,236.69,128.13]

        electricity_generation = 7
        startup_fuel = 7.50
        other_startup_cost = 5.61
        cycling_cost = other_startup_cost + startup_fuel*(coal_price/25.792)
        markdown = cycling_cost/16

        prices = []
        for eff in efficiencies:
            prices.append(coal_price/(eff*electricity_generation) - markdown)

        for i in range(len(prices)):
            price = prices[i]
            quantity = productions[i]
            if quantity > 0:
                self.bidding_prices_quantities.append({'price': price, 'quantity': quantity})

class DammedHydroBiddingStrategy(BiddingStrategy):
    def __init__(self, exogenous_data, training_data, **kwargs):
        super().__init__()
        self.exogenous_data = exogenous_data
        self.training_data = training_data
        self.num_bids = 1000

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        exog_row = self.exogenous_data[self.exogenous_data['Date'] == date]
        dammed_hydro_kgup = exog_row['DammedHydroKgup'].values[0]
        residual_load = exog_row['ResidualLoad'].values[0]

        lag_1_time = date - timedelta(days=1)
        lag_1_price_row = self.training_data[self.training_data['Date'] == lag_1_time]

        lag_1_price = lag_1_price_row['Prices'].values[0]
        dammed_hydro_forecast = (342.313*(dammed_hydro_kgup/residual_load)+0.01274*residual_load+0.7737*lag_1_price+179.72)

        P_cost = 0
        mu = dammed_hydro_forecast
        P_max = 3000
        Q_max = 13520

        sigma1 = (P_cost - mu)/norm.ppf(0.01)
        sigma2 = (P_max - mu)/norm.ppf(0.99)
        sigma = (abs(sigma1)+abs(sigma2))/2

        price_range = np.linspace(P_cost, P_max, self.num_bids)
        cdf_values = norm.cdf(price_range, loc=mu, scale=sigma/2)
        production = [max(((2*dammed_hydro_kgup - Q_max)+(2*Q_max-2*dammed_hydro_kgup)*cdf),0) for cdf in cdf_values]

        for i in range(self.num_bids):
            if i==0:
                price = price_range[i]
                quantity = production[i]
            else:
                price = price_range[i]
                quantity = production[i]-production[i-1]
            if quantity >0:
                self.bidding_prices_quantities.append({'price': price,'quantity':quantity})

class ZeroBiddingStrategy(BiddingStrategy):
    def __init__(self, exogenous_data, training_data, **kwargs):
        super().__init__()
        self.exogenous_data = exogenous_data
        self.training_data = training_data

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        zero_bid_row = self.exogenous_data[self.exogenous_data['Date']==date]
        zero_bid_amount = zero_bid_row['ZeroBidQuantity'].values[0]
        self.bidding_prices_quantities.append({'price':0,'quantity':zero_bid_amount})

class Bid:
    def __init__(self, agent, quantity, price):
        self.agent = agent
        self.quantity = quantity
        self.price = price

if __name__ == "__main__":
    start_date = pd.to_datetime('01.01.2023 00:00:00', dayfirst=True)
    end_date = pd.to_datetime('01.01.2023 14:00:00', dayfirst=True)

    historical_data_df = pd.read_excel('MarketData.xlsx', parse_dates=['Date'], dayfirst=True)
    historical_data_df.sort_values(by='Date', inplace=True)
    
    exogenous_data_df = pd.read_excel('ExogenousVariables.xlsx', parse_dates=['Date'], dayfirst=True)
    exogenous_data_df.sort_values(by='Date', inplace=True)

    consumer_bid_data_df = pd.read_excel('ConsumerBidData.xlsx', parse_dates=['Date'], dayfirst=True)
    consumer_bid_data_df.sort_values(by='Date', inplace=True)

    simulations = pd.ExcelFile('Agents.xlsx').sheet_names

    with pd.ExcelWriter('SimulationResults.xlsx') as writer:
        for simulation_name in simulations:
            print(f"Running Simulation: {simulation_name}")
            agent_df = pd.read_excel('Agents.xlsx', sheet_name=simulation_name)
            agent_df.fillna({'strategy_params': '{}'}, inplace=True)
            agent_attributes = agent_df.to_dict(orient='records')

            bidding_quantities_df = pd.read_excel('BiddingQuantities.xlsx', sheet_name=simulation_name, parse_dates=['Date'], dayfirst=True)
            bidding_quantities_df.fillna(0, inplace=True)
            bidding_quantities_df.sort_values(by='Date', inplace=True)

            validation_data = bidding_quantities_df[(bidding_quantities_df['Date'] >= start_date) & (bidding_quantities_df['Date'] <= end_date)].reset_index(drop=True)
            exogenous_data = exogenous_data_df[(exogenous_data_df['Date'] >= start_date) & (exogenous_data_df['Date'] <= end_date)].reset_index(drop=True)
            consumer_bid_data = consumer_bid_data_df[(consumer_bid_data_df['Date'] >= start_date) & (consumer_bid_data_df['Date'] <= end_date)].reset_index(drop=True)
            training_data = historical_data_df[historical_data_df['Date'] < start_date]

            simulation = Simulation(agent_attributes, training_data, validation_data, exogenous_data, consumer_bid_data)
            simulation.create_and_add_agents()
            simulation.run_day_ahead_market()

            final_results = []
            for result in simulation.market_results:
                date = result['Date']
                total_traded_quantity = result['total_traded_quantity']
                for agent_name, quantity in result['agent_trades'].items():
                    final_results.append({'Date': date, 'Agent': agent_name, 'Quantity': quantity, 'TotalTradedQuantity': total_traded_quantity})

            df_results = pd.DataFrame(final_results)
            df_results.to_excel(writer, sheet_name=simulation_name, index=False)

            print(f"Completed Simulation: {simulation_name}")

            actual_prices = []
            simulated_prices = []
            for d in simulation.dates:
                if d in simulation.actual_prices_map and d in simulation.simulated_prices_map:
                    actual_prices.append(simulation.actual_prices_map[d])
                    simulated_prices.append(simulation.simulated_prices_map[d])
            
            if len(actual_prices)>0:
                errors = [sim - act for sim, act in zip(simulated_prices, actual_prices)]
                squared_errors = [(sim - act)**2 for sim,act in zip(simulated_prices, actual_prices)]
                absolute_errors = [abs(act - sim) for act, sim in zip(actual_prices, simulated_prices)]
                sum_abs_errors = sum(absolute_errors)
                sum_actual = sum(abs(a) for a in actual_prices if a is not None)
                wmape = None if sum_actual==0 else sum_abs_errors/sum_actual
                mse = sum(squared_errors)/len(squared_errors)
                mean_errors = sum(errors)/len(errors)

                print("Metrics after simulation:")
                print(f" MSE: {mse}")
                print(f" Mean Error: {mean_errors}")
                if wmape is not None:
                    print(f" WMAPE: {wmape}")

                #for d, predicted, actual in zip(simulation.dates, simulated_prices, actual_prices):
                #    print(f"Date: {d}, Simulated: {predicted}, Actual: {actual}, Error: {predicted - actual}")
