import pandas as pd
import ast
import numpy as np
from datetime import timedelta
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Simulation:
    def __init__(self, agent_attributes, start_date, end_date, bidding_quantities, historical_data, exogenous_data, consumer_bid_data):
        self.agent_attributes = agent_attributes
        self.start_date = start_date
        self.end_date = end_date
        self.bidding_quantities = bidding_quantities
        self.exogenous_data = exogenous_data
        self.consumer_bid_data = consumer_bid_data
        self.historical_data = historical_data
        self.dates = pd.date_range(self.start_date, self.end_date, freq='H')
        self.agents = []
        self.market = DayAheadMarket()
        self.market_results = []
        self.actual_prices_map = {}
        self.simulated_prices_map = {}
        self.agent_trades = {}

    def create_and_add_agents(self):
        filtered_bq = self.bidding_quantities[self.bidding_quantities['Date'].isin(self.dates)]
        date_value_map = {}
        for col in filtered_bq.columns:
            if col != 'Date':
                date_value_map[col] = dict(zip(filtered_bq['Date'], filtered_bq[col]))

        for attributes in self.agent_attributes:
            agent_type = attributes['type'].lower()
            agent_name = attributes['name']
            strategy_class_name = attributes.get('bidding_strategy', '')
            strategy_params = attributes.get('strategy_params', {})
            if isinstance(strategy_params, str):
                strategy_params = ast.literal_eval(strategy_params)
            if agent_type == 'consumer':
                strategy = ConsumerBiddingStrategy(consumer_bid_data = self.consumer_bid_data)
                agent = Consumer(name=agent_name, bidding_strategy=strategy)
            else:
                if strategy_class_name == 'NaiveBiddingStrategy':
                    strategy = NaiveBiddingStrategy(historical_data=self.historical_data, bidding_quantities=self.bidding_quantities, **strategy_params)
                elif strategy_class_name == 'MovingAverageBiddingStrategy':
                    strategy = MovingAverageBiddingStrategy(historical_data=self.historical_data, bidding_quantities=self.bidding_quantities, **strategy_params)
                elif strategy_class_name == 'NaturalGasBiddingStrategy':
                    strategy = NaturalGasBiddingStrategy(exogenous_data=self.exogenous_data, historical_data=self.historical_data, bidding_quantities=self.bidding_quantities, **strategy_params)
                elif strategy_class_name == 'CoalBiddingStrategy':
                    strategy = CoalBiddingStrategy(exogenous_data=self.exogenous_data, historical_data=self.historical_data, bidding_quantities=self.bidding_quantities, **strategy_params)
                elif strategy_class_name == 'DammedHydroBiddingStrategy':
                    strategy = DammedHydroBiddingStrategy(exogenous_data=self.exogenous_data, historical_data=self.historical_data, bidding_quantities=self.bidding_quantities, **strategy_params)
                elif strategy_class_name == 'ZeroBiddingStrategy':
                    strategy = ZeroBiddingStrategy(exogenous_data=self.exogenous_data, historical_data=self.historical_data, bidding_quantities=self.bidding_quantities, **strategy_params)
                else:
                    strategy = BiddingStrategy()

                generation_schedule = date_value_map.get(agent_name, {})
                agent = Producer(name=agent_name, generation_schedule=generation_schedule, bidding_strategy=strategy)

            self.agents.append(agent)

        self.market.agents.extend(self.agents)
        self.agent_trades = {agent.name: {date:0 for date in self.dates} for agent in self.agents}

    def run_day_ahead_market(self):
        for date in self.dates:
            self.market.collect_bids(date)
            market_result = self.market.clear_market(date)
            self.market_results.append(market_result)
            actual_price_row = self.historical_data[self.historical_data['Date'] == date]
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
    def __init__(self, consumer_bid_data):
        super().__init__()
        self.consumer_bid_data = consumer_bid_data

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        date_row = self.consumer_bid_data[self.consumer_bid_data['date'] == date]
        date_row_sorted = date_row.sort_values('price', ascending = False)
        prices = date_row_sorted['price'].values
        quantities = date_row_sorted['demand'].values

        previous_quantity = 0
        for i in range(len(prices)):
            price = prices[i]
            q = quantities[i]
            bidding_quantity = abs(q - previous_quantity)
            previous_quantity = abs(q)
            if bidding_quantity > 0:
                self.bidding_prices_quantities.append({'price': price, 'quantity': bidding_quantity})

class NaiveBiddingStrategy(BiddingStrategy):
    def __init__(self, historical_data, bidding_quantities, **kwargs):
        super().__init__()
        self.period = int(kwargs.get('period', 1))
        self.historical_data = historical_data.sort_values('Date')
        self.bidding_quantities = bidding_quantities

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        price = self.historical_data[self.historical_data['Date'] == date - timedelta(days=self.period)]
        quantity = agent.generation_schedule.get(date, 0)
        self.bidding_prices_quantities.append({'price': price, 'quantity': quantity})

class MovingAverageBiddingStrategy(BiddingStrategy):
    def __init__(self, historical_data, bidding_quantities, **kwargs):
        super().__init__()
        self.window_size = int(kwargs.get('window_size', 3))
        self.historical_data = historical_data.sort_values('Date')
        self.bidding_quantities = bidding_quantities

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        ma_dates = [date - timedelta(days=i) for i in range(1, self.window_size+1)]
        price = self.historical_data[self.historical_data['Date'].isin(ma_dates)]['Prices'].mean()
        quantity = agent.generation_schedule.get(date, 0)
        self.bidding_prices_quantities.append({'price': price, 'quantity': quantity})

class NaturalGasBiddingStrategy(BiddingStrategy):
    def __init__(self, exogenous_data, historical_data, bidding_quantities, **kwargs):
        super().__init__()
        self.exogenous_data = exogenous_data
        self.historical_data = historical_data
        self.bidding_quantities = bidding_quantities
        self.num_bids = 1000
        self.exogenous_data['price_day_before'] = self.historical_data['Prices'].shift(24)
        self.exogenous_data['price_week_before'] = self.historical_data['Prices'].shift(168)
        self.exogenous_data = self.exogenous_data.dropna().reset_index(drop=True)
        self.historical_data['Prices'] = self.historical_data['Prices'].shift(168)
        self.historical_data = self.historical_data.dropna().reset_index(drop=True)

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        natural_gas_row = self.exogenous_data[self.exogenous_data['Date'] == date]
        natural_gas_kgup = natural_gas_row['NaturalgasKgupRegression'].values[0]
        natural_gas_kgup_normalized = natural_gas_row['NaturalgasKgup'].values[0]
        lag_1_price = natural_gas_row['price_day_before'].values[0] #self.historical_data[self.historical_data['Date'] == date - timedelta(days=1)]['Prices'].values[0]
        lag_7_price = natural_gas_row['price_week_before'].values[0] #self.historical_data[self.historical_data['Date'] == date - timedelta(days=7)]['Prices'].values[0]

        X = self.exogenous_data[['NaturalgasKgupRegression', 'price_day_before', 'price_week_before']].copy()
        X['NaturalgasKgupRegression'] = np.log(X['NaturalgasKgupRegression'])
        y = self.historical_data['Prices']


        model = LinearRegression()
        model.fit(X, y)

        X_pred_df = pd.DataFrame({
            'NaturalgasKgupRegression': [np.log(natural_gas_kgup)],
            'price_day_before': [lag_1_price],
            'price_week_before': [lag_7_price]
        })

        point_estimate = model.predict(X_pred_df)[0]

        std_dev = max(point_estimate * 0.05, 1e-3)
        price_range = np.linspace(0, 3000, 3001)
        pdf_values = norm.pdf(price_range, loc=point_estimate, scale=std_dev)
        cdf_at_mean = norm.cdf(point_estimate, loc=point_estimate, scale=std_dev)
        cdf_at_0 = norm.cdf(0, loc=point_estimate, scale=std_dev)
        cdf_diff = cdf_at_mean - cdf_at_0
        if cdf_diff == 0:
            return
        scale_factor = natural_gas_kgup_normalized / cdf_diff

        quantities = pdf_values * scale_factor
        for i in range(len(price_range)):
            price = price_range[i]
            quantity = quantities[i]
            if quantity > 0:
                self.bidding_prices_quantities.append({'price': price, 'quantity': quantity})

class CoalBiddingStrategy(BiddingStrategy):
    def __init__(self, exogenous_data, historical_data, bidding_quantities, **kwargs):
        super().__init__()
        self.exogenous_data = exogenous_data
        self.historical_data = historical_data
        self.bidding_quantities = bidding_quantities

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        #total_production = 2000                                                              # Adjust

        coal_price_row = self.exogenous_data[self.exogenous_data['Date'] == date]
        coal_price = coal_price_row['CoalPrice'].values[0]
        total_production = coal_price_row['CoalKgup'].values[0]/2
        efficiencies = [0.31,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49]
        mult = [0.020386693, 0.03329996, 0.061515957, 0.088816841, 0.100357438, 0.10335929, 0.085269833, 0.077181166, 0.083742451, 0.071588017, 0.068526615, 0.052327369, 0.039356713, 0.037069851, 0.025454545, 0.022527402, 0.015602837, 0.008446164, 0.005170858]        
        productions = [total_production * m for m in mult]

        electricity_generation = 7
        startup_fuel = 7.50
        other_startup_cost = 5.61
        cycling_cost = other_startup_cost + startup_fuel*(coal_price/25.792)
        markdown = cycling_cost/16

        prices = []
        for eff in efficiencies:
            prices.append((coal_price/(eff*electricity_generation)) - markdown)

        for i in range(len(prices)):
            price = prices[i]
            quantity = productions[i]
            self.bidding_prices_quantities.append({'price': price, 'quantity': quantity})

class DammedHydroBiddingStrategy(BiddingStrategy):
    def __init__(self, exogenous_data, historical_data, bidding_quantities, **kwargs):
        super().__init__()
        self.exogenous_data = exogenous_data
        self.historical_data = historical_data
        self.bidding_quantities = bidding_quantities
        self.num_bids = 1000

        self.exogenous_data = pd.merge(self.exogenous_data, self.historical_data[['Date','Prices']], on='Date', how='left')
        self.exogenous_data['price_day_before'] = self.exogenous_data['Prices'].shift(24)
        self.exogenous_data = self.exogenous_data.dropna().reset_index(drop=True)
        self.exogenous_data['DammedHydro/RL'] = self.exogenous_data['DammedHydroKgupRegression'] / self.exogenous_data['ResidualLoad']
        self.historical_data['Prices'] = self.historical_data['Prices'].shift(24)
        self.historical_data = self.historical_data.dropna().reset_index(drop=True)

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        exog_row = self.exogenous_data[self.exogenous_data['Date'] == date]
        dammed_hydro_kgup = exog_row['DammedHydroKgupRegression'].values[0]
        dammed_hydro_kgup_normalized = exog_row['DammedHydroKgup'].values[0]
        residual_load = exog_row['ResidualLoad'].values[0]
        dammed_over_RL = dammed_hydro_kgup/residual_load

        lag_1_price = self.exogenous_data[self.exogenous_data['Date'] == date - timedelta(days=1)]['Prices'].values[0]

        train_data = self.exogenous_data[(self.exogenous_data['Date'].dt.year == (date.year - 1)) & 
                                         (self.exogenous_data['Date'].dt.month == date.month)]

        X = self.exogenous_data[['DammedHydro/RL', 'DammedHydroKgupRegression', 'price_day_before']]
        y = self.historical_data['Prices']

        model = LinearRegression()
        model.fit(X, y)

        X_pred_df = pd.DataFrame({
            'DammedHydro/RL': [dammed_over_RL],
            'DammedHydroKgupRegression': [dammed_hydro_kgup],
            'price_day_before': [lag_1_price]
        })

        point_estimate = model.predict(X_pred_df)[0]

        std_dev = max(point_estimate * 0.05, 1e-3)
        price_range = np.linspace(0, 3000, 3001)
        pdf_values = norm.pdf(price_range, loc=point_estimate, scale=std_dev)
        cdf_at_mean = norm.cdf(point_estimate, loc=point_estimate, scale=std_dev)
        cdf_at_0 = norm.cdf(0, loc=point_estimate, scale=std_dev)

        cdf_diff = cdf_at_mean - cdf_at_0
        if cdf_diff == 0:
            return
        scale_factor = dammed_hydro_kgup_normalized / cdf_diff

        quantities = pdf_values * scale_factor
        for i in range(len(price_range)):
            price = price_range[i]
            quantity = quantities[i]
            if quantity > 0:
                self.bidding_prices_quantities.append({'price': price, 'quantity': quantity})

class ZeroBiddingStrategy(BiddingStrategy):
    def __init__(self, exogenous_data, historical_data, bidding_quantities, **kwargs):
        super().__init__()
        self.exogenous_data = exogenous_data
        self.historical_data = historical_data
        self.bidding_quantities = bidding_quantities

    def create_bid(self, date, agent):
        self.bidding_prices_quantities = []
        zero_bid_row = self.exogenous_data[self.exogenous_data['Date'] == date]
        zero_bid_amount = zero_bid_row['ZeroBidQuantity'].values[0]
        self.bidding_prices_quantities.append({'price':0,'quantity':zero_bid_amount})

class Bid:
    def __init__(self, agent, quantity, price):
        self.agent = agent
        self.quantity = quantity
        self.price = price

if __name__ == "__main__":
    start_date = pd.to_datetime('01.06.2024 00:00:00', dayfirst=True)
    end_date = pd.to_datetime('01.10.2024 00:00:00', dayfirst=True)

    historical_data_df = pd.read_excel('MarketData.xlsx')
    historical_data_df['Date'] = pd.to_datetime(historical_data_df['Date'], format='%d.%m.%y %H:%M:%S')
    historical_data_df.sort_values(by='Date', inplace=True)

    exogenous_data_df = pd.read_excel('ExogenousVariables.xlsx')
    exogenous_data_df['Date'] = pd.to_datetime(exogenous_data_df['Date'], format='%d.%m.%y %H:%M:%S')
    exogenous_data_df.sort_values(by='Date', inplace=True)

    consumer_bid_data_df = pd.read_csv('ConsumerBidData.csv')
    consumer_bid_data_df['date'] = pd.to_datetime(consumer_bid_data_df['date'], format='%d.%m.%Y %H:%M:%S')
    consumer_bid_data_df.sort_values(by='date', inplace=True)

    simulations = pd.ExcelFile('Agents.xlsx').sheet_names

    with pd.ExcelWriter('SimulationResults.xlsx') as writer:
        for simulation_name in simulations:
            print(f"Running Simulation: {simulation_name}")
            agent_df = pd.read_excel('Agents.xlsx', sheet_name=simulation_name)
            agent_df.fillna({'strategy_params': '{}'}, inplace=True)
            agent_attributes = agent_df.to_dict(orient='records')

            bidding_quantities_df = pd.read_excel('BiddingQuantities.xlsx', sheet_name=simulation_name)
            bidding_quantities_df['Date'] = pd.to_datetime(bidding_quantities_df['Date'], format='%d.%m.%y %H:%M:%S')
            bidding_quantities_df.fillna(0, inplace=True)
            bidding_quantities_df.sort_values(by='Date', inplace=True)

            simulation = Simulation(agent_attributes=agent_attributes, start_date=start_date, end_date=end_date, bidding_quantities=bidding_quantities_df, historical_data=historical_data_df, exogenous_data=exogenous_data_df, consumer_bid_data=consumer_bid_data_df)
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
                
                errors_data = {
                    'Date': simulation.dates, 
                    'Errors': errors,
                    'Actual Prices': actual_prices,
                    'Simulated Prices': simulated_prices
                               }

                errors_df = pd.DataFrame(errors_data) 
                errors_df.to_excel('errors.xlsx', index=False)

                print("Metrics after simulation:")
                print(f" MSE: {mse}")
                print(f" Mean Error: {mean_errors}")
                if wmape is not None:
                    print(f" WMAPE: {wmape}")

                plt.figure(figsize=(8, 6))
                plt.plot([i+1 for i in range(len(errors))], errors, marker='o', color='blue', linestyle='-', linewidth=2, markersize=6)

                plt.title("Errors in Simulation")
                plt.xlabel("Indices")
                plt.ylabel("Errors")

                plt.grid(True)
                plt.show()