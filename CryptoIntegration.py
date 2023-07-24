import requests
import csv
import json
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class CryptoIntegration:
    def __init__(self, api_key, api_endpoint="https://api.cryptocurrency.com"):
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.fetched_data = None
        self.processed_data = None

    def fetch_crypto_data(self, use_cache=True):
        if use_cache and self.fetched_data:
            print("Using cached cryptocurrency data.")
            return self.fetched_data

        try:
            url = f"{self.api_endpoint}/cryptocurrency_data"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise exception if response status is not 200
            self.fetched_data = response.json()
            return self.fetched_data
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch cryptocurrency data: {str(e)}")
            return None

    def process_data(self):
        if not self.fetched_data:
            print("No cryptocurrency data to process. Fetch data first.")
            return

        # Example data processing: Extracting relevant information
        relevant_data = []
        for item in self.fetched_data:
            relevant_info = {
                'name': item['name'],
                'symbol': item['symbol'],
                'price': item['price'],
                'market_cap': item['market_cap'],
                'timestamp': item['timestamp']  # Assuming the API provides a timestamp for each data point
            }
            relevant_data.append(relevant_info)

        self.processed_data = relevant_data

    def save_data_to_csv(self, file_path):
        if not self.processed_data:
            print("No processed data to save. Process data first.")
            return

        try:
            with open(file_path, 'w', newline='') as csvfile:
                fieldnames = ['name', 'symbol', 'price', 'market_cap', 'timestamp']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for item in self.processed_data:
                    writer.writerow(item)
            print("Cryptocurrency data saved to CSV successfully.")
        except IOError as e:
            print(f"Error while saving data to CSV: {str(e)}")

    def save_data_to_json(self, file_path):
        if not self.processed_data:
            print("No processed data to save. Process data first.")
            return

        try:
            with open(file_path, 'w') as json_file:
                json.dump(self.processed_data, json_file, indent=4)
            print("Cryptocurrency data saved to JSON successfully.")
        except IOError as e:
            print(f"Error while saving data to JSON: {str(e)}")

    def save_data_to_database(self, db_path):
        if not self.processed_data:
            print("No processed data to save. Process data first.")
            return

        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS cryptocurrency_data
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          name TEXT,
                          symbol TEXT,
                          price REAL,
                          market_cap REAL,
                          timestamp TEXT)''')

            for item in self.processed_data:
                c.execute("INSERT INTO cryptocurrency_data (name, symbol, price, market_cap, timestamp) VALUES (?, ?, ?, ?, ?)",
                          (item['name'], item['symbol'], item['price'], item['market_cap'], item['timestamp']))

            conn.commit()
            conn.close()
            print("Cryptocurrency data saved to the database successfully.")
        except sqlite3.Error as e:
            print(f"Error while saving data to the database: {str(e)}")

    def calculate_metrics(self):
        if not self.processed_data:
            print("No processed data to calculate metrics. Process data first.")
            return

        # Example: Calculate average price and market cap
        prices = [item['price'] for item in self.processed_data]
        market_caps = [item['market_cap'] for item in self.processed_data]
        average_price = sum(prices) / len(prices)
        average_market_cap = sum(market_caps) / len(market_caps)

        return {
            'average_price': average_price,
            'average_market_cap': average_market_cap
        }

    def visualize_data(self, metric='market_cap', top_n=10):
        if not self.processed_data:
            print("No processed data to visualize. Process data first.")
            return

        # Example: Visualize the top n cryptocurrencies by a specified metric
        df = pd.DataFrame(self.processed_data)
        top_n_data = df.nlargest(top_n, metric)
        plt.bar(top_n_data['name'], top_n_data[metric])
        plt.xlabel('Cryptocurrency')
        plt.ylabel(metric.capitalize())
        plt.title(f'Top {top_n} Cryptocurrencies by {metric.capitalize()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def time_series_analysis(self):
        if not self.processed_data:
            print("No processed data for time series analysis. Process data first.")
            return

        # Example: Perform time series analysis on price data
        df = pd.DataFrame(self.processed_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        price_series = df['price']

        # Plot time series
        plt.plot(price_series)
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.title('Cryptocurrency Price Time Series')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Compute rolling mean and plot
        window_size = 30
        rolling_mean = price_series.rolling(window=window_size).mean()
        plt.plot(price_series, label='Price')
        plt.plot(rolling_mean, label=f'{window_size}-Day Rolling Mean', color='red')
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.title('Cryptocurrency Price Time Series with Rolling Mean')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def filter_data_by_symbol(self, symbol):
        if not self.processed_data:
            print("No processed data to filter. Process data first.")
            return

        # Filter data by symbol
        filtered_data = [item for item in self.processed_data if item['symbol'] == symbol]
        return filtered_data

    def sort_data_by_metric(self, metric, ascending=True):
        if not self.processed_data:
            print("No processed data to sort. Process data first.")
            return

        # Sort data by the specified metric
        sorted_data = sorted(self.processed_data, key=lambda item: item[metric], reverse=not ascending)
        return sorted_data

    def aggregate_data_by_symbol(self):
        if not self.processed_data:
            print("No processed data to aggregate. Process data first.")
            return

        # Example: Aggregate data by symbol and calculate average price and market cap per symbol
        aggregated_data = {}
        for item in self.processed_data:
            symbol = item['symbol']
            if symbol not in aggregated_data:
                aggregated_data[symbol] = {
                    'average_price': item['price'],
                    'average_market_cap': item['market_cap'],
                    'count': 1
                }
            else:
                aggregated_data[symbol]['average_price'] += item['price']
                aggregated_data[symbol]['average_market_cap'] += item['market_cap']
                aggregated_data[symbol]['count'] += 1

        # Calculate averages
        for symbol, data in aggregated_data.items():
            data['average_price'] /= data['count']
            data['average_market_cap'] /= data['count']

        return aggregated_data

if __name__ == '__main__':
    api_key = input("Enter your cryptocurrency API key: ")

    crypto_integration = CryptoIntegration(api_key)
    crypto_data = crypto_integration.fetch_crypto_data()
    crypto_integration.process_data()
    crypto_integration.save_data_to_csv("crypto_data.csv")
    crypto_integration.save_data_to_json("crypto_data.json")
    crypto_integration.save_data_to_database("crypto_database.db")

    metrics = crypto_integration.calculate_metrics()
    print("Calculated Metrics:")
    print(metrics)

    crypto_integration.visualize_data(metric='market_cap', top_n=10)
    crypto_integration.time_series_analysis()

    filtered_data = crypto_integration.filter_data_by_symbol('BTC')
    print("Filtered Data for BTC:")
    print(filtered_data)

    sorted_data = crypto_integration.sort_data_by_metric('market_cap', ascending=False)
    print("Top 10 Cryptocurrencies by Market Cap:")
    print(sorted_data[:10])

    aggregated_data = crypto_integration.aggregate_data_by_symbol()
    print("Aggregated Data by Symbol:")
    print(aggregated_data)
