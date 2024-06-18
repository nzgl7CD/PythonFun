import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import starmap
from operator import mul
import timeit
from openpyxl import load_workbook, Workbook
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class PortfolioBackend:
    def __init__(self, expected_return=0.3, risk_free_rate=0.02, n=3, risk_aversion=3, *args):
        self.expected_return = expected_return
        self.risk_free_rate = risk_free_rate
        self.n = n
        self.risk_aversion = risk_aversion
        self.ds = self.get_data(n)
        self.returns = self.calculate_annualized_returns()  
        self.cov_matrix = self.compute_covariance_matrix()
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
    
    def get_data(self,n):
        ds=pd.read_excel('test.xlsx', sheet_name='input')
        ds['Date']=pd.to_datetime(ds['Date'])
        ds.iloc[:,1:]=ds.iloc[:,1:].pct_change()
        return ds.iloc[:,:n+1].dropna()
    
    def calculate_annualized_returns(self):
        returns = self.ds.iloc[:, 1:]
        compounded_returns = (returns + 1).prod() ** (12 / len(returns)) - 1
        return compounded_returns.values
    
    def compute_covariance_matrix(self):
        cov_matrix = self.ds.drop(columns=['Date']).cov()*12
        return cov_matrix
    
    def calculate_intermediate_quantities(self, target_return):
        u = np.ones(self.n)
        # A = []
        # for j in range(len(self.ds.drop(columns=['Date']).columns)):
        #     temp=[]
        #     for i in range(len(self.ds.drop(columns=['Date']).columns)):
        #         temp.append(u[i] * self.returns[j]*self.inv_cov_matrix[j,i])
        #     A.append(sum(temp))
        
        A = np.array([np.sum([u[i] * self.returns[j] * self.inv_cov_matrix[i, j] for i in range(self.n)]) for j in range(self.n)])
        B = np.array([np.sum([self.returns[i] * self.returns[j] * self.inv_cov_matrix[i, j] for i in range(self.n)]) for j in range(self.n)])
        C = np.array([np.sum([u[i] * u[j] * self.inv_cov_matrix[i, j] for i in range(self.n)]) for j in range(self.n)])
        M = np.array([np.sum([self.returns[i] * u[j] * self.inv_cov_matrix[i, j] for i in range(self.n)]) for j in range(self.n)])
        L = np.array([np.sum([self.returns[i] * self.returns[j] * self.inv_cov_matrix[i, j] for i in range(self.n)]) for j in range(self.n)])
        
        D = (B * C) - A**2
        
        G = (M * B - L * A) / D
        H = (L * C - M * A) / D
        G_plus_H=G+H
        
        # def rescale_weights(weights):
        #     if weights is not None:
        #         weights /= np.sum(weights)
        #         weights = np.clip(weights, 0, 1)
        #         return weights
        #     pass
        
        weights = G + H * target_return
        # weights=rescale_weights(weights)
        return weights
    
    def efficient_frontier(self):
        rets=[]
        risks=[]
        utils=[]
        shrations=[]
        weights_dictionary = {col_name: [] for col_name in self.ds.columns[1:]}
        for i in range(11):
            weights=self.calculate_intermediate_quantities(i/10)
            if weights is not None:
                portfolio_return = np.dot(self.returns, weights)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                utility = portfolio_return - (self.risk_aversion / 2) * portfolio_volatility ** 2
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                rets.append(portfolio_return)
                risks.append(portfolio_volatility)
                utils.append(utility)
                shrations.append(sharpe_ratio)
                for col_idx, col_name in enumerate(self.ds.columns[1:]):
                    weights_dictionary[col_name].append(weights[col_idx])
            else:
                rets.append(np.nan)
                risks.append(np.nan)
                utils.append(np.nan)
                shrations.append(np.nan)
        rets = np.array(rets)
        risks = np.array(risks)
        utils = np.array(utils)
        shrations = np.array(shrations)
        
        # Sort based on portfolio returns
        sort_indices = np.argsort(rets)
        rets = rets[sort_indices]
        risks = risks[sort_indices]
        utils = utils[sort_indices]
        shrations = shrations[sort_indices]
        
        return rets, risks, utils, shrations
    
    def plot_efficient_frontier(self, rets, risks):
        plt.figure(figsize=(10, 6))
        plt.plot(risks, rets, marker='o', linestyle='-', color='b', markersize=5)
        plt.title('Efficient Frontier')
        plt.xlabel('Portfolio Risk (Standard Deviation)')
        plt.ylabel('Portfolio Return')
        plt.grid(True)
        plt.show()

    def write_to_excel(self, output_file='test.xlsx'):
        
        rets, risks, utils, shrations = self.efficient_frontier()
        
        df = pd.DataFrame({
            'Return': rets,
            'Volatility': risks,
            'Utility': utils,
            'Sharpe Ratio': shrations
        })
        
        df.sort_values(by='Return', inplace=True)
        
        with pd.ExcelWriter(output_file, mode='a', engine="openpyxl",if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name='output', index=False)
        



class PortfolioFrontend:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Optimizer")
        self.create_widgets()

    def create_widgets(self):
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Portfolio size
        ttk.Label(self.frame, text="Number of Securities:").grid(column=1, row=1, sticky=tk.W)
        self.size_entry = ttk.Entry(self.frame, width=20)
        self.size_entry.grid(column=2, row=1)

        # Expected returns
        ttk.Label(self.frame, text="Expected Returns (comma-separated):").grid(column=1, row=2, sticky=tk.W)
        self.returns_entry = ttk.Entry(self.frame, width=50)
        self.returns_entry.grid(column=2, row=2)

        # Volatilities
        ttk.Label(self.frame, text="Volatilities (comma-separated):").grid(column=1, row=3, sticky=tk.W)
        self.volatilities_entry = ttk.Entry(self.frame, width=50)
        self.volatilities_entry.grid(column=2, row=3)

        # Correlations
        ttk.Label(self.frame, text="Correlation Matrix (comma-separated rows):").grid(column=1, row=4, sticky=tk.W)
        self.correlations_entry = ttk.Entry(self.frame, width=50)
        self.correlations_entry.grid(column=2, row=4)

        # Risk-free rate
        ttk.Label(self.frame, text="Risk-free Rate:").grid(column=1, row=5, sticky=tk.W)
        self.risk_free_entry = ttk.Entry(self.frame, width=20)
        self.risk_free_entry.grid(column=2, row=5)

        # Risk aversion coefficient
        ttk.Label(self.frame, text="Risk Aversion Coefficient:").grid(column=1, row=6, sticky=tk.W)
        self.risk_aversion_entry = ttk.Entry(self.frame, width=20)
        self.risk_aversion_entry.grid(column=2, row=6)

        # Run button
        ttk.Button(self.frame, text="Run Optimizer", command=self.run_optimizer).grid(column=1, row=7, sticky=tk.W)

    def parse_input(self, input_str, expected_length):
        try:
            values = list(map(float, input_str.split(',')))
            if len(values) != expected_length:
                raise ValueError(f"Expected {expected_length} values, but got {len(values)}.")
            return values
        except ValueError as e:
            raise ValueError(f"Invalid input: {e}")

    def parse_correlation_matrix(self, input_str, size):
        try:
            rows = input_str.split(';')
            if len(rows) != size:
                raise ValueError(f"Expected {size} rows for correlation matrix, but got {len(rows)}.")
            matrix = []
            for row in rows:
                values = list(map(float, row.split(',')))
                if len(values) != size:
                    raise ValueError(f"Expected {size} values in each row, but got {len(values)}.")
                matrix.append(values)
            return np.array(matrix)
        except ValueError as e:
            raise ValueError(f"Invalid correlation matrix: {e}")

    def run_optimizer(self):
        try:
            size = int(self.size_entry.get())
            if size < 2 or size > 12:
                raise ValueError("Number of securities must be between 2 and 12.")
            
            returns = self.parse_input(self.returns_entry.get(), size)
            volatilities = self.parse_input(self.volatilities_entry.get(), size)
            correlations = self.parse_correlation_matrix(self.correlations_entry.get(), size)
            risk_free_rate = float(self.risk_free_entry.get())
            risk_aversion = float(self.risk_aversion_entry.get())

            portfolio = PortfolioBackend(returns, volatilities, correlations, risk_free_rate, risk_aversion)
            portfolio.calculate_efficient_frontier()

            # You need to implement display and plotting results
            # self.display_results(portfolio)
            # self.plot_efficient_frontier(portfolio)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

      
def main():
    root = tk.Tk()
    app = PortfolioFrontend(root)
    root.mainloop()

if __name__ == "__main__":
    main()

    
        

   

