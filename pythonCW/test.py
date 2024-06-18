import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PortfolioOptimizer:
    def __init__(self, expected_return=0.3, risk_free_rate=0.02, n=6, risk_aversion=3):
        self.expected_return = expected_return
        self.risk_free_rate = risk_free_rate
        self.n = n
        self.risk_aversion = risk_aversion
        self.ds = self.get_data(n)
        self.returns = self.calculate_annualized_returns()  
        self.cov_matrix = self.compute_covariance_matrix()
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
    
    def get_data(self, n):
        ds = pd.read_excel('test.xlsx', sheet_name='input')
        ds['Date'] = pd.to_datetime(ds['Date'])
        ds.iloc[:, 1:] = ds.iloc[:, 1:].pct_change()
        return ds.iloc[:, :n + 1].dropna()
    
    def calculate_annualized_returns(self):
        returns = self.ds.iloc[:, 1:]
        compounded_returns = (returns + 1).prod() ** (12 / len(returns)) - 1
        return compounded_returns.values
    
    def compute_covariance_matrix(self):
        cov_matrix = self.ds.drop(columns=['Date']).cov() * 12
        return cov_matrix
    
    def calculate_intermediate_quantities(self):
        returns = self.returns
        n = self.n
        inv_cov_matrix = self.inv_cov_matrix
        
        u = np.ones(n)
        A = np.array([np.sum(u[i] * returns[j] * inv_cov_matrix[i, j] for i in range(n)) for j in range(n)])
        B = np.array([np.sum(returns[i] * returns[j] * inv_cov_matrix[i, j] for i in range(n)) for j in range(n)])
        C = np.array([np.sum(u[i] * u[j] * inv_cov_matrix[i, j] for i in range(n)) for j in range(n)])
        M = np.array([np.sum(returns[i] * u[j] * inv_cov_matrix[i, j] for i in range(n)) for j in range(n)])
        L = np.array([np.sum(returns[i] * inv_cov_matrix[i, j] for i in range(n)) for j in range(n)])
        
        D = B * C - A ** 2
        G = (M * C - L * A) / D
        H = (L * B - M * A) / D
        
        return A, B, C, D, G, H
    
    def calculate_portfolio_metrics(self, weights):
        portfolio_return = np.dot(weights, self.returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        return portfolio_return, portfolio_volatility
    
    def calculate_minimum_variance_portfolio(self):
        A, B, C, D, G, H = self.calculate_intermediate_quantities()
        weights = G / C
        return weights, *self.calculate_portfolio_metrics(weights)
    
    def calculate_optimum_variance_portfolio(self):
        A, B, C, D, G, H = self.calculate_intermediate_quantities()
        A_bar = np.sum(G * self.returns) / np.sum(G)
        weights = G + H * (self.expected_return - A_bar) / np.sum(H)
        return weights, *self.calculate_portfolio_metrics(weights)
    
    def calculate_efficient_frontier(self):
        A, B, C, D, G, H = self.calculate_intermediate_quantities()
        
        min_var_weights, min_var_return, min_var_volatility = self.calculate_minimum_variance_portfolio()
        opt_var_weights, opt_var_return, opt_var_volatility = self.calculate_optimum_variance_portfolio()
        
        # Minimum variance portfolio
        min_var_point = (min_var_return, min_var_volatility)
        
        # Optimum variance portfolio
        opt_var_point = (opt_var_return, opt_var_volatility)
        
        # Efficient frontier points
        target_returns = np.linspace(min_var_return, opt_var_return, 100)
        frontier_risks = []
        frontier_returns = []
        
        for t in target_returns:
            weights = G + H * t
            portfolio_return, portfolio_volatility = self.calculate_portfolio_metrics(weights)
            frontier_risks.append(portfolio_volatility)
            frontier_returns.append(portfolio_return)
        
        return frontier_risks, frontier_returns, min_var_point, opt_var_point
    
    def plot_efficient_frontier(self):
        frontier_risks, frontier_returns, min_var_point, opt_var_point = self.calculate_efficient_frontier()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(frontier_risks, frontier_returns, marker='o', color='b', label='Efficient Frontier')
        plt.plot(min_var_point[1], min_var_point[0], marker='o', color='g', markersize=10, label='Minimum Variance Portfolio')
        plt.plot(opt_var_point[1], opt_var_point[0], marker='o', color='r', markersize=10, label='Optimum Variance Portfolio')
        
        plt.title('Efficient Frontier')
        plt.xlabel('Portfolio Volatility (Risk)')
        plt.ylabel('Portfolio Return')
        plt.grid(True)
        plt.legend()
        plt.show()

    def run_optimizer(self):
        min_var_weights, min_var_return, min_var_volatility = self.calculate_minimum_variance_portfolio()
        opt_var_weights, opt_var_return, opt_var_volatility = self.calculate_optimum_variance_portfolio()
        frontier_risks, frontier_returns, min_var_point, opt_var_point = self.calculate_efficient_frontier()
        
        max_utility_index = np.argmax(frontier_returns - self.risk_aversion * np.array(frontier_risks) ** 2)
        max_utility_return = frontier_returns[max_utility_index]
        max_utility_risk = frontier_risks[max_utility_index]
        
        max_sharpe_index = np.argmax((frontier_returns - self.risk_free_rate) / np.array(frontier_risks))
        max_sharpe_return = frontier_returns[max_sharpe_index]
        max_sharpe_risk = frontier_risks[max_sharpe_index]
        
        return min_var_weights, min_var_return, min_var_volatility, \
               opt_var_weights, opt_var_return, opt_var_volatility, \
               max_utility_return, max_utility_risk, max_sharpe_return, max_sharpe_risk

# Example usage:
if __name__ == "__main__":
    optimizer = PortfolioOptimizer()
    optimizer.plot_efficient_frontier()
