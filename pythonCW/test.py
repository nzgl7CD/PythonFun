import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PortfolioOptimizer:
    def __init__(self, expected_return=0.5, risk_free_rate=0.02, n=2, risk_aversion=3):
        self.expected_return = expected_return
        self.risk_free_rate = risk_free_rate
        self.n = n
        self.risk_aversion = risk_aversion
        self.ds = self.get_data(n)
        # self.returns = self.calculate_annualized_returns() 
        self.returns=[0.1934,0.1575] 
        self.returns=np.asarray(self.returns)
        # self.cov_matrix = self.compute_covariance_matrix()
        self.cov_matrix=pd.DataFrame(data=[[0.09150625,0.023186625],[0.023186625,0.047961]])
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
    
    def get_data(self, n):
        ds = pd.read_excel('test.xlsx')
        ds['Date'] = pd.to_datetime(ds['Date'])
        ds.iloc[:, 1:] = ds.iloc[:, 1:].pct_change()
        return ds.iloc[:, :n + 1].dropna()
    
    def calculate_annualized_returns(self):
        returns = self.ds.iloc[:, 1:]
        compounded_returns = (returns + 1).prod() ** (12 / len(returns)) - 1
        return compounded_returns.values
    
    # Checked: Values are correct inversed
    
    def compute_covariance_matrix(self):
        cov_matrix = self.ds.drop(columns=['Date']).cov() * 12
        return cov_matrix
    
    # Checked: All values are correct according to the example
    def calculate_intermediate_quantities(self):
        u = np.ones(self.n)
        inv_cov_matrix = self.inv_cov_matrix
        A = np.sum([np.sum(u[i] * self.returns[j] * inv_cov_matrix[i, j] for i in range(self.n)) for j in range(self.n)])
        B = np.sum([np.sum(self.returns[i] * self.returns[j] * inv_cov_matrix[i, j] for i in range(self.n)) for j in range(self.n)])
        C = np.sum([np.sum(u[i] * u[j] * inv_cov_matrix[i, j] for i in range(self.n)) for j in range(self.n)])
        M = np.dot(np.ones(self.n), self.inv_cov_matrix)
        L = self.returns @ inv_cov_matrix
        D = B * C - A ** 2
        LA = np.dot(L, A)  # Vector L multiplied by matrix A
        MB = np.dot(M, B)  # Vector M multiplied by matrix B

        # Calculate G
        G = (1/D) * (MB - LA)

        LB = L * C  # Vector L multiplied by matrix B
        MA = M * A  # Vector M multiplied by matrix A

        # Calculate H
        H = (LB - MA) / D
        # print(G)
        # print(H)
        # print(G+H)
        return A, B, C, D, G, H

    def calculate_portfolio_metrics(self, weights):
        portfolio_return = np.sum(weights * self.returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        return portfolio_return, portfolio_risk
    
    def calculate_minimum_variance_portfolio(self):
        A, _, C, _, _, _ = self.calculate_intermediate_quantities()
        # min_var_return = A / C
        min_var_weights = np.dot(self.inv_cov_matrix, np.ones(self.n)) / C
        return min_var_weights, self.calculate_portfolio_metrics(min_var_weights)
    
    def calculate_optimum_variance_portfolio(self, target_return):
        _, _, _, _, G, H = self.calculate_intermediate_quantities()
        weights = G+(target_return*H)
        return weights, self.calculate_portfolio_metrics(weights)
    
    
    
    def calculate_mean_variance_efficient_frontier(self):
        min_var_weights, _ = self.calculate_minimum_variance_portfolio()
        frontier_weights = []
        for target_return in np.linspace(0, 1, 101):
            opt_var_weights, _ = self.calculate_optimum_variance_portfolio(target_return)
            weights = (1 - target_return) * min_var_weights + target_return * opt_var_weights
            frontier_weights.append(weights)
        frontier_metrics = [self.calculate_portfolio_metrics(w) for w in frontier_weights]
        return frontier_weights, frontier_metrics
    
    def plot_efficient_frontier(self):
        _, frontier_metrics = self.calculate_mean_variance_efficient_frontier()
        frontier_risks = [metric[1] for metric in frontier_metrics]
        frontier_returns = [metric[0] for metric in frontier_metrics]

        min_var_weights, _ = self.calculate_minimum_variance_portfolio()
        opt_var_weights, _ = self.calculate_optimum_variance_portfolio(self.expected_return)

        min_var_point = self.calculate_portfolio_metrics(min_var_weights)
        opt_var_point = self.calculate_portfolio_metrics(opt_var_weights)

        plt.figure(figsize=(10, 6))
        plt.scatter(frontier_risks, frontier_returns, marker='o', color='b', label='Efficient Frontier')
        plt.plot(min_var_point[1], min_var_point[0], marker='o', color='g', markersize=10, label='Minimum Variance Portfolio')
        plt.plot(opt_var_point[1], opt_var_point[0], marker='o', color='r', markersize=10, label='Optimum Variance Portfolio')

        plt.title('Mean-Variance Efficient Frontier')
        plt.xlabel('Portfolio Volatility (Risk)')
        plt.ylabel('Portfolio Return')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def write_to_excel(self, output_file='test.xlsx'):
        frontier_weights, frontier_metrics = self.calculate_mean_variance_efficient_frontier()

        weight_columns = [f'{col}_w' for col in self.ds.columns[1:]]
        data = {
            'Return': [metric[0] for metric in frontier_metrics],
            'Volatility': [metric[1] for metric in frontier_metrics],
            'Utility': [self.calculate_quadratic_utility(w) for w in frontier_weights],
            'Sharpe Ratio': [self.calculate_sharpe_ratio(w) for w in frontier_weights]
        }

        for i, col in enumerate(weight_columns):
            data[col] = [w[i] for w in frontier_weights]

        df = pd.DataFrame(data)
        df.sort_values(by='Return', inplace=True)
        numeric_columns = ['Return', 'Volatility', 'Utility', 'Sharpe Ratio'] + weight_columns
        df[numeric_columns] = df[numeric_columns].round(4)
        with pd.ExcelWriter(output_file, mode='a', engine="openpyxl",if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name='output', index=False)

    
    def calculate_quadratic_utility(self, weights):
        portfolio_return, portfolio_risk = self.calculate_portfolio_metrics(weights)
        utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_risk**2
        return utility

    def calculate_sharpe_ratio(self, weights):
        portfolio_return, portfolio_risk = self.calculate_portfolio_metrics(weights)
        excess_return = portfolio_return - self.risk_free_rate
        sharpe_ratio = excess_return / portfolio_risk
        return sharpe_ratio

# Example usage:
if __name__ == "__main__":
    optimizer = PortfolioOptimizer()
    optimizer.plot_efficient_frontier()