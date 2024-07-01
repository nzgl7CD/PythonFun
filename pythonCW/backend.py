import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PortfolioOptimizer:
    def __init__(self, expected_return=[0.193,0.1575], volatility=[0.3025], corr_matrix=[[1,0.35],[0.35,1]], risk_free_rate=0.045, portfolio_size=2, risk_aversion=3):

        """
        Initialize the PortfolioOptimizer instance.

        Parameters:
        - expected_return: list/array of expected returns for each security
        - volatility: list/array of volatilities for each security
        - corr_matrix: correlation matrix between securities
        - risk_free_rate: risk-free rate
        - portfolio_size: number of securities
        - risk_aversion: risk aversion parameter for utility calculation
        """

        self.expected_return = expected_return
        self.risk_free_rate = risk_free_rate
        self.portfolio_size = portfolio_size
        self.risk_aversion = risk_aversion
        self.dataframe=None
        if self.is_effectively_empty(expected_return, volatility,corr_matrix):
            self.returns=np.asarray(expected_return)
            corr_matrix = np.array(corr_matrix)
            stdv = np.array(volatility)
            self.cov_matrix = np.outer(stdv, stdv) * corr_matrix
        else:
            self.dataset = self.get_data(portfolio_size)
            self.returns = self.calculate_annualized_returns()
            self.cov_matrix = self.compute_covariance_matrix()
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        self.C, self.G, self.H=self.calculate_intermediate_quantities()
        
        
    def is_effectively_empty(self,expected_return,volatility,corr_matrix):
        if expected_return and len(expected_return)==self.portfolio_size and volatility and len(volatility)==self.portfolio_size and corr_matrix and len(corr_matrix)==self.portfolio_size:
            return True
        return False

    def get_data(self, n):
        ds = pd.read_excel('230023476PortfolioProblem.xlsx')
        ds['Date'] = pd.to_datetime(ds['Date'])
        ds.iloc[:, 1:] = ds.iloc[:, 1:].pct_change() # Generate returns from prices
        print(ds.iloc[:, :n + 1].dropna())
        return ds.iloc[:, :n + 1].dropna()

    def calculate_annualized_returns(self):
        returns = self.dataset.iloc[:, 1:] # Exclude dates
        compounded_returns = (returns + 1).prod() ** (12 / len(returns)) - 1
        return compounded_returns.values

    def compute_covariance_matrix(self):
        cov_matrix = self.dataset.drop(columns=['Date']).cov() * 12
        return cov_matrix

    def calculate_intermediate_quantities(self):
        # Calculates all variables from HL model
        #TODO Make sure the results equal the ons from the excel sheet provided
        u = np.ones(self.portfolio_size)
        inv_cov_matrix = self.inv_cov_matrix
        A = sum([sum(u[i] * self.returns[j] * inv_cov_matrix[i, j] for i in range(self.portfolio_size)) for j in range(self.portfolio_size)])
        B = sum([sum(self.returns[i] * self.returns[j] * inv_cov_matrix[i, j] for i in range(self.portfolio_size)) for j in range(self.portfolio_size)])
        C = sum([sum(u[i] * u[j] * inv_cov_matrix[i, j] for i in range(self.portfolio_size)) for j in range(self.portfolio_size)])
        M = np.dot(np.ones(self.portfolio_size), self.inv_cov_matrix)
        L = self.returns @ inv_cov_matrix
        D = B * C - A ** 2
        LA = np.dot(L, A)  # Vector L multiplied by matrix A
        MB = np.dot(M, B)  # Vector M multiplied by matrix B
        
        G = (1/D) * (MB - LA)
        
        LB = L * C  # Vector L multiplied by matrix B
        MA = M * A  # Vector M multiplied by matrix A

        H = (LB - MA) / D
        
        return C, G, H
    
    # Calculate all relevant values for a portfolio
    def calculate_portfolio_metrics(self, weights):
        portfolio_return = np.sum(weights * self.returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        excess_return = portfolio_return - self.risk_free_rate
        sharpe_ratio = excess_return / portfolio_risk
        utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_variance
        return portfolio_return, portfolio_risk, sharpe_ratio, utility

    # Calculate minimum variance weights
    def calculate_minimum_variance_portfolio(self):
        min_var_weights = np.dot(self.inv_cov_matrix, np.ones(self.portfolio_size)) /self.C
        return min_var_weights

    # Calculate optimal variance weights
    def calculate_optimum_variance_portfolio(self, target_return):
        weights = self.G+(target_return*self.H)
        return weights

    def calculate_mean_variance_efficient_frontier(self):
        """
        Calculate and returns one list of lists with all weights for the mean-variance optimal portfolio on target return.
        Returns the efficient frotnier portfolio_return, portfolio_risk, sharpe_ratio, utility for mean-variance optimal portfolio
        """
        min_var_weights = self.calculate_minimum_variance_portfolio()
        frontier_weights = []
        for target_return in np.linspace(0, 1, 101):
            opt_var_weights = self.calculate_optimum_variance_portfolio(target_return)
            weights = (1 - target_return) * min_var_weights + target_return * opt_var_weights
            frontier_weights.append(weights)
        frontier_metrics = [self.calculate_portfolio_metrics(w) for w in frontier_weights]
        return frontier_weights, frontier_metrics

    def plot_efficient_frontier(self):
        """
        Plot the mean-variance efficient frontier along with the min variance point
        and the max Sharpe ratio point.
        """
        _, frontier_metrics = self.calculate_mean_variance_efficient_frontier()
        frontier_risks = [metric[1] for metric in frontier_metrics]
        frontier_returns = [metric[0] for metric in frontier_metrics]
        sharpe_ratios = [metric[2] for metric in frontier_metrics]

        # Find index of the lowest standard deviation (min variance point)
        min_var_idx = np.argmin(frontier_risks)
        min_var_point = frontier_metrics[min_var_idx]
        
        # Find index of the greatest Sharpe ratio (max Sharpe ratio point)
        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_sharpe_point = frontier_metrics[max_sharpe_idx]

        # Plotting the efficient frontier and key points
        plt.figure(figsize=(12, 8))

        plt.plot(frontier_risks, frontier_returns, 'b-o', label='Efficient Frontier')

        # Highlighting the min variance point
        plt.scatter(min_var_point[1], min_var_point[0], color='green', marker='o', s=100, 
                zorder=5, label=f'Min Variance Stdv: {min_var_point[1]:.4f}')

        # Highlighting the max Sharpe ratio point
        plt.scatter(max_sharpe_point[1], max_sharpe_point[0], color='red', marker='o', s=100, 
                    zorder=5, label=f'Max Sharpe Ratio: {max_sharpe_point[2]:.4f}')

        # Additional plot settings for aesthetics
        plt.xlabel('Portfolio Volatility (Risk)', fontsize=12, fontweight='bold')
        plt.ylabel('Portfolio Return', fontsize=12, fontweight='bold')
        plt.title('Mean-Variance Efficient Frontier', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
        plt.xlim(0.0, max(frontier_risks))

        # Show the plot
        plt.tight_layout()  # Ensures labels do not overlap
        plt.show()

    def write_to_excel(self, output_file='230023476PortfolioProblem.xlsx'):
        
        frontier_weights, frontier_metrics = self.calculate_mean_variance_efficient_frontier()
        # Check if dataset exists or if the user input is the dataset for the weights columns
        if hasattr(self, 'ds'): 
            weight_columns = [f'w_{col}' for col in self.dataset.columns[1:]]
        else:
            weight_columns = [f'w{i+1}' for i in range(self.portfolio_size)]
        
        data = {
            'Return': [metric[0] for metric in frontier_metrics],
            'Volatility': [metric[1] for metric in frontier_metrics],
            'Utility': [metric[3] for metric in frontier_metrics],
            'Sharpe Ratio': [metric[2] for metric in frontier_metrics]
        }

        # Make columns for weights and round all data to 4 decimals
        for i, col in enumerate(weight_columns):
            data[col] = [w[i] for w in frontier_weights]

        df = pd.DataFrame(data)
        df.sort_values(by='Return', inplace=True)
        numeric_columns = ['Return', 'Volatility', 'Utility', 'Sharpe Ratio'] + weight_columns
        df[numeric_columns] = df[numeric_columns].round(4)
        
        # Write to excel sheet and replace existing output sheet if it exists
        with pd.ExcelWriter(output_file, mode='a', engine="openpyxl",if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name='output', index=False)
            workbook = writer.book
            worksheet = workbook['output']
            
            # Regular nested for loop as requested from assignment
            for column_cells in worksheet.columns:
                max_length = 0
                column = column_cells[0].column_letter  # Get the column name
                for cell in column_cells:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = (max_length + 2) * 1.2  # Adjust the width for autofitting
                worksheet.column_dimensions[column].width = adjusted_width
        
        # TODO Use self.dataset? 
        
        self.dataframe=df
    
    def print_values(self):
                
            print('\n' * 2)
            # Print maximum Sharpe Ratio
            max_sharpe_idx = self.dataframe['Sharpe Ratio'].idxmax()
            max_sharpe_return = self.dataframe.loc[max_sharpe_idx, 'Return']
            max_sharpe_volatility = self.dataframe.loc[max_sharpe_idx, 'Volatility']
            max_sharpe_value = self.dataframe.loc[max_sharpe_idx, 'Sharpe Ratio']
            
            print(f"Maximum Sharpe Ratio Portfolio:")
            print(f"Return: {max_sharpe_return:.4f}, Volatility: {max_sharpe_volatility:.4f}, Sharpe Ratio: {max_sharpe_value:.4f}")
            
            print('\n' * 2)
            
            # Print maximum Utility
            max_utility_idx = self.dataframe['Utility'].idxmax()
            max_utility_return = self.dataframe.loc[max_utility_idx, 'Return']
            max_utility_volatility = self.dataframe.loc[max_utility_idx, 'Volatility']
            max_utility_value = self.dataframe.loc[max_utility_idx, 'Utility']
            print(f"Maximum Utility Portfolio:")
            print(f"Return: {max_utility_return:.4f}, Volatility: {max_utility_volatility:.4f}, Utility: {max_utility_value:.4f}")
            
            print('\n' * 2)    
             
            # Print minimum volatility
            min_volatility_idx = self.dataframe['Volatility'].idxmin()
            min_volatility_return = self.dataframe.loc[min_volatility_idx, 'Return']
            min_volatility_volatility = self.dataframe.loc[min_volatility_idx, 'Volatility']
            
            print(f"Minimum Volatility Portfolio:")
            print(f"Return: {min_volatility_return:.4f}, Volatility: {min_volatility_volatility:.4f}")

            print('\n' * 2) 

def main():
    optimizer = PortfolioOptimizer()
    optimizer.plot_efficient_frontier()
    optimizer.write_to_excel()
    optimizer.print_values()
    
if __name__ == "__main__":
    main()
