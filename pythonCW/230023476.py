import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox

class PortfolioOptimizer:
    def __init__(self, expected_return, volatility, corr_matrix, risk_free_rate, n, risk_aversion):
        self.expected_return = expected_return
        self.risk_free_rate = risk_free_rate
        self.n = n
        self.risk_aversion = risk_aversion
        print(len(corr_matrix))
        if (self.is_effectively_empty(expected_return) and self.is_effectively_empty(volatility) and self.is_effectively_empty(corr_matrix)):
            self.returns=np.asarray(expected_return)
            corr_matrix = np.array(corr_matrix)
            stdv = np.array(volatility)
            self.cov_matrix = np.outer(stdv, stdv) * corr_matrix
        else:
            self.ds = self.get_data(n)
            self.returns = self.calculate_annualized_returns()
            self.cov_matrix = self.compute_covariance_matrix()
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        
    def is_effectively_empty(self,lst):
        if lst and len(lst)==self.n:
            return True
        return False

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
        optim=(A/C)-(D/C**2)/(self.risk_free_rate-A/C)
        return A, B, C, D, G, H
    
    # Gives correct calculation
    def calculate_portfolio_metrics(self, weights):
        portfolio_return = np.sum(weights * self.returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        excess_return = portfolio_return - self.risk_free_rate
        sharpe_ratio = excess_return / portfolio_risk
        utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_variance
        return portfolio_return, portfolio_risk, sharpe_ratio, utility

    # Gives correct calculation
    def calculate_minimum_variance_portfolio(self):
        _, _, C, _, _, _ = self.calculate_intermediate_quantities()
        min_var_weights = np.dot(self.inv_cov_matrix, np.ones(self.n)) / C
        return min_var_weights, self.calculate_portfolio_metrics(min_var_weights)

    # Gives correct calculation
    def calculate_optimum_variance_portfolio(self, target_return):
        _, _, _, _, G, H = self.calculate_intermediate_quantities()
        weights = G+(target_return*H)
        return weights, self.calculate_portfolio_metrics(weights)

    def calculate_mean_variance_efficient_frontier(self):
        min_var_weights, _ = self.calculate_minimum_variance_portfolio()
        frontier_weights = []
        for target_return in np.linspace(0, 1, 101):
            opt_var_weights, opt_metric = self.calculate_optimum_variance_portfolio(target_return)
            weights = (1 - target_return) * min_var_weights + target_return * opt_var_weights
            frontier_weights.append(weights)
            # if opt_metric[2]>self.optimal_portfolio_plot['Sharpe Ratio']:
            #     self.optimal_portfolio_plot['Return'],self.optimal_portfolio_plot['Risk'],self.optimal_portfolio_plot['Sharpe Ratio']=opt_metric[0],opt_metric[1],opt_metric[2] 
        frontier_metrics = [self.calculate_portfolio_metrics(w) for w in frontier_weights]
        return frontier_weights, frontier_metrics

    def plot_efficient_frontier(self):
        _, frontier_metrics = self.calculate_mean_variance_efficient_frontier()
        frontier_risks = [metric[1] for metric in frontier_metrics]
        frontier_returns = [metric[0] for metric in frontier_metrics]
        sharpe_ratios = [metric[2] for metric in frontier_metrics]
        
        _, min_var_point = self.calculate_minimum_variance_portfolio()
        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_sharpe_point = frontier_metrics[max_sharpe_idx]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(frontier_risks, frontier_returns, marker='o', color='b', label='Efficient Frontier')
        plt.plot(min_var_point[1], min_var_point[0], marker='o', color='g', markersize=10, label='Minimum Variance Portfolio')
        plt.plot(max_sharpe_point[1], max_sharpe_point[0], marker='o', color='r', markersize=10, label='Maximum Sharpe Ratio Portfolio')

        # Add labels for Sharpe Ratio and Minimum Variance Portfolio
        plt.text(min_var_point[1], min_var_point[0], f'  Min Variance: {min_var_point[1]**2:.4f}', verticalalignment='bottom', horizontalalignment='right', color='green')
        plt.text(max_sharpe_point[1], max_sharpe_point[0], f'  Max Sharpe Ratio: {max_sharpe_point[2]:.4f}', verticalalignment='bottom', horizontalalignment='right', color='red')

        plt.title('Mean-Variance Efficient Frontier')
        plt.xlabel('Portfolio Volatility (Risk)')
        plt.ylabel('Portfolio Return')
        plt.grid(True)
        plt.legend()
        plt.xlim(0.0, max(frontier_risks))
        plt.show()

    def write_to_excel(self, output_file='test.xlsx'):
        frontier_weights, frontier_metrics = self.calculate_mean_variance_efficient_frontier()
        if hasattr(self, 'ds'):
            weight_columns = [f'w_{col}' for col in self.ds.columns[1:]]
        else:
            weight_columns = [f'w{i+1}' for i in range(self.n)]
        data = {
            'Return': [metric[0] for metric in frontier_metrics],
            'Volatility': [metric[1] for metric in frontier_metrics],
            'Utility': [metric[3] for metric in frontier_metrics],
            'Sharpe Ratio': [metric[2] for metric in frontier_metrics]
        }

        for i, col in enumerate(weight_columns):
            data[col] = [w[i] for w in frontier_weights]

        df = pd.DataFrame(data)
        df.sort_values(by='Return', inplace=True)
        numeric_columns = ['Return', 'Volatility', 'Utility', 'Sharpe Ratio'] + weight_columns
        df[numeric_columns] = df[numeric_columns].round(4)
        
        
        with pd.ExcelWriter(output_file, mode='a', engine="openpyxl",if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name='output', index=False)
            workbook = writer.book
            worksheet = workbook['output']
            for column_cells in worksheet.columns:
                max_length = 0
                column = column_cells[0].column_letter  # Get the column name
                for cell in column_cells:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = (max_length + 2) * 1.2  # Adjust the width
                worksheet.column_dimensions[column].width = adjusted_width
            
        print()
        print()
        
        # Print maximum Sharpe Ratio
        max_sharpe_idx = df['Sharpe Ratio'].idxmax()
        max_sharpe_return = df.loc[max_sharpe_idx, 'Return']
        max_sharpe_volatility = df.loc[max_sharpe_idx, 'Volatility']
        max_sharpe_value = df.loc[max_sharpe_idx, 'Sharpe Ratio']
        
        print(f"Maximum Sharpe Ratio Portfolio:")
        print(f"Return: {max_sharpe_return:.4f}, Volatility: {max_sharpe_volatility:.4f}, Sharpe Ratio: {max_sharpe_value:.4f}")
        print()
        print()
        
        # Print maximum Utility
        max_utility_idx = df['Utility'].idxmax()
        max_utility_return = df.loc[max_utility_idx, 'Return']
        max_utility_volatility = df.loc[max_utility_idx, 'Volatility']
        max_utility_value = df.loc[max_utility_idx, 'Utility']
        print(f"Maximum Utility Portfolio:")
        print(f"Return: {max_utility_return:.4f}, Volatility: {max_utility_volatility:.4f}, Utility: {max_utility_value:.4f}")
        print()
        print()      
    

class PortfolioFrontend:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Optimizer")
        self.create_widgets()

    def create_widgets(self):
        self.frame = ttk.Frame(self.root, padding="20")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Portfolio Size
        ttk.Label(self.frame, text="Portfolio Size (number of securities):").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.portfolio_size_entry = ttk.Entry(self.frame, width=10)
        self.portfolio_size_entry.grid(column=1, row=0, padx=10, pady=10)
        self.portfolio_size_entry.insert(tk.END, "2")  # Default value

        # Risk Aversion
        ttk.Label(self.frame, text="Risk Aversion:").grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
        self.risk_aversion_entry = ttk.Entry(self.frame, width=10)
        self.risk_aversion_entry.grid(column=1, row=1, padx=10, pady=10)
        self.risk_aversion_entry.insert(tk.END, "3.0")  # Default value

        # Risk-Free Rate
        ttk.Label(self.frame, text="Risk-Free Rate:").grid(column=0, row=2, padx=10, pady=10, sticky=tk.W)
        self.risk_free_rate_entry = ttk.Entry(self.frame, width=10)
        self.risk_free_rate_entry.grid(column=1, row=2, padx=10, pady=10)
        self.risk_free_rate_entry.insert(tk.END, "0.02")  # Default value

        # Button to update fields based on portfolio size
        update_button = ttk.Button(self.frame, text="Update Fields", command=self.update_fields)
        update_button.grid(column=2, row=0, padx=10, pady=10)

        # Text box for Volatilities
        ttk.Label(self.frame, text="Volatilities (comma-separated):").grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)
        self.volatilities_text = tk.Text(self.frame, width=60, height=1)
        self.volatilities_text.grid(column=1, row=3, columnspan=2, padx=10, pady=10)
        self.volatilities_text.insert(tk.END, "0.3025, 0.219")  # Default value

        # Text box for Expected Returns
        ttk.Label(self.frame, text="Expected Returns (comma-separated):").grid(column=0, row=4, padx=10, pady=10, sticky=tk.W)
        self.expected_returns_text = tk.Text(self.frame, width=60, height=1)
        self.expected_returns_text.grid(column=1, row=4, columnspan=2, padx=10, pady=10)
        self.expected_returns_text.insert(tk.END, "0.1934, 0.1575")  # Default value

        # Text box for Correlation Matrix
        ttk.Label(self.frame, text="Correlation Matrix (semicolon-separated rows):").grid(column=0, row=5, padx=10, pady=10, sticky=tk.W)
        self.correlation_matrix_text = tk.Text(self.frame, width=60, height=4)
        self.correlation_matrix_text.grid(column=1, row=5, columnspan=2, padx=10, pady=10)
        self.correlation_matrix_text.insert(tk.END, "1.0, 0.35; 0.35, 1.0")  # Default value

        # Run button
        run_button = ttk.Button(self.frame, text="Run Optimizer", command=self.run_optimizer)
        run_button.grid(column=0, row=6, columnspan=3, padx=10, pady=20, sticky=tk.W+tk.E)

        # Add a tooltip for run button
        tooltip = ttk.Label(self.frame, text="Click to run portfolio optimization", foreground="gray")
        tooltip.grid(column=0, row=7, columnspan=3, padx=10, pady=5, sticky=tk.W)

        # Bind hover events to show/hide tooltip
        run_button.bind("<Enter>", lambda e: tooltip.grid(row=7, column=0, columnspan=3))
        run_button.bind("<Leave>", lambda e: tooltip.grid_forget())

    def update_fields(self):
        try:
            portfolio_size = int(self.portfolio_size_entry.get())

            # Clear existing text boxes
            self.volatilities_text.delete(1.0, tk.END)
            self.expected_returns_text.delete(1.0, tk.END)
            self.correlation_matrix_text.delete(1.0, tk.END)

            # Default values for text boxes based on portfolio size
            default_volatilities = ",".join(["0.1"] * portfolio_size)
            default_returns = ",".join(["0.05"] * portfolio_size)
            default_correlation = ";".join([",".join(["1.0" if i == j else "0.3" for j in range(portfolio_size)]) for i in range(portfolio_size)])

            # Insert default values into text boxes
            self.volatilities_text.insert(tk.END, default_volatilities)
            self.expected_returns_text.insert(tk.END, default_returns)
            self.correlation_matrix_text.insert(tk.END, default_correlation)

        except ValueError:
            messagebox.showerror("Error", "Portfolio Size must be a valid integer.")

    def parse_values(self, input_str):
        try:
            values = list(map(float, input_str.strip().split(',')))
            return values
        except ValueError as e:
            raise ValueError(f"Invalid input: {e}")

    def parse_correlation_matrix(self, input_str):
        try:
            rows = input_str.strip().split(';')
            correlation_matrix = []
            for row in rows:
                values = list(map(float, row.strip().split(',')))
                correlation_matrix.append(values)
            return correlation_matrix
        except ValueError as e:
            raise ValueError(f"Invalid correlation matrix input: {e}")

    def run_optimizer(self):
        try:
            portfolio_size = int(self.portfolio_size_entry.get())
            risk_aversion = float(self.risk_aversion_entry.get())
            risk_free_rate = float(self.risk_free_rate_entry.get())

            # Parse inputs
            volatilities = self.parse_values(self.volatilities_text.get("1.0", tk.END))
            expected_returns = self.parse_values(self.expected_returns_text.get("1.0", tk.END))
            correlation_matrix = self.parse_correlation_matrix(self.correlation_matrix_text.get("1.0", tk.END))

            # Create optimizer instance
            optimizer = PortfolioOptimizer(expected_return=expected_returns,
                                           volatility=volatilities,
                                           corr_matrix=correlation_matrix,
                                           risk_free_rate=risk_free_rate,
                                           n=portfolio_size,
                                           risk_aversion=risk_aversion)

            # Plot efficient frontier and write to Excel
            optimizer.plot_efficient_frontier()
            optimizer.write_to_excel('test.xlsx')

            messagebox.showinfo("Optimizer", "Optimization complete!")

        except ValueError as e:
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = PortfolioFrontend(root)
    root.mainloop()

if __name__ == "__main__":
    main()