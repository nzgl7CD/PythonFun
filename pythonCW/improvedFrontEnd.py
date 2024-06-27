import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
class PortfolioOptimizer:
    def __init__(self, expected_return=[], volatility=[], corr_matrix=[[],[]], risk_free_rate=0.045, portfolio_size=2, risk_aversion=3):
        
        """
        Initialize the PortfolioOptimizer instance.

        Parameters:
        - expected_return: list/array of expected returns for each security
        - volatility: list/array of volatilities for each security
        - corr_matrix: correlation matrix between securities
        - risk_free_rate: risk-free rate
        - n: number of securities
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
            self.ds = self.get_data(portfolio_size)
            self.returns = self.calculate_annualized_returns()
            self.cov_matrix = self.compute_covariance_matrix(self.ds)
            
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        _, _, self.C, _, self.G, self.H=self.calculate_intermediate_quantities()
        
    def is_effectively_empty(self,expected_return,volatility,corr_matrix):
        if expected_return and len(expected_return)==self.portfolio_size or volatility and len(volatility)==self.portfolio_size or corr_matrix and len(corr_matrix)==self.portfolio_size:
            return True
        return False

    def get_data(self, n):
        ds = pd.read_excel('230023476PortfolioProblem.xlsx')
        ds['Date'] = pd.to_datetime(ds['Date'])
        ds.iloc[:, 1:] = ds.iloc[:, 1:].pct_change()
        return ds.iloc[:, :n + 1].dropna()

    def calculate_annualized_returns(self):
        returns = self.ds.iloc[:, 1:] # Exclude dates
        compounded_returns = (returns + 1).prod() ** (12 / len(returns)) - 1
        return compounded_returns.values

    def compute_covariance_matrix(self, dataset):
        cov_matrix = dataset.drop(columns=['Date']).cov() * 12
        return cov_matrix

    # Should we put them in safe as they are calculated once?
    def calculate_intermediate_quantities(self):
        u = np.ones(self.portfolio_size)
        inv_cov_matrix = self.inv_cov_matrix
        A = np.sum([np.sum(u[i] * self.returns[j] * inv_cov_matrix[i, j] for i in range(self.portfolio_size)) for j in range(self.portfolio_size)])
        B = np.sum([np.sum(self.returns[i] * self.returns[j] * inv_cov_matrix[i, j] for i in range(self.portfolio_size)) for j in range(self.portfolio_size)])
        C = np.sum([np.sum(u[i] * u[j] * inv_cov_matrix[i, j] for i in range(self.portfolio_size)) for j in range(self.portfolio_size)])
        M = np.dot(np.ones(self.portfolio_size), self.inv_cov_matrix)
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
        min_var_weights = np.dot(self.inv_cov_matrix, np.ones(self.portfolio_size)) / self.C
        return min_var_weights, self.calculate_portfolio_metrics(min_var_weights)

    # Gives correct calculation
    def calculate_optimum_variance_portfolio(self, target_return):
        weights = self.G+(target_return*self.H)
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

    def plot_efficient_frontier(self, ax):
        """
        Plot the mean-variance efficient frontier along with the min variance point
        and the max Sharpe ratio point.
        """

        # Data processing: Calculate the mean-variance efficient frontier
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
        # plt.figure(figsize=(12, 8))

        # Efficient frontier
        ax.plot(frontier_risks, frontier_returns, 'b-o', label='Efficient Frontier')

        # Highlighting the min variance point
        ax.scatter(min_var_point[1], min_var_point[0], color='green', marker='o', s=100, 
                zorder=5, label=f'Min Variance Stdv: {min_var_point[1]:.4f}')

        # Highlighting the max Sharpe ratio point
        ax.scatter(max_sharpe_point[1], max_sharpe_point[0], color='red', marker='o', s=100, 
                zorder=5, label=f'Max Sharpe Ratio: {max_sharpe_point[2]:.4f}')


        # Additional plot settings for aesthetics
        ax.set_xlabel('Portfolio Volatility (Risk)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Return', fontsize=12, fontweight='bold')
        ax.set_title('Mean-Variance Efficient Frontier', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlim(0.0, max(frontier_risks))

    def write_to_excel(self, output_file='230023476PortfolioProblem.xlsx'):
        frontier_weights, frontier_metrics = self.calculate_mean_variance_efficient_frontier()
        if hasattr(self, 'ds'):
            weight_columns = [f'w_{col}' for col in self.ds.columns[1:]]
        else:
            weight_columns = [f'w{i+1}' for i in range(self.portfolio_size)]
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
            
            # Regular nested for loop as requested
            
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
        
            self.dataframe=df
    
    def print_values(self):
        df = self.dataframe
        output_str = ""
        
        max_sharpe_idx = df['Sharpe Ratio'].idxmax()
        max_sharpe_return, max_sharpe_volatility, max_sharpe_value, max_sharpe_utility = df.loc[max_sharpe_idx, ['Return', 'Volatility', 'Sharpe Ratio', 'Utility']]
        
        max_utility_idx = df['Utility'].idxmax()
        max_utility_return, max_utility_volatility, max_utility_sharpe, max_utility_value = df.loc[max_utility_idx, ['Return', 'Volatility', 'Sharpe Ratio', 'Utility']]
        
        min_volatility_idx = df['Volatility'].idxmin()
        min_volatility_return, min_volatility_volatility, min_volatility_sharpe, min_volatility_utility = df.loc[min_volatility_idx, ['Return', 'Volatility', 'Sharpe Ratio', 'Utility']]
        
        return {'MaxSharpeRatio': [max_sharpe_return, max_sharpe_volatility, max_sharpe_value, max_sharpe_utility],
                'MaxUtility': [max_utility_return, max_utility_volatility, max_utility_sharpe, max_utility_value],
                'MinVar': [min_volatility_return, min_volatility_volatility, min_volatility_sharpe, min_volatility_utility]}
        
class PortfolioFrontend:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Optimizer")
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Helvetica', 12), padding=10, background='#4CAF50', foreground='#000000')  
        self.style.map('TButton', background=[('active', '#45a049')])
        self.create_widgets()
        self.metrics_dict=None

    def create_widgets(self):
        
        self.frame = ttk.Frame(self.root, padding="20")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Portfolio Size
        ttk.Label(self.frame, text="Portfolio Size (2-12 securities):").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.portfolio_size_entry = ttk.Entry(self.frame, width=10, font=('Helvetica', 12))
        self.portfolio_size_entry.grid(column=1, row=0, padx=10, pady=10)
        self.portfolio_size_entry.insert(tk.END, "2")

        # Risk Aversion
        ttk.Label(self.frame, text="Risk Aversion:").grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
        self.risk_aversion_entry = ttk.Entry(self.frame, width=10, font=('Helvetica', 12))
        self.risk_aversion_entry.grid(column=1, row=1, padx=10, pady=10)
        self.risk_aversion_entry.insert(tk.END, "3.0")

        # Risk-Free Rate
        ttk.Label(self.frame, text="Risk-Free Rate:").grid(column=0, row=2, padx=10, pady=10, sticky=tk.W)
        self.risk_free_rate_entry = ttk.Entry(self.frame, width=10, font=('Helvetica', 12))
        self.risk_free_rate_entry.grid(column=1, row=2, padx=10, pady=10)
        self.risk_free_rate_entry.insert(tk.END, "0.045")

        # Update Fields Button
        update_button = ttk.Button(self.frame, text="Update Fields", command=self.update_fields, style='TButton')
        update_button.grid(column=2, row=0, padx=10, pady=10)

        # Volatilities
        ttk.Label(self.frame, text="Volatilities (comma-separated):").grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)
        self.volatilities_text = tk.Text(self.frame, width=60, height=1, font=('Helvetica', 12))
        self.volatilities_text.grid(column=1, row=3, columnspan=2, padx=10, pady=10)
        self.volatilities_text.insert(tk.END, "0.3025, 0.219")

        # Expected Returns
        ttk.Label(self.frame, text="Expected Returns (comma-separated):").grid(column=0, row=4, padx=10, pady=10, sticky=tk.W)
        self.expected_returns_text = tk.Text(self.frame, width=60, height=1, font=('Helvetica', 12))
        self.expected_returns_text.grid(column=1, row=4, columnspan=2, padx=10, pady=10)
        self.expected_returns_text.insert(tk.END, "0.1934, 0.1575")

        # Correlation Matrix
        ttk.Label(self.frame, text="Correlation Matrix (semicolon-separated rows):").grid(column=0, row=5, padx=10, pady=10, sticky=tk.W)
        self.correlation_matrix_text = tk.Text(self.frame, width=60, height=4, font=('Helvetica', 12))
        self.correlation_matrix_text.grid(column=1, row=5, columnspan=2, padx=10, pady=10)
        self.correlation_matrix_text.insert(tk.END, "1.0, 0.35; 0.35, 1.0")

        # Button Frame
        button_frame = ttk.Frame(self.frame)
        button_frame.grid(column=0, row=6, columnspan=3, padx=10, pady=20, sticky=tk.W+tk.E)

        # Run Optimizer Button
        run_button = ttk.Button(button_frame, text="Run Optimizer", command=self.run_optimizer, style='TButton')
        run_button.grid(column=0, row=0, padx=5, pady=10)

        # Exit Button
        exit_button = ttk.Button(button_frame, text="Exit", command=self.root.quit, style='TButton')
        exit_button.grid(column=1, row=0, padx=5, pady=10)

        # Center buttons within the button_frame
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

    def update_fields(self):
        try:
            portfolio_size = int(self.portfolio_size_entry.get())

            if portfolio_size < 2 or portfolio_size > 12:
                messagebox.showerror("Error", "Portfolio Size must be between 2 and 12.")
                return

            self.volatilities_text.delete(1.0, tk.END)
            self.expected_returns_text.delete(1.0, tk.END)
            self.correlation_matrix_text.delete(1.0, tk.END)

            default_volatilities = ",".join(["0.1"] * portfolio_size)
            default_returns = ",".join(["0.05"] * portfolio_size)
            default_correlation = ";".join([",".join(["1.0" if i == j else "0.3" for j in range(portfolio_size)]) for i in range(portfolio_size)])

            self.volatilities_text.insert(tk.END, default_volatilities)
            self.expected_returns_text.insert(tk.END, default_returns)
            self.correlation_matrix_text.insert(tk.END, default_correlation)

        except ValueError:
            messagebox.showerror("Error", "Portfolio Size must be a valid integer.")

    def run_optimizer(self):
        try:
            portfolio_size = int(self.portfolio_size_entry.get())
            
            
            # Get risk aversion and risk free rate, setting defaults if not provided
            risk_aversion_entry = self.risk_aversion_entry.get()
            risk_free_rate_entry = self.risk_free_rate_entry.get()
            
            risk_aversion = float(risk_aversion_entry) if risk_aversion_entry else 3.0
            risk_free_rate = float(risk_free_rate_entry) if risk_free_rate_entry else 0.045

            volatilities = self.parse_vectors(self.volatilities_text.get("1.0", tk.END))
            expected_returns = self.parse_vectors(self.expected_returns_text.get("1.0", tk.END))
            correlation_matrix = self.parse_correlation_matrix(self.correlation_matrix_text.get("1.0", tk.END))
            
            def input_validation(volatilities, expected_returns, correlation_matrix):
                
                if not (len(volatilities) == portfolio_size and len(expected_returns) == portfolio_size and len(correlation_matrix) == portfolio_size):
                    messagebox.showinfo("Optimizer", "Inputs invalid. Using real data from Excel file.")
                    
                    # Load real data from Excel file
                    optimizer = PortfolioOptimizer(expected_return=[],  
                                                volatility=[],     
                                                corr_matrix=[],    
                                                risk_free_rate=risk_free_rate,
                                                portfolio_size=portfolio_size,
                                                risk_aversion=risk_aversion)
                else:
                    # Load input data from user
                    optimizer = PortfolioOptimizer(expected_return=expected_returns,
                                                volatility=volatilities,
                                                corr_matrix=correlation_matrix,
                                                risk_free_rate=risk_free_rate,
                                                portfolio_size=portfolio_size,
                                                risk_aversion=risk_aversion)
                return optimizer
            
            optimizer=input_validation(volatilities, expected_returns, correlation_matrix)
            
            self.show_plot(optimizer)
            optimizer.write_to_excel('230023476PortfolioProblem.xlsx')
            self.metrics_dict = optimizer.print_values()

        except ValueError as e:
            messagebox.showerror("Error", str(e))
    
    def show_plot(self, optimizer):
        if hasattr(self, 'plot_window') and self.plot_window.winfo_exists():
            self.plot_window.destroy()

        # Create a new plot window
        self.plot_window = tk.Toplevel(self.root)
        self.plot_window.title("Efficient Frontier")
        width, height = 800, 800
        x = (self.plot_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.plot_window.winfo_screenheight() // 2) - (height // 2)
        self.plot_window.geometry(f"{width}x{height}+{x}+{y}")

        # Create a frame to hold the plot and the exit button
        plot_frame = ttk.Frame(self.plot_window)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))

        # Embed the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Call the backend plot function with the axes
        optimizer.plot_efficient_frontier(ax)
        
        # To show the portfolio metrics when the plot is closed
        def close_plot_and_show_metrics():
            self.plot_window.destroy()
            self.show_portfolio_metrics(self.metrics_dict)

        ttk.Button(plot_frame, text="Exit", command=close_plot_and_show_metrics).pack(side=tk.BOTTOM, pady=10)

    def show_portfolio_metrics(self, metrics_dict):
        top = tk.Toplevel(self.root)
        top.title("Portfolio Metrics")
    
        # Calculate and set the geometry of the dialog based on content size
        rows = len(metrics_dict) + 1  # Including header row
        cols = 5 
        top.geometry(f"{cols * 150}x{rows * 70}")  
        
        formatted_str = ""

        try:
            formatted_str += "{:<25s}{:^15s}{:^15s}{:^15s}{:>15s}\n".format("Portfolio Type", "Return", "Volatility", "Sharpe Ratio", "Utility")
            formatted_str += "-" * (86) + "\n"
            for key in metrics_dict:
                formatted_str += "{:<25s}".format(key) 
                return_value = metrics_dict[key][0]
                volatility_value = metrics_dict[key][1]
                sharpe_ratio_value = metrics_dict[key][2]
                utility_value = metrics_dict[key][3]
                formatted_str += "{:<15.4f}{:^15.4f}{:^15.4f}{:>15.4f}\n".format(return_value, volatility_value, sharpe_ratio_value, utility_value)

        except Exception as e:
            formatted_str += f"Error occurred when formatting portfolio metrics: {e}\n"

        frame = ttk.Frame(top, padding=10)
        frame.pack(expand=True, fill='both')
        
        # Create a text widget with borders
        text_widget = tk.Text(frame, wrap=tk.NONE)
        text_widget.insert(tk.END, formatted_str)
        text_widget.configure(state='disabled', font=('Courier', 10), relief=tk.SOLID, borderwidth=1)
        text_widget.pack(expand=True, fill='both', padx=10, pady=10)

        # Button to close the window
        # ttk.Button(frame, text="Close", command=top.destroy).pack(pady=10)
        
        # Print to satisfy the assignment requirement
        print(f'\n{formatted_str}')

    def parse_vectors(self, input_str):
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
    
    
    
def main():
    root = tk.Tk()
    app = PortfolioFrontend(root)
    root.mainloop()
    
if __name__ == "__main__":
    main()