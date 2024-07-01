import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
class PortfolioOptimizer:
    def __init__(self, expected_return, volatility, corr_matrix, risk_free_rate, portfolio_size, risk_aversion):
        
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
        
        if not self.is_effectively_empty(expected_return, volatility,corr_matrix):
            self.returns=np.asarray(expected_return)
            corr_matrix = np.array(corr_matrix)
            stdv = np.array(volatility)
            self.cov_matrix = np.outer(stdv, stdv) * corr_matrix
            
            # Alternative to show essential coding skills and knowledge 
            # self.returns = list(expected_return)
            # corr_matrix = list(corr_matrix)
            # stdv = list(volatility)

            # # Initialize the covariance matrix with zeros
            # self.cov_matrix = [[0.0 for _ in range(len(stdv))] for _ in range(len(stdv))]

            # # Calculate the covariance matrix using nested loops
            # for i in range(len(stdv)):
            #     for j in range(len(stdv)):
            #         self.cov_matrix[i][j] = stdv[i] * stdv[j] * corr_matrix[i][j]
            
        else:
            # Use real data from excel input sheet
            self.ds = self.get_data(portfolio_size)
            self.returns = self.calculate_annualized_returns()
            self.cov_matrix = self.compute_covariance_matrix(self.ds)
            
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        self.C, self.G, self.H=self.calculate_intermediate_quantities()
        
    def is_effectively_empty(self,expected_return,volatility,corr_matrix):
        if expected_return and len(expected_return)==self.portfolio_size and volatility and len(volatility)==self.portfolio_size and corr_matrix and len(corr_matrix)==self.portfolio_size:
            return False
        return True

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

    def calculate_intermediate_quantities(self):
        # Calculates all variables from HL model
        #TODO 
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
        
        # Return variable used in other functions only
        
        return C, G, H
    
    # Calculate all relevant values for a portfolio
    def calculate_portfolio_metrics(self, weights):
        portfolio_return = np.sum(weights * self.returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        excess_return = portfolio_return - self.risk_free_rate
        sharpe_ratio = excess_return / portfolio_risk
        utility = portfolio_return - (0.5 * self.risk_aversion * portfolio_variance)
        return portfolio_return, portfolio_risk, sharpe_ratio, utility

    # Calculate minimum variance weights
    def calculate_minimum_variance_weights(self):
        min_var_weights = np.dot(self.inv_cov_matrix, np.ones(self.portfolio_size)) /self.C
        return min_var_weights

    # Calculate optimal variance weights
    def calculate_optimum_variance_weights(self, target_return):
        weights = self.G+(target_return*self.H)
        return weights

    def calculate_mean_variance_efficient_frontier(self):
        """
        Calculate and returns one list of lists with all weights for the mean-variance optimal portfolio on target return.
        Returns the efficient frotnier portfolio_return, portfolio_risk, sharpe_ratio, utility for mean-variance optimal portfolio
        """
        min_var_weights = self.calculate_minimum_variance_weights()
        frontier_weights = []
        for target_return in np.linspace(0, 1, 101):
            opt_var_weights = self.calculate_optimum_variance_weights(target_return)
            weights = (1 - target_return) * min_var_weights + target_return * opt_var_weights
            frontier_weights.append(weights)
        frontier_metrics = [self.calculate_portfolio_metrics(w) for w in frontier_weights]
        return frontier_weights, frontier_metrics

    def plot_efficient_frontier(self, ax):
        
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
        """
        summary_
        Writes the calculated mean-variance efficient frontier data to an Excel file.

        Steps:
        1. Calculates the efficient frontier weights and metrics by calling the 
        `calculate_mean_variance_efficient_frontier` method.
        2. Checks if a dataset (`self.ds`) is available to determine the column names 
        for the weights. If not, generates default weight column names based on 
        the portfolio size.
        3. Constructs a dictionary to store the data for Return, Volatility, Utility, 
        Sharpe Ratio, and weights.
        4. Creates a DataFrame from the data dictionary and rounds all numeric values 
        to 4 decimal places.
        5. Writes the DataFrame to an Excel file, replacing the existing 'output' sheet 
        if it exists.
        6. Adjusts the column widths in the Excel sheet to fit the content.
        """
        # get weights for columns and metrics for portfolio values
        frontier_weights, frontier_metrics = self.calculate_mean_variance_efficient_frontier()
        
        # Check if dataset exists or if the user input is the dataset for the weights columns
        if hasattr(self, 'ds'): 
            weight_columns = [f'w_{col}' for col in self.ds.columns[1:]]
        else:
            weight_columns = [f'w{i+1}' for i in range(self.portfolio_size)]
        
        data = {
            'Return': [metric[0] for metric in frontier_metrics],
            'Volatility': [metric[1] for metric in frontier_metrics],
            'Sharpe Ratio': [metric[2] for metric in frontier_metrics],
            'Utility': [metric[3] for metric in frontier_metrics]
            
        }

        # Make columns for weights and round all data to 4 decimals
        for i, col in enumerate(weight_columns):
            data[col] = [w[i] for w in frontier_weights]

        df = pd.DataFrame(data)
        df.sort_values(by='Return', inplace=True)
        numeric_columns = ['Return', 'Volatility', 'Sharpe Ratio', 'Utility'] + weight_columns
        df[numeric_columns] = df[numeric_columns].round(4)
        
        # Write to excel sheet and replace existing output sheet if it exists
        with pd.ExcelWriter(output_file, mode='a', engine="openpyxl",if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name='output', index=False)
            workbook = writer.book
            worksheet = workbook['output']
            
            # Fit cells with regular nested for loop as requested from assignment
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

            self.dataframe=df
    
    def print_values(self):
        """_summary_
            Collects return, volatility, sharpe ration and utility of three portolfios from the excel sheet
        Returns:
            Dictionary of relevant values
        """
        
        df = self.dataframe
        output_str = ""
        
        max_sharpe_idx = df['Sharpe Ratio'].idxmax()
        max_sharpe_return, max_sharpe_volatility, max_sharpe_value, max_sharpe_utility = df.loc[max_sharpe_idx, ['Return', 'Volatility', 'Sharpe Ratio', 'Utility']]
        
        max_utility_idx = df['Utility'].idxmax()
        max_utility_return, max_utility_volatility, max_utility_sharpe, max_utility_value = df.loc[max_utility_idx, ['Return', 'Volatility', 'Sharpe Ratio', 'Utility']]
        
        min_volatility_idx = df['Volatility'].idxmin()
        min_volatility_return, min_volatility_volatility, min_volatility_sharpe, min_volatility_utility = df.loc[min_volatility_idx, ['Return', 'Volatility', 'Sharpe Ratio', 'Utility']]
        
        return {'MaxSharpeRatio': [max_sharpe_idx,max_sharpe_return, max_sharpe_volatility, max_sharpe_value, max_sharpe_utility],
                'MaxUtility': [max_utility_idx, max_utility_return, max_utility_volatility, max_utility_sharpe, max_utility_value],
                'MinVar': [min_volatility_idx,min_volatility_return, min_volatility_volatility, min_volatility_sharpe, min_volatility_utility]}
        
class PortfolioFrontend:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Optimizer")
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Helvetica', 12), padding=10, background='#4CAF50', foreground='#000000')
        self.style.map('TButton', background=[('active', '#45a049')])
        self.create_widgets()
        self.metrics_dict = None

    def create_widgets(self):
        self.frame = ttk.Frame(self.root, padding="20")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Portfolio Size
        ttk.Label(self.frame, text="Portfolio Size (2-12 securities):").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.portfolio_size_entry = ttk.Entry(self.frame, width=10, font=('Helvetica', 12))
        self.portfolio_size_entry.grid(column=1, row=0, padx=10, pady=10)
        self.portfolio_size_entry.insert(tk.END, "2")
        
        # Risk-Free Rate
        ttk.Label(self.frame, text="Risk-Free Rate:").grid(column=0, row=2, padx=10, pady=10, sticky=tk.W)
        self.risk_free_rate_entry = ttk.Entry(self.frame, width=10, font=('Helvetica', 12))
        self.risk_free_rate_entry.grid(column=1, row=2, padx=10, pady=10)
        self.risk_free_rate_entry.insert(tk.END, "0.045")

        # Risk Aversion
        ttk.Label(self.frame, text="Risk Aversion:").grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
        self.risk_aversion_entry = ttk.Entry(self.frame, width=10, font=('Helvetica', 12))
        self.risk_aversion_entry.grid(column=1, row=1, padx=10, pady=10)
        self.risk_aversion_entry.insert(tk.END, "3.0")

        # Update Fields Button
        update_button = ttk.Button(self.frame, text="Update Fields", command=self.click_update_fields, style='TButton')
        update_button.grid(column=2, row=0, padx=10, pady=10)
        
        # Expected Returns
        ttk.Label(self.frame, text="Expected Returns (comma-separated):").grid(column=0, row=4, padx=10, pady=10, sticky=tk.W)
        self.expected_returns_text = tk.Text(self.frame, width=60, height=1, font=('Helvetica', 12))
        self.expected_returns_text.grid(column=1, row=4, columnspan=2, padx=10, pady=10)
        self.expected_returns_text.insert(tk.END, "0.1934, 0.1575") # Suggest standards from excel file provided
        
        # Volatilities
        ttk.Label(self.frame, text="Volatilities (comma-separated):").grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)
        self.volatilities_text = tk.Text(self.frame, width=60, height=1, font=('Helvetica', 12))
        self.volatilities_text.grid(column=1, row=3, columnspan=2, padx=10, pady=10)
        self.volatilities_text.insert(tk.END, "0.3025, 0.219") # Suggest standards from excel file provided

        # Correlation Matrix
        ttk.Label(self.frame, text="Correlation Matrix (semicolon-separated rows):").grid(column=0, row=5, padx=10, pady=10, sticky=tk.W)
        self.correlation_matrix_text = tk.Text(self.frame, width=60, height=4, font=('Helvetica', 12))
        self.correlation_matrix_text.grid(column=1, row=5, columnspan=2, padx=10, pady=10)
        self.correlation_matrix_text.insert(tk.END, "1.0, 0.35; 0.35, 1.0") # Suggest standards from excel file provided

        # Button Frame to centre the buttons 
        button_frame = ttk.Frame(self.frame)
        button_frame.grid(column=0, row=6, columnspan=3, padx=10, pady=20, sticky=tk.W + tk.E)

        # Run Optimizer Button for input data
        run_button = ttk.Button(button_frame, text="Run Optimizer", command=self.click_run_optimizer, style='TButton')
        run_button.grid(column=0, row=0, padx=5, pady=10)
        
        # Run Optimizer Using Real Data 
        run_button = ttk.Button(button_frame, text="Run: Real Data", command=self.click_run_w_real_data, style='TButton')
        run_button.grid(column=1, row=0, padx=5, pady=10)
        
        # Exit Button to quit program
        exit_button = ttk.Button(button_frame, text="Exit", command=self.exit_program, style='TButton')
        exit_button.grid(column=2, row=0, padx=5, pady=10)

        # Center buttons within the button_frame
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
    
    def click_update_fields(self):
        """_summary_
        1. Validates if portfolio size, aversion and risk free rate are valid numbers
        2. Updates remaining fields with lengths that match portfolio size with standardised values. 
        """
        try:
            portfolio_size = int(self.portfolio_size_entry.get())
            risk_aversion_entry = self.risk_aversion_entry.get()
            risk_free_rate_entry = self.risk_free_rate_entry.get()
            if self.validate_size_aversion_riskfree(portfolio_size, risk_aversion_entry,risk_free_rate_entry):
                self.volatilities_text.delete(1.0, tk.END)
                self.expected_returns_text.delete(1.0, tk.END)
                self.correlation_matrix_text.delete(1.0, tk.END)

                default_volatilities = ",".join(["0.1"] * portfolio_size)
                default_returns = ",".join(["0.1"] * portfolio_size)
                default_correlation = ";".join([",".join(["1.0" if i == j else "0.35" for j in range(portfolio_size)]) for i in range(portfolio_size)])
                
                self.volatilities_text.insert(tk.END, default_volatilities)
                self.expected_returns_text.insert(tk.END, default_returns)
                self.correlation_matrix_text.insert(tk.END, default_correlation)

        except ValueError:
            messagebox.showerror("Error", "Portfolio Size must be a valid integer.")
    
    #seperate function used by update and run optimizers
    def validate_size_aversion_riskfree(self, portfolio_size, risk_aversion, risk_free):
        try:
            if portfolio_size < 2 or portfolio_size > 12:
                messagebox.showerror("Error", "Portfolio Size must be between 2 and 12.")
                return False
            elif risk_free and not self.validate_input(risk_free,-1.0,1.0,'Risk free rate'):
                return False
            elif  risk_aversion and not self.validate_input(risk_aversion,0,100.0,'Risk aversion'):
                return False
            return True
        except ValueError:
            messagebox.showerror("Error", "Portfolio Size must be a valid integer.")
            
    def validate_input(self, value, min_val, max_val, field_name):
        try:
            val = float(value)
            if not (min_val <= val <= max_val):
                messagebox.showerror("Error", f"{field_name} must be between {min_val} and {max_val}.")
                return False
            return True
        
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            return False

    def process_user_inputs(self, use_user_inputs:bool)->bool:
        try:
            portfolio_size = int(self.portfolio_size_entry.get())
            risk_aversion_entry = self.risk_aversion_entry.get()
            risk_free_rate_entry = self.risk_free_rate_entry.get()
            
            # Check the three first inputs
            if self.validate_size_aversion_riskfree(portfolio_size,risk_aversion_entry,risk_free_rate_entry):
                if use_user_inputs:
                    volatilities = self.volatilities_text.get("1.0", tk.END).strip().split(',')
                    expected_returns = self.expected_returns_text.get("1.0", tk.END).strip().split(',')
                    correlation_rows = self.correlation_matrix_text.get("1.0", tk.END).strip().split(';')
                    
                    if len(volatilities) != portfolio_size or len(expected_returns) != portfolio_size or len(correlation_rows) != portfolio_size:
                        messagebox.showerror("Error", "The number of entries must match the portfolio size.")
                        return False

                    for vol in volatilities:
                        if not self.validate_input(vol, 0.0, 1.0, "Volatility"):
                            return False

                    for ret in expected_returns:
                        if not self.validate_input(ret, -1.0, 1.0, "Expected Return"):
                            return False
                        
                    for row in correlation_rows:
                        correlations = row.split(',')
                        if len(correlations) != portfolio_size:
                            messagebox.showerror("Error", "The correlation matrix must be square and match the portfolio size.")
                            return False
                        for corr in correlations:
                            if not self.validate_input(corr, -1.0, 1.0, "Correlation"):
                                return False

                    self.volatilities = list(map(float, volatilities))
                    self.expected_returns = list(map(float, expected_returns))
                    self.correlation_matrix = [list(map(float, row.split(','))) for row in correlation_rows]
                return True

        except ValueError:
            messagebox.showerror("Error", "Portfolio Size must be a valid integer.")
            return False
    
    # generates object from backend class with paramteres if click_run_optimizer()
    def generate_portfolio_optimizer(self,expected_returns=[],volatilities=[],correlation_matrix=[]):
        portfolio_size = int(self.portfolio_size_entry.get())
        risk_aversion_entry = self.risk_aversion_entry.get()
        risk_free_rate_entry = self.risk_free_rate_entry.get()
        risk_aversion = float(risk_aversion_entry) if risk_aversion_entry else 3.0
        risk_free_rate = float(risk_free_rate_entry) if risk_free_rate_entry else 0.045
        
        optimizer = PortfolioOptimizer(expected_return=expected_returns,  
                                        volatility=volatilities,     
                                        corr_matrix=correlation_matrix,    
                                        risk_free_rate=risk_free_rate,
                                        portfolio_size=portfolio_size,
                                        risk_aversion=risk_aversion)
        return optimizer
    
    def click_run_optimizer(self):
        try:
            if self.process_user_inputs(True):
                volatilities = self.parse_vectors(self.volatilities_text.get("1.0", tk.END))
                expected_returns = self.parse_vectors(self.expected_returns_text.get("1.0", tk.END))
                correlation_matrix = self.parse_correlation_matrix(self.correlation_matrix_text.get("1.0", tk.END))
                
                self.optimizer=self.generate_portfolio_optimizer(expected_returns,volatilities,correlation_matrix)
                
                self.run_optimizer() 
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            return False
        
    def click_run_w_real_data(self):
        try:
            if self.process_user_inputs(False):
                self.optimizer=self.generate_portfolio_optimizer()
                self.run_optimizer() 
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            return False
        
    def run_optimizer(self):
        if self.optimizer is not None:
            try:
                self.show_plot(self.optimizer)
                self.optimizer.write_to_excel('230023476PortfolioProblem.xlsx')
                self.metrics_dict = self.optimizer.print_values()

            except ValueError as e:
                messagebox.showerror("Error", str(e))
    
    def show_plot(self, optimizer):
        if hasattr(self, 'plot_window') and self.plot_window.winfo_exists():
            self.plot_window.destroy()

        # Create a new plot window for aesthetics with exit button 
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
        cols = 6 
        top.geometry(f"{cols * 135}x{rows * 150}")  
        
        formatted_str = "" 

        try:
            formatted_str += "{:<15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}\n".format("Portfolio Type","Portolfio No.", "Return", "Volatility", "Sharpe Ratio", "Utility")
            formatted_str += "-" * (92) + "\n"
            for key in metrics_dict:
                formatted_str += "{:<15s}".format(key) 
                portfolio_index=metrics_dict[key][0]
                return_value = metrics_dict[key][1]
                volatility_value = metrics_dict[key][2]
                sharpe_ratio_value = metrics_dict[key][3]
                utility_value = metrics_dict[key][4]
                formatted_str += "{:^15.0f}{:^15.4f}{:^15.4f}{:^15.4f}{:^15.4f}\n".format(portfolio_index+1, return_value, volatility_value, sharpe_ratio_value, utility_value)

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
        ttk.Button(frame, text="Close", command=top.destroy).pack(pady=10)
        
        # Print to terminal to satisfy the assignment requirement
        print(f'\n{formatted_str}')

    # Used to strip vecror values for expected return and volatility
    def parse_vectors(self, input_str):
        try:
            values = list(map(float, input_str.strip().split(',')))
            return values
        except ValueError as e:
            raise ValueError(f"Invalid input: {e}")
        
    #used to strip both vectors and their values in corr matrix
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
    
    def exit_program(self):
        self.root.destroy()
        self.root.quit()
    
def main():
    try:
        root = tk.Tk()
        app = PortfolioFrontend(root)
        root.mainloop()
    except Exception as e:
        print(f"An error occurred: {e}")
    
if __name__ == "__main__":
    main()
