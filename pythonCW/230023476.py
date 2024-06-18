import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PortfolioBackend:
    def __init__(self, returns, volatilities, correlations, risk_free_rate, risk_aversion):
        self.returns = np.array(returns)
        self.volatilities = np.array(volatilities)
        self.correlations = np.array(correlations)
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion
        self.cov_matrix = self.calculate_covariance_matrix()
        self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        self.n = len(returns)

    def calculate_covariance_matrix(self):
        stddevs = np.diag(self.volatilities)
        cov_matrix = stddevs @ self.correlations @ stddevs
        return cov_matrix

    def calculate_intermediate_quantities(self):
        u = np.ones(self.n)
        inv_cov_matrix = self.inv_cov_matrix

        A = np.array([np.sum(u[i] * self.returns[j] * inv_cov_matrix[i, j] for i in range(self.n)) for j in range(self.n)])
        B = np.array([np.sum(self.returns[i] * self.returns[j] * inv_cov_matrix[i, j] for i in range(self.n)) for j in range(self.n)])
        C = np.array([np.sum(u[i] * u[j] * inv_cov_matrix[i, j] for i in range(self.n)) for j in range(self.n)])
        M = np.array([np.sum(self.returns[i] * u[j] * inv_cov_matrix[i, j] for i in range(self.n)) for j in range(self.n)])
        L = np.array([np.sum(self.returns[i] * inv_cov_matrix[i, j] for i in range(self.n)) for j in range(self.n)])

        D = B * C - A ** 2

        G = (M * C - L * A) / D
        H = (L * B - M * A) / D

        return G, H

    def calculate_portfolio_metrics(self, weights):
        portfolio_return = np.dot(weights, self.returns)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        return portfolio_return, portfolio_risk

    def calculate_minimum_variance_portfolio(self):
        G, _ = self.calculate_intermediate_quantities()
        weights = G
        return weights, *self.calculate_portfolio_metrics(weights)

    def calculate_optimum_variance_portfolio(self):
        G, H = self.calculate_intermediate_quantities()
        target_return = self.returns.mean()  # Choose mean return as the target for simplicity
        weights = G + H * target_return
        return weights, *self.calculate_portfolio_metrics(weights)

    def calculate_efficient_frontier(self):
        G, H = self.calculate_intermediate_quantities()
        target_returns = np.linspace(0, max(self.returns), 100)
        risks = []
        returns = []

        for t in target_returns:
            weights = G + H * t
            portfolio_return, portfolio_risk = self.calculate_portfolio_metrics(weights)
            risks.append(portfolio_risk)
            returns.append(portfolio_return)

        return np.array(risks), np.array(returns)

    def plot_efficient_frontier(self):
        risks, returns = self.calculate_efficient_frontier()
        plt.figure(figsize=(10, 6))
        plt.plot(risks, returns, marker='o', linestyle='-', color='b', markersize=5)
        plt.xlabel('Risk (Standard Deviation)')
        plt.ylabel('Return')
        plt.title('Efficient Frontier')
        plt.grid(True)
        plt.show()


class PortfolioFrontend:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Optimizer")
        self.create_widgets()

    def create_widgets(self):
        self.frame = ttk.Frame(self.root, padding="20")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Portfolio size
        ttk.Label(self.frame, text="Number of Securities:").grid(column=0, row=0, padx=10, pady=10, sticky=tk.W)
        self.size_entry = ttk.Entry(self.frame, width=20)
        self.size_entry.grid(column=1, row=0, padx=10, pady=10)
        self.size_entry.insert(tk.END, "3")
        ttk.Label(self.frame, text="(2 to 12)").grid(column=2, row=0, padx=10, sticky=tk.W)

        # Expected returns
        ttk.Label(self.frame, text="Expected Returns (comma-separated):").grid(column=0, row=1, padx=10, pady=10, sticky=tk.W)
        self.returns_entry = ttk.Entry(self.frame, width=50)
        self.returns_entry.grid(column=1, row=1, columnspan=2, padx=10, pady=10)
        self.returns_entry.insert(tk.END, "0.05, 0.07, 0.09")

        # Volatilities
        ttk.Label(self.frame, text="Volatilities (comma-separated):").grid(column=0, row=2, padx=10, pady=10, sticky=tk.W)
        self.volatilities_entry = ttk.Entry(self.frame, width=50)
        self.volatilities_entry.grid(column=1, row=2, columnspan=2, padx=10, pady=10)
        self.volatilities_entry.insert(tk.END, "0.1, 0.12, 0.15")

        # Correlations
        ttk.Label(self.frame, text="Correlation Matrix (semicolon-separated rows):").grid(column=0, row=3, padx=10, pady=10, sticky=tk.W)
        self.correlations_entry = ttk.Entry(self.frame, width=50)
        self.correlations_entry.grid(column=1, row=3, columnspan=2, padx=10, pady=10)
        self.correlations_entry.insert(tk.END, "1.0; 0.3, 1.0; 0.2, 0.4, 1.0")

        # Risk-free rate
        ttk.Label(self.frame, text="Risk-free Rate:").grid(column=0, row=4, padx=10, pady=10, sticky=tk.W)
        self.risk_free_entry = ttk.Entry(self.frame, width=20)
        self.risk_free_entry.grid(column=1, row=4, padx=10, pady=10)
        self.risk_free_entry.insert(tk.END, "0.02")

        # Risk aversion coefficient
        ttk.Label(self.frame, text="Risk Aversion Coefficient:").grid(column=0, row=5, padx=10, pady=10, sticky=tk.W)
        self.risk_aversion_entry = ttk.Entry(self.frame, width=20)
        self.risk_aversion_entry.grid(column=1, row=5, padx=10, pady=10)
        self.risk_aversion_entry.insert(tk.END, "3.0")

        # Run button
        run_button = ttk.Button(self.frame, text="Run Optimizer", command=self.run_optimizer)
        run_button.grid(column=0, row=6, columnspan=3, padx=10, pady=20, sticky=tk.W+tk.E)

        # Add a tooltip for run button
        tooltip = ttk.Label(self.frame, text="Click to run portfolio optimization", foreground="gray")
        tooltip.grid(column=0, row=7, columnspan=3, padx=10, pady=5, sticky=tk.W)

        # Bind hover events to show/hide tooltip
        run_button.bind("<Enter>", lambda e: tooltip.grid(row=7, column=0, columnspan=3))
        run_button.bind("<Leave>", lambda e: tooltip.grid_forget())

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
            # Parse inputs
            size = int(self.size_entry.get())
            if size < 2 or size > 12:
                raise ValueError("Number of securities must be between 2 and 12.")
            
            expected_returns = self.parse_input(self.returns_entry.get(), size)
            volatilities = self.parse_input(self.volatilities_entry.get(), size)
            correlation_matrix = self.parse_correlation_matrix(self.correlations_entry.get(), size)
            risk_free_rate = float(self.risk_free_entry.get())
            risk_aversion = float(self.risk_aversion_entry.get())
            
            # Example of what to do next (integration with PortfolioOptimizer class)
            # optimizer = PortfolioOptimizer(expected_returns, volatilities, correlation_matrix, risk_free_rate, risk_aversion)
            # optimizer.run_optimization()

            messagebox.showinfo("Optimizer", "Optimization complete!")
        
        except ValueError as e:
            messagebox.showerror("Error", str(e))
#corr4 0.3,0.4,0.5,0.7 ; 0.3,0.4,0.5,0.7 ; 0.3,0.4,0.5,0.7 ; 0.3,0.4,0.5,0.7 
# Example usage:
if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioFrontend(root)
    root.mainloop()
