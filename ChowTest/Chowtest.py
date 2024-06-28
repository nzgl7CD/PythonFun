import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import chi2

def set_data():
    ds = pd.read_excel('data.xlsx')

    ds['delta_lnWctv'] = np.log(ds['WCTW'] / ds['WCTW'].shift(1))
    ds['delta_lnP'] = np.log(ds['P'] / ds['P'].shift(1))
    ds['delta_lnProd'] = np.log(ds['PROD'] / ds['PROD'].shift(1))
    ds['delta_lnCpi'] = np.log(ds['CPI'] / ds['CPI'].shift(1))
    # ds['delta_NH'] = ds['NH'] - ds['NH'].shift(1)    
    ds['lnWS'] = np.log(ds['WS'])
    ds['lnUREG'] = np.log(ds['Ureg'])

    # Create lagged variables
    ds['delta_lnWctvLag'] = ds['delta_lnWctv'].shift(1)
    ds['lnWSLag'] = ds['lnWS'].shift(1)
    ds['delta_lnPLag'] = ds['delta_lnP'].shift(1)
    ds['delta_lnProdLag'] = ds['delta_lnProd'].shift(1)
    ds['lnUregLag'] = ds['lnUREG'].shift(1)
    ds['delta_lnCpiLag'] = ds['delta_lnCpi'].shift(1)
    
    # Clean up for later use of chowtest
    columns = ['Year','delta_lnWctv', 'delta_lnWctvLag', 'lnWSLag', 'delta_lnP', 'delta_lnPLag',
               'delta_lnProd', 'delta_lnProdLag', 'lnUREG', 'lnUregLag', 'delta_lnCpi', 
               'delta_lnCpiLag', 'delta_NH', 'STOP']
    # columns = ['Year','delta_lnWctv', 'delta_lnWctvLag', 'lnWSLag', 'delta_lnP', 'delta_lnPLag',
    #            'delta_lnProd', 'delta_lnProdLag', 'lnUREG', 'lnUregLag', 'delta_lnCpi', 
    #            'delta_lnCpiLag']
    ds = ds[columns].dropna()

    return ds

def regression(data, model):
    regression = ols(model, data).fit()
    return regression.summary()


def ChowTest(data, y_variable_name, subgroup_variable_name):

    X = data.drop(y_variable_name, axis=1)
    X = sm.add_constant(X)
    y = data[y_variable_name]
    X_1 = X[data[subgroup_variable_name] == 1].drop(subgroup_variable_name, axis=1)
    y_1 = data[data[subgroup_variable_name] == 1][y_variable_name]
    X_2 = X[data[subgroup_variable_name] == 0].drop(subgroup_variable_name, axis=1)
    y_2 = data[data[subgroup_variable_name] == 0][y_variable_name]
    J = X.shape[1]
    k = X_1.shape[1]
    N1 = X_1.shape[0]
    N2 = X_2.shape[0]
    model_dummy = sm.OLS(y, X).fit()
    RSSd = model_dummy.ssr
    model_1 = sm.OLS(y_1, X_1).fit()
    RSS1 = model_1.ssr
    model_2 = sm.OLS(y_2, X_2).fit()
    RSS2 = model_2.ssr
    chow = ((RSSd - (RSS1 + RSS2)) / J) / ((RSS1 + RSS2) / (N1 + N2 - 2 * k))
    p_value = 1 - scipy.stats.f.cdf(chow, J, N1 + N2 - 2 * k)
    
    results = pd.DataFrame({
        'RSSr': [RSSd],
        'RSS1': [RSS1],
        'RSS2': [RSS2],
        'Chow Statistic': [chow],
        'p-value': [p_value]
    })
    
    return results

def save_data(data, output_path):
        if data is not None:
            data.to_excel(output_path, index=False)
        else:
            print("No data to save. Please load and clean the data first.")

data = set_data()
model = 'delta_lnWctv ~ delta_lnWctvLag + lnWSLag + delta_lnP + delta_lnPLag + delta_lnProd + delta_lnProdLag + lnUREG + lnUregLag + delta_lnCpi + delta_lnCpiLag + delta_NH + STOP'
# model = 'delta_lnWctv ~ delta_lnWctvLag + lnWSLag + delta_lnP + delta_lnPLag + delta_lnProd + delta_lnProdLag + lnUREG + lnUregLag + delta_lnCpi + delta_lnCpiLag'
# Create the PostCorona dummy variable
print(regression(data, model))
data['postCorona'] = data['Year'].apply(lambda x: 1 if x >= 1990 else 0)
# Run the Chow Test
chow_test_results = ChowTest(data, 'delta_lnWctv', 'postCorona')
print(chow_test_results)

# Save to output file
output_path=r'output.xlsx'
save_data(data,output_path)



