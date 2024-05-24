import pandas as pd						
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.tsa.api as smt
import statsmodels.stats.diagnostic as smdiag
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.stattools as stats
class task9:
    def __init__(self) -> None:
        global RDdata 
        RDdata=pd.read_excel("dataanalysis\RD25firmsexports.xlsx")
    
    def model(self):
        model='RD~PROFITS+INTEREST+FIRMS+FIRMS*EXPORTS'
        regression=smf.ols(model,RDdata).fit()
        corrmat=[['PROFITS', 'INTEREST', 'FIRMS', 'FIRMS*EXPORTS']]
        vif_values=pd.DataFrame() #create empty DataFrame to store results
        vif_values["Regressors"]=x.columns #first variable "regressors" is filled with labels of X variables
        vif_values["VIF"] = [oi.variance_inflation_factor(corrmat.values, i) for i in range(len(corrmat.columns))] #VIFs
        yield vif_values
        print(corrmat.corr())
        ts=((regression.params[4]-(0))/regression.bse[4])
        p_value=stats.t.sf(abs(ts),regression.df_resid)
        yield ts, p_value
        

        
    
    
        
    
        

    
task9().model()