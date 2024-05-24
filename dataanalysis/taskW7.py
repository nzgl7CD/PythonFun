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




RDdata=pd.read_excel("dataanalysis\RD25firms.xlsx")
model='RD~PROFITS+INTEREST+FIRMS'
corrMat=RDdata[['PROFITS','INTEREST', 'FIRMS']]



def vif(x):
    #Task 1a
    print(x.corr())
    #Task 1b
    vif_values=pd.DataFrame() #create empty DataFrame to store results
    vif_values["Regressors"]=x.columns #first variable "regressors" is filled with labels of X variables
    vif_values["VIF"] = [oi.variance_inflation_factor(x.values, i) for i in range(len(x.columns))] #VIFs
    return vif_values
print(vif(corrMat))

regression=smf.ols(model,RDdata).fit()
#Task 2a
print(regression.rsquared)
#Task 2b
def t_test(beta,model):
    ts=((model.params[beta]-(0))/model.bse[beta])
    pVal=getPvalue(model,ts,1)
    print(f't-statistic: {ts}')
    print(f'p-value:{pVal}')
    print("from Students t with df:", model.df_resid)
    return {"P-value":pVal, "Test Statistic":ts, "Degrees of Freedom": model.df_resid}
def getPvalue(reg, ts, sided):
    p_value=0
    p_value=2*stats.t.sf(abs(ts),reg.df_resid)

    return p_value
def tes():
    max=0
    #Task 2c
    biggestImpact=""
    for i in range(1, 4):
        if abs(t_test(i,regression)["Test Statistic"])>abs(max):
            max=t_test(i,regression)["Test Statistic"]
            biggestImpact=corrMat[i-1]
        yield t_test(i,regression)["Test Statistic"]
    return biggestImpact

residuals=regression.resid
predicted=regression.fittedvalues
plt.xlabel('Predicted values of RD')
plt.ylabel('Residuals')
plt.scatter(predicted, residuals)
plt.show()


white_test = smdiag.het_white(regression.resid,  regression.model.exog)
print("LM test", white_test[0])
print("p-value", white_test[1])
print("F test", white_test[2])
print("p-value", white_test[3])

#regression with robust se 

regression=smf.ols(model,RDdata).fit(cov_type='HC1')


    


