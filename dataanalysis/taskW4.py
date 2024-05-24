import pandas as pd						
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.stats.diagnostic as smdiag
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.stattools as stats
from pyfiglet import Figlet


f = Figlet(font="slant", width=1000)
print(f.renderText("Session 4 - Data Analysis"))
print()

def getData():
    CEOdata=pd.read_excel('dataanalysis\CEOunbal.xlsx').dropna()
    CEOdata['LSALARY']=np.log(CEOdata['SALARY'])
    CEOdata['LSALES']=np.log(CEOdata['SALES'])
    CEOdata['ROExSALES']=CEOdata['ROE']*CEOdata['SALES']
    return CEOdata
def regression(data, model):
    regression=smf.ols(model, data).fit()
    return {"model":regression,"regression":regression.summary(), "AIC":regression.aic, "BIC":regression.bic}
def vif(x):
    print(x.corr())
    vif_values=pd.DataFrame() #create empty DataFrame to store results
    vif_values["Regressors"]=x.columns #first variable "regressors" is filled with labels of X variables
    vif_values["VIF"] = [oi.variance_inflation_factor(x.values, i) for i in range(len(x.columns))] #VIFs
    return vif_values
def ramseyReset(regMod):
    return oi.reset_ramsey(regMod, degree=3)

def solution():
    CEOdata=getData()
    model1='SALARY~ROE+ROS+SALES'
    model2='LSALARY~ROE+ROS+LSALES'
    model3='SALARY~ROE+ROS+ROExSALES'
    print()
    print(regression(CEOdata, model1)["regression"]) #regression on model 1
    print()
    print(f'For model 1 the BIC is: {regression(CEOdata, model1)["BIC"]} and the AIC is: {regression(CEOdata, model1)["AIC"]}')
    print()
    print(regression(CEOdata, model2)["regression"]) #regression on model 2
    print()
    print(f'For model 2 the BIC is: {regression(CEOdata, model2)["BIC"]} and the AIC is: {regression(CEOdata, model1)["AIC"]}')
    print()
    print(regression(CEOdata, model3)["regression"]) #regression on model 3
    print()
    print(f'For model 3 the BIC is: {regression(CEOdata, model3)["BIC"]} and the AIC is: {regression(CEOdata, model1)["AIC"]}')

    print()

    x=CEOdata[['ROE', 'ROS', 'SALES']]
    y=CEOdata[['ROE', 'ROS', 'LSALES']]
    z=CEOdata[['ROE', 'ROS', 'ROExSALES']]
    print("The VIF for ROE, ROS, SALES")
    print(vif(x))
    print("The VIF for ROE, ROS and LSALES")
    print(vif(y))
    print("The VIF for ROE, ROS and ROExSALES")
    print(vif(z))   

    print()

    model4='SALARY~ROE+ROS+SALES+ROExSALES'
    print("Regression on model 4")
    print(regression(CEOdata,model4)["regression"])
    o=CEOdata[['ROE', 'ROS', 'SALES', 'ROExSALES']]
    print("The VIF for ROE+ROS+SALES+ROExSALES")
    print(vif(o))

    #Ramsey reset test 
    print(ramseyReset(regression(CEOdata, model1)["model"]))
    print(ramseyReset(regression(CEOdata, model2)["model"]))
    print(ramseyReset(regression(CEOdata, model3)["model"]))

solution()




