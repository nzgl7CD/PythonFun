import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import numpy as np
import scipy.stats as stats

def getData():
    CAPMdata=pd.read_excel('Book1.xlsx')
    CAPMdata['deAn1TBILL']=CAPMdata['1TBILL']/365
    CAPMdata['lTSLA']=100*np.log(CAPMdata['TSLA']/CAPMdata['TSLA'].shift(2))
    CAPMdata['lAZ']=100*np.log(CAPMdata['AZ']/CAPMdata['AZ'].shift(2))
    CAPMdata['lMKT']=100*np.log(CAPMdata['MKT']/CAPMdata['MKT'].shift(2))
    CAPMdata['TESLAER']= CAPMdata['lTSLA']-CAPMdata['deAn1TBILL']
    CAPMdata['AZER']= CAPMdata['lAZ'] -  CAPMdata['deAn1TBILL']
    CAPMdata['MKTER']=CAPMdata['lMKT']-CAPMdata['deAn1TBILL']
    CAPMdata=CAPMdata.dropna()
    return CAPMdata

def regress(data, model):
    regression=sm.ols(model,data).fit()
    return regression

def buildmodel(string):
    model1='TESLAER~'
    for i in range(len(string)):
        if i==len(string)-1:
            model1+=string[i]
        else:
            model1+=string[i]+"+"
    return model1

def getPvalue(reg, ts):
    p_value=2*stats.t.sf(abs(ts),reg.df_resid)
    return p_value

def checkSlopes():
    CAPMdata=getData()
    regressors='MKTER+SMB+HML+deAn1TBILL+AZER'
    splittedregressors=regressors.split("+")
    significant=int(input("Significant: "))
    model1=buildmodel(splittedregressors)
    regression=regress(CAPMdata, model1)
    c=0
    while c<len(splittedregressors):
        ts=((regression.params[c+1]-(0))/regression.bse[c+1])
        pVal=getPvalue(regression,ts)
        if pVal>significant:
            splittedregressors.remove(splittedregressors[c])
            model1=buildmodel(splittedregressors)
            regression=regress(CAPMdata, model1)
            c=0
        else:
            c+=1
            continue
    return regression.summary()
            
print(checkSlopes())


