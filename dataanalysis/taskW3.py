import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import numpy as np
import scipy.stats as stats


def getPvalue(reg, ts, sided):
    p_value=0
    if sided==1:
        p_value=2*stats.t.sf(abs(ts),reg.df_resid)
    elif sided ==2:
          p_value=stats.t.sf(abs(ts),reg.df_resid)
    else:
        p_value=1-stats.t.sf(abs(ts),reg.df_resid)
    return p_value
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

def t_test(h0,beta,model,sided):
    ts=((model.params[beta]-(h0))/model.bse[beta])
    pVal=getPvalue(model,ts,sided)
    print("test: H_0 MKTER beta = 1; H_A !=1:")
    print(f't-statistic: {ts}')
    print(f'p-value:{pVal}')
    print("from Students t with df:", model.df_resid)
    return {"P-value":pVal, "Test Statistic":ts, "Degrees of Freedom": model.df_resid}

def solution():
    print()
    CAPMdata=getData()

    #fama french model
    model1='TESLAER~MKTER+SMB+HML'
    model2='AZER~MKTER+SMB+HML'
    regression1=sm.ols(model1,CAPMdata).fit()
    regression2=sm.ols(model2, CAPMdata).fit()
    #test: H_0 MKTER beta = 1; H_A !=1
    dict=t_test(1, 1, regression1, 1)
    significants=[0.1,0.05,0.01]
    if dict["P-value"]<significants[0]:
        if dict["P-value"]<significants[1]:
            if dict["P-value"]<significants[2]:
                print(f'We reject H0 with {significants[2]*100}% significans')
            else:
                print(f'We reject H0 with {significants[1]*100}% significans')
        else:
            print(f'We reject H0 with {significants[0]*100}% significans')


    #f-test H_0: SPX beta = 1; H_A: SPX beta!=1
    res='MKTER=1'
    fTest=regression1.f_test(res)
    print()
    print('The F-test results for H_0: SPX beta = 1; H_A: SPX beta!=1:',"\n",f'{fTest.summary()}', "\n")

solution()








    
