import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

def rSquared(rss, varY, varE, t, k):
    r2=1-rss/(varY*(t-1))
    r2100=1-rss/(varY*(100-1))
    r2adj100=r2Adj=1-(100-1)/(100-k)*(1-r2)
    r2Adj=1-(t-1)/(t-k)*(1-r2)
    r2adjfrom100=r2Adj/r2adj100

    aic=np.log(rss)/t+(2*k)/t
    aic100=np.log(rss)/100+(2*k)/100
    aicfrom100=aic/aic100


    return "r2", round(r2, 3),"adr2", round(r2Adj, 3),"aic", aic, r2adjfrom100/100, aicfrom100/100

x=[i*467 for i in range(1,10)]
print(rSquared(1693, 5.2086**2, 0, 30, 3))

