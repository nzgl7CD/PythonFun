from linearmodels import PooledOLS
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tsap
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_white
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
def getData():
        CEOdata = pd.read_excel('bendik/dfBRP.xlsx')
        headers = ['Company', 'Year','Buy Out Indicator', 'Profit per employee', 'Int Cov Ratio', 'ROE', 'ROA', 'Gross margin']
        df = pd.DataFrame(columns=headers)
        yearIndexer=[2005,2006,2007,2008,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2019,2020,2021,2022,2023]
        for index, row in CEOdata.iterrows():
                company = row['Company']
                year=2005
                dateaquestition=row['Date buyout']
                dateaquestition=str(dateaquestition).split("-")
                dateaquestition=int(dateaquestition[0])
                buyoutindicator=0
                counter=0
                for i in range(0,19):
                        buyoutindicator=1 if dateaquestition <=year else 0
                        if buyoutindicator ==1:
                                if counter==5:
                                        buyoutindicator=0
                                else:
                                        counter=counter+1
                        temp={"Company": company, "Year": year, "Buy Out Indicator": buyoutindicator, "Profit per employee": row["Profit per Empl "+str(yearIndexer[i])], 'Int Cov Ratio': row["Int Cov Ratio "+str(yearIndexer[i])], 'ROE':row["ROE "+str(yearIndexer[i])], 'ROA':row["ROA "+str(yearIndexer[i])], 'Gross margin': row['Gross margin '+str(yearIndexer[i])]  }
                        df.loc[len(df)] = temp
                        year=year+1
                counter=0
        print(df.head(20))
        return df
def regression():
        df=getData()
        df=df.dropna()
        y='Profit per employee'
        x=['Buy Out Indicator']
        pooled_y=df[y]
        pooled_X=df[x]
        pooled_X = sm.add_constant(pooled_X)
        pooled_olsr_model = sm.OLS(endog=pooled_y, exog=pooled_X)
        pooled_olsr_model_results = pooled_olsr_model.fit()
        print(pooled_olsr_model_results.summary())
        
regression()