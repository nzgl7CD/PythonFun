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
import statsmodels.tsa.ar_model as sar



class ExamPrep:
    def getData():
        CEOdata=pd.read_excel('dataanalysis\CEOunbal.xlsx').dropna()
        CEOdata['LSALARY']=np.log(CEOdata['SALARY'])
        CEOdata['LSALES']=np.log(CEOdata['SALES'])
        CEOdata['ROExSALES']=CEOdata['ROE']*CEOdata['SALES']
        return CEOdata
    def getTimeSerie():
        UKHPIdata=pd.read_excel(r'dataanalysis\UKHOUSE.xlsx').dropna()
        UKHPIdata['CPINF']=100*np.log(UKHPIdata['CPI']/UKHPIdata['CPI'].shift(12))
        UKHPIdata['FXCHG']=100*np.log(UKHPIdata['USGBP']/UKHPIdata['USGBP'].shift(12))
       
        UKHPIdata=UKHPIdata.dropna()
        return UKHPIdata

    def regression(data, model):
        regression=smf.ols(model, data).fit()
        return {"model":regression,"regression":regression.summary(), "AIC":regression.aic, "BIC":regression.bic}
    def vif(x):
        print("Correlation Matrix")
        print(x.corr())
        for i in range(3): print()
        vif_values=pd.DataFrame() #create empty DataFrame to store results
        vif_values["Regressors"]=x.columns #first variable "regressors" is filled with labels of X variables
        vif_values["VIF"] = [oi.variance_inflation_factor(x.values, i) for i in range(len(x.columns))] #VIFs
        return vif_values
    def ramseyReset(regMod):
        return oi.reset_ramsey(regMod, degree=3)
    
    def sumData(): #from week 3 
        CEOdata=ExamPrep.getData()  
        CEOdata.corr()					                     
        Y=CEOdata['SALARY']			        
        X=CEOdata[['ROE',  'ROS', 'SALES']]    
        print(Y.head())							
        print(Y.describe())	
        print(X.head)
        print(X.describe())

    
    def multicoll(): #from week 5
        CEOdata=ExamPrep.getData() 
        model='SALARY~ROE+ROS+SALES+ROExSALES'
        print(ExamPrep.regression(CEOdata,model)["regression"])
        o=CEOdata[['ROE', 'ROS', 'SALES', 'ROExSALES']]
        print(ExamPrep.vif(o))
        for i in range(3): print()
        print("Ramsey test to check for misspecification. H0 is that there's a good fit", end='\n \n')
        print(ExamPrep.ramseyReset(ExamPrep.regression(CEOdata, model)["model"]))
        for i in range(3): print()
    
    def autoCorr(): #from week 8
        UKHPIdata=ExamPrep.getTimeSerie()
        UKHPIdata['UNEMPL3']=(UKHPIdata['UNEMPL'].shift(1)+UKHPIdata['UNEMPL'].shift(2)+UKHPIdata['UNEMPL'].shift(3))/3
        UKHPIdata['INTEREST3']=(UKHPIdata['INTEREST'].shift(1)+UKHPIdata['INTEREST'].shift(2)+UKHPIdata['INTEREST'].shift(3))/3
        UKHPIdata['INDPRO3']=(UKHPIdata['INDPRO'].shift(1)+UKHPIdata['INDPRO'].shift(2)+UKHPIdata['INDPRO'].shift(3))/3
        UKHPIdata['CPINF3']=(UKHPIdata['CPINF'].shift(1)+UKHPIdata['CPINF'].shift(2)+UKHPIdata['CPINF'].shift(3))/3
        UKHPIdata['FXCHG3']=(UKHPIdata['FXCHG'].shift(1)+UKHPIdata['FXCHG'].shift(2)+UKHPIdata['FXCHG'].shift(3))/3
        UKHPIdata['GEPU3']=(UKHPIdata['GEPU'].shift(1)+UKHPIdata['GEPU'].shift(2)+UKHPIdata['GEPU'].shift(3))/3

        UKHPIdata=UKHPIdata.dropna()
        
        X=UKHPIdata[['UNEMPL3','INTEREST3', 'INDPRO3', 'CPINF3', 'FXCHG3', 'GEPU3']]
        print()
        print(ExamPrep.vif(X))

        #Histogram
        UKHPIdata.set_index(UKHPIdata['month'], inplace=True) #use month variable as index for X axis
        plt.plot(UKHPIdata['HPINF'], label='House Price Inflation')
        plt.xlabel('time')
        plt.ylabel('Year-on-Year montly UK House Price Changes')
        plt.grid()
        plt.legend()
        plt.show
        for i in range(3): print()

        #3.	OLS estimation of a dynamic time-series model and residual tests
        model='HPINF~UNEMPL3+INTEREST3+INDPRO3+CPINF3+FXCHG3+GEPU3'
        regression1=smf.ols(model, UKHPIdata).fit()
        print(regression1.summary())
        print(regression1)
        print()
        print("from the summary we can see that Durbin-Watson does suggest a positiv 1st order autocorrelation", end='\n')
        
        [print("-", end='-') for i in range(45)]
        print()
        #3.1	 Testing for residual autocorrelation --> 
        # Next we run two formal tests: the Breusch-Godfrey LM test and Ljung-Box Q test.
        #These two are designed differently to test the same hypotheses: H0: No autocorr up to order p
        test=sms.acorr_breusch_godfrey(regression1, 12)
        print("LM test", test[0])
        print("p-value", test[1])
        print()
        print("F test", test[2])
        print("p-value", test[3], end='\n \n')
        print("The p-values are essentially 0 and we strongly reject Ho.")
        print("There is significant autocorrelation in the residuals of some order from 1 to 12 for at least one", end='\n \n')
        [print("-", end='-') for i in range(45)]
        print()
        #Ljung box test
        print("Next we implement the Ljung-Box test", end='\n \n')
        lb_test = sms.acorr_ljungbox(regression1.resid, lags=[12]) 
        print(lb_test)
        print("P-value is essentially 0")
        [print("-", end='-') for i in range(45)]
        print()


ExamPrep.autoCorr()
ExamPrep.multicoll() 



