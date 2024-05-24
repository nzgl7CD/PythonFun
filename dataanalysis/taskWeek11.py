import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt   #graphs library
import seaborn as sns  #more graphs
import linearmodels as panel   #panel estimation library

    
GDPdata=pd.read_excel('dataanalysis\GDPpanel.xlsx') 
GDPdata=GDPdata.set_index(['COUNTRY', 'YEAR'])  #to indicate the panel structure (entity, time)
FEregression= panel.PanelOLS.from_formula('DGDP~DCAPITAL + EntityEffects +TimeEffects', GDPdata).fit()

effects=FEregression.estimated_effects.unstack(level=0)
print(effects)



dict=effects.to_dict(orient="list")
diffEffect={} #Difference between belgium and UK collected in new hashmap
year=1992
for i in range(len(dict['estimated_effects', 'Belgium'])):
    print(dict['estimated_effects', 'UK'][i]-dict['estimated_effects', 'Belgium'][i])
    diffEffect[str(year)]=dict['estimated_effects', 'Belgium'][i]-dict['estimated_effects', 'UK'][i]
    year=year+1
print(diffEffect)
