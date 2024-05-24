import pandas as pd						

def Solution():

    DUMMdata=pd.read_excel('dummies2000_2023.xlsx').dropna()
    UKHPIdata=pd.read_excel('UKHOUSE.xlsx').dropna()
    UKHPIdata['Brexit']=DUMMdata['Brexit']
    print(UKHPIdata)
    with pd.option_context('display.max_rows', None):print(UKHPIdata)
    # with pd.option_context('display.max_rows', None):print(DUMMdata)
Solution()