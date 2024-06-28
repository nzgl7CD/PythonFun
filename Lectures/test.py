def fvOrdinaryAnnuity(pmt:int, r, n:int):
    return pmt*((1+r)**n-1)/r
def pvOrdinaryAnnuity(pmt:int, r, n:int):
    return (pmt/r)*(1-(1/(1+r)**n))

def pvOrdinaryAnnuityWGrowth(pmt:int, r, n:int,g):
    return True

def max():
    cf0=250
    cfn=200
    r=1.1
    sum=-250
    
    for i in range(1,14):
        if i <4:
            sum=sum-cf0/r**i
            print(sum)
        else:
            sum=sum+cfn/r**i
            print(sum)
    return sum 

def score():
    score=100
    wrong=0
    while score>77:
        score-=6
        wrong+=1
    return wrong


def pvsom():
    c=250
    r=0.1
    cf=200

    sum=0

    for i in range(0,4):
        if i ==0:
            sum-=c
        else:
            sum-=250/(1+r)**i
    for i in range(4,14):
        sum+=200/(1+r)**i
    return sum

def pvno():
    sum=-250
    sum+=(-250/0.1)*(1-(1/1.1**3))
    sum+=200/0.1*(1-(1/1.1**13))
    sum-=200/0.1*(1-(1/1.1**3))
    return sum

def npv():
    yr=10
    npv=10
    marketshare=0.08
    population=42.4
    wacc=0.12
    sum=-90-80/1.12**2
    for i in range(3,yr+1):
        population=population*1.03
        sum+=population*marketshare*npv/(1+wacc)**i
        
    return sum
print(npv())


def ArbitrageOpportunity():
    K= 100 #Strike price on put and call
    S=105 #share price
    rf=0.02 #risk free of 2%
    C= 8.92 #call premium 
    P=3.25 #put premium
    maturity=165 #days to maturity
    shortPortfolio=[]
    longPortfolio=[]
    shortPortfolio.append(S, P) if S+P>K+C else shortPortfolio.append(S, C)
    if len(shortPortfolio) !=0:
        if P in shortPortfolio:
            longPortfolio.append(C)
            if S in shortPortfolio:
                longPortfolio.append((P+S)*(1+rf*maturity/365))

    return shortPortfolio,longPortfolio
