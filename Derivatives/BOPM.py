import math 
def pricing1period(s,su,sd, r):
    U=su/s
    D=sd/s
    k=s
    cd=0 if sd<k else (sd,k)
    cu=max(0,su-k)
    delta=(cu-cd)/(su-sd) #delta on long call
    R=1+r if r<1 else 1+r/100
    q=(R-D)/(U-D) #risk-neutral probability”
    fairC=(1/R)*(q*cu+(1-q)*cd)
    
    return fairC, delta



#spot, volatility, periods, interest, years, call premium
def pricingNperiods(s,sigma,n,r,T,C):
    U=math.exp(sigma*math.sqrt(T/n))
    D=math.exp(-sigma*math.sqrt(T/n))
    R=math.exp(r*T/n)
    q=(R-D)/(U-D)

    cud=C*U*D if s*U*D>=s else 0
    cdd=C*D**2 if s*D**2>=s else 0
    
    cu=1/R*(q*C*U**2+(1-q)*C*U*D)
    cd=1/R*(q*cud+(1-q)*cdd)
    FVC=1/R*(q*cu+(1-q)*cd)
    return FVC
pricingNperiods(100,0.2,12,0.05,1,10.75)