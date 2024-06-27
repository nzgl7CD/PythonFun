##Draft solution by Dr. Adrian Euler
#-------------------------------------------------
from math import *
def e(x):
    r = 0.0
    tol = 0.01
    n = int(1.0/tol)
    for i in range(n):
        r += (pow(x, i)/factorial(i))
    return r
#---------------------------------------------------

def u(r, q, sigma, T, nsteps):
    dt = T/float(nsteps)
    x = (r-q-0.5*sigma**2)*dt+sigma*(dt**0.5)
    r = e(x)
    return r
#---------------------------------------------------
def payoff(r, q, s, x, sigma, T, nsteps):
    po = []
    uu = u(r, q, sigma, T, nsteps)
    dd = 1/uu
    for i in range(nsteps, -1, -1):
        po.append(max(s*uu**(nsteps-i) * dd** (i) -x, 0))
    return po
#---------------------------------------------------
def p(r, q, sigma, T, nsteps):
    up = u(r, q, sigma, T, nsteps)
    down = 1/up
    dt = T/nsteps
    x = (r-q)*dt
    prob = (e(x)-down)/(up-down)
    return prob
#-------------------------------------------------
def combin(n, k):
    nf = factorial(n)
    kf = factorial(k)
    n_k = factorial(n-k)
    c = nf/(kf*n_k)
    return c
#------------------------------------------------
def pdf(r, q, sigma, T, nsteps):
    prob = p(r, q, sigma, T, nsteps)
    pstar = 1 - prob
    k = []
    for i in range(nsteps, -1, -1):
        k.append(i)
    r = []
    temp = 0.0
    for  j in range(nsteps, -1, -1):
        temp = combin(nsteps, k[j])*(pow(prob, k[j])*pow(pstar, (nsteps-k[j])))
        r.append(temp)
    return r
#-------------------------------------------------------------------------------------
def optval(r, T, payoff = [], prob = []):
    tot = 0.0
    n = len(prob)
    for i in range(n):
        tot += (payoff[i]*prob[i])
        #print(payoff[i], "\t", prob[i])
    return tot*e(-r*T)
#------------------------------------------------------------------------------------
#The part below is provided by Dr. Adrian Euler
#stdents should use to test their user-defined functions.

from datetime import datetime
import time as clock

def main():
    start = clock.time()
    message = \
    '''
      \t-----------------------------------------------------------------------------------
      \tIndivual test for SMM283 - Introduction to Python
      \t----------------------------------------------------------
      \tStudent should test the user-defined functions using \n\ta  main function you must create.
      \tMake sure you enter your student name and student id. \n\tDo not enter the name/student id
      \tof anyone else...
      \tYour code must run.
      \tCode will be inspected further...
      \t-----------------------------------------------------------------------------------
    '''
    print(message)
    s = 100.0      #share price
    x = 95.0       #exercise price
    r = 0.08       #short rate
    q = 0.03       #dividend yield
    T = 0.5        #option expiration
    sigma = 0.20   #volatility
    nsteps = 9          #number of steps in the bionomial tree
    iopt = 1       #i-option (+1 for call, -1 for put)
    while True:
        try:
            name = input("\tEnter your full name: ")
            yn = input("\t\tDid you enter your name correctly (y or n): ")
            if yn.lower()=='y':
                break
            else:
                continue
            break
        except:
            continue
    while True:
        try:
            stdId = input("\tEnter your (student) ID: ")
            yn = input("\t\tDid you enter your student ID correctly (y or n): ")
            if yn.lower()=='y':
                break
            else:
                continue
            break
        except:
            continue
    payvals = payoff(r, q, s, x, sigma, T, nsteps)
    pvals = []
    result1 = factorial(nsteps)
    print("\n\t------------------------------------------------------------")
    print("\t", name, "/", stdId)
    print("\t--------------------------------------------------------------------------")
    print("\tNumber of steps is", nsteps, "; its factorial is ", result1)
    result2 = round(e(nsteps), 3)
    print("\t--------------------------------------------------------------------------")
    print("\texp(nsteps, 3) = exp(9, 3) = ", result2)
    result3 = round(u(r, q, sigma, T, nsteps), 4)
    print("\t--------------------------------------------------------------------------")
    print("\tu(r, q, sigma, T, nsteps) = u(0.08, 0.03, 0.20, 0.5, 9) = ", result3)  
    result4 = round(p(r, q, sigma, T, nsteps), 4)
    print("\t--------------------------------------------------------------------------")
    print("\tp(r, q, sigma, T, nsteps) = p(0.08, 0.030.20, 0.5, 9) = ", result4)  
    result5 = combin(10, 4)
    print("\t--------------------------------------------------------------------------")
    print("\tcombin(10, 4) = ", result5) 
    i = 9
    result6 = pdf(r, q, sigma, T, nsteps)
    pvals = result6
    print("\t--------------------------------------------------------------------------")
    print("\tpdf(r, q, sigma, T, nsteps) = u(0.08, 0.03, 0.20, 0.5, 9) = \n\t")
    for i in range(nsteps, -1, -1):
        if i % 1 == 0:
            print("\t")
        print("\t", pvals[i], "\t")
    result7 = round(optval(r, T, payvals, pvals), 2)
    print("\n\t------------------------------------------------------------------------")
    print("\tFor the data given, the binomial call option price is: £{0} ".format(result7))
    print("\t--------------------------------------------------------------------------")
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")
    print("\n\tTested by {0} with ID {1} on date and time {2} ".format(name, stdId, dt))
    end = clock.time()
    print("\n\tIt took {0}seconds to run the program!".format(end-start))
    input("\n\n\tPress ENTER Key to Exit!")
main()
