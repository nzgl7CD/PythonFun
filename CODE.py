import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# csv_file_path_A = 'C:/Users/casig/OneDrive/Desktop/MarketMicrostructure/SSignals.csv'
# csv_file_path_R = 'C:/Users/casig/OneDrive/Desktop/MarketMicrostructure/Returns.csv'
# signals= pd.read_csv(csv_file_path_A).dropna()
# Ret= pd.read_csv(csv_file_path_R).dropna()


x= 0.01
y=0.01

# print(Ret)
crra = 4

IC = 0.02 + x
stdev = 0.08 + y
# alpha = IC * stdev * signals

# sigma = Ret.dropna().cov()
# print(sigma)
# inv_sigma = np.linalg.inv(sigma)
# print(inv_sigma)
myOnes = np.ones(34)
i=0
weights = []
for i in range(len(alpha)):
    A = alpha.iloc[i]
    theta = (A.T @ inv_sigma @ myOnes - crra) / (myOnes.T @ inv_sigma @ myOnes)
    w = inv_sigma @ (A - myOnes * theta) / crra
    weights.append(w)
    i += 1

weights_array = np.array(weights)
Results_weights = pd.DataFrame(weights_array)

T_Results_df = Results_weights.transpose()
# T_Results_df.to_excel('C:/Users/casig/OneDrive/Desktop/MarketMicrostructure/Results.xlsx')
print(Results_weights.T)
Ret=Ret.iloc[59:].reset_index(drop=True)
if Ret.shape[1] != Results_weights.shape[0]:
    Results_weights = Results_weights.T
Portofolio_Ret = Ret.dot(Results_weights)
print(Portofolio_Ret)


"""""
#del weights
plt.plot(weights)
plt.title("Simple weights")
plt.show()
plt.close()

print("Sum of weights is " + str(sum(weights)))
#print("Sum of absolute weights is " + str(sum(abs(weights))))
"""