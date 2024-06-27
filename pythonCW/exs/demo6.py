betaOfStockA = 1.13
betaOfMarket = betaOfStockA // 1.0
relativeDiff = betaOfStockA % betaOfMarket
print(betaOfMarket, round(relativeDiff, 2))
