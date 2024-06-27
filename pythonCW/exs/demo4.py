## A simple program: prints a set of stock prices
## and call option payout on an Excel file
s = 0
k = 70
somefile = open("calloption.xls", "r+")
somefile.write("\n")
somefile.write("Stock Price\tCall Option Payout\n")
for i in range(11):
     payout = max(s-k, 0)
     somefile.write(str(s) + "\t" + str(payout) +  "\n") 
     s += 10
somefile.close()
