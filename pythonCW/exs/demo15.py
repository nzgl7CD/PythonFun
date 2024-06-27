from __future__ import print_function

## Demonstrate use of the format method
print("The {0:s} weighted average cost of capital, as a decimal, is {1:,.2f}".format(" Apple's", 0.23))


str1 = "The  {0:s} weighted average cost of capital, as a percentage, is {1:.2%}"
print(str1.format("Apple's", 0.23))
