from __future__ import print_function

## Demonstrate justification of output.
print("0123456789012345678901234567")
print("{0:^5s}{1:<20s}{2:>3s}".format("No#", "Trader", "CAP"))
print("{0:^5n}{1:<12s}{2:>3s}".format(1, "Ryan Bonds", "76M"))
print("{0:^5n}{1:<20s}{2:>3s}".format(2, "Barry Safe", "40M"))
print("{0:^5n}{1:<20s}{2:>3s}".format(3, "Vera Silver", "80M"))
