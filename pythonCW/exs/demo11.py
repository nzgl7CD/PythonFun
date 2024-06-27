from __future__ import input_function

## Break a name into two parts -- the last name and the first name
fullName = input("Enter your full name: ")
n = fullName.rfind(" ")     # index of the space preceding the last name
# Display the desired information
print("Last name:", fullName[n+1:])
print("First name(s):", fullName[:n])
