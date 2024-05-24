from examRev import ExamPrep
from pyfiglet import Figlet
f = Figlet(font="slant", width=1000)
print(f.renderText("Exam Revision - Data Analysis"))
print()
#If you would like to check basic outputs
# ExamPrep.sumData()
#If you want to test for multicollinearitys
ExamPrep.multicoll()