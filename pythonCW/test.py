import numpy as np
import os

try:
    import pandas as pd
except ImportError as e:
     e = str(e)[15:]
     e = e.strip().replace("'", "")
     os.system('py -m pip install %s' % (e))
     
class X:
        
    def test():
        print("testing this shit")
        x=[1,2,3,4]
        print(np.max(x))
X.test()

