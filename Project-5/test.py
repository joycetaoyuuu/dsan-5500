#* test.py
#*
#*  ANLY 555 Fall 2023
#*  Project Data Science Python Toolbox (Deliveriable 4)
#*
#*  Due on: 12/5/2023
#*  Author: Yu Tao
#*
#*
#*  In accordance with the class policies and Georgetown's
#*  Honor Code, I certify that, with the exception of the
#*  class resources and those items noted below, I have neither
#*  given nor received any assistance on this project other than
#*  the TAs, professor, textbook and teammates.
#*
#*  References not otherwise commented within the program source code.
#*  Note that you should not mention any help from the TAs, the professor,
#*  or any code taken from the class textbooks.
#*



from toolbox import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    
    # testing on quantdata
    df = QuantDataSet()
    df = df.clean()

    x = df.data.iloc[:,:8]
    y = df.data['Outcome']

    # reduce data size for faster computing
    x = x.iloc[:300,:]
    y = y.iloc[:300]

    e = Experiment(x,y)
    e.runCrossVal(2)
    score = e.score()
    print(score)
    e.confusion_matrix()
    










