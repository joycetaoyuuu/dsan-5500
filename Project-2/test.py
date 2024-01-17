#* test.py
#*
#*  ANLY 555 Fall 2023
#*  Project Data Science Python Toolbox
#*
#*  Due on: 10/05/2023
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

if __name__ == "__main__":
    # test for quantdata
    data =QuantDataSet()
    data.clean()
    data.explore()
    print("\nTest Completed!\n")

    # test for qualdata
    data = QualDataSet()
    data.clean()
    data.explore()
    print("\nTest Completed!\n")

    # tst for ts data
    data = TimeSeriesDataSet()
    data.clean()
    data.explore()
    print("\nTest Completed!\n")

    #test for text data
    data = TextDataSet()
    data.clean()
    data.explore()
    print("\nTest Completed!\n")







