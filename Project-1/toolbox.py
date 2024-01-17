#* toolbox.py
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

import pandas as pd

class DataSet:
    """
    Represent a dataset for further analysis
    """
    def __init__(self,filename,datatype):
        """Initialize the file name with the name provided"""
        self.filename = filename
        self.datatype = datatype

    def __readFromCSV(self,filename):
        """Read data from a csv file"""
        self = pd.read_csv(filename)
        #print('successfully read the file from ',filename)

    def __load(self,filename):
        """Prompt the user to enter file name and file type to load data"""
        filename = input("Please enter the file name.")
        datatype = input("Please enter the datatype for the given file.")
        #print('successfully read the file from ', filename)

    def clean(self):
        """Clean the loaded data"""
        print('data cleaning finished')

    def explore(self):
        """Explore cleaned data"""
        print('data exploring finished')

class TimeSeriesDataSet(DataSet):
    """
    Represent a time series dataset, inherit from DataSet
    """
    def __init__(self,filename,datatype):
        super().__init__(filename)
        super().__init__(datatype)



class TextDataSet(DataSet):
    """
    Represent a text dataset, inherit from DataSet
    """
    def __init__(self,filename):
        super().__init__(filename)

class QuantDataSet(DataSet):
    """
    Represent a quantitive (numerical) dataset, inherit from DataSet
    """
    def __init__(self,filename,datatype):
        super().__init__(filename)
        super().__init__(datatype)

    def clean(self):
        columns = self.columns.tolist()
        try:
            for col in columns:
                mean = self[col].mean()
                self[col].fillna(value = mean, inplace = True)
        except ValueError:
            del columns[0]
            for col in columns:
                mean = self[col].mean()
                self[col].fillna(value = mean, inplace = True)

class QualDataSet(DataSet):
    """
    Represent a qualitative (descriptive) data set, inherit from DataSet
    """
    def __init__(self,filename,datatype):
        super().__init__(filename)
        super().__init__(datatype)

    def clean(self):
        self.fillna(value = self.mode(),inplace = True)



class ClassifierAlgorithm:
    """
    Represent the classifier for given data.
    """

    def __init__(self):
        pass

    def train(self):
        """Train the data on given dataset"""
        print('data training finished')

    def test(self):
        """Test the data on given dataset"""
        print('data testing finished')

class simpleKNNClassifier(ClassifierAlgorithm):
    """
    Represent the algorithm for KNN classifier, inherited from ClassifierAlgorithm.
    """
    def __init__(self):
        super().__init__()

class kdTreeKNNClassifier(ClassifierAlgorithm):
    """
    Represent the algorithm for KDTree classifier.
    """
    def __init__(self):
        super().__init__()

class Experiment:
    """
    Represent the model evaluation part.
    """
    def runCrossVal(self,k):
        """Run cross validation for a given k"""
        print('run cross validation for k=',k,' finished')
    
    def score(self):
        """Return the score for models """
        print('The score is x')

    def __confusionMatrix(self):
        """Return the confusion matrix for classifiers"""
        print('get confusion matrix')


