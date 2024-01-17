#* test.py
#*
#*  ANLY 555 Fall 2023
#*  Project Data Science Python Toolbox
#*
#*  Due on: 9/18/2023
#*  Author(s): Yu Tao
#*
#*
#*  In accordance with the class policies and Georgetown's
#*  Honor Code, I certify that, with the exception of the
#*  class resources and those items noted below, I have neither
#*  given nor received any assistance on this project other than
#*  the TAs, professor, textbook and teammates.
#*



from toolbox import *

if __name__ == "__main__":
    print('Test for DataSet class\n')
    data = DataSet('data.csv')
    print('file name is ',data.filename)
    data.clean()
    data.explore()
    print('\n')

    print('Test for time series data\n')
    ts_data = TimeSeriesDataSet('ts_data.csv')
    print('file name is ',ts_data.filename)
    ts_data.clean()
    ts_data.explore()
    print('\n')

    print('Test for text data\n')
    text_data = TextDataSet('text_data.csv')
    print('file name is ',ts_data.filename)
    ts_data.clean()
    ts_data.explore()
    print('\n')

    print('Test for quantitive data\n')
    quant_data = TextDataSet('quant_data.csv')
    print('file name is ',quant_data.filename)
    quant_data.clean()
    quant_data.explore()
    print('\n')

    print('Test for qualitative data\n')
    qual_data = TextDataSet('qual_data.csv')
    print('file name is ',qual_data.filename)
    qual_data.clean()
    qual_data.explore()
    print('\n')

    print('Test for classifier')
    classifier = ClassifierAlgorithm()
    classifier.train()
    classifier.test()
    print('\n')

    print('Test for KNN Classifier')
    knn_classifier = simpleKNNClassifier()
    knn_classifier.train()
    knn_classifier.test()
    print('\n')

    print('Test for KDTreeClassifier')
    kdtree_classifier = kdTreeKNNClassifier()
    kdtree_classifier.train()
    kdtree_classifier.test()
    print('\n')

    print('Test for Experiment class')
    exp = Experiment()
    exp.runCrossVal(5)
    exp.score()




