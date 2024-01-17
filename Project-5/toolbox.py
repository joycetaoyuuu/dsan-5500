#* toolbox.py
#*
#*  ANLY 555 Fall 2023
#*  Project Data Science Python Toolbox (Deliveriable 5)
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud
import heapq

try: 
    class DataSet:
        """
        Represent a dataset.
        """
        def __init__(self,filename = None):
            """Initialize the file name with the name provided"""
            # call the load method to get file name and date type
            self.filename, self.datatype = self.__load(filename)
            # call the read method to read data into self.data
            self.data = self.__readFromCSV(self.filename)

        def __readFromCSV(self,filename):
            """Read data from a csv file, filename is provided"""
            # set the first column as index
            data = pd.read_csv(filename,index_col=0,low_memory=False)
            return data

        def __load(self,filename):
            """Prompt the user to enter file name and file type to load data"""
            print("Please enter the name of the file.")
            filename = input()
            print("Please enter the datatype for the given file.")
            datatype = input()

            # add extension if file type is not included in the input
            if '.csv' not in filename:
                filename = filename+'.csv'

            return filename,datatype

        def clean(self):
            """Clean the loaded data"""

        def explore(self):
            """Visualization for cleaned data"""

        def __len__(self):
            return len(self.data)


    class TimeSeriesDataSet(DataSet):
        """
        Represent a time series dataset, inherit from DataSet
        """
        def __init__(self,filename = None):
            super().__init__(filename)
            if ((self.datatype != "timeseries") and (self.datatype != "ts")):
                raise ValueError("Invalid datatype, expected 'timeseries'.")
            
        def clean(self):
            """
            Override the clean method for time series data.
            Replaced the time stap from date-time to date only.
            Median filter with window size 15 is applied for smoothing.
            """
            # keep date only, remove times
            self.data.index = pd.to_datetime(self.data.index)
            self.data.index = self.data.index.date

            filter_size = 15
            colnames = self.data.columns.tolist()

            # median filter with window size = 15
            for col in colnames:
                records = self.data[col]
                median_filter = [0]*(len(records)-(filter_size-1))

                for i in range(len(median_filter)):
                    start = i
                    end = i + filter_size
                    window = records[start:end]
                    median = sorted(window)[filter_size//2]
                    median_filter[i] = median

                nas = [None]*(filter_size//2)

                new_col_name = col+'_with_median_filter'
                self.data[new_col_name] = nas+median_filter+nas

            return self

        def explore(self):
            """
            Override the explore method for time series data.
            Provides two time series plots.
            One for original data and the cleaned data (after median filter).
            Another one for a smaller window.
            """

            # plot the data for the whole time period
            origin_data = self.data['Close']
            new_data = self.data['Close_with_median_filter']

            plt.plot(self.data.index,origin_data,label = "Original Data")
            plt.plot(self.data.index,new_data,label = "Filtered Data")
            plt.legend()
            plt.title('Time Series Plot')
            plt.show()

            # plot the time series for a random year
            dates = pd.to_datetime(self.data.index)
            years = dates.year.tolist()
            self.data['year'] = years
            unique_years = list(set(years))
            selected_year = np.random.choice(unique_years,1)[0]

            selected_df = self.data[self.data['year']==selected_year]

            plt.plot(selected_df.index,selected_df['Close'],label ='Original Data')
            plt.plot(selected_df.index,selected_df['Close_with_median_filter'],label='Filtered Data')
            plt.legend()
            plt.title(f'Time Series Plot for {selected_year}')
            plt.show()


    class TextDataSet(DataSet):
        """
        Represent a text dataset, inherit from DataSet.
        """
        def __init__(self,filename = None):
            super().__init__(filename)
            if ((self.datatype !="text") and (self.datatype !='textdata')):
                raise ValueError("Invalid datatype, 'text' expected.")
            
        def clean(self):
            """
            Override the clean method specific for text data.
            Text data cleaning was applied through this method.
            Including lower case, contraction replacement, punctuation removing, stopwords removing and word lemmatization.
            Returns a new column in data with cleaned words in list.
            """

            stop_words = set(nltk.corpus.stopwords.words('english'))

            Apos_dict={"'s":" is","'t":" not","'m":" am","'ll":" will",
            "'d":" would","'ve":" have","'re":" are"}
            
            lemmatizer = nltk.stem.WordNetLemmatizer()

            text = self.data['text'] # get text data in the dataframe
            clean_text = []

            for t in text:
                t = t.lower() # lower case

                for key,value in Apos_dict.items(): # contraction replacement
                    if key in t:
                        t = t.replace(key,value)

                # remove punctuation
                t = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), t))
                words = t.split() # split string to list of words
                words = [word for word in words if word not in stop_words] # remove stopwords
                words = [lemmatizer.lemmatize(word) for word in words] #lemmatize words

                clean_text.append(words)

            self.data['clean_text'] = clean_text
            return self

        def explore(self):
            """
            Override the explore method specific for text data.
            Returns a wordcolud for visualization.
            Returns a box plot for data distributed in different category.
            """
            try:
                # combine all the text together for exploration
                words_str = ' '.join([(" ").join(word_list) for word_list in self.data['clean_text']])
                wordcloud = WordCloud(width = 1000, height = 800,background_color ='white',colormap ="YlOrRd").generate(words_str)

                plt.figure(figsize = (10,8))
                plt.imshow(wordcloud)
                plt.axis("off")
                plt.title("Word Could Plot")
                plt.show()

                # text length analysis
                text_length = [len(word_list) for word_list in self.data['clean_text']]
                rating = self.data['stars'].astype("category").tolist()
                df = pd.DataFrame({'text_length':text_length,'rating':rating})

                df.boxplot(by ='rating')
                plt.title("Boxplot for text length in different categories")
                plt.xlabel('Ratings')
                plt.ylabel('Text Length')
                plt.show()

            except:
                print('Please clean the dataset first before exploration!')

            
    class QuantDataSet(DataSet):
        """
        Represent a quantitive (numerical) dataset, inherit from DataSet
        """
        def __init__(self,filename = None):
            super().__init__(filename)
            if ((self.datatype != "quantdata") and (self.datatype != "quant")):
                raise ValueError("Invalid datatype")

        def clean(self):
            """
            Override the clean method specific for quantitive data.
            Filled missing values with mean.
            """

            # fill missing values with mean
            columns = self.data.columns.tolist()

            for col in columns:
                mean = self.data[col].mean()
                self.data[col].fillna(value = mean, inplace = True)
            
            return self

        def explore(self):
            """
            Override the explore method specific for quantitive data.
            Returns one 2-D scatrer plot bwtween two variables.
            Returns one 3-D plot between three normalized variables.
            """

            columns = self.data.columns.tolist()

            # randomly select two variables for scatter plot
            var1,var2 = np.random.choice(columns,2,replace=False)

            plt.scatter(self.data[var1],self.data[var2])
            plt.title(f"Scatter Plot for {var1} and {var2}")
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.show()

            # randomly select 3 variables for 3-d plot
            x,y,z = np.random.choice(columns,3,replace=False)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.scatter(self.data[x], self.data[y], self.data[z])  
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)
            ax.set_title("3D Plot")
            plt.show()


    class QualDataSet(DataSet):
        """
        Represent a qualitative (descriptive) data set, inherit from DataSet
        """
        def __init__(self,filename = None):
            super().__init__(filename)
            if ((self.datatype != "qualdata") and (self.datatype != "qual")):
                raise ValueError("Invalid datatype")

        def clean(self):
            """
            Override the clean method for qualitative datasets.
            Filled the missing values with mode.
            """
            # set the first column as colnames
            self.data.columns = self.data.iloc[0]
            self.data = self.data.iloc[1:,:]
            columns = self.data.columns.tolist()
            
            # fill missing values with mode
            for col in columns:
                mode = self.data[col].mode()
                self.data[col].fillna(value = mode, inplace = True)

            return self
        
        def explore(self):
            """
            Override the explore method for qualitative datasets.
            Returns a histogram for the first column.
            Returns a bar plot for combinations inside different columns.
            """

            # histogram plot for the first variable
            col_name = self.data.columns.tolist()[0]
            values = self.data[col_name]
            plt.figure(figsize=(6,8))
            plt.hist(values,rwidth = 0.5,color = 'lightblue')
            plt.title(col_name)
            plt.xticks(fontsize=8)
            plt.show()

            # bar plot for top user info
            idx = [0,1,3,4]
            df = self.data.iloc[:,idx]
            colnames = [self.data.columns.tolist()[i] for i in idx]

            count_top_5 = df.groupby(colnames).size().reset_index(name='count').nlargest(5,'count')
            count_top_5.columns = ['Age','Gender','Country','Education','count']
            count_top_5 = count_top_5.sort_values(by = 'count',ascending = True)
            plt.figure(figsize=(12,6))
            plt.barh(count_top_5['Age']+","+count_top_5['Gender']+","+count_top_5['Country']+","+count_top_5['Education'],count_top_5['count'],color = 'cornflowerblue',height = 0.5)
            plt.title("Top 5 combinations who attended the servey")
            plt.tight_layout()
            plt.show()

    class TransactionDataSet(DataSet):
        """
        Represent a transaction data set, inherit from DataSet
        """
        def __init__(self,filename = None):
            """
            Inherit the initialization method from Dataset class, do both read data and load data
            """
            super().__init__(filename)
            if ((self.datatype != "transaction") and (self.datatype != "trans")):
                raise ValueError("Invalid datatype, expected 'transaction'.")

        def clean(self):
            """
            Clean the dataset, drop NA rows, and modify the transaction data from string to list of items
            """
            self.data = self.data.dropna() # drop nas
            self.data = self.data.iloc[:,0].str.split(",") # split the transaction data into lists
            return self

        def explore(self):
            """
            Explore the transaction dataset, call the __ARM__() method to implement Association Rule Mining
            """
            # reduce threshold value for returning more outputs on testing data
            return self.__ARM__(0.2)

        def __ARM__(self,supportThreshold):
            """
            Association Rule Mining using Apriori algorithm. Generates a dataframe with 10 rules that meets the requirement of support threshold.
            supportThreshold: a decimal.
            """

            self._transactions_matrix = self.data.to_list() # transfer series data to 2-d matrix
            self._transactions_list = sum(self._transactions_matrix,[]) # flat the 2-d matrix to 1-d array
            item_set = set(self._transactions_list) # get unique items in the dataset

            rule_table = Rule().generate_rules(list(item_set)) # call the rule class to generate all the possible rules

            for i in range(len(rule_table)): # go through the table by index number
                x = rule_table['x'][i] # get antecedent of the rule
                y = rule_table['y'][i] # get consequent of the rule

                # check the support rate for a single item
                if ((self.get_support(x)<supportThreshold) or (self.get_support(y))<supportThreshold):
                    # remove the record if it does not meet the threshold
                    rule_table.drop(index = i, inplace=True)

                else:
                    # check the support rate of a combination of two items
                    support = self.get_support(x,y)

                    # remove the combination if it does meet the threshold
                    if support < supportThreshold:
                        rule_table.drop(index = i,inplace=True)

                    # case that the requirements meet
                    else: 
                        # calculate lift and confidence 
                        lift = self.get_lift(x,y)
                        confidence = self.get_confidence(x,y)

                        # save values into dataframe
                        rule_table['support'][i] = round(support,2)
                        rule_table['confidence'][i] = round(confidence,2)
                        rule_table['lift'][i] = round(lift,2)

            # save the top 10 rules base on confidence value
            rule_table = rule_table.sort_values('confidence',ascending=False).head(10).reset_index(drop=True)
            return rule_table
        
        def get_freq(self,x,y=None):
            """
            A method takes item x and item y as input and returns frequency for one element (x) or two elements (x and y).
            """
            items = [x] if y is None else [x,y] # convert string to list
            count = 0
            for transaction in self._transactions_matrix: # go through every transaction records
                if all(item in transaction for item in items): # check if items is a subset of transaction
                    count += 1
            return count
        
        def get_support(self,x,y = None):
            """
            A method takes item x and item y as input and returns support value (in decimals).
            """
            support = self.get_freq(x,y)/len(self._transactions_matrix)
            return support
        
        def get_confidence(self,x,y):
            """
            A method takes item x and item y as input and returns the confidence value of the rule between x and y
            """
            conf = self.get_freq(x,y)/self.get_freq(x)
            return conf

        def get_lift(self,x,y):
            """
            A method takes item x and item y as input and returns the lift value of the rule between x and y.
            """
            lift = self.get_support(x,y)/(self.get_support(x)*self.get_support(y))
            return lift
         

except Exception as e:
    print("There is an error with: ",e)

#----------------------------------- end of DataSet Class -----------------------------------

class Rule:
    """
    Represent rules.
    """
    def __init__(self):
        pass

    def generate_rules(self,data):
        """
        Generate rules for transaction data, takes list of unique items as input, and returns a dataframe with all possible combination of rules under level 2.
        """
        df = pd.DataFrame(columns=['x', 'y','support','confidence','lift'])
        for antecedent in data:
            for consequent in data:
                if antecedent != consequent: # skip the duplicated elements in the combination for rules.
                    df.loc[len(df.index)] = [antecedent,consequent,None,None,None] 
        return df

#----------------------------------- end of Rule Class -----------------------------------

class ClassifierAlgorithm:
    """
    Run a classifier based on given data.
    """

    def __init__(self):
        pass

    def train(self):
        """Train the data on given dataset"""
        pass

    def test(self):
        """Test the data on given dataset"""
        pass


class simpleKNNClassifier(ClassifierAlgorithm):
    """
    Represent a KNN classifier, inherited from ClassifierAlgorithm.
    """
    def __init__(self):
        super().__init__()

    def train(self,x_train,y_train):
        """
        Training method under simpleKNNClassifier class.
        Saves training data with variables(x) and labels(y)
        x_train: Training data, in the format of dataframe
        y_train: Training labels, in the format of pandas series.
        """
        self._x_train = x_train
        self._y_train = y_train.to_list()

    def test(self,x_test,k):
        """
        Implement knn classifier and return predicted labels.
        Calculating distances between each test record and all training records.
        Return the test label based on the mode of k nearest records (in training data).

        x_test: Testing data, in the format of dataframe
        k: k nearest neighbour, an integer
        """
        self._x_test = x_test
        self._k = k
        y_train = self._y_train
        x_train = self._x_train
        y_pred = []

        for i in range(len(x_test)): # go through every row
            t = pd.DataFrame(columns = ['labels','dist']) # generate a df with labels and distance variables
            x_test_data = x_test.iloc[i,:].to_list() # get the testing data (in the format of list)

            for j in range(len(x_train)): # go through each training records
                label = y_train[j]        # get the label of corresponsing training data
                x_train_data = x_train.iloc[j,:].to_list() # get the training data (in the format of list)
                dist = self.get_distance(x_test_data,x_train_data) # calculate the distance between two observations
                t.loc[j] = [label,dist] # save label and distance in the dataframe

            t = t.sort_values(by =['dist']) # sort the dataframe based on distance
            t = t.loc[:k-1] # get the top k rows
            labels = t['labels'] # get the labels of top k rows
            mode_label = labels.mode().iloc[0] # get the mode label of cloest records, as the y_pred
            y_pred.append(mode_label) # save the value into y_pred variable
            # end of one x_test, go to the next test record (if applicable)


        return y_pred # end of all testing values, return the prediction for all testing records


    def get_distance(self,vector1,vector2): # calculate the distance between two observations
        d = 0
        for i in range(len(vector1)): # go through each coordinate
            d += (vector1[i]-vector2[i])**2 # calculate the distance in the same coordinate, and add together
        return (d)**0.5 # return the distance
    

class lshKNNclassifier(ClassifierAlgorithm):
    """Represent a lshKNN Classifier"""
    def __init__(self):
        super().__init__()

    def train(self,x_train,y_train):
        """Train method for the lshKNN algorithm, save the values and labels for the training data"""
        self._x_train = x_train
        self._y_train = y_train.to_list()

    def test(self,x_test,k_nearest_neighbors):
        """
        test method for the lshKNN algorithm,
        returns the prediction labels for each test data
        x_test: the test data
        k_nearest_neighbors: the parameter that controls how many neighbors should be considered
        """

        self.k = k_nearest_neighbors
        self._x_test = x_test
        y_train = self._y_train
        x_train = self._x_train
        y_pred = []

        # save values after projection
        train_projection_table = pd.DataFrame()
        test_projection_table = pd.DataFrame()

        dimensions = len(x_train.iloc[0])


        # Implement the LSH for 5 times
        for i in range(5):

            hash_table = HashTable(dimensions) # call the hashtable class, the random_vector is determined here
            col_name = f'projection_{i+1}'
            val = []
            for j1 in range(len(x_train)): # go through each training records
                # calculate the bucket after random projection
                val.append(hash_table.random_projection(x_train.iloc[j1].to_list()))
            train_projection_table[col_name] = val # save values to a table
            val = []
            for j2 in range(len(x_test)): # gothrough each testing records
                # calculate the bucket after random projection
                val.append(hash_table.random_projection(x_test.iloc[j2].to_list()))
            test_projection_table[col_name] = val # save values to a table

        # implement KNN below
        for i in range(len(x_test)): # go through each test record
            distance = []
            str1 = "".join(test_projection_table.iloc[i]) # convert the row in the table to string
            for j in range(len(x_train)):
                str2 = "".join(train_projection_table.iloc[j])
                distance.append(self.hamming_distance(str1,str2)) # calculate hamming distance
            
            top_k = self.heapsort(distance)[:self.k] # heapsort, and keep the top k values only

            indicies = [distance.index(top_element) for top_element in top_k] # get the index number

            labels = [y_train[index] for index in indicies] # get labels with corresponding index number

            pred_label = max(labels, key = labels.count) # calculate mode

            y_pred.append(pred_label) # save the label to prediction values

        return y_pred

    def hamming_distance(self,str1,str2):
        """Calculating the hamming distance between two bit strings"""
        return sum(c1 != c2 for c1,c2 in zip(str1,str2))
    
    def heapsort(self,input_list):
        """Implement a heapsort for an input_list"""
        h = []
        for val in input_list:
            heapq.heappush(h,val)
        return [heapq.heappop(h) for i in range(len(h))]


class HashTable:
    """
    Represent a hash table for LSH algorithm, useing a rendom vector to partition data.
    """
    def __init__(self,d):
        """hashtable class takes dimensions as one parameter,
        d: the number of dimensions of the data (number of columns)"""
        self.dimensions = d

        # generate a projection vector that follows gaussian distribution
        self.projection_vector = np.random.randn(self.dimensions,)

    def random_projection(self,vector):
        """Apply dot product between two vectors, return the bucked identified"""
        bucket = int((self.dot_product(vector,self.projection_vector.tolist())>0))
        return str(bucket)
    
    def dot_product(self,vec1,vec2):
        """Dot product between two vectors"""
        result = 0
        for i in range(len(vec2)): # go through each element in the lists
            result += vec1[i] * vec2[i] # vec(a) <a1,b1,c2> +vec(b) <a2,b2,c2> = a1a2+b1b2+c1c2
        return result


#----------------------------------- end of Classifier Class -----------------------------------

class Experiment:
    """
    Represent the experiment part, including cross validation, get accuracy score and print confusion matrix.

    dataset: input of independent variables
    labels: input of dependent variables
    classifiers: a list of classifiers provided in this toolbox
    """
    def __init__(self,dataset,labels,classifiers = None):
        self.dataset = dataset
        self.labels = labels
        self.classifiers = [simpleKNNClassifier(),lshKNNclassifier()]

        self.clf_dict = {0:'Simple KNN Classifier',1:'LSH-KNN Classifier'}

    def runCrossVal(self,k):
        """
        Run a k-fold cross validation for all the classifiers.
        Returns a matrix including all predicted values, one row for each available classifier.
        """

        fold_size = len(self.dataset)//k # define a fold size (number of observations in a fold)
        indices = list(range(len(self.dataset))) # generate index number from 0 to (length of dataset -1)
        folds = []
        pred_labels = []

        # k-fold cross validation index generation
        for i in range(k):
            if i==(k-1): # the last fold may contains records more than one fold_size (to make prediction and ground truth same length)
                test_idx = indices[i*fold_size:]
                train_idx = indices[:i*fold_size]
                folds.append((train_idx,test_idx))
            else:
                test_idx = indices[i*fold_size:(i+1)*fold_size] # generate test index
                train_idx = indices[:i * fold_size] + indices[(i + 1) * fold_size:] # take others as train index
                folds.append((train_idx,test_idx)) # save to folds

        # get data and training
        for train_idx,test_idx in folds: # go through each folds
            x_train,y_train = self.dataset.iloc[train_idx],self.labels.iloc[train_idx] # get training data and training labels
            x_test,y_test =  self.dataset.iloc[test_idx],self.labels.iloc[test_idx] # get testing data and testing labels
            y_test = y_test.to_list() # series to list

            for classifier in self.classifiers: # apply different classifiers for given fold
                try:
                    clf = classifier # construct the clf
                    clf.train(x_train,y_train) # train on given data
                    y_pred = clf.test(x_test,10) # fit the model, get predictions (in list)
                    pred_labels.append(y_pred) # save the predictions in list [[pred_fold1_clf1],[pred_fold1_clf2],[pred_fold2_clf1],...]
                except Exception as e:
                    print("There is an error with: ",e)
        
        # reshape the prediction matrix
        self.matrix = []
        #self.prob_matrix = []
        n = len(self.classifiers)
        for i in range(n):
            pred_same_clf = [val for idx,val in enumerate(pred_labels) if idx%n==i] # get the elements based on index number
            pred_same_clf = sum(pred_same_clf,[]) # flatten the 2-d list to 1-d ==> number of samples
            self.matrix.append(pred_same_clf) 

        return self.matrix
            
    def score(self):
        """
        Calculates the accuracy score.
        Returns a table including the name of classifier and the corresponsing accuracy score.
        """

        # create dataframe saving accuracy scores
        accuracy_report = pd.DataFrame(columns=['Classifier','Accuracy'])
        actual = self.labels.to_list() # labels will be actual values

        # calculate accuracy for each classifiers
        for idx, val in enumerate(self.matrix):
            predicted = val # values in matrix will be predicted values
            num = 0

            for i in range(len(actual)): # to through each pair
                if (actual[i]==predicted[i]):
                    num = num+1 # count
            accuracy = (num/len(actual)) # count accuracy
            clf_name = self.clf_dict.get(idx) # get clf name
            accuracy_report.loc[idx] = [clf_name,accuracy] # save info

        return accuracy_report

    def __confusionMatrix(self):
        """Return the confusion matrix for each classifier"""
        actual = self.labels.to_list()
        labels = set(actual) # save distinct labels
        cm_dict = dict()
        cm_list = []
        for i, val in enumerate(labels):  # build a dictionary
            cm_dict[val] = i

        for idx, val in enumerate(self.matrix): # go through predictions in each classifier
            cm = [[0]*len(labels) for _ in range(len(labels))] # generate a confusion matrix for saving data,
            predicted = val
            for i in range(len(actual)):
                x = cm_dict[actual[i]]
                y = cm_dict[predicted[i]]
                cm[y][x] += 1 # put values inside matrix

            # sace values in a list of dict, with classifier name associated with confusion matrix
            cm_list.append({'classifier':self.clf_dict.get(idx),
                            'matrix':cm})

        return cm_list
            
    def confusion_matrix(self):
        """
        Print the confusion matrix by line
        """
        cm_list = self.__confusionMatrix() # pass the list
        for d in cm_list: # go through each dictionary (each classifier)
            cm = d.get('matrix') # get the confusion matrix for this classifier
            for row in cm: # go through each row of the confusion matrix
                print(row) # print the row

    def ROC(self):
        """
        Visualize the ROC curves for given confusion matrix.
        """

        for d in self.__confusionMatrix(): # go through each classifier with its confusion matrix
            cm = d.get('matrix') # get the confusion matrix for a given classifier

            if len(cm) == 2: # if it is a 2-class problem

                # get tp, tn,fn,fp values
                tp = cm[0][0]
                tn = cm[1][1]
                fn = cm[1][0]
                fp = cm[0][1]

                #calculate true positive rate and false positive rate
                tpr = tp/(tp+fn)
                fpr = fp/(tn+fp)

                # make plots
                plt.plot([0,fpr,1],[0,tpr,1])
                plt.plot([0,1],[0,1],label= 'baseline',linestyle='dashed')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC curve")
                plt.show()

            else: # if there are more than 2 classes
                roc_curves = []
                for i in range(len(cm)): # go through each class
                    tp = cm[i][i]
                    fp = sum([row[i] for row in cm]) - tp
                    fn = sum(cm[i]) - tp
                    tn = sum(sum(cm,[])) - tp - fp - fn

                    # Calculate true positive rate and false positive rates
                    if tp+fn ==0:
                        tpr = 0
                    else:
                        tpr = tp/(tp+fn)
                        fpr = fp/(tn+fp)

                    x = [0,fpr,1]
                    y = [0,tpr,1]

                    roc_curves.append([x,y]) # save each class curve in list

                for idx,val in enumerate(roc_curves): # go through each class
                    x,y = val # unpack the list to get x and y
                    plt.plot(x,y,label = f'ROC curve of class {idx}') # print multiple roc curves
                plt.plot([0,1],[0,1],label= 'baseline',linestyle='dashed') # print baseline
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC curve")
                plt.legend()
                plt.show()

    



#------------------------------------------------------------------------------------------
# Computational Complexity Analysis
#------------------------------------------------------------------------------------------
# Time Complexity Analysis
# Assume the input dataset has m rows and n columns, the in worst case:
# lshKNNclassifier train method:

# The training method contains two lines, 
# In the first line, the x_train values was passed into self._x_train, which will take 1 step count.
# In the second line, y_train was converted into a list and then saved into variable self._y_train, this will take m step count.
# The step count be T(m,n) = 1 + m = O(m)

# lshKNNclassifier test method:
# T(n) = 7 [assining values for the beginning couple of rows] + n [get dimensions based on the length of one row] 
# + 5 [run LSH five times] * ( ( ( n [generate a projection vector with length n]) + 2 [assigning values] + m [ a loop run m times, go through each record under x_train]* n [length of one row] * (n + 3) [dot product goes through each element in the list] ) * 2 [do one time to get buckets under test data]) 
# + m [go through testing records] * ( 1 + 5 [get str1] + m [inner loop, go through training records] * ( 5 + 5 [hamming distance, comparing each pairs] + n(logn) [heapsort] + k [go through top_k list]] + k[go through indicies list] + 1 + 1))
# = 10mn^2 + (12 + nlogn + 2k)m^2 + 30mn + 6m + 11n +1
# = (nlogn)m^2 + 10mn^2 + 30mn + 6m + 11n
# T(m,n) = O(nlogn(m^2)+mn^2)


#------------------------------------------------------------------------------------------
# Space Complexity Analysis

# lshKNNclassifier train method:
# S(m,n) = m*n + m = O(mn)


# lshKNNclassifier test method:
# S(m,n) = 1 [variable k] + mn [x_test with shape m*n] + m [y_train with length m] + mn [x_train with shape m*n] + 5m [projecttion dataframe with shape 5*m] + 5m [projection dataframe with shape 5*m] + 1 [variable dimensions] + (n+1) [hash_table object, containing d and vector with size m] + 1 [variable col_name] + 1 [variable val] + m [variable distance with size m after appending] + k [variable top-k] + k [variable indicies] + k [variable labels] + m [variable y_pred]
# = 2mn + 13m + n + 5 + 3k
# = O(mn)


#------------------------------------------------------------------------------------------
# Compare with simpleKNN algorithm:
# T(m,n) for simpleKNN = O(mn^2+mn+n); T(m,n) for lshKNN = O(nlogn(m^2)+mn^2)
# The lshKNN method applied a dimensionality reduction before perform KNN method, so, for the data with high dimensions, lshKNN can successfully reduce the calculating time. 
# Also, a heapsort is applied for sorting in lshKNN method instead of the dataframe sorting used in simpleKNN method, this can improve the time complexity as well.








