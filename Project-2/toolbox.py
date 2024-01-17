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
import numpy as np
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud

try: 
    class DataSet:
        """
        Represent a dataset for analysis.
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

            filename = input("Please enter the name of the file.              ")
            datatype = input("Please enter the datatype for the given file.   ")

            # add extension if file type is not included in the input
            if '.csv' not in filename:
                filename = filename+'.csv'

            return filename,datatype

        def clean(self):
            """Clean the loaded data"""

        def explore(self):
            """Visualization for cleaned data"""

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

            return self.data

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
            return self.data

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

        def explore(self):
            """
            Override the explore method specific for quantitive data.
            Returns one 2-D scatrer plot bwtween two variables.
            Returns one 3-D plot between three normalized variables.
            """

            # split data into normalized data and un-normalized data
            columns = self.data.columns.tolist()
            normalized = [col for col in columns if "Normalized" in col]
            others = [col for col in columns if "Normalized" not in col]

            # randomly select two variables for scatter plot
            var1,var2 = np.random.choice(others,2,replace=False)

            plt.scatter(self.data[var1],self.data[var2])
            plt.title(f"Scatter Plot for {var1} and {var2}")
            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.show()

            # randomly select 3 variables for 3-d plot
            x,y,z = np.random.choice(normalized,3,replace=False)
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
            return self.data
        
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

except Exception as e:
    print("There is an error with: ",e)

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


