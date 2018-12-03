# Prediction Experiments
from sklearn.svm import SVC
#import numpy as np
import pandas as pd



class experiments_on_model:
    
    # initilization requires just the initial files
    def __init__(self,train_file,test_file):
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        # default parameters
        self.parameters = self.update_params()
                
    # perform data_preprocessing steps here    
    def data_processing(data):
        # cleaning/ imputation/ bla bla bla code
        return data    
        
        
    # update this if we want to test out various parameter settings.
    # update model     
    def update_params(self,C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None):  
        self.parameters = {
        'C' : C,
        'kernel': kernel, 
        'degree':degree,
        'gamma':gamma ,
        'coef0':coef0,
        'shrinking':shrinking ,
        'probability':probability,
        'tol':tol,
        'cache_size':cache_size,
        'class_weight':class_weight,
        'verbose':verbose,
        'max_iter':max_iter, 
        'decision_function_shape':decision_function_shape,
        'random_state':random_state
        }
    
        
    # train, test, run model on data and params
    def experiments(self):   
        
        # instance of prediction model
        self.clf = SVC(self.parameters)
                
        # data preprocessing of the training data
        self.train_data = self.data_preprocessing(self.training_data)
        
        # training the model
        self.clf.train(self.train_data)

        # data preprocessing of the training data
        self.test_data = self.data_preprocessing(self.test_data)
        
        # testing the model
        output = self.clf.test(self.test_data)        
        
        print(output)
        return output

    
    # input - results to print to stderr as a list
    # prints to stderr
    def print_results(self,list_results):
        for res in list_results:
            print(res)
    
    
    
    # for calculating accuracy and stuff
    def analysis(self,output):
        pass        

    # Plot various graphs here
    def plot_graphs(self):
        pass
        

            
if __name__ == "main":
    
    # experimental pipeline
    
    experiment0 = experiments_on_model("../dataset/train.csv","../dataset/test.csv")
        
    experiment0.update_params() # use this to run various tests 
    
    # performs train, and test and returns test output
    output = experiment0.experiments()
    print(output)
    # prints results
    experiment0.print_results()
    
    # plots graphs
    experiment0.plot_graphs()
    
    