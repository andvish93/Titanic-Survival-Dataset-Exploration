#!/usr/bin/env python3

#NAME:svm_example.py

import pandas as pd
import numpy as np
from sklearn.svm import SVC
import itertools


__NUM_ITERATIONS__ = 5  # default number of iteratios to display histograms of
__TRAINING_TEST_SPLIT__ = 0.8
__PREDICTOR_VARIABLES__ = ['Sex_female','Pclass_3']

# Read in SVM example data
df = pd.read_csv("train_split.csv")


# split
accuracy = list()
for _ in itertools.repeat(None, __NUM_ITERATIONS__): 
    dfsvc_train = df.sample(frac = __TRAINING_TEST_SPLIT__)
    dfsvc_test = pd.concat([dfsvc_train, df]).loc[dfsvc_train.index.symmetric_difference(df.index)] 
        
    # extract class info
    y_train = np.array(dfsvc_train['Survived'])
    y_test = np.array(dfsvc_test['Survived'])
    
    # Extract data values
    X_train = dfsvc_train[__PREDICTOR_VARIABLES__]
    X_test = dfsvc_test[__PREDICTOR_VARIABLES__]
    
    # Defining SVC classifier
    clf = SVC(C = 0.5, gamma = 'auto', class_weight=None, coef0=0.0, kernel = 'linear')
    clf.fit(X_train,y_train) 
    
    # predicitng for test set
    female_list = list(X_test['Sex_female'])
    Pclass__list = list(X_test['Pclass_3'])
    correct_pred_count = 0
    for i in range(len(X_test)):
        female = female_list[i]
        Pclass_3 = Pclass__list[i]
        if y_test[i] == clf.predict([[female, Pclass_3]])[0]:
            correct_pred_count += 1
    accuracy.append(correct_pred_count/len(y_test))
print("The Best Accuracy is " + str(max(accuracy)))
    

    # # sanity check
    # for female in [0, 1]:
    #     for Pclass_3 in [0, 1]:
    #         print("Class prediction of {}: {}".format("female="+ str(female) + " Pclass_3=" + str(Pclass_3) + " ", clf.predict([[female, Pclass_3]])[0]))
    # print()
    
"""
print("Support:\n", clf.support_)
print()
print("Support vectors:\n", clf.support_vectors_)
print()
print("Dual coefficients:\n", clf.dual_coef_)
print()
print("Intercept: ", clf.intercept_)
print()
print("Coefficients:\n", clf.coef_[0])
print()
print("result:", (clf.coef_[0][0]*0.948) + (clf.coef_[0][1]*0.4103) + clf.intercept_[0])
"""
print("DONE")
