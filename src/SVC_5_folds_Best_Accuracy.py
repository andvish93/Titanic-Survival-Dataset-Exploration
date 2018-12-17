# this code assumes that the data has Class Column spelled as "Class" as the first column
# the remaining columns are the predictors.

# to run 
# python Amer_Rez_chk_data_normality.py -n <name of the data set>

# packages needed
# pandas
# sklearn
# matplotlib

#!/usr/bin/env python3

#NAME:Amer_Rez_chk_data_normality.py

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import sys
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats.mstats import normaltest
from textwrap import wrap
from decimal import *

#=============
#Helpers
#=============
# Compute vector sum
def comp_vec_sum(nums):
    return np.sqrt(np.sum([x*x for x in nums])) 
    
#==============
program_name = sys.argv[0]
arguments = sys.argv[1:]

fig_ctr = 1
__PLOT_SIZE_X__ = 8
__PLOT_SIZE_Y__ = 6
__MATPLOTLIP_TITLE_WIDTH__ = 60

# Read in SVM example data
data_set_name = sys.argv[1]
data_set_name = "train_split.csv"
df = pd.read_csv(data_set_name)

df_num_rows = len(df.index)
df_num_cols = len(df.columns)

# Data Values
__PREDICTOR_VARIABLES__ = ['Sex_female','Pclass_3']
X = df[__PREDICTOR_VARIABLES__]

print()
print("SVM Example Dataset:\n", df)
print()

# extract class info
y = np.array(df['Survived'])

# Scale the data set from -1 to 1
print ("   Scaling training data set between [-1., 1.]" )
scaler = MinMaxScaler(feature_range = (-1., 1.))
X_scaled = scaler.fit_transform(X)

# Generate histograms for both classes in both the training and test data sets
# First compute vector sum of samples for training set
print("   Deterining the degree of fit between training and test data to a normal distribution.")
col_names = X.columns
df_X_scaled = pd.DataFrame(X_scaled, columns = col_names)

# Make copy of data frames and compute vector sum in preparation to 
# generate histograms
df_X_scaled_vecsum = df_X_scaled
df_X_scaled_vecsum['vec_sum'] = df_X_scaled_vecsum.apply(comp_vec_sum, axis = 1)

# Extract the vector sum info from the train and test data sets
X_scaled_hist_data = df_X_scaled_vecsum['vec_sum']

# Compute degree of match of data to normal dist
X_scaled_hr = normaltest(X_scaled_hist_data)
X_scaled_hr_match = X_scaled_hr[0]
X_scaled_hr_match_pvalue = X_scaled_hr[1]

print("    Data set match to normal dist: %.1f  with p-value: %.4E" % \
        (X_scaled_hr_match, Decimal(X_scaled_hr_match_pvalue)))

print("Displaying histograms for the data set.")

fig = plt.figure(fig_ctr, figsize = (__PLOT_SIZE_X__, __PLOT_SIZE_Y__)) 
fig_ctr = 1 + fig_ctr
plt.gcf().clear()
        
X_scaled_hist_data.hist(normed = True)
X_scaled_hist_data.plot(kind = 'kde', linewidth = 2, \
                        color = 'r', label = 'Distribution Of The Data')

# find minimum and maximum of xticks, so we know
# where we should compute theoretical distribution
xt = plt.xticks()[0]  
xmin, xmax = min(xt), max(xt)  
lnspc = np.linspace(xmin, xmax, len(X_scaled_hist_data))

# Now display the normal distribution over the histogram of the 
# training data
m, s = stats.norm.fit(X_scaled_hist_data) # get mean and standard deviation  
pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
plt.plot(lnspc, pdf_g, label="Normal Distribution", color = 'k', linewidth = 2) # plot it

plt.xlabel("Data feature vector distance/magnitude.")
plt.ylabel("Frequency.")
# match_val = '%.2f' % (X_scaled_hr_match)
# match_p_val = '%.4E' % (X_scaled_hr_match_pvalue)

# title_str = "Histrogram and Distribution of  data overlayed with normal distribution. " \
#     + "  Degree of match = " + match_val + " with p-value = " + match_p_val + "."
# plt.title("\n".join(wrap(title_str, __MATPLOTLIP_TITLE_WIDTH__)))
title_str = "Histrogram and Distribution of  data overlayed with normal distribution. "
plt.title("\n".join(wrap(title_str, __MATPLOTLIP_TITLE_WIDTH__)))

leg = plt.legend(loc = 'best', ncol = 1, shadow = True, fancybox = True)
leg.get_frame().set_alpha(0.5)

plt.show()


# Defining SVC classifier
clf = SVC(C = 1, gamma = 'auto', class_weight=None, coef0=0.0, kernel = 'linear')
clf.fit(X,y) 

# print("Predicting data point: ({}, {})".format(0.948, 0.4103))
# print()
# print("Class prediction of {}: {}".format("(0.948, 0.4103)", clf.predict([[0.948, 0.4103]])[0]))
# print()
# """
# print("Support:\n", clf.support_)
# print()
# print("Support vectors:\n", clf.support_vectors_)
# print()
# print("Dual coefficients:\n", clf.dual_coef_)
# print()
# print("Intercept: ", clf.intercept_)
# print()
# print("Coefficients:\n", clf.coef_[0])
# print()
# print("result:", (clf.coef_[0][0]*0.948) + (clf.coef_[0][1]*0.4103) + clf.intercept_[0])
# """
# print("DONE")
