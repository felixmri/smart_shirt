# region Import Dependencies
## Import Dependencies:
# For data processing:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks

# For ML:
import time as timer  # for timing code cells.
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector  # For sequential feature sel.
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression, Perceptron
from sklearn.pipeline import make_pipeline  # for grid-search over parameters pipeline.
# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Only for GPU based computing:
#from cuml.svm import SVC
#from cuml.ensemble import RandomForestClassifier as cuml_ran_fo
#import cuml.metrics
# endregion

# region Load data
## Load data:
s1_1 = pd.read_csv('data/11.csv', header=None)  # first subject, sensor 1
ch1 = s1_1.iloc[:, 1]
ch2 = s1_1.iloc[:, 2]
r1_1_ch1 = (-ch1 * 100) / (ch1 - 1)  # channel one (normal breathing)
r1_1_ch2 = (-ch2 * 100) / (ch2 - 1)  # channel two (normal breathing) [This channel was not used]
time = np.arange(0, .015 * len(r1_1_ch1), .015)  # 15 ms per time-point

s1_2 = pd.read_csv('data/12.csv', header=None)  # first subject, sensor 2
ch1 = s1_2.iloc[:, 1]
ch2 = s1_2.iloc[:, 2]
r1_2_ch1 = (-ch1 * 100) / (ch1 - 1)  # channel one (coughing)
r1_2_ch2 = (-ch2 * 100) / (ch2 - 1)  # channel two (coughing) [This channel was not used]

s2_1 = pd.read_csv('data/21.csv', header=None)
ch1 = s2_1.iloc[:, 1]
ch2 = s2_1.iloc[:, 2]
r2_1_ch1 = (-ch1 * 100) / (ch1 - 1)
r2_1_ch2 = (-ch2 * 100) / (ch2 - 1)

s2_2 = pd.read_csv('data/22.csv', header=None)
ch1 = s2_2.iloc[:, 1]
ch2 = s2_2.iloc[:, 2]
r2_2_ch1 = (-ch1 * 100) / (ch1 - 1)
r2_2_ch2 = (-ch2 * 100) / (ch2 - 1)

s3_1 = pd.read_csv('data/41.csv', header=None)
ch1 = s3_1.iloc[:, 1]
ch2 = s3_1.iloc[:, 2]
r3_1_ch1 = (-ch1 * 100) / (ch1 - 1)
r3_1_ch2 = (-ch2 * 100) / (ch2 - 1)

s3_2 = pd.read_csv('data/42.csv', header=None)
ch1 = s3_2.iloc[:, 1]
ch2 = s3_2.iloc[:, 2]
r3_2_ch1 = (-ch1 * 100) / (ch1 - 1)
r3_2_ch2 = (-ch2 * 100) / (ch2 - 1)
# endregion

# region Plot Data
## Plot data:
fig, axs = plt.subplots(4, 1)
fig.suptitle('Subject 1')
axs[0].plot(time, r1_1_ch1)
axs[1].plot(time, r1_1_ch2)
axs[2].plot(time, r1_2_ch1)
axs[3].plot(time, r1_2_ch2)
plt.show()

fig, axs = plt.subplots(4, 1)
fig.suptitle('Subject 2')
axs[0].plot(time, r2_1_ch1)
axs[1].plot(time, r2_1_ch2)
axs[2].plot(time, r2_2_ch1)
axs[3].plot(time, r2_2_ch2)
plt.show()

fig, axs = plt.subplots(4, 1)
fig.suptitle('Subject 3')
axs[0].plot(time, r3_1_ch1)
axs[1].plot(time, r3_1_ch2)
axs[2].plot(time, r3_2_ch1)
axs[3].plot(time, r3_2_ch2)
plt.show()

# endregion

# region Create labels for coughing events
## Create labels for coughing events:
label1 = (time > 20) * (time < 25)
label2 = (time > 60) * (time < 65)
label3 = (time > 95) * (time < 100)
label_c = label1 + label2 + label3
label_c = np.int32(label_c)
# Turn into Dataframe
label_c = pd.DataFrame(label_c)
label_c = label_c.iloc[:, 0]

# Label for channels with no coughing (all 0's):
label_nc = np.zeros(len(label_c))
label_nc = pd.DataFrame(label_nc)
label_nc = label_nc.iloc[:, 0]

# Plot subjects with red labeled as cough:
labels = ['r1_2_ch1', 'r2_2_ch1', 'r3_2_ch1']
fig, axs = plt.subplots(3, 1)
fig.suptitle("Labeled Events")
for i in range(3):
    data = locals()[labels[i]]
    axs[i].plot(time, data, 'b')
    axs[i].plot(time, data * label_c, 'r')
    axs[i].set_ylim([np.max(data)-10, np.max(data)])
plt.show()
# endregion

# region Partition into windows
## Partition into windows

# Organize data:
set_11 = r1_1_ch1  # normal breathing
set_12 = r1_2_ch1  # coughs
set_21 = r2_1_ch1
set_22 = r2_2_ch1
set_31 = r3_1_ch1
set_32 = r3_2_ch1

dataset = [set_11, set_12, set_21, set_22, set_31, set_32]
labels = [label_nc, label_c, label_nc, label_c, label_nc, label_c]
data = pd.concat([pd.concat(dataset), pd.concat(labels)], axis=1)
# Rename columns of dataframe:
name_list = ['Data', 'Labels']
data.columns = name_list

# Partition data:
window_size = 100
overlap = 0
step_size = window_size - overlap
# Initialize list to place windows:
window_list = []
label_list = []

# Do partitioning:
for i in range(0, len(data), step_size):
    xs = data['Data'].values[i:i + window_size]
    lab = stats.mode(data['Labels'].values[i:i + window_size])[0][0]

    window_list.append(xs)
    label_list.append(lab)

# Subtract mean to reduce drift artifact:
window_list_cent = []
for i in range(len(window_list)):
    window_list_cent.append(window_list[i] - np.mean(window_list[i]))


# endregion

# region Transform features
## Transform features
# Initialize DF:
X = pd.DataFrame()
# Statistical Features on signal in time domain

# std dev
X['x_std'] = pd.Series(window_list_cent).apply(lambda x: x.std())
# min
X['x_min'] = pd.Series(window_list_cent).apply(lambda x: x.min())
# max
X['x_max'] = pd.Series(window_list_cent).apply(lambda x: x.max())
# median
X['x_median'] = pd.Series(window_list_cent).apply(lambda x: np.median(x))
# number of peaks
X['x_peak_count'] = pd.Series(window_list_cent).apply(lambda x: len(find_peaks(x)[0]))
# skewness
X['x_skewness'] = pd.Series(window_list_cent).apply(lambda x: stats.skew(x))
# kurtosis
X['x_kurtosis'] = pd.Series(window_list_cent).apply(lambda x: stats.kurtosis(x))
# energy
X['x_energy'] = pd.Series(window_list_cent).apply(lambda x: np.sum(x ** 2) / 100)

# Statistical Features on signal in freq domain:
lim = int(window_size / 2)
window_list_cent_fft = pd.Series(window_list_cent).apply(lambda x: np.abs(np.fft.fft(x))[0:lim])

# std dev
X['x_std_fft'] = pd.Series(window_list_cent_fft).apply(lambda x: x.std())
# min
X['x_min_fft'] = pd.Series(window_list_cent_fft).apply(lambda x: x.min())
# max
X['x_max_fft'] = pd.Series(window_list_cent_fft).apply(lambda x: x.max())
# median
X['x_median_fft'] = pd.Series(window_list_cent_fft).apply(lambda x: np.median(x))
# number of peaks
X['x_peak_count_fft'] = pd.Series(window_list_cent_fft).apply(lambda x: len(find_peaks(x)[0]))
# skewness
X['x_skewness_fft'] = pd.Series(window_list_cent_fft).apply(lambda x: stats.skew(x))
# kurtosis
X['x_kurtosis_fft'] = pd.Series(window_list_cent_fft).apply(lambda x: stats.kurtosis(x))
# energy
X['x_energy_fft'] = pd.Series(window_list_cent_fft).apply(lambda x: np.sum(x ** 2) / 100)
# endregion

# region Split into train and test
## Split into train and test (2/3,1/3)
label_list = pd.Series(label_list)

X_train = X.iloc[0:320, :]  # First 320 points
label_train = label_list.iloc[0:320]

X_test = X.iloc[320:X.shape[0], :]
label_test = label_list.iloc[320:label_list.shape[0]]
# endregion

# region Implement ML: SVC




# endregion
##
from sklearn.svm import SVC
start = timer.time()
# Also store parameters and f_1 scores for each dimension loop:
k_list = list()
f_1 = list()
parameters = list()
# Iterate through number of features (dimensions) 2-16:
for k in range(12, 14):
    df_train_fs = X_train[:]
   # df_test_fs = df_test[:]
    k_list.append(k + 2)  # from k = 2 to 16
    print(f"Applying SFS with {k+2} dimensions")
    # Apply sequential feature selection keeping k dimensions:
    rdg_cls = RidgeClassifier(class_weight='balanced')
    sfs = SequentialFeatureSelector(rdg_cls, n_features_to_select=k + 2, scoring='f1')
    sfs.fit(df_train_fs, np.array(label_train).ravel())

    # Apply to test and train:
    df_train_fs = sfs.transform(df_train_fs)
    df_train_fs = pd.DataFrame(df_train_fs, columns=sfs.get_feature_names_out())

    #df_test_fs = sfs.transform(df_test_fs)
    #df_test_fs = pd.DataFrame(df_test_fs, columns=sfs.get_feature_names_out())

    # For each dimension k, do a grid search over the best parameters of the SVC regressor:
    params = [
        {'standardscaler': ['passthrough', StandardScaler()],
         'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['rbf'], 'svc__gamma': [0.001, 0.0001],
         'svc__class_weight': [None, 'balanced']},
        {'standardscaler': ['passthrough', StandardScaler(), MinMaxScaler()],
         'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['linear'], 'svc__class_weight': [None, 'balanced']}
    ]

    pipe = make_pipeline(StandardScaler(), SVC())
    gs = GridSearchCV(pipe, params, scoring='f1')
    gs.fit(df_train_fs, np.array(label_train).ravel())

    # Store best f1 score for each dimension k:
    f_1.append(gs.best_score_)
    parameters.append(gs.best_params_)

print(f'The optimal number of dimension is {k_list[np.array(f_1).argmax()]} with'
      f' an F1 score of {np.round(np.array(f_1).max(), 3)}'
      f' with parameters {parameters[np.array(f_1).argmax()]}')
end = timer.time()

print(f'Elapsed time with no GPU is {end-start}')