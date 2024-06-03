import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
from scipy.io import savemat
from sklearn.preprocessing import MultiLabelBinarizer

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = ''
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

def aggregate_subdiagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_subclass)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

mlb = MultiLabelBinarizer()
one_hot_y_test = mlb.fit_transform(y_test)
one_hot_y_train = mlb.fit_transform(y_train)

x_train_mat = X_train[:,:,[1, 7, 3, 4]]
x_test_mat = X_test[:,:,[1, 7, 3, 4]]

# mdic = {"x_train": x_train_mat, 'y_train':one_hot_y_train, "x_test": x_test_mat, 'y_test': one_hot_y_test}
# savemat("matlab_ptb.mat", mdic)

np.save('ptb_np.npy', mdic)

print('exit')