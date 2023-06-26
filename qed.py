#%% imports

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.ensemble import GradientBoostingClassifier

#%% read data

train_df = pd.read_csv("cybersecurity_training.csv", sep="|")
test_df = pd.read_csv("cybersecurity_test.csv", sep="|")

#%% first insight

train_df.info
train_df.describe()
train_df.shape



#%% looking for nans

for column in train_df.columns:
    col = train_df[column]
    print(column)
    print(col.isna().sum())
    
# the only nan values are in n1 - n2 columns
# nan count is the same for all of these

#%% changing na with other values

train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)

#%% value counts

for column in train_df.columns:
    value_counts = train_df[column].value_counts().head(5)
    print("Column:", column)
    print(value_counts)
    print()
    
# alert_ids to be dropped (all unique)
# grandparent category to be dropped (nearly no information)
# n7, n8, n10 only zeros

# all categoric data

#%% cols to remove

cols_to_remove = ['alert_ids', 'grandparent_category', 'n7', 'n8', 'n10']

train_df.drop(cols_to_remove, axis=1, inplace=True)

test_df.drop(cols_to_remove, axis=1, inplace=True)

#%% more insight

temp_df = train_df[['categoryname', 'overallseverity']]

temp_df.groupby(['categoryname', 'overallseverity']).value_counts()

train_df.nunique()

#%% checking correlation

def calculate_correlation(dataframe):
    columns = dataframe.columns
    correlation_matrix = pd.DataFrame(np.zeros((len(columns), len(columns))), columns=columns, index=columns)

    for col1 in columns:
        for col2 in columns:
            if col1 != col2:
                cross_table = pd.crosstab(dataframe[col1], dataframe[col2])
                chi2 = chi2_contingency(cross_table)[0]
                correlation = np.sqrt(chi2 / (dataframe.shape[0] * min(cross_table.shape) - 1))
                correlation_matrix.loc[col1, col2] = correlation

    return correlation_matrix


corr_matrix = calculate_correlation(train_df)

#%% correlation conclusion

# correlatedcount - timestamp_dist 0.97

# ip - categoryname 0.85
# ip - ipcategory_name 0.94
# ip - ipcategory_scope 0.86
# ip - parent_category 0.89
# leave ip only

# ipcategory_scope - ipcategory_name 0.86
# parent_category - ipcategory_name 0.89

# timestamp_dist high correlation (over 0.9) with:
    # srcip_cd
    # dstid_cd
    # dstport_cd
    # alerttype_cd
    # reportingdevice_cd
    # domain_cd
    # protocol_cd
    # username_cd
    # p9

# alerttype_cd - p6 0.94

cols_to_remove = ['categoryname', 'ipcategory_name', 'ipcategory_scope',
                  'dstport_cd', 'alerttype_cd', 'reportingdevice_cd',
                  'domain_cd', 'protocol_cd', 'username_cd', 'p6', 'p9']

train_df.drop(cols_to_remove, axis=1, inplace=True)

test_df.drop(cols_to_remove, axis=1, inplace=True)

#%%

# srcipcategory_dominate, dstipcategory_dominate - one hot encoding
# weekday - label encoding

train_df['ip'].nunique()
train_df['ip'].head(30)

train_df.drop(['client_code'], axis=1, inplace=True)
test_df.drop(['client_code'], axis=1, inplace=True)

#%% extracting data from ip

def process_ip(df):
    df[['ip1', 'ip2', 'ip3', 'ip4']] = df['ip'].str.split('.', expand=True)

    df.drop(['ip'], axis=1, inplace=True)

    le = LabelEncoder()

    df['ip1'] = le.fit_transform(df['ip1'])
    df['ip2'] = le.fit_transform(df['ip2'])
    df['ip3'] = le.fit_transform(df['ip3'])
    df['ip4'] = le.fit_transform(df['ip4'])

process_ip(train_df)
process_ip(test_df)

#%% variable encoding

def variable_encoding(df):
    src_category = df['srcipcategory_dominate']
    dst_category = df['dstipcategory_dominate']

    src_encoded = pd.get_dummies(src_category)
    dst_encoded = pd.get_dummies(dst_category)

    src_encoded = src_encoded.add_suffix('_1')
    dst_encoded = dst_encoded.add_suffix('_2')

    df = pd.concat([df, src_encoded, dst_encoded], axis=1)

    df.drop(['srcipcategory_dominate', 'dstipcategory_dominate'], axis=1, inplace=True)

    weekday = df['weekday']

    mapping = {
        'Mon': 0,
        'Tue': 1,
        'Wed': 2,
        'Thu': 3,
        'Fri': 4,
        'Sat': 5,
        'Sun': 6
    }

    result = pd.Series([mapping[i] for i in weekday])

    df['weekday'] = result

variable_encoding(train_df)

variable_encoding(test_df)

#%% splitting data for feedback before submission

Y = train_df['notified']

train_df.drop(['notified'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(train_df, Y, test_size=0.2, random_state=42)

#%% random forest (not used)

model = RandomForestClassifier(n_estimators=1000,
                               max_features=None)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu: {:.2f}%".format(accuracy * 100))

auc = roc_auc_score(y_test, y_pred)
print("AUC:", auc)

#%% logistic regression (not used)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu: {:.2f}%".format(accuracy * 100))

auc = roc_auc_score(y_test, y_pred)
print("Metryka AUC: {:.2f}".format(auc))

#%% gradient boosting (not used)

model = GradientBoostingClassifier(n_estimators=1000,
                                   max_depth=None,
                                   learning_rate=0.01)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu: {:.2f}%".format(accuracy * 100))

auc = roc_auc_score(y_test, y_pred)
print("Metryka AUC: {:.2f}".format(auc))
# 0.63 best so far, but still pretty poor

#%% svm (not used)

from sklearn import svm

model = svm.SVC(kernel='rbf')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_pred)
print("Wartość AUC:", auc)

#%% bayes (not used)

from sklearn.naive_bayes import GaussianNB

naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

y_pred = naive_bayes.predict(X_test)

auc = roc_auc_score(y_test, y_pred)
print("Wartość AUC:", auc)

#%% lightGBM (selected model)

import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

model = lgb.train(params, train_data, num_boost_round=100)

y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_pred)
print("Wartość AUC:", auc)

#%% final model (used right data)

import lightgbm as lgb

train_data = lgb.Dataset(train_df, label=Y)
test_data = lgb.Dataset(test_df)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

model = lgb.train(params, train_data, num_boost_round=100)

y_pred = model.predict(test_df)

auc = roc_auc_score(y_test, y_pred)
print("Wartość AUC:", auc)

np.savetxt('result_1.txt', y_pred, fmt='%.4f')
