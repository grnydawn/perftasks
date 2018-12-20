
import os

import pandas as pd
import shap

from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from sklearn.ensemble import RandomForestRegressor

datapath = os.path.join(os.path.dirname(__file__), "combine_data_since_2000_PROCESSED_2018-04-26.csv")
data_df = pd.read_csv(datapath)

# onyl get players that have been in the league for 3 years
data_df2 = data_df.loc[data_df.Year <= 2015].copy()

# calculate the player AV percentiles by position
data_df2['AV_pctile'] = data_df2.groupby('Pos').AV.rank(pct=True, method='min', ascending=True)

# Get the data for the position we want, in this case it's DE
pos_df = data_df2.loc[data_df2.Pos=='DE'].copy().reset_index(drop=True)

# Combine measurables
features = ['Forty',
            'Wt',
            'Ht',
            'Vertical',
            'BenchReps',
            'BroadJump',
            'Cone',
            'Shuttle']
# what we want to predict
target = 'AV_pctile'

# Split the data into train and test sets
train_df = pos_df.loc[pos_df.Year <= 2011]
test_df = pos_df.loc[pos_df.Year.isin([2012, 2013, 2014, 2015])]

X = train_df[features].values
y = train_df[target].values

X_test = test_df[features].values
y_test = test_df[target].values

# best parameter set
pipe = Pipeline([("imputer", Imputer(strategy='median')), ("estimator",
    RandomForestRegressor( max_features=6, min_samples_split=63,
           n_estimators=500, random_state=420))])

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    #search.fit(X, y) 
    pipe.fit(X, y)

estimator = pipe.named_steps['estimator']
imputer = pipe.named_steps['imputer']

# create our SHAP explainer
shap_explainer = shap.TreeExplainer(estimator)

test_X_imp = imputer.transform(X_test)

# calculate the shapley values for our test set
test_shap_vals = shap_explainer.shap_values(test_X_imp)

# load JS in order to use some of the plotting functions from the shap
# package in the notebook
shap.initjs()

test_X_imp = imputer.transform(X_test)

test_X_imp_df = pd.DataFrame(test_X_imp, columns=features)

# plot the explanation for a single prediction
#shap.force_plot(test_shap_vals[0, :], test_X_imp_df.iloc[0, :])
#shap.force_plot(test_X_imp_df.iloc[0, :], test_shap_vals[0, :])

# visualize the first prediction's explanation
shap.force_plot(shap_explainer.expected_value, test_shap_vals[0,:], test_X_imp_df.iloc[0,:])

