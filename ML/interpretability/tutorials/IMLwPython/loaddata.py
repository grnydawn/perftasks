import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.model_selection import  cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor

from skll.metrics import spearman

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

import warnings

import perftask

class LoadData(perftask.TaskFrame):

    def perform(self):

        # setting up the styling for the plots in this notebook
        sns.set(style="white", palette="colorblind", font_scale=1.2, rc={"figure.figsize":(12,9)})
        RANDOM_STATE = 420
        N_JOBS=8

        import pdb; pdb.set_trace()
