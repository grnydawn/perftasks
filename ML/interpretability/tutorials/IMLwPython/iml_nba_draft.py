
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer
from sklearn.model_selection import  cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from sklearn.ensemble import RandomForestRegressor

from skll.metrics import spearman

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

import eli5
from eli5.sklearn import PermutationImportance

from sklearn.tree import export_graphviz
import graphviz
import pydotplus
from io import StringIO, BytesIO

import treeinterpreter.treeinterpreter as ti

from pycebox.ice import ice, ice_plot

from perftask import TaskBase

RANDOM_STATE = 420
N_JOBS=8

from concurrent.futures import ProcessPoolExecutor

def multiproc_iter_func(max_workers, an_iter, func, item_kwarg, **kwargs):
    """
    A helper functions that applies a function to each item in an iterable using
    multiple processes. 'item_kwarg' is the keyword argument for the item in the
    iterable that we pass to the function.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_results = [executor.submit(func, **{item_kwarg: item}, **kwargs)
                          for item in an_iter]

        results = [future.result() for future in future_results]
        
    return results

# a hacky function that plots a swarmplot along with a colorbar
# based off the code found here:
# https://stackoverflow.com/questions/40814612/map-data-points-to-colormap-with-seaborn-swarmplot
def swarmplot_with_cbar(cmap, cbar_label, *args, **kwargs):
    fig = plt.gcf()
    ax = sns.swarmplot(*args, **kwargs)
    # remove the legend, because we want to set a colorbar instead
    ax.legend().remove()
    ## create colorbar ##
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="3%", pad=0.05)
    fig.add_axes(ax_cb)
    cb = ColorbarBase(ax_cb, cmap=cmap, orientation='vertical')
    cb.set_label(cbar_label, labelpad=10)
    
    return fig

def double_heatmap(data1, data2, cbar_label1, cbar_label2,
                   title='', subplot_top=0.86, cmap1='viridis', cmap2='magma', 
                   center1=0.5, center2=0, grid_height_ratios=[1,4],
                   figsize=(14,10)):
    # do the actual plotting
    # here we plot 2 seperate heatmaps one for the predictions and actual percentiles
    # the other for the contributions
    # the reason I chose to do this is because of the difference in magnitudes
    # between the percentiles and the contributions
    fig, (ax,ax2) = plt.subplots(nrows=2, figsize=figsize, 
                                 gridspec_kw={'height_ratios':grid_height_ratios})

    fig.suptitle(title)
    fig.subplots_adjust(hspace=0.02, top=subplot_top)

    # heatmap for actual and predicted percentiles
    sns.heatmap(data1, cmap="viridis", ax=ax, xticklabels=False, center=center1,
                cbar_kws={'location':'top', 
                          'use_gridspec':False, 
                          'pad':0.1,
                          'label': cbar_label1})
    ax.set_xlabel('')

    # heatmap of the feature contributions
    sns.heatmap(data2, ax=ax2, xticklabels=False, center=center2, cmap=cmap2,
                cbar_kws={'location':'bottom', 
                          'use_gridspec':False, 
                          'pad':0.07, 
                          'shrink':0.41,
                          'label': cbar_label2})
    ax2.set_ylabel('');
    return fig

def create_ordered_joint_contrib_df(contrib):
    """
    Creates a dataframe from the joint contribution info, where the
    feature combinations are ordered (in descending fashion) by the absolute
    value of the joint contribution.
    """
    df = pd.DataFrame(contrib, columns=['feat_interaction', 'contribution'])
    # get the reordered index    
    new_idx = (df.contribution.abs()
                              .sort_values(inplace=False, ascending=False)
                              .index)
    df = df.reindex(new_idx).reset_index(drop=True)
    return df

class Interperting(TaskBase):

    def __init__(self, parent, url, argv):

        self.add_data_argument("name", nargs="?", help="analysis name")
        self.add_option_argument("-l", "--list-analysis", action="store_true", help="list available analyses")

        datapath = os.path.join(os.path.dirname(__file__), "combine_data_since_2000_PROCESSED_2018-04-26.csv")
        data_df = pd.read_csv(datapath)

        # setting up the styling for the plots in this notebook
        sns.set(style="white", palette="colorblind", font_scale=1.2, 
                rc={"figure.figsize":(12,9)})

        # onyl get players that have been in the league for 3 years
        data_df2 = data_df.loc[data_df.Year <= 2015].copy()

        # calculate the player AV percentiles by position
        data_df2['AV_pctile'] = data_df2.groupby('Pos').AV.rank(pct=True, method='min', ascending=True)

        # Get the data for the position we want, in this case it's DE
        pos_df = data_df2.loc[data_df2.Pos=='DE'].copy().reset_index(drop=True)

        # Split the data into train and test sets
        self.train_df = pos_df.loc[pos_df.Year <= 2011]
        self.test_df = pos_df.loc[pos_df.Year.isin([2012, 2013, 2014, 2015])]

        # Combine measurables
        self.features = ['Forty',
                    'Wt',
                    'Ht',
                    'Vertical',
                    'BenchReps',
                    'BroadJump',
                    'Cone',
                    'Shuttle']
        # what we want to predict
        target = 'AV_pctile'

        self.X = self.train_df[self.features].values
        self.y = self.train_df[target].values


        # best parameter set
        self.pipe = Pipeline([("imputer", Imputer(strategy='median')), ("estimator",
            RandomForestRegressor( max_features=6, min_samples_split=63,
                   n_estimators=500, random_state=420))])

# Uncomment and use searcher for better parameter set
#        # the modeling pipeline
#        pipe = Pipeline([("imputer", Imputer()),
#                         ("estimator", RandomForestRegressor(random_state=RANDOM_STATE))])
#
#        # We use spearman's rank correlation as the scoring metric since
#        # we are concerned with ranking the players
        self.spearman_scorer = make_scorer(spearman)
#
#        # the hyperparamters to search over, including different imputation strategies
#        rf_param_space = {
#            'imputer__strategy': Categorical(['mean', 'median', 'most_frequent']),
#            'estimator__max_features': Integer(1, 8),
#            'estimator__n_estimators': Integer(50, 500), 
#            'estimator__min_samples_split': Integer(2, 200),
#        }
#        # create our search object
#        search = BayesSearchCV(pipe, 
#                              rf_param_space, 
#                              cv=10,
#                              n_jobs=N_JOBS, 
#                              verbose=0, 
#                              error_score=-9999, 
#                              scoring=spearman_scorer, 
#                              random_state=RANDOM_STATE,
#                              return_train_score=True, 
#                              n_iter=75)
        # fit the model
        # I get some funky warnings, possibly due to the spearman scorer,
        # I choose to suppress them
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            #search.fit(X, y) 
            self.pipe.fit(self.X, self.y) 

        # test set data
        self.X_test = self.test_df[self.features].values
        self.y_test = self.test_df[target].values
        # predictions
        #y_pred = search.predict(X_test)
        self.y_pred = self.pipe.predict(self.X_test)
        # evaluation
        #model_test_score = spearman_scorer(search, X_test, y_test)
        model_test_score = self.spearman_scorer(self.pipe, self.X_test, self.y_test)

#        # create percentiles for nfl draft picks
#        # Lower numerical picks (e.g. 1, 2, 3) are ranked closer to 1
#        # Higher numerical picks (e.g. 180, 200, etc) are ranked closer to 0
#        draft_pick_pctile = test_df.Pick.rank(pct=True, method='min', ascending=False, na_option='top')
#        #spearman(y_test, draft_pick_pctile)

        #####################
        #  Feature Importance
        #####################

        ###### Mean Decrease Impurity ######

        # get the estimator and imputer from our pipeline, which will be used
        # as we try and interpret our model
        # if you use search
        #estimator = search.best_estimator_.named_steps['estimator']
        #imputer = search.best_estimator_.named_steps['imputer']
        self.estimator = self.pipe.named_steps['estimator']
        self.imputer = self.pipe.named_steps['imputer']

    def perform(self):
        """Analyze draft data"""

        if self.targs.name:

            try:
                func = getattr(self, "analyze_{}".format(self.targs.name))

                func()

                plt.show()
            except AttributeError:
                print("'{}' is not supported.".format(self.targs.name))

            #title = getattr(func, "__doc__")
            #if title:
            #    plt.title(title, fontsize=20)


        elif self.targs.list_analysis:
            for attr in dir(self):
                if attr.startswith("analyze_") and callable(getattr(self, attr)):
                    name = attr[8:]
                    desc = getattr(getattr(self, attr), "__doc__")
                    if desc:
                        print("* ", attr[8:], ": ", desc)
                    else:
                        print("* ", attr[8:])

        #return 0, {"data_df": self.data_df, "data_df2": self.data_df2}


    def analyze_fi_mdi(self):
        "Feature Importance - Mean Decrease Impurity"


        feat_imp_df = eli5.explain_weights_df(self.estimator, feature_names=self.features)

        #import pdb; pdb.set_trace()
        # get the feature importances from each tree and then visualize the
        # distributions as boxplots
        all_feat_imp_df = pd.DataFrame(data=[tree.feature_importances_ for tree in self.estimator], columns=self.features)

        sns.boxplot(data=all_feat_imp_df).set(title='Feature Importance Distributions', ylabel='Importance')

    def analyze_fi_pi(self):
        "Feature Importance - Permutation Importance"

        # we need to impute the data first before calculating permutation importance
        train_X_imp = self.imputer.transform(self.X)
        # set up the met-estimator to calculate permutation importance on our training
        # data
        perm_train = PermutationImportance(self.estimator, scoring=self.spearman_scorer, n_iter=50, random_state=RANDOM_STATE)
        # fit and see the permuation importances
        perm_train.fit(train_X_imp, self.y)
        eli5.explain_weights_df(perm_train, feature_names=self.features)

        # plot the distributions
        perm_train_feat_imp_df = pd.DataFrame(data=perm_train.results_, columns=self.features)
        sns.boxplot(data=perm_train_feat_imp_df) .set(title='Permutation Importance Distributions (training data)', ylabel='Importance')

    def analyze_fc_dp(self):
        "Feature Contribution - Decision Path"

        # source for plotting decision tree
        # https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176
        # Get all trees of depth 2 in the random forest
        depths2 = [tree for tree in self.estimator.estimators_ if tree.tree_.max_depth==2]
        # grab the first one
        tree = depths2[0]
        # plot the tree
        dot_data = StringIO()
        export_graphviz(tree, out_file=dot_data, feature_names=self.features, filled=True, rounded=True, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        png_str = graph.create_png(prog='dot')

        # treat the dot output string as an image file
        bio = BytesIO()
        bio.write(png_str)
        bio.seek(0)
        img = mpimg.imread(bio)

        # plot the image
        imgplot = plt.imshow(img, aspect='equal')

        # simple exmaple of a player with a 4.6 Forty and a Wt of 260 lbs
        example = np.array([4.6, 260, 0, 0, 0, 0, 0, 0])
        eli5.explain_prediction_df(tree, example, feature_names=features)

        example_prec = tree.predict(example.reshape(1,-1))

    def analyze_fc_bp(self):
        "Feature Contribution - Box Plot"

        # we need to impute the data first before calculating permutation importance
        train_X_imp = self.imputer.transform(self.X)

        # Contibrutions for training set predictions
        # construct a list of all contributions for the entire train set
        train_expl_list = multiproc_iter_func(N_JOBS, train_X_imp, eli5.explain_prediction_df,
            'doc', estimator=self.estimator, feature_names=self.features)
        # concatenate them into 1 large dataframe, with the proper player name as an
        # index
        train_expl_df = pd.concat(train_expl_list, keys=self.train_df.Player, names=['Player'])

        # Contributions for test set predictions
        # we need to impute the missing values in the test set
        test_X_imp = self.imputer.transform(self.X_test)
        # now repeat what we did with the training data on the test data
        test_expl_list = multiproc_iter_func(N_JOBS, test_X_imp, eli5.explain_prediction_df,
            'doc', estimator=self.estimator, feature_names=self.features)

        test_expl_df = pd.concat(test_expl_list, keys=self.test_df.Player, names=['Player'])

        # double check
        y_pred_sums = test_expl_df.groupby('Player').weight.sum()
        assert np.allclose(self.y_pred, y_pred_sums), "y_pred is not equal to y_red_sums"

        # I"m creating one big dataframe that includes both train and test
        # to plot them on same plot using seaborn's boxplot
        train_expl_df.rename(columns={'weight': 'contribution'}, inplace=True)
        test_expl_df.rename(columns={'weight': 'contribution'}, inplace=True)
        train_expl_df['data'] = 'train'
        test_expl_df['data'] = 'test'
        train_test_expl_df = pd.concat([train_expl_df, test_expl_df])
        sns.boxplot(x='feature', y='contribution', hue='data', order=self.features,
            data=train_test_expl_df.loc[train_test_expl_df.feature!=''], palette={'train': 'salmon', 'test':'deepskyblue'})
        plt.legend(loc=9)
        plt.title('Distributions of Feature Contributions');

    def analyze_fc_sp(self):
        "Feature Contribution - Swamp Plot"

        train_X_imp = self.imputer.transform(self.X)

        train_expl_list = multiproc_iter_func(N_JOBS, train_X_imp, eli5.explain_prediction_df,
            'doc', estimator=self.estimator, feature_names=self.features)

        train_expl_df = pd.concat(train_expl_list, keys=self.train_df.Player, names=['Player'])
        train_expl_df.rename(columns={'weight': 'contribution'}, inplace=True)
        train_expl_df['data'] = 'train'

        # min-max scaling of the feature values allows us to use a colorbar
        # to indicate high or low feature values
        train_scaled_feat_vals = (train_expl_df.groupby('feature')
                                               .value
                                               .transform(lambda x: x/x.max()))

        train_expl_df['scaled_feat_vals'] = train_scaled_feat_vals

        cmap = plt.get_cmap('viridis')
        cbar_label = 'Feature Value %ile'

        plt.title('Distribution of Feature Contributions (training data)')
        swarmplot_with_cbar(cmap, cbar_label,  x='feature', y='contribution',
                            hue='scaled_feat_vals', palette='viridis', order=self.features,
                            data=train_expl_df.loc[train_expl_df.feature!='']);

        plt.show()

        test_X_imp = self.imputer.transform(self.X_test)

        test_expl_list = multiproc_iter_func(N_JOBS, test_X_imp, eli5.explain_prediction_df,
            'doc', estimator=self.estimator, feature_names=self.features)

        test_expl_df = pd.concat(test_expl_list, keys=self.test_df.Player, names=['Player'])
        y_pred_sums = test_expl_df.groupby('Player').weight.sum()
        test_expl_df.rename(columns={'weight': 'contribution'}, inplace=True)
        test_expl_df['data'] = 'test'

        test_scaled_feat_vals = (test_expl_df.groupby('feature')
                                              .value
                                              .transform(lambda x: x/x.max()))

        test_expl_df['scaled_feat_vals'] = test_scaled_feat_vals

        plt.title('Distribution of Feature Contributions (test data)')
        swarmplot_with_cbar(cmap, cbar_label,  x='feature', y='contribution',
                            hue='scaled_feat_vals', palette='viridis', order=self.features,
                            data=test_expl_df.loc[test_expl_df.feature!='']);


    def analyze_fc_fv(self):
        "Feature Contribution - Feature Values (training data)"

        train_X_imp = self.imputer.transform(self.X)

        train_expl_list = multiproc_iter_func(N_JOBS, train_X_imp, eli5.explain_prediction_df,
            'doc', estimator=self.estimator, feature_names=self.features)

        train_expl_df = pd.concat(train_expl_list, keys=self.train_df.Player, names=['Player'])
        train_expl_df.rename(columns={'weight': 'contribution'}, inplace=True)
        train_expl_df['data'] = 'train'

        train_scaled_feat_vals = (train_expl_df.groupby('feature')
                                               .value
                                               .transform(lambda x: x/x.max()))


        train_expl_df['scaled_feat_vals'] = train_scaled_feat_vals

        fg = sns.lmplot(x='value', y='contribution', col='feature',
                        data=train_expl_df.loc[train_expl_df.feature!=''], 
                        col_order=self.features, sharex=False, col_wrap=3, fit_reg=False,
                        size=4, scatter_kws={'color':'salmon', 'alpha': 0.5, 's':30})
        fg.fig.suptitle('Feature Contributions vs Feature Values (training data)')
        fg.fig.subplots_adjust(top=0.90);

        test_X_imp = self.imputer.transform(self.X_test)

        test_expl_list = multiproc_iter_func(N_JOBS, test_X_imp, eli5.explain_prediction_df,
            'doc', estimator=self.estimator, feature_names=self.features)

        test_expl_df = pd.concat(test_expl_list, keys=self.test_df.Player, names=['Player'])
        test_expl_df.rename(columns={'weight': 'contribution'}, inplace=True)
        test_expl_df['data'] = 'test'

        test_scaled_feat_vals = (test_expl_df.groupby('feature')
                                               .value
                                               .transform(lambda x: x/x.max()))

        test_expl_df['scaled_feat_vals'] = test_scaled_feat_vals

        fg = sns.lmplot(x='value', y='contribution', col='feature',
                        data=test_expl_df.loc[test_expl_df.feature!=''], 
                        col_order=self.features, sharex=False, col_wrap=3, fit_reg=False, 
                        size=4, scatter_kws={'color':'salmon', 'alpha': 0.5, 's':30})
        fg.fig.suptitle('Feature Contributions vs Feature Values (testing data)')
        fg.fig.subplots_adjust(top=0.90);


    def analyze_fc_fi(self):
        "Feature Contribution - Feature Interaction"

        train_X_imp = self.imputer.transform(self.X)

        train_expl_list = multiproc_iter_func(N_JOBS, train_X_imp, eli5.explain_prediction_df,
            'doc', estimator=self.estimator, feature_names=self.features)

        train_expl_df = pd.concat(train_expl_list, keys=self.train_df.Player, names=['Player'])
        train_expl_df.rename(columns={'weight': 'contribution'}, inplace=True)
        train_expl_df['data'] = 'train'

        train_scaled_feat_vals = (train_expl_df.groupby('feature')
                                               .value
                                               .transform(lambda x: x/x.max()))


        train_expl_df['scaled_feat_vals'] = train_scaled_feat_vals

        # before we actually plot anything we need to do a bit of data manipulation
        # let's pivot the data and create a new dataframe where the columns are
        # the feature contributions and each row is a player, with the player
        # name as the index value
        # here are different ways to pivot column values to columns
        # https://stackoverflow.com/questions/26255671/pandas-column-values-to-columns
        # based on running %%timeit, the groupby method was fastest 
        train_contrib_df = (train_expl_df.groupby(['Player','feature'])
                                         .contribution
                                         .aggregate('first')
                                         .unstack())
        # add in the feature values
        train_feat_contrib_df = train_contrib_df.merge(self.train_df[['Player'] + self.features],
                                                       how='left', left_index=True, 
                                                       right_on='Player',
                                                       suffixes=('_contrib', '_value'))

        cmap = plt.get_cmap('viridis')

        # now we can plot
        plt.scatter(x='Forty_value', y='Forty_contrib', c='Wt_value', cmap=cmap,
                    data=train_feat_contrib_df)
        plt.xlabel('Forty')
        plt.ylabel('contribution')
        plt.colorbar(label='Wt');


    def analyze_fc_hm(self):
        "Feature Contribution - Heatmap"

        test_X_imp = self.imputer.transform(self.X_test)

        test_expl_list = multiproc_iter_func(N_JOBS, test_X_imp, eli5.explain_prediction_df,
            'doc', estimator=self.estimator, feature_names=self.features)

        test_expl_df = pd.concat(test_expl_list, keys=self.test_df.Player, names=['Player'])
        test_expl_df.rename(columns={'weight': 'contribution'}, inplace=True)
        test_expl_df['data'] = 'test'

        test_scaled_feat_vals = (test_expl_df.groupby('feature')
                                               .value
                                               .transform(lambda x: x/x.max()))

        test_expl_df['scaled_feat_vals'] = test_scaled_feat_vals

        # get the prediction and actual target values to plot
        y_test_and_pred_df = pd.DataFrame(np.column_stack((self.y_test, self.y_pred)),
                                          index=self.test_df.Player,
                                          columns=['true_AV_pctile', 'pred_AV_pctile'])

        # let's pivot the data such that the feature contributions are the columns
        test_heatmap_df = (test_expl_df.groupby(['Player','feature'])
                                       .contribution
                                       .aggregate('first')
                                       .unstack())

        # there may be some NaNs if a feature did not contribute to a prediction, 
        # so fill them in with 0s
        test_heatmap_df = test_heatmap_df.fillna(0)

        # merge our predictions with the the contributions
        test_heatmap_df = test_heatmap_df.merge(y_test_and_pred_df, how='left',
                                                right_index=True, left_index=True)
        # sort by predictions
        test_heatmap_df.sort_values('pred_AV_pctile', ascending=True, inplace=True)

        title = 'Feature contributions to predicted AV %ile \nfor each player in the testing data'
        fig = double_heatmap(test_heatmap_df[['true_AV_pctile', 'pred_AV_pctile']].T,
                             test_heatmap_df[self.features].T, '%ile', 'contribution',
                             title=title)


    def analyze_fc_jc(self):
        "Feature Contribution - Joint Contribution"

        depths2 = [tree for tree in self.estimator.estimators_ if tree.tree_.max_depth==2]
        # grab the first one
        tree = depths2[0]

        example = np.array([4.6, 260, 0, 0, 0, 0, 0, 0])

        example_pred, example_bias, example_contrib = ti.predict(tree,
                                                             example.reshape(1, -1),
                                                             joint_contribution=True)
        test_X_imp = self.imputer.transform(self.X_test)

        joint_pred, joint_bias, joint_contrib = ti.predict(self.estimator,
                                                   test_X_imp,
                                                   joint_contribution=True)


        # add the names of the feats to the joint contributions
        joint_contrib_w_feat_names = []
        # for each observation in the join contributions
        for obs in joint_contrib:
            # create a list
            obs_contrib = []
            # for each tuple of column indexes
            for k in obs.keys():
                # get the associated feature names
                feature_combo = [self.features[i] for i in k]
                # get the contribution value
                contrib = obs[k]
                # store that information in the observation individual list
                obs_contrib.append([feature_combo, contrib])
            # append that individual to the large list containing each observations
            # joint feature contributions
            joint_contrib_w_feat_names.append(obs_contrib)

        # create an ordered dataframe for each player
        joint_contrib_dfs = [create_ordered_joint_contrib_df(contrib)
                             for contrib in joint_contrib_w_feat_names]
        # now combine them all
        joint_contrib_df = pd.concat(joint_contrib_dfs, keys=self.test_df.Player, names=['Player'])

        # edit feat_interaction column so the values are strings and not lists
        joint_contrib_df['feat_interaction'] = joint_contrib_df.feat_interaction.apply(' | '.join) 


        # first get the sum of the absolute values for each joint feature contribution
        abs_imp_joint_contrib = (joint_contrib_df.groupby('feat_interaction')
                                                  .contribution
                                                  .apply(lambda x: x.abs().sum())
                                                   .sort_values(ascending=False))

        # then calculate the % of total contribution by dividing by the sum of all absolute vals
        rel_imp_join_contrib = abs_imp_joint_contrib / abs_imp_joint_contrib.sum()

        rel_imp_join_contrib.head(15)[::-1].plot(kind='barh', color='salmon', 
                                                      title='Joint Feature Importances');
        plt.show()

        plt.ylabel('Features')
        plt.xlabel('% of total joint contributions');

        top_feat_interactions = rel_imp_join_contrib.head(15).index
        top_contrib_mask = joint_contrib_df.feat_interaction.isin(top_feat_interactions)
        sns.boxplot(y='feat_interaction', x='contribution', 
                    data=joint_contrib_df.loc[top_contrib_mask],
                    orient='h', order=top_feat_interactions);

        plt.show()

        joint_contrib_heatmap_df = (joint_contrib_df[top_contrib_mask]
                                       .groupby(['Player','feat_interaction'])
                                       .contribution
                                       .aggregate('first')
                                       .unstack())
        joint_contrib_heatmap_df = joint_contrib_heatmap_df.fillna(0)

        # get the prediction and actual target values to plot
        y_test_and_pred_df = pd.DataFrame(np.column_stack((self.y_test, self.y_pred)),
                                          index=self.test_df.Player,
                                          columns=['true_AV_pctile', 'pred_AV_pctile'])

        joint_contrib_heatmap_df = joint_contrib_heatmap_df.merge(y_test_and_pred_df, 
                                                                  how='left',
                                                                  right_index=True, 
                                                                  left_index=True)
        # sort by predictions
        joint_contrib_heatmap_df.sort_values('pred_AV_pctile', ascending=True, 
                                             inplace=True)

        title = 'Top 15 Joint Feature Contributions to predicted AV %ile\n(testing data)'
        fig = double_heatmap(joint_contrib_heatmap_df[['true_AV_pctile', 'pred_AV_pctile']].T,
                             joint_contrib_heatmap_df[top_feat_interactions].T, 
                             cbar_label1='%ile', cbar_label2='contribution', 
                             title=title, grid_height_ratios=[1, 7], figsize=(14, 12),
                             subplot_top=0.89)


    def analyze_ice(self):
        "Individual Conditional Expectation"

        # pcyebox likes the data to be in a DataFrame so let's create one with our imputed data
        # we first need to impute the missing data

        train_X_imp = self.imputer.transform(self.X)

        train_X_imp_df = pd.DataFrame(train_X_imp, columns=self.features)


        forty_ice_df = ice(data=train_X_imp_df, column='Forty', 
                           predict=self.pipe.predict)

        ice_plot(forty_ice_df, c='dimgray', linewidth=0.3)
        plt.ylabel('Pred. AV %ile')
        plt.xlabel('Forty');

    def analyze_ice_fi(self):
        "Individual Conditional Expectation - Feature Interaction"

        # pcyebox likes the data to be in a DataFrame so let's create one with our imputed data
        # we first need to impute the missing data

        train_X_imp = self.imputer.transform(self.X)

        train_X_imp_df = pd.DataFrame(train_X_imp, columns=self.features)

        forty_ice_df = ice(data=train_X_imp_df, column='Forty', 
                           predict=self.pipe.predict)

        # new colormap for ICE plot
        cmap2 = plt.get_cmap('OrRd')
        # set color_by to Wt, in order to color each curve by that player's weight
        ice_plot(forty_ice_df, linewidth=0.5, color_by='Wt', cmap=cmap2)
        # ice_plot doesn't return a colorbar so we have to add one
        # hack to add in colorbar taken from here:
        # https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots/11558629#11558629
        wt_vals = forty_ice_df.columns.get_level_values('Wt').values
        sm = plt.cm.ScalarMappable(cmap=cmap2, 
                                   norm=plt.Normalize(vmin=wt_vals.min(), 
                                                      vmax=wt_vals.max()))
        # need to create fake array for the scalar mappable or else we get an error
        sm._A = []
        plt.colorbar(sm, label='Wt')
        plt.ylabel('Pred. AV %ile')
        plt.xlabel('Forty');

