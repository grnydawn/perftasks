
import os
import itertools

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.text import Text

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
from sklearn.metrics import mean_squared_error, r2_score

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

# Not working
#from pdpbox import pdp

import lime
from lime.lime_tabular import LimeTabularExplainer

import shap

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
        future_results = [executor.submit(func, **{item_kwarg: item}, **kwargs) for item in an_iter]
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

def plot_ice_grid(dict_of_ice_dfs, data_df, features, ax_ylabel='', nrows=3, 
                  ncols=3, figsize=(12, 12), sharex=False, sharey=True, 
                  subplots_kws={}, rug_kws={'color':'k'}, **ice_plot_kws):
    """A function that plots ICE plots for different features in a grid."""
    fig, axes = plt.subplots(nrows=nrows, 
                             ncols=ncols, 
                             figsize=figsize,
                             sharex=sharex,
                             sharey=sharey,
                             **subplots_kws)
    # for each feature plot the ice curves and add a rug at the bottom of the 
    # subplot
    for f, ax in zip(features, axes.flatten()):
        ice_plot(dict_of_ice_dfs[f], ax=ax, **ice_plot_kws)
        # add the rug
        sns.distplot(data_df[f], ax=ax, hist=False, kde=False, 
                     rug=True, rug_kws=rug_kws)
        ax.set_title('feature = ' + f)
        ax.set_ylabel(ax_ylabel)
        sns.despine()
        
    # get rid of blank plots
    for i in range(len(features), nrows*ncols):
        axes.flatten()[i].axis('off')

    return fig

def plot_2d_pdp_grid(pdp_inters, feature_pairs,
                     ncols=3, nrows=4, figsize=(13, 16),
                     xaxis_font_size=12, yaxis_font_size=12,
                     contour_line_fontsize=12,
                     tick_labelsize=10, x_quantile=None, 
                     plot_params=None, subplots_kws={}):
    """Plots a grid of 2D PDP plots."""
    # create our subplots to plot our PDPs on
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                             figsize=figsize, **subplots_kws)

    # for each feature pair, plot the 2-D pdp
    for pdp_inter, feat_pair, ax in zip(pdp_inters, feature_pairs, axes.flatten()):
    
        # use pdpbox's _pdp_contour_plot function to actually plot the 2D pdp
        pdp._pdp_contour_plot(pdp_inter, feat_pair, 
                              x_quantile=x_quantile, ax=ax, 
                              plot_params=plot_params,
                              fig=None)
        # adjust some font sizes
        ax.tick_params(labelsize=tick_labelsize)
        ax.xaxis.get_label().set_fontsize(xaxis_font_size)
        ax.yaxis.get_label().set_fontsize(yaxis_font_size)
    
        # set the contour line fontsize
        for child in ax.get_children():
            if isinstance(child, Text):
                child.set(fontsize=contour_line_fontsize)   
    
    # get rid of empty subplots
    for i in range(len(pdp_inters), nrows*ncols):
        axes.flatten()[i].axis('off')
        
    return fig

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
            except AttributeError as err:
                import pdb; pdb.set_trace()
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
        eli5.explain_prediction_df(tree, example, feature_names=self.features)

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

    def analyze_ice_center(self):
        "Individual Conditional Expectation - Centered Feature Interaction"

        # pcyebox likes the data to be in a DataFrame so let's create one with our imputed data
        # we first need to impute the missing data

        train_X_imp = self.imputer.transform(self.X)

        train_X_imp_df = pd.DataFrame(train_X_imp, columns=self.features)

        forty_ice_df = ice(data=train_X_imp_df, column='Forty', 
                           predict=self.pipe.predict)

        # new colormap for ICE plot
        cmap2 = plt.get_cmap('OrRd')
        # set color_by to Wt, in order to color each curve by that player's weight
        ice_plot(forty_ice_df, linewidth=0.5, color_by='Wt', cmap=cmap2, plot_pdp=True,
            pdp_kwargs={'c': 'k', 'linewidth': 5}, centered=True)
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
        plt.ylabel('Pred. AV %ile (centered)')
        plt.xlabel('Forty');



    def analyze_ice_grid(self):
        "Individual Conditional Expectation - Feature Interaction Grid"

        train_X_imp = self.imputer.transform(self.X)

        train_X_imp_df = pd.DataFrame(train_X_imp, columns=self.features)

        # create dict of ICE data for grid of ICE plots
        train_ice_dfs = {feat: ice(data=train_X_imp_df, column=feat, predict=self.estimator.predict) 
                         for feat in self.features}

        fig = plot_ice_grid(train_ice_dfs, train_X_imp_df, self.features,
                            ax_ylabel='Pred. AV %ile', alpha=0.3, plot_pdp=True,
                            pdp_kwargs={'c': 'red', 'linewidth': 3},
                            linewidth=0.5, c='dimgray')
        fig.tight_layout()
        fig.suptitle('ICE plots (training data)')
        fig.subplots_adjust(top=0.89);

    def analyze_ice_gc(self):
        "Individual Conditional Expectation - Centered Feature Interaction Grid"

        train_X_imp = self.imputer.transform(self.X)

        train_X_imp_df = pd.DataFrame(train_X_imp, columns=self.features)

        # create dict of ICE data for grid of ICE plots
        train_ice_dfs = {feat: ice(data=train_X_imp_df, column=feat, predict=self.estimator.predict) 
                         for feat in self.features}

        fig = plot_ice_grid(train_ice_dfs, train_X_imp_df, self.features, 
                            ax_ylabel='Pred AV %ile (centered)',
                            alpha=.2, plot_points=False, plot_pdp=True,
                            pdp_kwargs={"c": "red", "linewidth": 3},
                            linewidth=0.5, c='dimgray', centered=True,
                            sharey=False, nrows=4, ncols=2, figsize=(11,16))
        fig.tight_layout()
        fig.suptitle('Centered ICE plots (training data)')
        fig.subplots_adjust(top=0.9)


#    def analyze_pdp_2d(self):
#        "Partial Dependence Plot - 2D Grid"
#
#        train_X_imp = self.imputer.transform(self.X)
#
#        train_X_imp_df = pd.DataFrame(train_X_imp, columns=self.features)
#
#        # get each possible feature pair combination
#        feature_pairs = [list(feat_pair) for feat_pair in itertools.combinations(self.features, 2)]
#
#        # we will only plot the feature iteractions that invlove either Forty or Wt
#        # just to avoid making soooo many plots
#        forty_wt_feat_pairs = [fp for fp in feature_pairs if 'Forty' in fp or 'Wt' in fp]
#        # now calculate the data for the pdp interactions
#        # we can do that with pdpbox's pdp_interact function
#        # in the current development version on github, parallelization is supported
#        # but it didn't work for me so I resorted to using that multiprocess helper
#        # function from before
#
#        train_feat_inters = multiproc_iter_func(N_JOBS, forty_wt_feat_pairs, 
#                                                pdp.pdp_interact, 'features',
#                                                model=self.estimator, dataset=train_X_imp_df, model_features=self.features)
#        # and now plot a grid of PDP interaction plots
#        # NOTE that the contour colors do not represent the same values
#        # across the different subplots
#        fig = plot_2d_pdp_grid(train_feat_inters, forty_wt_feat_pairs)
#        fig.tight_layout()
#        fig.suptitle('PDP Interaction Plots (training data)', fontsize=20)
#        fig.subplots_adjust(top=0.95);

    def analyze_lime(self):
        "Local Interpretable Model-agnostic Explanamtions"

        train_X_imp = self.imputer.transform(self.X)

        train_X_imp_df = pd.DataFrame(train_X_imp, columns=self.features)

        # create the explainer by passing our training data, 
        # setting the correct modeling mode, pass in feature names and
        # make sure we don't discretize the continuous features
        explainer = LimeTabularExplainer(train_X_imp_df, mode='regression', 
                                         feature_names=self.features, 
                                         random_state=RANDOM_STATE, 
                                         discretize_continuous=False) 

        test_X_imp = self.imputer.transform(self.X_test)

        test_X_imp_df = pd.DataFrame(test_X_imp, columns=self.features)

        # the number of features to include in our predictions
        num_features = len(self.features)
        # the index of the instance we want to explaine
        exp_idx = 2
        exp = explainer.explain_instance(test_X_imp_df.iloc[exp_idx,:].values, 
                                         self.estimator.predict, num_features=num_features)

        # a plot of the weights for each feature
        exp.as_pyplot_figure();

        plt.show()

        lime_expl = test_X_imp_df.apply(explainer.explain_instance, 
                                        predict_fn=self.estimator.predict, 
                                        num_features=num_features,
                                        axis=1)

        # get all the lime predictions
        lime_pred = lime_expl.apply(lambda x: x.local_pred[0])
        # RMSE of lime pred
        mean_squared_error(self.y_pred, lime_pred)**0.5

        # r^2 of lime predictions
        r2_score(self.y_pred, lime_pred)


        # new explainer with smaller kernel_width
        better_explainer = LimeTabularExplainer(train_X_imp_df, mode='regression', 
                                                feature_names=self.features, 
                                                random_state=RANDOM_STATE, 
                                                discretize_continuous=False,
                                                kernel_width=1) 

        better_lime_expl = test_X_imp_df.apply(better_explainer.explain_instance, 
                                               predict_fn=self.estimator.predict, 
                                               num_features=num_features,
                                               axis=1)

        # get all the lime predictions
        better_lime_pred = better_lime_expl.apply(lambda x: x.local_pred[0])
        # RMSE of lime pred
        mean_squared_error(self.y_pred, better_lime_pred)**0.5

        # r^2 of lime predictions
        r2_score(self.y_pred, better_lime_pred)

        # construct a DataFrame with all the feature weights and bias terms from LIME
        # create an individual dataframe for each explanation
        lime_dfs = [pd.DataFrame(dict(expl.as_list() + [('bias', expl.intercept[0])]), index=[0]) 
                    for expl in better_lime_expl]
        # then concatenate them into one big DataFrame
        lime_expl_df = pd.concat(lime_dfs, ignore_index=True)

        lime_expl_df.head()

        # scale the data
        scaled_X = (test_X_imp_df - explainer.scaler.mean_) / explainer.scaler.scale_
        # calc the lime feature contributions
        lime_feat_contrib = lime_expl_df[self.features] * scaled_X

        # get the prediction and actual target values to plot
        y_test_and_pred_df = pd.DataFrame(np.column_stack((self.y_test, self.y_pred)),
                                          index=self.test_df.Player,
                                          columns=['true_AV_pctile', 'pred_AV_pctile'])

        # add on bias term, actual av %ile and predicted %ile
        other_lime_cols = ['bias', 'true_AV_pctile', 'pred_AV_pctile']
        lime_feat_contrib[other_lime_cols] = pd.DataFrame(np.column_stack((lime_expl_df.bias,
                                                                           y_test_and_pred_df)))

        lime_feat_contrib.sort_values('pred_AV_pctile', inplace=True)

        lime_feat_contrib.head()

        title = 'LIME Feature Contributions for each prediction in the testing data'
        fig = double_heatmap(lime_feat_contrib[['true_AV_pctile', 'pred_AV_pctile']].T,
                             lime_feat_contrib.loc[:, :'bias'].T, title=title,
                             cbar_label1='%ile', cbar_label2='contribution', 
                             subplot_top=0.9)
        # set the x-axis label for the bottom heatmap
        # fig has 4 axes object, the first 2 are the heatmaps, the other 2 are the colorbars
        fig.axes[1].set_xlabel('Player');


    def analyze_shap(self):
        "SHapley Additive exPlanations"

        # create our SHAP explainer
        shap_explainer = shap.TreeExplainer(self.estimator)

        test_X_imp = self.imputer.transform(self.X_test)

        # calculate the shapley values for our test set
        test_shap_vals = shap_explainer.shap_values(test_X_imp)

        # load JS in order to use some of the plotting functions from the shap
        # package in the notebook
        #shap.initjs()

        test_X_imp = self.imputer.transform(self.X_test)

        test_X_imp_df = pd.DataFrame(test_X_imp, columns=self.features)

        # plot the explanation for a single prediction
        #shap.force_plot(test_shap_vals[0, :], test_X_imp_df.iloc[0, :])
        #shap.force_plot(test_X_imp_df.iloc[0, :], test_shap_vals[0, :])

        # visualize the first prediction's explanation
        shap.force_plot(shap_explainer.expected_value, test_shap_vals[0,:], test_X_imp_df.iloc[0,:])


#In v0.20 force_plot now requires the base value as the first parameter!
#Try shap.force_plot(explainer.expected_value, shap_values)
#for multi-output models try shap.force_plot(explainer.expected_value[0], shap_values[0]).

