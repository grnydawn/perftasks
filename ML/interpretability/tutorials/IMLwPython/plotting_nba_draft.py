
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import perftask

class Plotting(perftask.TaskFrame):

    def __init__(self, parent, url, argv):

        self.add_data_argument("plotname", nargs="?", help="plot name")
        self.add_option_argument("-l", "--list-plot", action="store_true", help="list available plots")

        datapath = os.path.join(os.path.dirname(__file__), "draft_data_1966_to_2016.csv")
        self.draft_df = pd.read_csv(datapath, index_col=0)

    def perform(self):
        """Plot draft data"""

        if self.targs.plotname:

            plotfunc = getattr(self, "plot_{}".format(self.targs.plotname))

            plotfunc()

            title = getattr(plotfunc, "__doc__")
            if title:
                plt.title(title, fontsize=20)

            plt.show()

        elif self.targs.list_plot:
            for attr in dir(self):
                if attr.startswith("plot_") and callable(getattr(self, attr)):
                    name = attr[5:]
                    desc = getattr(getattr(self, attr), "__doc__")
                    if desc:
                        print("* ", attr[5:], ": ", desc)
                    else:
                        print("* ", attr[5:])

        return 0, {"draft_df": self.draft_df}


    def plot_ws48_numdrafts(self):
        'The Number of Players Drafted and Average Career WS/48 for each Draft (1966-2016)'

        WS48_yrly_avg = self.draft_df.groupby('Draft_Yr').WS_per_48.mean()
        players_drafted = self.draft_df.groupby('Draft_Yr').Pk.count()

        sns.set_style("white")  

        # change the mapping of default matplotlib color shorthands (like 'b' 
        # or 'r') to default seaborn palette 
        sns.set_color_codes()

        # set the x and y values for our first line
        x_values = self.draft_df.Draft_Yr.unique() 
        y_values_1 = players_drafted

        # plt.subplots returns a tuple containing a Figure and an Axes
        # fig is a Figure object and ax1 is an Axes object
        # we can also set the size of our plot
        fig, ax1 = plt.subplots(figsize=(12,9))  
        # plt.xlabel('Draft Pick', fontsize=16)

        # Create a series of grey dashed lines across the each
        # labled y-value of the graph
        plt.grid(axis='y',color='grey', linestyle='--', lw=0.5, alpha=0.5)

        # Change the size of tick labels for x-axis and left y-axis
        # to a more readable font size for
        plt.tick_params(axis='both', labelsize=14)

        # Plot our first line with deals with career WS/48 per draft
        # We assign it to plot 1 to reference later for our legend
        # We alse give it a label, in order to use for our legen
        plot1 = ax1.plot(x_values, y_values_1, 'b', label='No. of Players Drafted')
        # Create the ylabel for our WS/48 line
        ax1.set_ylabel('Number of Players Drafted', fontsize=18)
        # Set limits for 1st y-axis
        ax1.set_ylim(0, 240)
        # Have tick color match corrsponding line color
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        # Now we create the our 2nd Axes object that will share the same x-axis
        # To do this we call the twinx() method from our first Axes object
        ax2 = ax1.twinx()
        y_values_2 = WS48_yrly_avg
        # Create our second line for the number of picks by year
        plot2 = ax2.plot(x_values, y_values_2, 'r', 
                         label='Avg WS/48')
        # Create our label for the 2nd y-axis
        ax2.set_ylabel('Win Shares Per 48 minutes', fontsize=18)
        # Set the limit for 2nd y-axis
        ax2.set_ylim(0, 0.08)
        # Set tick size for second y-axis
        ax2.tick_params(axis='y', labelsize=14)
        # Have tick color match corresponding line color
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        # Limit our x-axis values to minimize white space
        ax2.set_xlim(1966, 2016.15)

        # create our legend 
        # First add our lines together
        lines = plot1 + plot2
        # Then create legend by calling legend and getting the label for each line
        ax1.legend(lines, [l.get_label() for l in lines])

        # Create evenly ligned up tick marks for both y-axis
        # np.linspace allows us to get evenly spaced numbers over
        # the specified interval given by first 2 arguments,
        # Those 2 arguments are the the outer bounds of the y-axis values
        # the third argument is the number of values we want to create
        # ax1 - create 9 tick values from 0 to 240
        ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 9))
        # ax2 - create 9 tick values from 0.00 to 0.08
        ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 9))

        # need to get rid of spines for each Axes object
        for ax in [ax1, ax2]:
            ax.spines["top"].set_visible(False)  
            ax.spines["bottom"].set_visible(False)  
            ax.spines["right"].set_visible(False)  
            ax.spines["left"].set_visible(False)  
            
        # Create text by calling the text() method from our figure object    
        fig.text(0.1, 0.02,
                 'Data source: http://www.basketball-reference.com/draft/'
                '\nAuthor: Youngsung', fontsize=10)

    def plot_numdrafts(self):
        'The Number of players Drafted in each Draft (1966-2016)'

        players_drafted = self.draft_df.groupby('Draft_Yr').Pk.count()

        sns.set_style("white")  
        plt.figure(figsize=(12,9))
        x_values = self.draft_df.Draft_Yr.unique()  
        y_values = players_drafted
        plt.ylabel('Number of Players Drafted', fontsize=18)
        plt.xlim(1966, 2016.5)
        plt.ylim(0, 250)
        plt.grid(axis='y',color='grey', linestyle='--', lw=0.5, alpha=0.5)
        plt.tick_params(axis='both', labelsize=14) 
        sns.despine(left=True, bottom=True) 
        plt.plot(x_values, y_values)

        plt.text(1966, -35,
                 'Primary Data Source: http://www.basketball-reference.com/draft/'
                 '\nAuthor: Youngsung Kim',
                  fontsize=12)

    def plot_ws48(self):
        "Average Career Win Shares Per 48 minutes by Draft Year (1966-2016)"

        #WS48_yrly_avg = [self.draft_df[self.draft_df['Draft_Yr']==yr]['WS_per_48'].mean()
        #         for yr in self.draft_df.Draft_Yr.unique() ]
        WS48_yrly_avg = self.draft_df.groupby('Draft_Yr').WS_per_48.mean()

        # use seaborn to set our graphing style
        # the style 'white' creates a white background for
        # our graph
        sns.set_style("white")  

        # Set the size to have a width of 12 inches
        # and height of 9
        plt.figure(figsize=(12,9))

        # get the x and y values
        x_values = self.draft_df.Draft_Yr.unique()  
        y_values = WS48_yrly_avg

        # add a title
        #title = ('Average Career Win Shares Per 48 minutes by Draft Year (1966-2016)')
        #plt.title(title, fontsize=20)

        # Label the y-axis
        # We don't need to label the year values
        plt.ylabel('Win Shares Per 48 minutes', fontsize=18)

        # Limit the range of the axis labels to only
        # show where the data is. This helps to avoid
        # unnecessary whitespace.
        plt.xlim(1966, 2016.5)
        plt.ylim(0, 0.08)

        # Create a series of grey dashed lines across the each
        # labled y-value of the graph
        plt.grid(axis='y',color='grey', linestyle='--', lw=0.5, alpha=0.5)

        # Change the size of tick labels for both axis
        # to a more readable font size
        plt.tick_params(axis='both', labelsize=14)
          
        # get rid of borders for our graph using seaborn's
        # despine function
        sns.despine(left=True, bottom=True) 

        # plot the line for our graph
        plt.plot(x_values, y_values)

        # Provide a reference to data source and credit yourself
        # by adding text to the bottom of the graph
        # the first 2 arguments are the x and y axis coordinates of where
        # we want to place the text
        # The coordinates given below should place the text below
        # the xlabel and aligned left against the y-axis

        plt.text(1966, -0.008,
                 'Primary Data Source: http://www.basketball-reference.com/draft/'
                 '\nAuthor: Youngsung Kim',
                  fontsize=12)

    def plot_ws48_top60(self):
        'Average Career Win Shares Per 48 minutes for Top 60 Picks by Draft Year (1966-2016)'

        top60 = self.draft_df[(self.draft_df['Pk'] < 61)]
        top60_yrly_WS48 = top60.groupby('Draft_Yr').WS_per_48.mean()

        sns.set_style("white")  

        plt.figure(figsize=(12,9))
        x_values = self.draft_df.Draft_Yr.unique() 
        y_values = top60_yrly_WS48
        plt.ylabel('Win Shares Per 48 minutes', fontsize=18)
        plt.xlim(1966, 2016.5)
        plt.ylim(0, 0.08)
        plt.grid(axis='y',color='grey', linestyle='--', lw=0.5, alpha=0.5)
        plt.tick_params(axis='both', labelsize=14)
        sns.despine(left=True, bottom=True) 
        plt.plot(x_values, y_values)
        plt.text(1966, -0.012,
                 'Primary Data Source: http://www.basketball-reference.com/draft/'
                 '\nAuthor: Youngsung Kim'
                 '\nNote: Drafts from 1989 to 2006 have less than 60 draft picks',
                  fontsize=12)

    def plot_ws48_top60(self):
        'Average Career Win Shares Per 48 minutes for Top 60 Picks by Draft Year (1966-2016)'



fg = sns.lmplot(x='value', y='contribution', col='feature',
                data=train_expl_df.loc[train_expl_df.feature!=''], 
                col_order=features, sharex=False, col_wrap=3, fit_reg=False,
                size=4, scatter_kws={'color':'salmon', 'alpha': 0.5, 's':30})
fg.fig.suptitle('Feature Contributions vs Feature Values (training data)')
fg.fig.subplots_adjust(top=0.90);

