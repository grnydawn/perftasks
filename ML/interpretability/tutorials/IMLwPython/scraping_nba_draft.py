
import sys
if sys.version_info[0] < 3:
    from urllib import urlopen
else:
    from urllib.request import urlopen

from bs4 import BeautifulSoup
import pandas as pd

import certifi

import perftask

class Scraping(perftask.TaskFrame):

    def perform(self):

        # url that we are scraping
        #url = "https://www.basketball-reference.com/draft/NBA_2014.html"
        url_template = "http://www.basketball-reference.com/draft/NBA_{year}.html"

        # create an empty DataFrame
        draft_df = pd.DataFrame()

        
        for year in range(1966, 2016):  # for each year
            url = url_template.format(year=year)  # get the url

            if sys.version_info[0] < 3:
                html = urlopen(url)
            else:
                html = urlopen(url, cafile=certifi.where())

            soup = BeautifulSoup(html, 'html5lib')

            column_headers = [th.getText() for th in soup.findAll('tr', limit=2)[1].findAll('th')[1:]]
            data_rows = soup.findAll('tr')[2:]
            player_data = [[td.getText() for td in data_rows[i].findAll('td')] for i in range(len(data_rows))]

            df = pd.DataFrame(player_data, columns=column_headers)
            df = df[df.Player.notnull()]
            df.rename(columns={'WS/48':'WS_per_48'}, inplace=True)
            df.columns = df.columns.str.replace('%', '_Perc')
            df.columns.values[13:17] = [df.columns.values[13:17][col] + "_per_G" for col in range(4)]
            df = df.convert_objects(convert_numeric=True)
            df = df[:].fillna(0)
            df.loc[:,'Yrs':'AST'] = df.loc[:,'Yrs':'AST'].astype(int)
            df.insert(0, 'Draft_Yr', year)  
            draft_df = draft_df.append(df, ignore_index=True)

        draft_df.to_csv("draft_data_1966_to_2016.csv")
