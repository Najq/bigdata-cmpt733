#!/usr/bin/env python
# coding: utf-8


#converted jupyter notebook to py file

import pandas as pd
import os
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot


def global_tempChange():
	#read file
	master_df = pd.read_csv('tmax-2/tmax-2/part-00000.csv.gz',header=None, compression='gzip', sep=',', quotechar='"', error_bad_lines=False)
	master_df.columns = ["station", "date", "latitude", "longitude","elevation","tmax"]
	#append all different files in the folder
	for file in os.listdir('tmax-2/tmax-2'):
		if(file != 'part-00000.csv.gz'):
			df = pd.read_csv('tmax-2/tmax-2/'+file, compression='gzip', header=None, sep=',', quotechar='"', error_bad_lines=False)
			df.columns = ["station", "date", "latitude", "longitude","elevation","tmax"]
			master_df = master_df.append(df, ignore_index = True)
	master_df = master_df.dropna()
	#converting to date type
	master_df['date'] = master_df['date'].astype('datetime64[ns]') 
	#extracting year
	master_df['year_of_date'] = master_df['date'].dt.year
	#extracting decade
	master_df['decade'] = master_df['date'].astype(str).str[:3].astype(int)*10

	#generating 1 dataframe by selecting decades and grouping by station name
	decade_1 = master_df[(master_df['decade'] >= 1910) & (master_df['decade'] <= 1960) ]
	grouped_df_1 = decade_1.groupby(['station','latitude','longitude','elevation'],as_index=False).agg({'tmax':'mean'})

	#generating 2nd dataframe by selecting decades and grouping by station name
	decade_2 = master_df[(master_df['decade'] > 1960) & (master_df['decade'] <= 2010)]
	grouped_decade_2 = decade_2.groupby(['station','latitude','longitude','elevation'],as_index=False).agg({'tmax':'mean'})

	#finding difference betweeen the tmax of 2 dataframes
	diff_df = pd.merge( grouped_decade_2, grouped_df_1, how='inner', on=['station','longitude','latitude','elevation'])
	diff_df['temp_diff'] = diff_df['tmax_x']-diff_df['tmax_y']

	#plot graph
	layout=dict(title="Average Temperature Change in stations, between 1910 and 1960 and 1960 and 2010",
            geo=dict(landcolor='rgb(220,220,220)')
           )
	fig = go.Figure(data=go.Scattergeo(lon = diff_df['longitude'],lat = diff_df['latitude'],mode = 'markers',
                                   marker={'showscale':True,'color':diff_df['temp_diff'],'cmin':diff_df['temp_diff'].min(),
                                           'cmax':diff_df['temp_diff'].max(),
                                           'colorbar':{'title':"Temperature Difference"}}),layout=layout)
	fig.show()

	plot(fig, filename='Temperature_change.html')


def generate_heatMap():
	#this file was generated from Spark and saved as a Csv
	master_df = pd.read_csv('FinalFile.csv', sep=',', quotechar='"', error_bad_lines=False)

	#defining plot characteristics for heat map
	data = [dict(
	    type='scattergeo',
	    lon=master_df['longitude'],
	    lat=master_df['latitude'],
	    text=master_df['prediction'],
	    mode="markers",
	    marker=dict(
	    color=master_df['prediction'],
	    colorscale="temps",
	    opacity=0.4,
	    size=2,
	        colorbar=dict(
	        titleside="right",outlinecolor="rgba(68,68,68,0)")
	    )
	    )
	]

	layout = dict(
	    title="Density heatmap showing the predicted max temperature at each latitude and longitude",
	    geo = dict(
	        scope="world",
	        projection=dict(type="natural earth"),
	        showland=True,
	        landcolor="rgb(250,250,250)",
	        subunitcolor="rgb(217,217,217)",
	        countrycolor="rgb(217,217,217)",
	        countrywidth=0.5,
	        subunitwidth=0.5
	    ),
	)

	fig = dict(data=data,layout=layout)
	#the maps take time load and may cause jupyter to crash, hence saving it as an html to open in a different browser window
	plot(fig, filename='fig.html') 



def generate_regErrorMap():
	# showing the regression error (part b2)
	# this file was generated using Spark and saved as a csv
	testError_df = pd.read_csv('Test_error.csv', sep=',', quotechar='"', error_bad_lines=False)


	#defining plot characteristics for map
	data = [dict(
	    type='scattergeo',
	    lon=testError_df['longitude'],
	    lat=testError_df['latitude'],
	    text=testError_df['reg_error'],
	    mode="markers",
	    marker=dict(
	    color=testError_df['prediction'],
	    colorscale="temps",
	    size=2,
	        colorbar=dict(
	        titleside="right",outlinecolor="rgba(68,68,68,0)")
	    )
	)]

	layout=dict(
	    title="Error between predicted tmax values and test tmax values",
	     geo = dict(
	        scope="world",
	        projection=dict(type="equirectangular"),
	         showland=True,
	     ),
	)

	fig = dict(data=data,layout=layout)
	
	#the maps take time load and may cause jupyter to crash, hence saving it as an html to open in a different browser window
	plot(fig, filename='Reg_Error.html') 



if __name__ == '__main__':

	#generate first plot 
	global_tempChange()
	#generate heatmap
	generate_heatMap()
	#genrate regression error map
	generate_regErrorMap()


