# -*- coding: utf-8 -*-
# @Author: Najeeb Qazi
# @Date:   2020-01-25 12:50:51
# @Last Modified by:   najeebq
# @Last Modified time: 2020-01-26 12:05:02
import sys
import elevation_grid as eg
import pandas as pd
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
import numpy as np
from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('tmax model tester').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import griddata
#import plotly.graph_objects as go
#from scipy.interpolate import griddata
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator


tmax_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.DateType()),
    types.StructField('latitude', types.FloatType()),
    types.StructField('longitude', types.FloatType()),
    types.StructField('elevation', types.FloatType()),
    types.StructField('tmax', types.FloatType()),
])

elev_fname = 'elevations_latlon.npy'
tile_size = 100
tile_degrees = 10


import gzip
try:
    try:
        fh = gzip.open(elev_fname+'.gz','rb')
    except:
        fh = open(elev_fname,'rb')
    elevgrid = np.load(fh)
    fh.close()
except:
    print("Warning: There was a problem initializing the elevation array from {}[.gz]".format(elev_fname))
    print("         Consider to run make_elevation_grid()")


def get_elevations(latlons):
    """For latlons being a N x 2 np.array of latitude, longitude pairs, output an
       array of length N giving the corresponding elevations in meters.
    """
    lli = ((latlons + (90,180))*(float(tile_size)/tile_degrees)).astype(int)
    return elevgrid[lli[:,0],lli[:,1]]

def get_elevation(lat, lon, get_elevations=get_elevations):
    """Lookup elevation in m"""
    return get_elevations(np.array([[lat,lon]]))[0]


#function produced the grid file in csv format, which is later used to plot using pandas and plotyl
def make_testfile():

	#produce lats and lons, pass to elevation function
	lats, lons = np.meshgrid(np.arange(-90,90,.5),np.arange(-180,180,.5))
	latlons = np.stack([lats.flatten(), lons.flatten()]).T;
	latlons = np.concatenate([latlons, latlons+.1])
	
	elevs = [eg.get_elevations(np.array([late,lone]).T) for late,lone in zip(lats,lons)]
	columns = ["station","latitude","longitude","elevation","tmax"]
	cols = ["latitude","longitude","elevation"]


	#flatten the latitudes and longitudes
	lats_flattened = np.reshape(lats.flatten(), (len(lats.flatten()),1))
	lons_flattened = np.reshape(lons.flatten(), (len(lons.flatten()),1))

	#flatten the elevation list
	elevs_arr = np.array(elevs)
	elevs_flattened = np.reshape(elevs_arr.flatten(), (len(elevs_arr.flatten()),1))

	#combine all the three
	combined_arr = np.concatenate([lats_flattened, lons_flattened, elevs_flattened],axis=1)

	combined_df = pd.DataFrame(combined_arr, columns=['latitude', 'longitude', 'elevation'])

	length_of_DF= len(combined_df)
	station_df = pd.DataFrame(np.arange(length_of_DF), columns=['station'])
	station_df["station"] = "SFU"
	df_date = pd.DataFrame(np.arange(length_of_DF), columns=['date'])
	df_date['date'] = datetime.datetime.now().date()
	tmax_df = pd.DataFrame(np.arange(length_of_DF), columns=['tmax'])
	tmax_df['tmax'] = 0.0
	combined_df =  pd.concat([station_df, df_date, combined_df, tmax_df], axis=1)
	
	return combined_df




def test_model(model_file, inputs):

	test_tmax = spark.createDataFrame(inputs, schema = tmax_schema)
	
	# load the model
	model = PipelineModel.load(model_file)
	# use the model to make predictions
	predictions = model.transform(test_tmax)
	#evaluate the predictions
	r2_evaluator = RegressionEvaluator(predictionCol='prediction',labelCol='tmax',metricName='r2')
	r2 = r2_evaluator.evaluate(predictions)
	rmse_evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='tmax',metricName='rmse')
	rmse = rmse_evaluator.evaluate(predictions)
	print('r2 =', r2)
	print('rmse =', rmse)

	# If you used a regressor that gives .featureImportances, maybe have a look..
	print(model.stages[-1].featureImportances)
	predictions = predictions.drop('features')
	
	return predictions

#function produces the regression error between model and tmax-test
def model_againstTest(model_file,inputs):
	test_tmax = spark.read.csv(inputs, schema=tmax_schema)
	model = PipelineModel.load(model_file)
	# use the model to make predictions
	predictions = model.transform(test_tmax)
	predictions = predictions.withColumn("reg_error", predictions['prediction'] - predictions['tmax'])
	return predictions
    



if __name__ == '__main__':
    model_file = sys.argv[1]
    #tmax test file is inputs 
    inputs = sys.argv[2]
    
    #make the file for the grid
    combined_df = make_testfile()
    #run the predictions
    predictions = test_model(model_file,combined_df)
    #convert to a pandas dataframe
    predictions = predictions.toPandas()    
    predictions.to_csv("FinalFile.csv", index = None, header=True)
    print("...File Generated for heat map modeling")
    test_error_df =  model_againstTest(model_file,inputs)
    test_error_df = test_error_df.toPandas()    
    test_error_df.to_csv("Test_error.csv", index = None, header=True)
    print("...File Generated for regression error map")










