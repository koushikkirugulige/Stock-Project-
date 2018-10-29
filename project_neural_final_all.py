#STOCK MARKET PREDICTION USING HISTORIC DATA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from keras import optimizers
import datetime
from operator import itemgetter
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Reshape,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv3D
from keras.layers.recurrent import LSTM
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.advanced_activations import LeakyReLU, PReLU,ELU,ThresholdedReLU
from keras import *
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as graph
import csv
import requests
import operator
import sys
import numpy as np
import os
from sklearn import metrics
from datetime import date
from datetime import datetime
import shutil
from datetime import timedelta
from optparse import OptionParser, SUPPRESS_HELP
from keras.models import model_from_json


#Directory to be used for storing stock data
master_directory="./stocks"

#Used to scan the command-line arguments for options and values
def generate_opt_parser():
	usage_format = "usage: %prog [options] arg1 arg2"
	parser = OptionParser(usage=usage_format)
	parser.add_option("-n","--name",action="store",type="string",dest="stock_name",default="EMPTY",help="Name of the stock")
	parser.add_option("-e","--epoch",action="store",type="int",dest="epoch",default=300,help="Number of epochs for training the model")
	parser.add_option("-b","--batch_size",action="store",type="int",dest="batch_size",default=16,help="Batch Size for training model")
	parser.add_option("-r","--reload_data",action="store_true",dest="reload_data",default=False,help="Reloads training data and model")
	parser.add_option("-t","--retrain_model",action="store_true",dest="retrain_model",default=False,help="Retrains the model")
	parser.add_option("-s","--test_size",action="store",type="int",dest="test_size",default=30,help="Number of days to test the model with")
	parser.add_option("-a","--sample_case",action="store_true",dest="sample_mode",default=False,help="Show Sample Results(Demo)")
	parser.add_option("-x","--train_size",action="store",type="int",dest="train_size",default=365,help="Adjust training set size")
	
	(options,args)=parser.parse_args()
	
	#If no stock has been given in the commandline, then default to DEMO mode
	if options.stock_name=="EMPTY":
		options.sample_mode=True
		
	#If the model has to be retrained
	if options.retrain_model==True:
		model_loc_json=master_directory+"/"+str(options.stock_name)+"/model/"+str(options.stock_name)+".json"
		model_loc_h5=master_directory+"/"+str(options.stock_name)+"/model/"+str(options.stock_name)+".h5"
		if(os.path.exists(model_loc_json)):
			os.remove(model_loc_json)
		if(os.path.exists(model_loc_h5)):
			os.remove(model_loc_h5)

	#If the data of the stock has to be downloaded again	
	if options.reload_data==True:
		name=master_directory+"/"+str(options.stock_name)
		if os.path.exists(name):	
			shutil.rmtree(name)
		
	return (options,args)	

(options,args)=generate_opt_parser()
test_size=options.test_size
train_size=options.train_size

#Used to get the next market-working day after 'date'
def get_next_date(date):
	today=datetime.strptime(date,"%Y-%m-%d")
	today+=timedelta(1)
	if today.isoweekday() in set((6,7)):
		today=today+timedelta(8-today.isoweekday())
	return str(today.strftime("%Y-%m-%d"))

#Used to create the directory structure for the stock
#If the Directory already exists, no change is made	
def construct_path(stock_name):
	name=master_directory+"/"+str(stock_name)
	if os.path.exists(name) and os.path.exists(name+"/data") and os.path.exists(name+"/output_image") and os.path.exists(name+"/model"):				
		return 
	if os.path.exists(name):	
		shutil.rmtree(name)
	os.mkdir(name)
	os.mkdir(name+"/data")
	os.mkdir(name+"/output_image")
	os.mkdir(name+"/model")

#Fetches stock data from the internet
#Places the data in the relevant folder.If the file already exists, it is NOT redownloaded	
def populate_data(stock_name):
	name=str(stock_name)
	path_data=master_directory+"/"+name+"/data"
	file_path=path_data+"/"+name+"_details.csv"
	CSV_URL='https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=4QTXZKSRGCINTK64&datatype=csv'%(name)
	if os.path.exists(file_path):
		return
	with requests.Session() as s:
		download = s.get(CSV_URL)	
	decoded_content = download.content.decode('utf-8')
	cr = csv.reader(decoded_content.splitlines(), delimiter=',')
	cr=sorted(cr,key=operator.itemgetter(0))
	my_list = list(cr)
	f1=open(file_path,"w")
	for i in range(0,len(my_list)-1):		
		if my_list[i][3]!='0.0000':
			f1.write(my_list[i][0]+","+my_list[i][1]+","+my_list[i][2]+","+my_list[i][3]+","+my_list[i][4]+","+my_list[i][5]+"\n")
	f1.close()

#The directory of the stock data, the hstoric data are all populated	
def preprocess(stock_name):
	name=str(stock_name)
	if os.path.exists(master_directory)==False :
		os.mkdir("stocks")
	construct_path(stock_name)
	populate_data(stock_name)	
		
#This function returns the required column of data 
#The column names are hashed to their respective index values
def get_data(stock_name,column):
		column_id={"Date": 0,
			   "Open": 1,
			   "High": 2,
			   "Low": 3,
			   "Close": 4,
			   "Volume": 5}
		name=str(stock_name)
		path_data=master_directory+"/"+name+"/data/"+name+"_details.csv"
		table=pd.read_csv(path_data,header=None)
		return table[column_id[column]]

#Function to generate and compile an instance of the defined model
def build():
	d=0.2
	model=Sequential()
	model.add(LSTM(128,input_shape=(1,1),return_sequences=True))
	#model.add(Dropout(d))
	model.add(LSTM(64,input_shape=(1,1),return_sequences=False))
	#model.add(Dropout(d))
	#model.add(LSTM(32,input_shape=(1,1),return_sequences=False))
	#model.add(Dropout(d))
	model.add(Dense(16,kernel_initializer='uniform',activation='relu'))
	model.add(Dense(1,kernel_initializer='uniform',activation='linear'))
	model.compile(loss='mse',optimizer='adam',metrics=[])
	return model		

#This function returns the model of a stock that has been trained on historic data
#If the model already exists, then it is not retrained. 
#The trained model is also stored before being returned			
def get_model(stock,train_x,train_y):
	stock_name=str(stock)
	model_loc_json=master_directory+"/"+stock_name+"/model/"+stock_name+".json"
	model_loc_h5=master_directory+"/"+stock_name+"/model/"+stock_name+".h5"
	if os.path.exists(model_loc_json) and os.path.exists(model_loc_h5):	
		json_file = open(model_loc_json, 'r')
		loaded_model_json=json_file.read()
		json_file.close()
		model=model_from_json(loaded_model_json)
		model.load_weights(model_loc_h5)
		print("Loaded an existing model for "+stock_name)
		return model
	
	model=build()
	train_x=train_x.reshape(train_x.shape[0],1,1)
	print("Training "+str(stock))
	model.fit(train_x,train_y,batch_size=options.batch_size,epochs=options.epoch,validation_split=0.1,verbose=1)	
	model_json=model.to_json()
	with open(model_loc_json,"w") as json_file:
		json_file.write(model_json)
	model.save_weights(model_loc_h5)
	return model
	
#This function returns the predicted values based on the model and set of values sent	
def predict_model_val(model,predict_x):
	predict_x=predict_x.reshape(predict_x.shape[0],1,1)
	result=model.predict(predict_x)
	result=np.array(result)
	return result

#This function is used to select the last 'num' number of entries
def downsize(data,num):
	data=data[data.shape[0]-num:data.shape[0]]
	return data

def get_arr(a):
	p=[]
	for i in range(0,a.shape[0]):
		p.append(a.item(i))
	return p

#This function partitions the data-set into training and test data set
#The model is trained on the training data
#The model is tested and the result is pushed to a graph
#The accuracy of each data point is calculated and pushed to the 'Accuracy' graph
def get_output(stock_name,color_graph):
	preprocess(stock_name)
	data=get_data(stock_name,"Close")
	dates=get_data(stock_name,"Date")
	test_size=int(0.1*data.shape[0])
	graph_image_loc=master_directory+"/"+stock_name+"/output_image/"+stock_name+".png"	
	train_x=np.array(data[0:data.shape[0]-test_size])
	train_y=np.array(data[1:data.shape[0]-test_size+1])
	test=np.array(data[data.shape[0]-test_size:data.shape[0]-1])
	predict_x=np.array(test)
	predict_test_y=np.array(data[data.shape[0]-test_size+1:data.shape[0]])
	predict_date_y=np.array(dates[dates.shape[0]-test_size+1:dates.shape[0]])
	model=get_model(stock_name,train_y,train_x)
	predict_y=predict_model_val(model,predict_x)
	accuracy=np.array([])
	for i in range(0,len(test)):
		accuracy=np.append(accuracy,((predict_test_y[i]-abs(predict_test_y[i]-predict_y.item(i)))/predict_test_y[i])*100)
	graph.figure(stock_name)
	graph.plot(predict_y,label="prediction "+stock_name)
	graph.plot(predict_test_y,label="actual "+stock_name)
	graph.legend(loc='upper right')
	graph.savefig(graph_image_loc)
	#graph.figure("Accuracy")
	#graph.plot(predict_date_y,accuracy,label="accuracy "+stock_name,color=color_graph)
	#graph.legend(loc='upper right')	
	
	
if options.sample_mode==False: 		
	#For execution of a single stock
	stock_name=str(options.stock_name)
	get_output(stock_name,"blue")
else:
	#DEMO mode shows summary of all the stocks in the array 'stocks'
	stocks=["SBIN.NS","AAPL","MSFT","TATAMOTORS.NS","BHARTIARTL.NS","GOOGL"]
	colors=["blue","green","red","cyan","magenta","yellow","black"]
	for i in range(0,len(stocks)):
		get_output(stocks[i],colors[i])

#graph.figure("Accuracy")
#graph.savefig("./stocks/accuracy.png")		
graph.show()


