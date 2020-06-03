#Keras implementation of 1D CNN model
#import libraries
import tensorflow as tf
import keras
from keras.models import Model,Sequential
from keras.layers import Conv1D,MaxPooling1D,Dense,Flatten,BatchNormalization,Dropout, Activation
from model_config import *
from collections import OrderedDict

#To add modularity later on, first the default model
class EnvNet(Model):
	def __init__(self,classes,config):
		super(EnvNet,self).__init__()
		if(config==1):
			self.feature_extractor = FeatureBlock1()
		elif(config==2):
			self.feature_extractor = FeatureBlock2()
		elif(config==3):
			self.feature_extractor = FeatureBlock3()
		elif(config==4):
			self.feature_extractor = FeatureBlock4()
		elif(config==5):
			self.feature_extractor = FeatureBlock5()
		else:
			self.feature_extractor = FeatureBlock6()															
		self.flatten = Flatten()
		self.FC1 = Dense(units=128)
		self.relu1 = Activation(activation="relu")
		self.dropout1 = Dropout(rate=0.25)
		self.FC2 = Dense(units=64)
		self.relu2 = Activation(activation="relu")
		self.dropout2 = Dropout(rate=0.25)
		self.FC3 = Dense(classes)
		self.outputs = Activation(activation="softmax")

	def call(self,inputs):
		x = self.feature_extractor.call(inputs)
		x = self.flatten(x)
		x = self.FC1(x)
		x = self.relu1(x)
		x = self.dropout1(x)
		x = self.FC2(x)
		x = self.relu2(x)
		x = self.dropout2(x)
		x = self.FC3(x)
		x = self.outputs(x)
		return x

		
