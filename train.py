#Training the model and outputing the metrics
import keras
import tensorflow as tf
import numpy as np
from keras.callbacks import LambdaCallback 

#train the model for a single fold
def train_model(model,loss,optimizer,train_data,num_epochs,batch_size):
	model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
	X_train = train_data['x_train']
	Y_train = train_data['y_train']  
	#print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()[0].shape))
	history = model.fit(X_train,Y_train,batch_size=batch_size,epochs=num_epochs,validation_split=0.11)
	return history


