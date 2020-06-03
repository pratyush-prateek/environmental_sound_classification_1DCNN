#Module to process the audio data into frames of equal lengths with a given overlapping percentage
import os
import numpy as np
import librosa

def make_frames(filename,folder,frame_length,overlapping_fraction):
	''' takes the .wav file name, frame length and overlapping percentage and returns numpy arrays of frames and classes'''
	class_id = filename.split('-')[1]
	filename = './dataset/audio'+'/'+folder + '/'+filename
	data,sample_rate = librosa.load(filename,sr=16000)
	stride = int((1-overlapping_fraction)*frame_length)
	num_frames = int((len(data)-frame_length)/stride)+1
	temp = np.array([data[i*stride:i*stride+frame_length] for i in range(num_frames)])
	if(len(temp.shape)==2):
		res = np.zeros(shape=(num_frames,frame_length+1),dtype=np.float64)
		res[:temp.shape[0],:temp.shape[1]] = temp
		res[:,frame_length]=np.array([class_id]*num_frames)
		return res

def make_frames_folder(folders,frame_length,overlapping_fraction):
	''' takes a list of folders and makes frames for all audio files in that folder'''
	data = []
	for folder in folders:
		files = os.listdir('./dataset/audio'+'/'+folder)
		for file in files:
			res = make_frames(file,folder,frame_length,overlapping_fraction)
			if res is not None:
				data.append(res)
	dataset = data[0]
	for i in range(1,len(data)):
		dataset = np.vstack((dataset,data[i]))
	return dataset	




