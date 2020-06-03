#custom initializer for the convolutional layer
import tensorflow as tf
import numpy as np

def erb_scale_2_freq_hz(freq_erb):
    # Convert frequency on ERB scale to frequency in Hertz
    freq_hz = (np.exp(freq_erb/9.265)-1)*24.7*9.265
    return freq_hz

def freq_hz_2_erb_scale(freq_hz):
    # Convert frequency in Hertz to frequency on ERB scale
    freq_erb = 9.265*np.log(1+freq_hz/(24.7*9.265))
    return freq_erb

def normalize_filters(filterbank):
    # Normalizes a filterbank such that all filters
    # have the same root mean square (RMS).
    rms_per_filter = np.sqrt(np.mean(np.square(filterbank), axis=1))
    rms_normalization_values = 1. / (rms_per_filter/np.amax(rms_per_filter))
    normalized_filterbank = filterbank * rms_normalization_values[:, np.newaxis]
    return normalized_filterbank

def gammatone_impulse_response(samplerate_hz, length, center_freq_hz, phase_shift,p):
    # Generate single parametrized gammatone filter
    erb = 24.7 + 0.108*center_freq_hz # equivalent rectangular bandwidth
    divisor = (np.pi * np.math.factorial(2*p-2) * np.power(2, float(-(2*p-2))) )/ np.square(np.math.factorial(p-1))
    b = erb/divisor # bandwidth parameter
    a = 1.0 # amplitude. This is varied later by the normalization process.
    length_in_seconds = (1./samplerate_hz)*length
    t = np.linspace(1./samplerate_hz, length_in_seconds, length)
    gammatone_ir = a * np.power(t, p-1)*np.exp(-2*np.pi*b*t) * np.cos(2*np.pi*center_freq_hz*t + phase_shift)
    return gammatone_ir

def generate_filters(num_filters,length,samplerate_hz,min_center_freq,p):
	filterbank = np.zeros((num_filters,length))
	curr_center_freq = min_center_freq
	for i in range(num_filters):
		filterbank[i,:] = gammatone_impulse_response(samplerate_hz,length,curr_center_freq,0,p)
		curr_center_freq = erb_scale_2_freq_hz(freq_hz_2_erb_scale(curr_center_freq)+1)
	filterbank = normalize_filters(filterbank)
	return filterbank


class GammatoneInit(tf.keras.initializers.Initializer):
    def __init__(self,sample_rate,min_center_freq,order):
        self.sample_rate = sample_rate
        self.min_center_freq = min_center_freq
        self.order = order

    def __call__(self,shape,dtype=None):
        print(shape)
        filters = generate_filters(shape[2],shape[0],self.sample_rate,self.min_center_freq,self.order)
        filters = filters.reshape(filters.shape[1],1,filters.shape[0])
        return tf.Variable(filters,dtype=dtype)
        

