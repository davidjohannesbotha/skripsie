import numpy as np 
from numpy import array
from numpy import hstack
import tensorflow as tf
import numpy as np
import math
import pandas
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Bidirectional
import kerastuner


# after proving that more datapoints = better performance, my hypothotis is that if I increase the amount of data points in the code that I will achieve higher 
# generalization ability. This is simply due to lower gradients of extrapolation required..... MAYBEEE



# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


global_force_array = []
global_time_array = []
global_output_array = []

#for this example I will have 10 "tests" at 10 seconds of data at 500Hz...

force_input_array= []
force_input_value = 10
force_steps = 10/19
force_input_array.append(10)
for i in range(19):
    force_input_value= force_input_value + force_steps
    force_input_array.append(force_input_value)

## here I need to create a half-simusoid input. The input needs to be insanely sort relative to the response of the beam...

#The period of the response is about 0.7s as a benchmark, lets divide that value by 10... 0.07seconds 
time = np.linspace(0, 16/500, 352)
for x in range(len(force_input_array)):
    force = []
    force = np.append(force, force_input_array[x]*np.sin((125*math.pi/4)*time))
    global_force_array = np.append(global_force_array ,force)


time_array= []
time_value = 0
timesteps = 1/(500)
time_array.append(0)
for i in range(351):
    time_value = time_value+timesteps
    time_array.append(time_value)

for p in range(len(force_input_array)): #this creates a concatenated version that has enough
    global_time_array = global_time_array + time_array

print(len(global_time_array))
numpy_global_time_array= array(global_time_array)
numpy_global_time_array = numpy_global_time_array.reshape((len(numpy_global_time_array), 1))



for u in range(len(force_input_array)):
    for k in range(352): #this is for the times
        out_seq= (10/4471)*force_input_array[u]*math.exp(-0.1*time_array[k])*math.sin(4.4710*time_array[k]) 
        global_output_array.append(out_seq)


print(len(global_output_array))
numpy_global_output_array = array(global_output_array)
numpy_global_output_array =  numpy_global_output_array.reshape((len(numpy_global_output_array), 1))

global_force_array = global_force_array.reshape((len(global_force_array), 1))

##formatting of the dataset
dataset = hstack([numpy_global_time_array,numpy_global_output_array, global_force_array])

#formatting of the force datset
#this is done to pass to the minmax function and get the minmax of the input force alone
dataset_force = hstack([global_force_array])

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Contrive small dataset
dataset = dataset
# Calculate min and max for each column
minmax = dataset_minmax(dataset)
minmax_force = dataset_minmax(dataset_force)
#normalize the training dataset
normalize_dataset(dataset, minmax)

# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)


print(X,y)
#automatically generate the amount of features in the dataset
n_features = X.shape[2]
tf.keras.backend.clear_session()

#this testing sequence is to feed to the model
testing_sequence = []
for k in range(352): #this is for the times
    out_seq= (10/4471)*50*math.exp(-0.1*time_array[k])*math.sin(4.4710*time_array[k]) 
    testing_sequence.append(out_seq)

plt.plot(testing_sequence, label = "response data")

#The testing sequence however needs to be normalized on the same scale as was suplied to the network during training
testing_sequence = array(testing_sequence)
testing_sequence = testing_sequence.reshape((len(testing_sequence), 1))

time_array= array(time_array)
time_array = time_array.reshape((len(time_array), 1))

testingdataset= hstack([time_array, testing_sequence])

#normalize the dataset
normalize_dataset(testingdataset, minmax)

#the normalized dataset can now be passed to the model for prediction
time_array =testingdataset[:,0]
testing_sequence = testingdataset[:,1]



#in order for comparison, the true response is also plotted. This true response needs to be scaled as well (with the same weights of course)
testarray = []
for t in range(352):
	inp = 50*np.sin((125*math.pi/4)*time[t])
	testarray.append(inp)  

teststack = array(testarray)
teststack = teststack.reshape((len(teststack), 1))
teststack = hstack([teststack])
normalize_dataset(teststack, minmax_force)
#the normalized dataset can then be plotted on the same axis.
plt.plot(teststack, label='True')
plt.legend()



#this setup has proven to have the interpolation covered. 
#however, in order for this to be a successfull skripsie, I need to extrapolate. 
model = tf.keras.models.load_model("simple_add.h5")
results = []
for h in range((len(testing_sequence) -2)):
    x_input = array([[time_array[h], testing_sequence[h]], [time_array[h+1], testing_sequence[h+1]], [time_array[h+2] , testing_sequence[h+2]]])
    x_input = x_input.reshape((1, n_steps, n_features))
    #x_input = np.expand_dims(x_input, axis = 0)
    yhat = model.predict(x_input, verbose=0)
    print(yhat[0][0])
    results.append(yhat[0][0])

plt.plot(results, label='This is my prediction')
plt.legend()
plt.show()


model = tf.keras.models.load_model("simple_add_80.h5")
results = []
for h in range((len(testing_sequence) -2)):
    x_input = array([[time_array[h], testing_sequence[h]], [time_array[h+1], testing_sequence[h+1]], [time_array[h+2] , testing_sequence[h+2]]])
    x_input = x_input.reshape((1, n_steps, n_features))
    #x_input = np.expand_dims(x_input, axis = 0)
    yhat = model.predict(x_input, verbose=0)
    print(yhat[0][0])
    results.append(yhat[0][0])

plt.plot(results, label='This is my prediction simple_add_80')
plt.legend()




# define model
model = tf.keras.models.Sequential()

model.add(Bidirectional(tf.keras.layers.LSTM(80, return_sequences= True), input_shape = (n_steps, n_features)))
model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(X, y, epochs=400, verbose=0)
# demonstrate prediction
plt.plot(history.history['loss'], label='train')
model.save("80_return_seq_1.h5")
plt.show()

print("SAVED")
















#this setup has proven to have the interpolation covered. 
#however, in order for this to be a successfull skripsie, I need to extrapolate. 
model = tf.keras.models.load_model("simple_add.h5")
results = []
for h in range((len(testing_sequence) -2)):
    x_input = array([[time_array[h], testing_sequence[h]], [time_array[h+1], testing_sequence[h+1]], [time_array[h+2] , testing_sequence[h+2]]])
    x_input = x_input.reshape((1, n_steps, n_features))
    #x_input = np.expand_dims(x_input, axis = 0)
    yhat = model.predict(x_input, verbose=0)
    print(yhat[0][0])
    results.append(yhat[0][0])

plt.plot(results, label='This is my prediction')
plt.legend()



model = tf.keras.models.load_model("simple_add_80.h5")
results = []
for h in range((len(testing_sequence) -2)):
    x_input = array([[time_array[h], testing_sequence[h]], [time_array[h+1], testing_sequence[h+1]], [time_array[h+2] , testing_sequence[h+2]]])
    x_input = x_input.reshape((1, n_steps, n_features))
    #x_input = np.expand_dims(x_input, axis = 0)
    yhat = model.predict(x_input, verbose=0)
    print(yhat[0][0])
    results.append(yhat[0][0])

plt.plot(results, label='This is my prediction 80')
plt.legend()

model = tf.keras.models.load_model("simple_add_80_drop.h5")
results = []
for h in range((len(testing_sequence) -2)):
    x_input = array([[time_array[h], testing_sequence[h]], [time_array[h+1], testing_sequence[h+1]], [time_array[h+2] , testing_sequence[h+2]]])
    x_input = x_input.reshape((1, n_steps, n_features))
    #x_input = np.expand_dims(x_input, axis = 0)
    yhat = model.predict(x_input, verbose=0)
    print(yhat[0][0])
    results.append(yhat[0][0])

plt.plot(results, label='This is my simple_add_80_drop 80')
plt.legend()


plt.show()




#1 Dropout
# Start by dropout in the hidden layers
#Possibly add dropout on the dataset as well
#play twith the dropout rates.

#2 Increase the capacity of the NN. 
# - More neurons
# - More layers

#2 Add more data aand repeat step 1
#What data will I add??
# Start by adding a longer timeframe (ie twice as long), but still keeping 20 input forces.


#4 Add white noise.
#White noise can be added to the data in MATLAB. 
#As far as scaling goes I do not believe I need any additional scaling for this model.'

for k in range(350): #this is for the times
    out_seq= (10/4471)*10*math.exp(-0.1*time_array[k])*math.sin(4.4710*time_array[k]) 
    testing_sequence.append(out_seq)


model = tf.keras.models.load_model("inverse_ltsm_1_350steps.h5")
results = []
for h in range((len(testing_sequence) -2)):
    x_input = array([[time_array[h], testing_sequence[h]], [time_array[h+1], testing_sequence[h+1]], [time_array[h+2] , testing_sequence[h+2]]])
    x_input = x_input.reshape((1, n_steps, n_features))
    #x_input = np.expand_dims(x_input, axis = 0)
    yhat = model.predict(x_input, verbose=0)
    print(yhat[0][0])
    results.append(yhat[0][0])

plt.plot(results)
print(results)



model = tf.keras.models.load_model("22_step_ligthart.h5")
results = []
for p in range(16):  
    x_input = []
    for h in range(22):
        x_input.append([time_array[p+h], testing_sequence[p + h]]) 
    x_input = array([x_input])
    #print(x_input)
    x_input = x_input.reshape((1, n_steps, n_features))

    yhat = model.predict(x_input, verbose=0)
    results.append(yhat[0][0])

plt.plot(results, label = "22_step_ligthart")

plt.show()






model = tf.keras.models.load_model("335_step_16neuron_1000epoch.h5")
results = []
for p in range(16):  
    x_input = []
    for h in range(335):
        x_input.append([time_array[p+h], testing_sequence[p + h]]) 
    x_input = array([x_input])
    #print(x_input)
    x_input = x_input.reshape((1, n_steps, n_features))

    yhat = model.predict(x_input, verbose=0)
    results.append(yhat[0][0])

plt.plot(results, label = "current model 25 percent higher")


model = tf.keras.models.load_model("335_step_10neuron_750epoch_nodrop.h5")
results = []
for p in range(16):  
    x_input = []
    for h in range(335):
        x_input.append([time_array[p+h], testing_sequence[p + h]]) 
    x_input = array([x_input])
    #print(x_input)
    x_input = x_input.reshape((1, n_steps, n_features))

    yhat = model.predict(x_input, verbose=0)
    results.append(yhat[0][0])

plt.plot(results, label = "335_step_10neuron_750epoch_nodrop")

model = tf.keras.models.load_model("335_step_32neuron_1000epoch_H2l.h5")
results = []
for p in range(16):  
    x_input = []
    for h in range(335):
        x_input.append([time_array[p+h], testing_sequence[p + h]]) 
    x_input = array([x_input])
    #print(x_input)
    x_input = x_input.reshape((1, n_steps, n_features))

    yhat = model.predict(x_input, verbose=0)
    results.append(yhat[0][0])

plt.plot(results, label = "335_step_32neuron_1000epoch_H2l")





