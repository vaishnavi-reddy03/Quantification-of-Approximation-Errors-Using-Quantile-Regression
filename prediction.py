import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ANN_Adam import ANNModel
import pandas as pd


'''
Predicts the open-loop/multistep prediction for the trained ANN model
'''

states = pd.read_csv('states.csv').drop(columns=['Unnamed: 0'])
inputs = pd.read_csv('inputs.csv').drop(columns=['Unnamed: 0'])

scaler_states = StandardScaler()
scaler_states.mean_ = np.load('scaler_states.npy')
scaler_states.scale_ = np.load('scaler_states_scale.npy')

scaler_inputs = StandardScaler()
scaler_inputs.mean_ = np.load('scaler_inputs.npy')
scaler_inputs.scale_ = np.load('scaler_inputs_scale.npy')

states_normalized = scaler_states.transform(states)
inputs_normalized = scaler_inputs.transform(inputs)

initial_state = states_normalized[0]
initial_input = inputs_normalized[0]

input_dim = 6
output_dim = 4


model = ANNModel(input_dim, output_dim)
model.load_state_dict(torch.load('ann_model.pth'))
model.eval()


def generate_future_predictions(model, initial_state, initial_input, future_steps):
    model.eval()
    predictions = []
    current_input = np.hstack((initial_state, initial_input))
    input_tensor = torch.FloatTensor(np.array([current_input]))
    for i in range(future_steps):
        with torch.no_grad():
            output = model(input_tensor)
            predictions.append(output.numpy().flatten())
            next_state = output.numpy().flatten()
            
            #creates the input data such that the states are taken from the output of the model and inputs from the csv dataset
            current_input = np.hstack((next_state, inputs_normalized[i+1]))  
            input_tensor = torch.FloatTensor(np.array([current_input]))
    return np.array(predictions)

#number of steps to predict 
future_steps = 200
predictions = generate_future_predictions(model, initial_state, initial_input, future_steps)


predictions_denorm = scaler_states.inverse_transform(predictions)  #denormalize

targets = states.to_numpy()

true_values = []
true_values.append(targets)
true_values = np.vstack(true_values)

plt.figure(figsize=(12, 8))
y_labels = ['C_A', 'C_B', 'T_R', 'T_J']
for i in range(output_dim):
    plt.subplot(output_dim, 1, i + 1)
    plt.plot(predictions_denorm[:, i], label='Predicted Future Values')
    plt.plot(true_values[:future_steps, i], label='True Value')
    plt.xlabel('Time')
    plt.ylabel(y_labels[i])
    plt.legend()
plt.tight_layout()
plt.show()
