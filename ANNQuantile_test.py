import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ANNQuantile import ANNModel


'''
Same structure as ANNQuantile.py
'''
class ErrorDataset(Dataset):
    def __init__(self, data_file, scaler=None):
        self.data_df = pd.read_csv('error_mapping_test.csv')

        self.state_df = self.data_df[['0','1','2','3']]
        self.inputs_df = self.data_df[['4','5']]
        self.error_df = self.data_df[['error_0','error_1',"error_2",'error_3']]
        
        # Use StandardScaler for normalization
        if scaler is None:
            self.state_df = StandardScaler()
            self.inputs_df = StandardScaler()
            self.error_df = StandardScaler()
            self.state_df = self.state_df.fit_transform(self.state_df)
            self.inputs_df = self.inputs_df.fit_transform(self.inputs_df)
            self.error_df = self.error_df.fit_transform(self.error_df)
        else:
            self.scaler_state, self.scaler_inputs, self.scaler_error = scaler
            self.state = self.scaler_state.transform(self.state_df)
            self.inputs = self.scaler_inputs.transform(self.inputs_df)
            self.error = self.scaler_error.transform(self.error_df)

        #self.data = np.hstack((self.inputs[:-1], self.error[:-1]))
        self.data = np.hstack((self.state, self.inputs))
        self.targets = self.error
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class pinballLoss(nn.Module):
    def __init__(self):
        super(pinballLoss, self).__init__()

    def forward(self,predictions,targets,tau):
        self.error = targets - predictions
        self.loss = torch.max(tau * self.error, (tau - 1) * self.error)
        return torch.mean(self.loss)


'''
Calculates accuracy for pinball loss. 
Eg: If there are 100 samples and we train the mode for tau=0.9, 10 samples should be greater than the predicted value.
'''
def calculate_above_predictions(true_values, predicted_values):
    num_above = np.sum(true_values > predicted_values)
    return num_above

if __name__ == "__main__":
    # Load scalers
    scaler_state = StandardScaler()
    scaler_state.mean_ = np.load('scaler_state_quantile.npy')
    scaler_state.scale_ = np.load('scaler_state_quantile_scale.npy')

    scaler_inputs = StandardScaler()
    scaler_inputs.mean_ = np.load('scaler_inputs_quantile.npy')
    scaler_inputs.scale_ = np.load('scaler_inputs_quantile_scale.npy')

    scaler_error = StandardScaler()
    scaler_error.mean_ = np.load('scaler_error_quantile.npy')
    scaler_error.scale_ = np.load('scaler_error_quantile_scale.npy')

    scaler = (scaler_state, scaler_inputs, scaler_error)

    # Load test dataset
    test_dataset = ErrorDataset('error_mapping_test', scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    input_dim = 6
    output_dim = 4
    model = ANNModel(input_dim, output_dim)
    model.load_state_dict(torch.load('ann_model_quantile_10.pth'))
    model.eval()


    # Visualize the predictions
    total_samples = len(test_dataset)
    data_iter = iter(DataLoader(test_dataset, batch_size=total_samples, shuffle=False))
    inputs, targets = next(data_iter)

    inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = model(inputs)
    


    inputs_np = inputs.numpy()
    targets_np = targets.numpy()
    outputs_np = outputs.detach().numpy()

    # Denormalize the outputs for comparison
    outputs_np_denorm = scaler_error.inverse_transform(outputs_np)
    true_values = scaler_error.inverse_transform(targets_np)


    plt.figure(figsize=(12, 6))
    y_labels = ['C_A', 'C_B', 'T_R', 'T_J']


    i = 0
    len = 500
    plt.xlabel("Time")
    plt.ylabel("C_A")
    plt.plot(true_values[:len, i], 'r+',label='True Value')
    plt.plot(outputs_np_denorm[:len, i], label='Predicted Value', linestyle='--')
    plt.legend()
    plt.show()
