import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ANN_Adam import CSTRDataset, ANNModel
from ANNQuantile import ANNModel as ANNModelQ


'''
Plots everything together. Also contains the conformal prediction part
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


if __name__ == "__main__":
    # Load scalers
    scaler_states = StandardScaler()
    scaler_states.mean_ = np.load('scaler_states.npy')
    scaler_states.scale_ = np.load('scaler_states_scale.npy')

    scaler_inputs = StandardScaler()
    scaler_inputs.mean_ = np.load('scaler_inputs.npy')
    scaler_inputs.scale_ = np.load('scaler_inputs_scale.npy')

    scaler = (scaler_states, scaler_inputs)

    # Load test dataset
    test_dataset = CSTRDataset('states_test.csv', 'inputs_test.csv', scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model for state prediction
    input_dim = 6
    output_dim = 4
    model = ANNModel(input_dim, output_dim)
    model.load_state_dict(torch.load('ann_model.pth'))
    model.eval()

    # Test the model
    criterion = nn.MSELoss()

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
    outputs_np_denorm = scaler_states.inverse_transform(outputs_np)
    true_values = scaler_states.inverse_transform(targets_np)


#################################################################################

    scaler_error = StandardScaler()
    scaler_error.mean_ = np.load('scaler_error_quantile.npy')
    scaler_error.scale_ = np.load('scaler_error_quantile_scale.npy')

    scaler = (scaler_states, scaler_inputs, scaler_error)

    test_dataset = ErrorDataset('error_mapping_test', scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    #Load model for 90th quantile
    modelQ90 = ANNModelQ(input_dim, output_dim)
    modelQ90.load_state_dict(torch.load('ann_model_quantile_90.pth'))
    modelQ90.eval()

   
    total_samples = len(test_dataset)
    data_iter = iter(DataLoader(test_dataset, batch_size=total_samples, shuffle=False))
    inputs, targets = next(data_iter)

    inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = modelQ90(inputs)

    inputs_np = inputs.numpy()
    targets_np = targets.numpy()
    outputs_npQ90 = outputs.detach().numpy()

    outputs_np_denormQ90 = scaler_error.inverse_transform(outputs_npQ90)

    #Load model for 10th quantile
    modelQ10 = ANNModelQ(input_dim, output_dim)
    modelQ10.load_state_dict(torch.load('ann_model_quantile_10.pth'))
    modelQ10.eval()

    
    total_samples = len(test_dataset)
    data_iter = iter(DataLoader(test_dataset, batch_size=total_samples, shuffle=False))
    inputs, targets = next(data_iter)

    inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = modelQ10(inputs)

    inputs_np = inputs.numpy()
    targets_np = targets.numpy()
    outputs_npQ10 = outputs.detach().numpy()

    outputs_np_denormQ10 = scaler_error.inverse_transform(outputs_npQ10)

    uBound = outputs_np_denorm+outputs_np_denormQ90
    lBound = outputs_np_denorm+outputs_np_denormQ10

    
    ##########Conformal Prediction#################
    # Split data into calibration and test sets
    calibration_set_size = int(0.2 * len(test_dataset))
    calibration_set, test_set = torch.utils.data.random_split(test_dataset, [calibration_set_size, len(test_dataset) - calibration_set_size])

    # Get calibration predictions
    calibration_inputs, calibration_targets = next(iter(DataLoader(calibration_set, batch_size=len(calibration_set))))
    calibration_inputs = torch.tensor(calibration_inputs, dtype=torch.float32)
        
    with torch.no_grad():
        calibration_lower = modelQ10(calibration_inputs).numpy()
        calibration_upper = modelQ90(calibration_inputs).numpy()

    # Calculate nonconformity scores
    nonconformity_scores = np.maximum(calibration_targets.numpy() - calibration_upper, calibration_lower - calibration_targets.numpy())
    quantile_level = (1 - 0.80) / 2     #80% confidence
    nonconformity_quantile = np.quantile(nonconformity_scores, 1 - quantile_level)
    

    adjusted_lower = lBound - nonconformity_quantile
    adjusted_upper = uBound + nonconformity_quantile
    

    y_labels = ['C_A', 'C_B', 'T_R', 'T_J']
    i = 2
    len = 199

    plt.figure(figsize=(12, 6))
    for i in range(output_dim):
        plt.subplot(output_dim, 1, i + 1)
        
        #true values
        plt.plot(true_values[:len, i], label='True Value')
        
        #state predictions
        plt.plot(outputs_np_denorm[:len, i], label='Predicted Value', linestyle='--')
        
        #quantile regression
        plt.fill_between(
        range(len), lBound[:len, i], uBound[:len, i], color='r', alpha=0.3, label='Prediction Interval'
        )
       
        #conformalization
        plt.fill_between(
            np.arange(len), adjusted_lower[:len, i], adjusted_upper[:len, i], color='gray', alpha=0.2, label='Conformalized Prediction Interval'
        )
        plt.xlabel('Time')
        plt.ylabel(y_labels[i])
        plt.legend()
    
    plt.tight_layout()
    plt.show()

    


