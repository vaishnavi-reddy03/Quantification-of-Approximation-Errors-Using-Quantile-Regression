import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split


'''
Follows the same structure as ANN_Adam. Difference being it used pinball loss criterion instead of MSE
'''

class ErrorDataset(Dataset):
    def __init__(self, data_file, scaler=None):

        #Instead of next state, the target is the error retrived from the error dataset
        self.data_df = pd.read_csv('error_mapping.csv')

        self.state_df = self.data_df[['0','1','2','3']]
        self.inputs_df = self.data_df[['4','5']]
        self.error_df = self.data_df[['error_0','error_1',"error_2",'error_3']]
        
        # Use StandardScaler for normalization
        if scaler is None:
            self.state_df_scaler = StandardScaler()
            self.inputs_df_scaler = StandardScaler()
            self.error_df_scaler = StandardScaler()
            self.state = self.state_df_scaler.fit_transform(self.state_df)
            self.inputs = self.inputs_df_scaler.fit_transform(self.inputs_df)
            self.error = self.error_df_scaler.fit_transform(self.error_df)
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

'''
Custom loss function in pytorch
'''
class pinballLoss(nn.Module):
    def __init__(self):
        super(pinballLoss, self).__init__()

    def forward(self,predictions,targets,tau):
        self.error = targets - predictions
        self.loss = torch.max(tau * self.error, (tau - 1) * self.error)
        return torch.mean(self.loss)


class ANNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, output_dim)

        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(self.leaky_relu(self.fc2(x)))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=25, early_stopping_patience=5):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = 0
        for inputs, targets in train_dataloader:
            inputs = torch.tensor(inputs, dtype=torch.float32)
            targets = torch.tensor(targets, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets,tau)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs = torch.tensor(inputs, dtype=torch.float32)
                targets = torch.tensor(targets, dtype=torch.float32)
                outputs = model(inputs)
                loss = criterion(outputs, targets, tau)
                val_loss += loss.item()

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break


if __name__ == "__main__":
    data_df = pd.read_csv('error_mapping.csv')

    state_df = data_df[['0','1','2','3']]
    inputs_df = data_df[['4','5']]
    error_df = data_df[['error_0','error_1',"error_2",'error_3']]

    scaler_state = StandardScaler().fit(state_df)
    scaler_inputs = StandardScaler().fit(inputs_df)
    scaler_error = StandardScaler().fit(error_df)

    train_dataset = ErrorDataset('error_mapping.csv', scaler=(scaler_state,scaler_inputs,scaler_error))


    # Split dataset into training and validation sets without shuffling
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(train_dataset)))

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

   
    input_dim = 6
    output_dim = 4
    model = ANNModel(input_dim, output_dim)

    #redefine tau as per the required quantile
    tau = 0.9
    criterion = pinballLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)  

    train_model(model, train_dataloader, val_dataloader, optimizer, criterion, epochs = 100)

    # Save the model and the scalers
    torch.save(model.state_dict(), 'ann_model_quantile_90.pth')
    np.save('scaler_state_quantile.npy', scaler_state.mean_)
    np.save('scaler_state_quantile_scale.npy', scaler_state.scale_)
    np.save('scaler_inputs_quantile.npy', scaler_inputs.mean_)
    np.save('scaler_inputs_quantile_scale.npy', scaler_inputs.scale_)
    np.save('scaler_error_quantile.npy', scaler_error.mean_)
    np.save('scaler_error_quantile_scale.npy', scaler_error.scale_)

    # Denormalize the model predictions for plotting
    scaler_outputs = scaler_error
    model_predictions = []
    true_values = []

    model.eval()
    with torch.no_grad():
         for inputs, targets in val_dataloader:
             inputs = torch.tensor(inputs, dtype=torch.float32)
             outputs = model(inputs).numpy()
             model_predictions.append(outputs)
             true_values.append(targets.numpy())

    model_predictions = np.vstack(model_predictions)
    true_values = np.vstack(true_values)
    
    #outputs_np_denorm = model_predictions * scaler_outputs.scale_ + scaler_outputs.mean_
    outputs_np_denorm = scaler_error.inverse_transform(model_predictions)
    true_values = true_values * scaler_outputs.scale_ + scaler_outputs.mean_


    #Plotting the results
    plt.figure(figsize=(12, 6))
    y_labels = ['C_A', 'C_B', 'T_R', 'T_J']

    for i in range(output_dim):
        plt.subplot(output_dim, 1, i + 1)
        plt.plot(true_values[:10000, i], label='True Value')
        plt.plot(outputs_np_denorm[:10000, i], label='Predicted Value', linestyle='--')
        plt.xlabel('Sample Index')
        plt.ylabel(y_labels[i])
        plt.legend()
    
    plt.tight_layout()
    plt.show()
