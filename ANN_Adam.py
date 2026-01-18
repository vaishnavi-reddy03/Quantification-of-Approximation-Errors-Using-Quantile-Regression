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
Takes the data as a Dataset object and preprocess it to be ready for training.
'''
class CSTRDataset(Dataset):
    def __init__(self, states_file, inputs_file, scaler=None):

        #Removing the header row from the csv file
        self.states = pd.read_csv(states_file).drop(columns=['Unnamed: 0'])     
        self.inputs = pd.read_csv(inputs_file).drop(columns=['Unnamed: 0'])
        
        #Some basic checks to check for invalid data
        assert not self.states.isnull().values.any(), "NaN values found in states data"
        assert not self.inputs.isnull().values.any(), "NaN values found in inputs data"
        assert np.isfinite(self.states.values).all(), "Infinite values found in states data"
        assert np.isfinite(self.inputs.values).all(), "Infinite values found in inputs data"

        #Using standard scalar to normalize the data
        if scaler is None:
            self.scaler_states = StandardScaler()
            self.scaler_inputs = StandardScaler()
            self.states = self.scaler_states.fit_transform(self.states)
            self.inputs = self.scaler_inputs.fit_transform(self.inputs)
        else:
            self.scaler_states, self.scaler_inputs = scaler
            self.states = self.scaler_states.transform(self.states)
            self.inputs = self.scaler_inputs.transform(self.inputs)

        #Stacking data to required format (x1,x2,x3,x4,u1,u2)
        self.data = np.hstack((self.states[:-1], self.inputs[:-1]))

        #Delaying the states by 1 so that the output to current data is the next time step data
        self.targets = self.states[1:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

'''
Defines the pytorch ANN model. Model consists of
    -input layer of 6 neurons
    -first hidden layer of 16 neurons with leaky relu activation
    -second hidden layer of 32 neurons with dropout of 0.5 and leaky relu activation
    -third hidden layer of 16 neurons with leaky relu activation
    -output layer of 4 neurons with linear activation
'''
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


'''
Trains the model for defined criterion
'''
def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=25, early_stopping_patience=5):
    model.train()
    best_val_loss = float('inf')

    #patience counter for earlystopping
    patience_counter = 0

    #standard training procedure
    for epoch in range(epochs):
        train_loss = 0
        for inputs, targets in train_dataloader:
            inputs = torch.tensor(inputs, dtype=torch.float32)
            targets = torch.tensor(targets, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
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
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        #earlystopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break



'''
Calculates the accuracy for each state with tolerance of 5%
'''
def calculate_accuracy(model, dataloader, criterion, tolerance=0.05):
    model.eval()
    total_loss = 0
    correct = np.zeros(4)  # 4 states
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = torch.tensor(inputs, dtype=torch.float32)
            targets = torch.tensor(targets, dtype=torch.float32)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

            for i in range(4):
                correct[i] += torch.sum(torch.abs((outputs[:, i] - targets[:, i]) / targets[:, i]) < tolerance).item()
            total += targets.size(0)  # Number of samples in the batch

    avg_loss = total_loss / len(dataloader)
    accuracies = (correct / total) * 100

    print(f'Average Loss: {avg_loss:.4f}')
    for i, acc in enumerate(accuracies):
        print(f'Accuracy for state {i + 1}: {acc:.2f}%')

    return accuracies


if __name__ == "__main__":

    #read the data
    states_df = pd.read_csv('states.csv').drop(columns=['Unnamed: 0'])
    inputs_df = pd.read_csv('inputs.csv').drop(columns=['Unnamed: 0'])

    #retrive the normalizing parameters(mean and sd)
    scaler_states = StandardScaler().fit(states_df)
    scaler_inputs = StandardScaler().fit(inputs_df)

    train_dataset = CSTRDataset('states.csv', 'inputs.csv', scaler=(scaler_states, scaler_inputs))

    # Split dataset into training and validation sets randomly
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size   
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    #create dataloader object
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    #define input and output dimension
    input_dim = 6
    output_dim = 4
    model = ANNModel(input_dim, output_dim)

    #define criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)  

    #start training
    train_model(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=100)

    # Save the model and the scalers
    torch.save(model.state_dict(), 'ann_model.pth')
    np.save('scaler_states.npy', scaler_states.mean_)
    np.save('scaler_states_scale.npy', scaler_states.scale_)
    np.save('scaler_inputs.npy', scaler_inputs.mean_)
    np.save('scaler_inputs_scale.npy', scaler_inputs.scale_)

    # Denormalize the model predictions for plotting
    scaler_outputs = scaler_states
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
    
    outputs_np_denorm = model_predictions * scaler_outputs.scale_ + scaler_outputs.mean_
    true_values = true_values * scaler_outputs.scale_ + scaler_outputs.mean_

    print("Train accuracy:")
    test_accuracy = calculate_accuracy(model, train_dataloader, criterion)

    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    y_labels = ['C_A', 'C_B', 'T_R', 'T_J']

 
    for i in range(output_dim):
        plt.subplot(output_dim, 1, i + 1)
        plt.plot(true_values[:200, i], label='True Value')
        plt.plot(outputs_np_denorm[:200, i], label='Predicted Value', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel(y_labels[i])
        plt.legend()
    
    plt.tight_layout()
    plt.show()
