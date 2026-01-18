import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ANN_Adam import CSTRDataset, ANNModel

'''
Follows the same structure and defination as ANN_Adam. Only difference it predicts and plots on Test data
'''

def calculate_accuracy(model, dataloader, criterion, tolerance=0.05):
    model.eval()
    total_loss = 0
    correct = np.zeros(4)  # Assuming there are 4 states
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
    # Load scalers
    scaler_states = StandardScaler()
    scaler_states.mean_ = np.load('scaler_states.npy')
    scaler_states.scale_ = np.load('scaler_states_scale.npy')

    scaler_inputs = StandardScaler()
    scaler_inputs.mean_ = np.load('scaler_inputs.npy')
    scaler_inputs.scale_ = np.load('scaler_inputs_scale.npy')

    scaler = (scaler_states, scaler_inputs)

    # Load test dataset
    test_dataset = CSTRDataset('states.csv', 'inputs.csv', scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    input_dim = 6
    output_dim = 4
    model = ANNModel(input_dim, output_dim)
    model.load_state_dict(torch.load('ann_model.pth'))
    model.eval()

    # Test the model
    criterion = nn.MSELoss()
    print("Test accuracy:")
    test_accuracy = calculate_accuracy(model, test_dataloader, criterion)

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


    plt.figure(figsize=(12, 6))
    y_labels = ['C_A', 'C_B', 'T_R', 'T_J']

    

    for i in range(output_dim):
        plt.subplot(output_dim, 1, i + 1)
        plt.plot(true_values[5000:6000, i], label='True Value')
        plt.plot(outputs_np_denorm[5000:6000, i], label='Predicted Value', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel(y_labels[i])
        plt.legend()
    plt.tight_layout()
    plt.show()
