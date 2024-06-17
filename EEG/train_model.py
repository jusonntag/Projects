import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from EEGModels import EEGNet, NFEEG, FMI_EEG
from data_augemtation import *
from fbcnet import FBCNet
from numba import cuda

def reset_gpu():
    # Reset GPU device using Numba
    device = cuda.get_current_device()
    device.reset()

    # Clear any previous TensorFlow sessions
    tf.keras.backend.clear_session()

    # Configure GPU memory growth for TensorFlow
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Clear GPU cache for PyTorch
    torch.cuda.empty_cache()

def train_eegnet(save_path_X, save_path_Y, labels_to_drop, test_size=0.2, random_state=42, batch_size=16, epochs=50, lr = 0.001):
    """
    function for training EEGNet + uses GPU 

    input:
    labels_to_drop (list): List of labels to drop.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.
    batch_size (int): Number of samples per gradient update.
    epochs (int): Number of epochs to train the model.

    output:
    model1: Trained EEGNet model.
    test_loss: Loss value of the model on the test data.
    test_accuracy: Accuracy of the model on the test data.
    """
    # reset GPU
    #reset_gpu()

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # Load data
    X = np.load(save_path_X)
    y = np.load(save_path_Y)
    
    # Select only motor imagery and drop motor execution
    indices_to_drop = np.where(np.isin(y, labels_to_drop))
    X = np.delete(X, indices_to_drop, axis=0)
    y = np.delete(y, indices_to_drop, axis=0)
    
    # Prepare y for model -> one hot encoding
    unique_labels = np.unique(y)
    nb_classes = len(unique_labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    indices = np.array([label_to_index[label] for label in y])
    y = tf.one_hot(indices, nb_classes).numpy()
    
    # Prepare X for model
    X /= X.flatten().max()  # Normalizing the data
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))  # Adding a channel dimension for EEGNet

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Define EEGNet model
    model1 = EEGNet(nb_classes=nb_classes, Chans=X.shape[1], Samples=X.shape[2], kernLength=64)
    model1.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model1.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    
    test_loss, test_accuracy = model1.evaluate(X_test, y_test)

    del model1
    tf.keras.backend.clear_session()
    
    return test_loss, test_accuracy


def train_fbcnet(save_path_X, save_path_Y, labels_to_drop, test_size=0.2, random_state=42, batch_size=16, epochs=50, lr = 0.001):
    """
    function for training FBCNet + uses GPU 

    input:
    labels_to_drop (list): List of labels to drop.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.
    batch_size (int): Number of samples per gradient update.
    epochs (int): Number of epochs to train the model.

    output:
    model1: Trained EEGNet model.
    test_loss: Loss value of the model on the test data.
    test_accuracy: Accuracy of the model on the test data.
    """
    
    '''device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True'''

    # reset GPU
    #reset_gpu()

    # Load data
    X = np.load(save_path_X)
    y = np.load(save_path_Y)
    
    # Select only motor imagery and drop motor execution
    indices_to_drop = np.where(np.isin(y, labels_to_drop))
    X = np.delete(X, indices_to_drop, axis=0)
    y = np.delete(y, indices_to_drop, axis=0)
    
    # Prepare y for model -> one hot encoding
    unique_labels = np.unique(y)
    nb_classes = len(unique_labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    indices = np.array([label_to_index[label] for label in y])
    y = indices

    # Prepare X for model
    X /= X.flatten().max()  # Normalizing the data
    X = np.expand_dims(X, axis = 1)  # Adding a channel dimension (Trials, Channel, Samples, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # FBCNet
    model2 = FBCNet(nChan=X.shape[2], nTime=X.shape[3], nBands=9, num_classes=nb_classes)
    model2.cuda()

    # Define your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model2.parameters(), lr=lr)

    x_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    # make train and test torch samples
    train_loader = DataLoader(TensorDataset(x_train.float(), y_train.long()), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test.float(), y_test.long()), batch_size=batch_size, shuffle=False)

    # use GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model2.cuda()

    # train FBCNet
    epochs = epochs
    for epoch in range(epochs):
        model2.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda() , labels.cuda()
            optimizer.zero_grad()
            outputs = model2(inputs)
            assert outputs.shape[1] == nb_classes, f"Outputs have incorrect shape: {outputs.shape}"
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_train_loss = total_loss / len(train_loader)
        # Validation
        model2.eval()
        total_loss_val = 0.0
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model2(inputs)
                loss = criterion(outputs, labels)
                total_loss_val += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        average_val_loss = total_loss_val / len(test_loader)
        correct += (predicted == labels).sum().item()
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}, Val Accuracy: {accuracy:.4f})#, lr:{optimizer.param_groups[0]["lr"]}')
    
    test_loss, test_accuracy = accuracy, average_val_loss

    del model2
    torch.cuda.empty_cache()

    return test_loss, test_accuracy
    

def train_nfeeg(save_path_X, save_path_Y, labels_to_drop, test_size=0.2, random_state=42, batch_size=8, epochs=50, lr = 0.0001):
    """
    function for training NFEEG + uses GPU 

    input:
    labels_to_drop (list): List of labels to drop.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.
    batch_size (int): Number of samples per gradient update.
    epochs (int): Number of epochs to train the model.

    output:
    model1: Trained EEGNet model.
    test_loss: Loss value of the model on the test data.
    test_accuracy: Accuracy of the model on the test data.
    """
    # reset GPU
    #reset_gpu()

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # Load data
    X = np.load(save_path_X)
    y = np.load(save_path_Y)
    
    # Select only motor imagery and drop motor execution
    indices_to_drop = np.where(np.isin(y, labels_to_drop))
    X = np.delete(X, indices_to_drop, axis=0)
    y = np.delete(y, indices_to_drop, axis=0)
    
    # Prepare y for model -> one hot encoding
    unique_labels = np.unique(y)
    nb_classes = len(unique_labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    indices = np.array([label_to_index[label] for label in y])
    y = tf.one_hot(indices, nb_classes).numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # data augmentation to artificially incresea data size
    noise_std = [0.03, 0.07, 0.1]
    scaling_std = [0.03, 0.07, 0.1]
    X_train, y_train = augment_eeg_data_with_labels(X_train, y_train, noise_std, scaling_std)
    X_test, y_test = augment_eeg_data_with_labels(X_test, y_test, noise_std, scaling_std,)
    
    # prepare X for model (Trials, Samples, 1, Channel)
    X_train = X_train.transpose(0, 2, 1).reshape(X_train.shape[0], X_train.shape[2], 1, X_train.shape[1])
    X_test = X_test.transpose(0, 2, 1).reshape(X_test.shape[0], X_test.shape[2], 1, X_test.shape[1])
    
    # NFEEG
    model3 = NFEEG(nb_classes, X.shape[1], X.shape[2])
    model3.compile(optimizer = Adam(lr), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    # Train the model
    model3.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    
    test_loss, test_accuracy = model3.evaluate(X_test, y_test)
    
    del model3
    tf.keras.backend.clear_session()
    
    return test_loss, test_accuracy

    
