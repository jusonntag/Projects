import numpy as np
import mne
import pywt
from scipy.signal import stft

def augment_eeg_data_with_labels(eeg_data, labels, noise_std_values, scaling_std_values, noise_mean=0, scaling_mean=1):
    """
    Augments EEG data by adding Gaussian noise and applying random scaling, and extends the corresponding labels.
    
    Parameters:
    eeg_data (numpy.ndarray): The original EEG data of shape (trials, channels, timesteps).
    labels (numpy.ndarray): The corresponding labels of shape (trials, ...).
    noise_std_values (list): List of standard deviations for Gaussian noise.
    scaling_std_values (list): List of standard deviations for random scaling.
    noise_mean (float): Mean value for Gaussian noise (default is 0).
    scaling_mean (float): Mean value for random scaling (default is 1).
    
    Returns:
    tuple: Augmented EEG data and the extended labels.
    """
    
    def add_gaussian_noise(data, std, mean=0):
        noise = np.random.normal(mean, std, data.shape)
        return data + noise

    def apply_random_scaling(data, std, mean=1):
        scalars = np.random.normal(mean, std, (data.shape[0], 1, 1))  # Generate scalars for each trial
        return data * scalars
    
    augmented_data = [eeg_data]  # Start with original data
    extended_labels = [labels]  # Start with original labels

    # Apply Gaussian noise with different STD values
    for std in noise_std_values:
        noisy_data = add_gaussian_noise(eeg_data, std, noise_mean)
        augmented_data.append(noisy_data)
        extended_labels.append(labels)  # Extend labels

    # Apply random scaling with different STD values
    for std in scaling_std_values:
        scaled_data = apply_random_scaling(eeg_data, std, scaling_mean)
        augmented_data.append(scaled_data)
        extended_labels.append(labels)  # Extend labels
    
    # Combine original data and augmented data
    augmented_data = np.concatenate(augmented_data, axis=0)
    extended_labels = np.concatenate(extended_labels, axis=0)

    return augmented_data, extended_labels


def data_augmentation_MF_CNN(X_data, labels, rows):
    csp = mne.decoding.CSP(rows*2, reg=None, log=None, cov_est='concat')
    csp.fit(X_data, labels)
    csp_features = csp.transform(X_data)
    csp_cwt_features = []
    for trial in range(csp_features.shape[0]):
        trial_cwt = []
        for csp_row in range(csp_features.shape[1]):
            signal = csp_features[trial, csp_row]
            coeff, _ = pywt.cwt(signal, np.arange(8,31), 'morl')
            trial_cwt.append(coeff)
        csp_cwt_features.append(trial_cwt)
    return csp_features, csp_cwt_features
    