import os
import numpy as np
from scipy.signal import welch

class EEGProcessor:
    """
    Class for processing EEG data from the STEW dataset.
    Provides functionality for loading data, creating windows,
    and extracting spectral features.
    """
    def __init__(self, sampling_rate=128, window_size=512):
        """
        Initialize the EEG processor.
        
        Parameters:
        -----------
        sampling_rate : int
            Sampling rate of the EEG data in Hz (default: 128)
        window_size : int
            Size of the window for feature extraction (default: 512, ~4 seconds)
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30)
        }
        
    def load_dataset(self, data_path):
        """
        Load the STEW dataset from the specified directory.
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing the STEW dataset files
            
        Returns:
        --------
        dict
            Dictionary containing subjects data, filenames, and ratings if available
        """
        # Find all subject files
        subject_files = [f for f in os.listdir(data_path) if f.startswith("sub")]
        subject_files.sort()
        
        # Load ratings if available
        ratings = {}
        ratings_file = os.path.join(data_path, "ratings.txt")
        if os.path.exists(ratings_file):
            try:
                with open(ratings_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 3:
                            subject = int(parts[0].strip())
                            rest_rating = int(parts[1].strip())
                            test_rating = int(parts[2].strip())
                            ratings[subject] = {
                                'rest': rest_rating,
                                'test': test_rating
                            }
            except Exception as e:
                print(f"Error loading ratings: {e}")
        
        # Create a dataset dictionary with metadata and file paths
        dataset = {
            'files': subject_files,
            'file_paths': [os.path.join(data_path, f) for f in subject_files],
            'ratings': ratings,
            'subjects': {}
        }
        
        return dataset
    
    def load_subject_data(self, file_path):
        """
        Load EEG data for a single subject.
        
        Parameters:
        -----------
        file_path : str
            Path to the subject's EEG data file
            
        Returns:
        --------
        numpy.ndarray or None
            EEG data if successfully loaded, None otherwise
        """
        try:
            data = np.loadtxt(file_path)
            
            # Validate shape (should have 14 channels)
            if data.shape[1] != 14:
                print(f"Invalid data shape: {data.shape}")
                return None
            
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_windows(self, data):
        """
        Create fixed-size windows from the EEG data.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Raw EEG data
            
        Returns:
        --------
        list
            List of EEG data windows
        """
        windows = []
        
        # Generate windows with specified size
        num_windows = (data.shape[0] - self.window_size) // self.window_size + 1
        for i in range(num_windows):
            start = i * self.window_size
            end = start + self.window_size
            if end > data.shape[0]:
                break  # Safety check
                
            window = data[start:end, :]
            windows.append(window)
            
        return windows
    
    def compute_band_power(self, data):
        """
        Compute average power in EEG frequency bands for each channel.
        
        Parameters:
        -----------
        data : numpy.ndarray
            EEG window (time x channels)
            
        Returns:
        --------
        numpy.ndarray
            Flattened band power features
        """
        n_channels = data.shape[1]
        features = []
        
        for ch in range(n_channels):
            f, psd = welch(data[:, ch], fs=self.sampling_rate, nperseg=256)
            band_powers = []
            for band in self.bands.values():
                mask = (f >= band[0]) & (f <= band[1])
                band_power = np.mean(psd[mask]) if np.any(mask) else 0
                band_powers.append(band_power)
            features.extend(band_powers)
            
        return np.array(features)
    
    def compute_band_power_matrix(self, data):
        """
        Compute band power matrix for visualization.
        
        Parameters:
        -----------
        data : numpy.ndarray
            EEG window (time x channels)
            
        Returns:
        --------
        tuple
            Band power matrix and band names
        """
        n_channels = data.shape[1]
        band_power_matrix = np.zeros((n_channels, len(self.bands)))
        band_names = list(self.bands.keys())
        
        for ch in range(n_channels):
            f, psd = welch(data[:, ch], fs=self.sampling_rate, nperseg=256)
            for i, band in enumerate(self.bands.values()):
                mask = (f >= band[0]) & (f <= band[1])
                band_power_matrix[ch, i] = np.mean(psd[mask]) if np.any(mask) else 0
        
        return band_power_matrix, band_names
    
    def extract_features(self, data_path, cache=True):
        """
        Load files, create time windows, compute features and labels.
        
        Parameters:
        -----------
        data_path : str
            Path to EEG dataset
        cache : bool
            Whether to cache the results in memory
            
        Returns:
        --------
        tuple
            (raw_windows, spectral_features), labels
        """
        # Ensure data_path exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The specified data path does not exist: {data_path}")
            
        print(f"Processing data from: {data_path}")
        print(f"Directory contents: {os.listdir(data_path)}")
        
        dataset = self.load_dataset(data_path)
        
        X_raw, X_spec, y = [], [], []
        metadata = []

        for idx, file in enumerate(dataset['files']):
            full_path = dataset['file_paths'][idx]
            
            try:
                print(f"Processing file: {file} at {full_path}")
                
                # Check if file exists
                if not os.path.exists(full_path):
                    print(f"File not found: {full_path}")
                    continue
                    
                data = self.load_subject_data(full_path)
                if data is None:
                    print(f"Failed to load data for {file}")
                    continue
                
                # Extract subject number and condition
                parts = file.split('_')
                subject_id = parts[0][3:]  # Remove 'sub' prefix
                condition = parts[1].split('.')[0]  # Remove file extension
                
                print(f"Creating windows for subject {subject_id}, condition {condition}")
                
                # Generate windows and compute features
                windows = self.create_windows(data)
                for window in windows:
                    X_raw.append(window)
                    X_spec.append(self.compute_band_power(window))
                    
                    # Create label and metadata
                    if condition == "hi":
                        y.append(1)
                    elif condition == "lo":
                        y.append(0)
                    else:
                        print(f"Unknown condition in filename: {file}")
                        continue
                        
                    metadata.append({
                        'subject': int(subject_id),
                        'condition': condition,
                        'file': file
                    })
                
                print(f"Successfully processed {file}, extracted {len(windows)} windows")
                    
            except Exception as e:
                print(f"Error processing {file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Check if we have any data
        if len(X_raw) == 0:
            print("No data was extracted. Please check the dataset files.")
            # Return empty results with the right structure
            return {
                'X_raw': np.array([]),
                'X_spec': np.array([]),
                'y': np.array([]),
                'metadata': [],
                'dataset': dataset
            }
            
        # Normalize spectral features
        X_spec = np.array(X_spec)
        X_spec = (X_spec - X_spec.mean(axis=0)) / (X_spec.std(axis=0) + 1e-8)
        
        results = {
            'X_raw': np.array(X_raw),
            'X_spec': X_spec,
            'y': np.array(y),
            'metadata': metadata,
            'dataset': dataset
        }
        
        print(f"Feature extraction complete. Extracted {len(X_raw)} windows.")
        return results