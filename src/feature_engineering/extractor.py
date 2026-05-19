import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew
import pywt
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Extracts features from time-series sensor data for predictive maintenance.
    Includes statistical, frequency-domain, and time-frequency features.
    """

    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize the FeatureEngineer.

        Args:
            sampling_rate: Sampling rate of the sensor data in Hz.
        """
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2.0

    def extract_statistical_features(self, signal_data: np.ndarray) -> dict:
        """
        Extract statistical features from a 1D signal.

        Args:
            signal_data: 1D numpy array of signal values.

        Returns:
            Dictionary of statistical feature names and values.
        """
        if len(signal_data) == 0:
            return {}

        features = {
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'var': np.var(signal_data),
            'rms': np.sqrt(np.mean(signal_data**2)),
            'peak_to_peak': np.ptp(signal_data),
            'skewness': skew(signal_data),
            'kurtosis': kurtosis(signal_data),
            'crest_factor': np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2)),
            'impulse_factor': np.max(np.abs(signal_data)) / np.mean(np.abs(signal_data)),
            'shape_factor': np.sqrt(np.mean(signal_data**2)) / np.mean(np.abs(signal_data)),
        }
        return features

    def extract_frequency_features(self, signal_data: np.ndarray) -> dict:
        """
        Extract frequency-domain features using FFT.

        Args:
            signal_data: 1D numpy array of signal values.

        Returns:
            Dictionary of frequency-domain feature names and values.
        """
        if len(signal_data) < 2:
            return {}

        # Compute FFT
        fft_vals = np.fft.rfft(signal_data)
        fft_freq = np.fft.rfftfreq(len(signal_data), d=1/self.sampling_rate)
        magnitude = np.abs(fft_vals)

        # Avoid division by zero
        if np.sum(magnitude) == 0:
            return {
                'spectral_centroid': 0.0,
                'spectral_spread': 0.0,
                'spectral_flux': 0.0,
                'spectral_rolloff': 0.0,
                'zero_crossings': 0.0,
            }

        # Spectral centroid
        spectral_centroid = np.sum(fft_freq * magnitude) / np.sum(magnitude)

        # Spectral spread
        spectral_spread = np.sqrt(np.sum(((fft_freq - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))

        # Spectral flux (using difference in magnitude spectrum)
        # For simplicity, we use the sum of squared differences from the mean magnitude
        spectral_flux = np.sum(np.diff(magnitude) ** 2)

        # Spectral rolloff (frequency below which 85% of the magnitude spectrum lies)
        cumsum_magnitude = np.cumsum(magnitude)
        rolloff_threshold = 0.85 * cumsum_magnitude[-1]
        rolloff_index = np.where(cumsum_magnitude >= rolloff_threshold)[0]
        spectral_rolloff = fft_freq[rolloff_index[0]] if len(rolloff_index) > 0 else 0.0

        # Zero crossings
        zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)

        features = {
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'spectral_flux': spectral_flux,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossings': zero_crossings,
            'dominant_frequency': fft_freq[np.argmax(magnitude)] if len(magnitude) > 0 else 0.0,
            'spectral_energy': np.sum(magnitude ** 2),
        }
        return features

    def extract_time_frequency_features(self, signal_data: np.ndarray, 
                                       wavelet: str = 'db4', level: int = 5) -> dict:
        """
        Extract time-frequency features using wavelet decomposition.

        Args:
            signal_data: 1D numpy array of signal values.
            wavelet: Wavelet type to use (default: 'db4').
            level: Decomposition level (default: 5).

        Returns:
            Dictionary of wavelet-based feature names and values.
        """
        if len(signal_data) < (2 ** level):
            return {}

        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(signal_data, wavelet, level=level)

            features = {}
            # Energy of each detail coefficient
            for i, coeff in enumerate(coeffs[1:], start=1):
                energy = np.sum(np.array(coeff) ** 2)
                features[f'wavelet_energy_level_{i}'] = energy

            # Relative energy (normalized by total energy)
            total_energy = np.sum(np.array(signal_data) ** 2)
            if total_energy > 0:
                for i, coeff in enumerate(coeffs[1:], start=1):
                    energy = np.sum(np.array(coeff) ** 2)
                    features[f'wavelet_relative_energy_level_{i}'] = energy / total_energy

            return features
        except Exception as e:
            logger.warning(f"Wavelet decomposition failed: {e}")
            return {}

    def extract_all_features(self, signal_data: np.ndarray) -> dict:
        """
        Extract all available features from a signal.

        Args:
            signal_data: 1D numpy array of signal values.

        Returns:
            Dictionary containing all extracted features.
        """
        features = {}
        features.update(self.extract_statistical_features(signal_data))
        features.update(self.extract_frequency_features(signal_data))
        features.update(self.extract_time_frequency_features(signal_data))
        return features

    def extract_features_from_dataframe(self, df: pd.DataFrame, 
                                       sensor_columns: List[str]) -> pd.DataFrame:
        """
        Extract features for multiple sensor columns in a DataFrame.

        Args:
            df: DataFrame containing sensor data.
            sensor_columns: List of column names to extract features from.

        Returns:
            DataFrame with original columns plus extracted features.
        """
        feature_rows = []

        for idx, row in df.iterrows():
            features = {'index': idx}
            for col in sensor_columns:
                signal_data = df[col].values
                col_features = self.extract_all_features(signal_data)
                # Prefix feature names with column name to avoid collisions
                for feat_name, feat_value in col_features.items():
                    features[f"{col}_{feat_name}"] = feat_value
            feature_rows.append(features)

        features_df = pd.DataFrame(feature_rows)
        features_df.set_index('index', inplace=True)
        # Join with original dataframe
        return df.join(features_df, how='left')
