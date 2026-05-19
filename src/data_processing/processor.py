import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data ingestion, cleaning, and preprocessing for sensor data."""

    def __init__(self):
        self.scaler = None
        self.feature_columns = None

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        Load raw sensor data from CSV or JSON file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame with raw sensor readings
        """
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw sensor data by handling missing values and outliers.
        
        Args:
            df: Raw sensor data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Handle missing values - forward fill for time series
        cleaned_df = cleaned_df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove obvious outliers using IQR method
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing to maintain sequence length
            cleaned_df[col] = cleaned_df[col].clip(lower_bound, upper_bound)
            
        logger.info(f"Data cleaning completed. Shape: {cleaned_df.shape}")
        return cleaned_df

    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 30, 
                        target_column: Optional[str] = None) -> tuple:
        """
        Create sequences for time-series modeling.
        
        Args:
            df: Cleaned DataFrame with sensor readings
            sequence_length: Length of each sequence for LSTM input
            target_column: Column to predict (if None, predict all features)
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns 
                                  if col not in ['timestamp', 'equipment_id']]
        
        feature_data = df[self.feature_columns].values
        
        if target_column and target_column in df.columns:
            target_data = df[target_column].values
        else:
            target_data = feature_data  # Autoencoder-style prediction
        
        X, y = [], []
        
        for i in range(len(feature_data) - sequence_length):
            X.append(feature_data[i:(i + sequence_length)])
            y.append(target_data[i + sequence_length])
            
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences of length {sequence_length}")
        return X, y

    def scale_features(self, train_data: np.ndarray, 
                      test_data: Optional[np.ndarray] = None) -> tuple:
        """
        Scale features using StandardScaler.
        
        Args:
            train_data: Training data to fit scaler on
            test_data: Optional test data to transform
            
        Returns:
            Scaled training and test data
        """
        self.scaler = StandardScaler()
        train_scaled = self.scaler.fit_transform(train_data)
        
        if test_data is not None:
            test_scaled = self.scaler.transform(test_data)
            return train_scaled, test_scaled
        
        return train_scaled, None
