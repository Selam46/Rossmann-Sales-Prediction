import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from pathlib import Path
import yaml
import os

logger = logging.getLogger(__name__)

class RossmannDataLoader:
    def __init__(self, config_path: str = None):
        if config_path is None:
            root_dir = Path(__file__).parent.parent.parent
            config_path = str(root_dir / "config" / "model_config.yaml")
        
        self.root_dir = Path(config_path).parent.parent
        self.config = self._load_config(config_path)


    @staticmethod
    def _load_config(config_path: str) -> dict:
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}. Using default configuration.")
            return {
                'data_paths': {
                    'train': "data/raw/train.csv",
                    'test': "data/raw/test.csv",
                    'store': "data/raw/store.csv"
                }
            }
        
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            train_path = self.root_dir / self.config['data_paths']['train']
            test_path = self.root_dir / self.config['data_paths']['test']
            
            train_df = pd.read_csv(train_path, parse_dates=['Date'])
            test_df = pd.read_csv(test_path, parse_dates=['Date'])
            logger.info("Successfully loaded training and test data")
            return train_df, test_df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    

    def load_store_data(self) -> pd.DataFrame:
        try:
            store_path = self.root_dir / self.config['data_paths']['store']
            store_df = pd.read_csv(store_path)
            logger.info("Successfully loaded store data")
            return store_df
        except Exception as e:
            logger.error(f"Error loading store data: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_config = self.config.get('data_preprocessing', {}).get('missing_values', {})
        
        if missing_config.get('competition_distance') == 'median':
            df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
        
        for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']:
            df[col].fillna(missing_config.get('competition_open_since', 0), inplace=True)
            
        for col in ['Promo2SinceWeek', 'Promo2SinceYear']:
            df[col].fillna(missing_config.get('promo2_since', 0), inplace=True)
            
        df['PromoInterval'].fillna(missing_config.get('promo_interval', ''), inplace=True)
        
        logger.info("Successfully handled missing values")
        return df
    
    
    def detect_outliers(self, df: pd.DataFrame, column: str) -> pd.Series:
        outlier_config = self.config.get('data_preprocessing', {}).get('outliers', {})
        threshold = outlier_config.get('threshold', 1.5)
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[column] < (Q1 - threshold * IQR)) | (df[column] > (Q3 + threshold * IQR))
        logger.info(f"Detected {outliers.sum()} outliers in {column}")
        return outliers
    

    def create_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        date_features = self.config.get('features', {}).get('datetime', [])
        
        feature_mapping = {
            'year': ('Year', lambda x: x.dt.year),
            'month': ('Month', lambda x: x.dt.month),
            'day': ('Day', lambda x: x.dt.day),
            'week_of_year': ('WeekOfYear', lambda x: x.dt.isocalendar().week),
            'weekday': ('DayOfWeek', lambda x: x.dt.dayofweek),
            'is_weekend': ('IsWeekend', lambda x: x.dt.dayofweek.isin([5, 6]).astype(int))
        }
        
        for feature in date_features:
            if feature in feature_mapping:
                col_name, transform = feature_mapping[feature]
                df[col_name] = transform(df['Date'])
        
        logger.info("Successfully created date features")
        return df
    

    def merge_store_data(self, train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
        merged_df = train_df.merge(store_df, on='Store', how='left')
        logger.info("Successfully merged training and store data")
        return merged_df 