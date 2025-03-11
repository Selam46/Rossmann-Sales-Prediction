import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RossmannFeatureEngineer:  
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Basic date components
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        
        # Weekend feature
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Month period features
        df['DayOfMonth'] = df['Date'].dt.day
        df['IsMonthStart'] = (df['DayOfMonth'] <= 5).astype(int)
        df['IsMonthEnd'] = (df['DayOfMonth'] >= 26).astype(int)
        df['IsMidMonth'] = ((df['DayOfMonth'] > 5) & (df['DayOfMonth'] < 26)).astype(int)
        
        # Quarter feature
        df['Quarter'] = df['Date'].dt.quarter
        
        # Create season mapping (Dec-Feb: Winter, Mar-May: Spring, Jun-Aug: Summer, Sep-Nov: Fall)
        df['Season'] = df['Month'].map(lambda x: 0 if x in [12, 1, 2] else  
                                              1 if x in [3, 4, 5] else  
                                              2 if x in [6, 7, 8] else  
                                              3)  
        
        logger.info("Successfully extracted datetime features")
        return df
    

    def calculate_holiday_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Create holiday indicator (1 for any type of holiday)
        df['IsHoliday'] = (df['StateHoliday'] != '0').astype(int)
        
        # Initialize distance columns
        df['DaysToHoliday'] = 0
        df['DaysAfterHoliday'] = 0
        
        # Calculate for each store
        for store in df['Store'].unique():
            store_mask = df['Store'] == store
            store_dates = df[store_mask].sort_values('Date')
            
            # Find holiday dates
            holiday_dates = store_dates[store_dates['IsHoliday'] == 1]['Date'].values
            
            if len(holiday_dates) > 0:
                for idx, row in store_dates.iterrows():
                    # Days until next holiday
                    next_holidays = holiday_dates[holiday_dates > row['Date']]
                    if len(next_holidays) > 0:
                        df.loc[idx, 'DaysToHoliday'] = (next_holidays[0] - row['Date']).days
                    
                    # Days since last holiday
                    prev_holidays = holiday_dates[holiday_dates < row['Date']]
                    if len(prev_holidays) > 0:
                        df.loc[idx, 'DaysAfterHoliday'] = (row['Date'] - prev_holidays[-1]).days
        
        logger.info("Successfully calculated holiday distances")
        return df
    

    def create_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate competition duration
        df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
                               (df['Month'] - df['CompetitionOpenSinceMonth'])
        
        # Competition duration categories
        df['CompetitionDuration'] = pd.cut(
            df['CompetitionOpen'],
            bins=[-float('inf'), 0, 12, 24, float('inf')],
            labels=['Not_Open', 'New', 'Established', 'Old']
        )
        
        # Convert to numeric
        duration_map = {'Not_Open': 0, 'New': 1, 'Established': 2, 'Old': 3}
        df['CompetitionDuration'] = df['CompetitionDuration'].map(duration_map)
        
        # Distance categories
        df['CompetitionDistanceCategory'] = pd.qcut(
            df['CompetitionDistance'].fillna(df['CompetitionDistance'].max()),
            q=4,
            labels=['Very_Close', 'Close', 'Far', 'Very_Far']
        )
        
        # Convert to numeric
        distance_map = {'Very_Close': 0, 'Close': 1, 'Far': 2, 'Very_Far': 3}
        df['CompetitionDistanceCategory'] = df['CompetitionDistanceCategory'].map(distance_map)
        
        logger.info("Successfully created competition features")
        return df


    def create_promo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Promo duration
        df['Promo2Open'] = 12 * (df['Year'] - df['Promo2SinceYear']) + \
                          (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0
        
        # Create PromoInterval mapping with common variations
        month_map = {
            'Jan': 1, 'January': 1,
            'Feb': 2, 'February': 2,
            'Mar': 3, 'March': 3,
            'Apr': 4, 'April': 4,
            'May': 5,
            'Jun': 6, 'June': 6,
            'Jul': 7, 'July': 7,
            'Aug': 8, 'August': 8,
            'Sep': 9, 'Sept': 9, 'September': 9,
            'Oct': 10, 'October': 10,
            'Nov': 11, 'November': 11,
            'Dec': 12, 'December': 12
        }
        
        # Check if current month is in promo interval
        df['IsPromoMonth'] = 0
        for interval in df['PromoInterval'].unique():
            if isinstance(interval, str):
                try:
                    promo_months = [month_map[m.strip()] for m in interval.split(',')]
                    mask = df['PromoInterval'] == interval
                    df.loc[mask, 'IsPromoMonth'] = df.loc[mask, 'Month'].isin(promo_months).astype(int)
                except KeyError as e:
                    logger.warning(f"Unknown month format found in PromoInterval: {e}")
                    continue
        
        logger.info("Successfully created promotion features")
        return df
    

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Store type encoding
        df['StoreType'] = df['StoreType'].map({'a': 0, 'b': 1, 'c': 2, 'd': 3})
        
        # Assortment encoding
        df['Assortment'] = df['Assortment'].map({'a': 0, 'b': 1, 'c': 2})
        
        # State holiday encoding
        df['StateHoliday'] = df['StateHoliday'].map({'0': 0, 'a': 1, 'b': 2, 'c': 3})
        
        logger.info("Successfully encoded categorical features")
        return df
    

    def scale_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None, 
                      features_to_scale: list = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if features_to_scale is None:
            features_to_scale = ['CompetitionDistance', 'CompetitionOpen', 'Promo2Open']
            
        # Create copies
        train_scaled = train_df.copy()
        test_scaled = test_df.copy() if test_df is not None else None
        
        # Fit scaler on training data and transform
        train_scaled[features_to_scale] = self.scaler.fit_transform(train_df[features_to_scale])
        
        # Transform test data if provided
        if test_scaled is not None:
            test_scaled[features_to_scale] = self.scaler.transform(test_df[features_to_scale])
        
        logger.info("Successfully scaled features")
        return (train_scaled, test_scaled) if test_scaled is not None else train_scaled
    

    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Process training data
        train_processed = train_df.copy()
        train_processed = self.extract_datetime_features(train_processed)
        train_processed = self.calculate_holiday_distances(train_processed)
        train_processed = self.create_competition_features(train_processed)
        train_processed = self.create_promo_features(train_processed)
        train_processed = self.encode_categorical_features(train_processed)
        
        # Process test data if provided
        if test_df is not None:
            test_processed = test_df.copy()
            test_processed = self.extract_datetime_features(test_processed)
            test_processed = self.calculate_holiday_distances(test_processed)
            test_processed = self.create_competition_features(test_processed)
            test_processed = self.create_promo_features(test_processed)
            test_processed = self.encode_categorical_features(test_processed)
            
            # Scale features
            train_processed, test_processed = self.scale_features(train_processed, test_processed)
            
            logger.info("Successfully completed preprocessing pipeline")
            return train_processed, test_processed
        
        # If no test data, only return processed training data
        train_processed = self.scale_features(train_processed)
        logger.info("Successfully completed preprocessing pipeline")
        return train_processed 