import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class RossmannAnalyzer:
    """Class containing utility functions for Rossmann data analysis."""
    
    @staticmethod
    def analyze_promotions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        Analyze promotion distribution in training and test sets.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Dictionary containing promotion statistics
        """
        train_promo = train_df['Promo'].value_counts(normalize=True)
        test_promo = test_df['Promo'].value_counts(normalize=True)
        
        stats = {
            'train_promo_dist': train_promo.to_dict(),
            'test_promo_dist': test_promo.to_dict(),
            'train_promo_count': train_df['Promo'].value_counts().to_dict(),
            'test_promo_count': test_df['Promo'].value_counts().to_dict()
        }
        
        logger.info("Completed promotion analysis")
        return stats
    
    @staticmethod
    def analyze_holiday_sales(df: pd.DataFrame) -> Dict:
        """
        Analyze sales behavior around holidays.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing holiday sales statistics
        """
        # Group sales by StateHoliday
        holiday_stats = df.groupby('StateHoliday')['Sales'].agg(['mean', 'std', 'count']).to_dict()
        
        # Calculate sales before and after holidays
        df['NextDayHoliday'] = df.groupby('Store')['StateHoliday'].shift(-1) != '0'
        df['PrevDayHoliday'] = df.groupby('Store')['StateHoliday'].shift(1) != '0'
        
        before_holiday = df[df['NextDayHoliday']]['Sales'].mean()
        after_holiday = df[df['PrevDayHoliday']]['Sales'].mean()
        
        stats = {
            'holiday_stats': holiday_stats,
            'before_holiday_avg': before_holiday,
            'after_holiday_avg': after_holiday
        }
        
        logger.info("Completed holiday sales analysis")
        return stats
    
    @staticmethod
    def analyze_seasonal_patterns(df: pd.DataFrame) -> Dict:
        """
        Analyze seasonal patterns in sales.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing seasonal statistics
        """
        # Monthly patterns
        monthly_stats = df.groupby('Month')['Sales'].agg(['mean', 'std']).to_dict()
        
        # Weekly patterns
        weekly_stats = df.groupby('DayOfWeek')['Sales'].agg(['mean', 'std']).to_dict()
        
        # Holiday season (December) vs rest
        holiday_season = df[df['Month'] == 12]['Sales'].mean()
        regular_season = df[df['Month'] != 12]['Sales'].mean()
        
        stats = {
            'monthly_stats': monthly_stats,
            'weekly_stats': weekly_stats,
            'holiday_season_avg': holiday_season,
            'regular_season_avg': regular_season
        }
        
        logger.info("Completed seasonal pattern analysis")
        return stats
    
    @staticmethod
    def analyze_store_patterns(df: pd.DataFrame) -> Dict:
        """
        Analyze store-specific patterns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing store pattern statistics
        """
        # Stores open all weekdays
        always_open = df.groupby('Store')['Open'].mean() == 1
        always_open_stores = always_open[always_open].index.tolist()
        
        # Sales by store type
        store_type_stats = df.groupby('StoreType')['Sales'].agg(['mean', 'std']).to_dict()
        
        # Sales by assortment
        assortment_stats = df.groupby('Assortment')['Sales'].agg(['mean', 'std']).to_dict()
        
        stats = {
            'always_open_stores': always_open_stores,
            'store_type_stats': store_type_stats,
            'assortment_stats': assortment_stats
        }
        
        logger.info("Completed store pattern analysis")
        return stats
    
    @staticmethod
    def analyze_competition_impact(df: pd.DataFrame) -> Dict:
        """
        Analyze impact of competition on sales.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing competition impact statistics
        """
        # Create distance bins
        df['DistanceBin'] = pd.qcut(df['CompetitionDistance'], q=5, labels=['Very Close', 'Close', 'Medium', 'Far', 'Very Far'])
        
        # Sales by distance bin
        distance_stats = df.groupby('DistanceBin')['Sales'].agg(['mean', 'std']).to_dict()
        
        # Impact of new competition
        df['CompetitionAge'] = (
            (df['Year'] - df['CompetitionOpenSinceYear']) * 12 +
            (df['Month'] - df['CompetitionOpenSinceMonth'])
        )
        
        new_competition = df[df['CompetitionAge'] <= 3]['Sales'].mean()
        established_competition = df[df['CompetitionAge'] > 3]['Sales'].mean()
        
        stats = {
            'distance_stats': distance_stats,
            'new_competition_avg': new_competition,
            'established_competition_avg': established_competition
        }
        
        logger.info("Completed competition impact analysis")
        return stats
    
    @staticmethod
    def analyze_promo_effectiveness(df: pd.DataFrame) -> Dict:
        """
        Analyze effectiveness of promotions.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing promotion effectiveness statistics
        """
        # Overall promo impact
        promo_stats = df.groupby('Promo')['Sales'].agg(['mean', 'std', 'count']).to_dict()
        
        # Promo impact by store type
        promo_store_stats = df.groupby(['StoreType', 'Promo'])['Sales'].mean().unstack().to_dict()
        
        # Customer count during promos
        customer_promo_stats = df.groupby('Promo')['Customers'].agg(['mean', 'std']).to_dict()
        
        # Sales per customer during promos
        df['SalesPerCustomer'] = df['Sales'] / df['Customers']
        sales_per_customer_stats = df.groupby('Promo')['SalesPerCustomer'].agg(['mean', 'std']).to_dict()
        
        stats = {
            'promo_stats': promo_stats,
            'promo_store_stats': promo_store_stats,
            'customer_promo_stats': customer_promo_stats,
            'sales_per_customer_stats': sales_per_customer_stats
        }
        
        logger.info("Completed promotion effectiveness analysis")
        return stats 