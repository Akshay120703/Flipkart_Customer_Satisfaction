import pandas as pd
import numpy as np
from datetime import datetime
import re

class DataProcessor:
    """Data processing and cleaning utilities"""
    
    def __init__(self):
        self.processed_data = None
    
    def clean_data(self, df):
        """Clean and preprocess the dataset"""
        try:
            # Create a copy to avoid modifying original data
            df_clean = df.copy()
            
            # Handle missing values
            df_clean = self._handle_missing_values(df_clean)
            
            # Clean column names
            df_clean = self._clean_column_names(df_clean)
            
            # Process date columns
            df_clean = self._process_dates(df_clean)
            
            # Clean categorical variables
            df_clean = self._clean_categorical_data(df_clean)
            
            # Handle numeric columns
            df_clean = self._process_numeric_data(df_clean)
            
            # Remove duplicates
            df_clean = self._remove_duplicates(df_clean)
            
            # Validate data
            df_clean = self._validate_data(df_clean)
            
            self.processed_data = df_clean
            return df_clean
            
        except Exception as e:
            print(f"Error in data cleaning: {str(e)}")
            return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Fill missing CSAT scores with median
        if 'CSAT Score' in df.columns:
            df['CSAT Score'] = df['CSAT Score'].fillna(df['CSAT Score'].median())
        
        # Fill missing categorical variables with 'Unknown'
        categorical_cols = ['channel_name', 'category', 'Sub-category', 'Agent_name', 
                           'Manager', 'Supervisor', 'Tenure Bucket', 'Agent Shift']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Fill missing numeric variables with median
        numeric_cols = ['connected_handling_time', 'Item_price']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _clean_column_names(self, df):
        """Clean column names"""
        # Replace spaces with underscores and remove special characters
        df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_')
        df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
        return df
    
    def _process_dates(self, df):
        """Process date columns"""
        date_columns = ['order_date_time', 'Issue_reported_at', 'issue_responded', 
                       'Survey_response_Date']
        
        for col in date_columns:
            if col in df.columns:
                try:
                    # Try to parse dates
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    # Extract useful date features
                    if col == 'Issue_reported_at':
                        df['issue_hour'] = df[col].dt.hour
                        df['issue_day_of_week'] = df[col].dt.dayofweek
                        df['issue_month'] = df[col].dt.month
                    
                except Exception as e:
                    print(f"Error processing date column {col}: {str(e)}")
                    continue
        
        return df
    
    def _clean_categorical_data(self, df):
        """Clean categorical data"""
        # Clean text columns
        text_columns = ['Customer_Remarks', 'channel_name', 'category', 'Sub_category']
        
        for col in text_columns:
            if col in df.columns:
                # Remove extra whitespace
                df[col] = df[col].astype(str).str.strip()
                
                # Standardize case
                if col in ['channel_name', 'category']:
                    df[col] = df[col].str.title()
        
        # Standardize tenure buckets
        if 'Tenure_Bucket' in df.columns:
            df['Tenure_Bucket'] = df['Tenure_Bucket'].str.replace('On Job Training', 'Training')
        
        return df
    
    def _process_numeric_data(self, df):
        """Process numeric columns"""
        # Ensure CSAT Score is numeric and within valid range
        if 'CSAT_Score' in df.columns:
            df['CSAT_Score'] = pd.to_numeric(df['CSAT_Score'], errors='coerce')
            df['CSAT_Score'] = df['CSAT_Score'].clip(1, 5)
        
        # Process handling time
        if 'connected_handling_time' in df.columns:
            df['connected_handling_time'] = pd.to_numeric(df['connected_handling_time'], errors='coerce')
            # Remove negative values
            df['connected_handling_time'] = df['connected_handling_time'].clip(0, None)
        
        # Process item price
        if 'Item_price' in df.columns:
            df['Item_price'] = pd.to_numeric(df['Item_price'], errors='coerce')
            df['Item_price'] = df['Item_price'].clip(0, None)
        
        return df
    
    def _remove_duplicates(self, df):
        """Remove duplicate records"""
        initial_count = len(df)
        df = df.drop_duplicates()
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} duplicate records")
        
        return df
    
    def _validate_data(self, df):
        """Validate data quality"""
        # Check for required columns
        required_columns = ['CSAT_Score', 'channel_name', 'category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
        
        # Check CSAT Score validity
        if 'CSAT_Score' in df.columns:
            invalid_csat = df[(df['CSAT_Score'] < 1) | (df['CSAT_Score'] > 5)]
            if len(invalid_csat) > 0:
                print(f"Warning: {len(invalid_csat)} records with invalid CSAT scores")
        
        return df
    
    def create_features(self, df):
        """Create additional features for analysis"""
        df_featured = df.copy()
        
        # Create satisfaction binary flag
        if 'CSAT_Score' in df_featured.columns:
            df_featured['is_satisfied'] = (df_featured['CSAT_Score'] >= 4).astype(int)
        
        # Create handling time buckets
        if 'connected_handling_time' in df_featured.columns:
            df_featured['handling_time_bucket'] = pd.cut(
                df_featured['connected_handling_time'], 
                bins=[0, 5, 15, 30, float('inf')], 
                labels=['Quick', 'Standard', 'Long', 'Extended']
            )
        
        # Create price buckets
        if 'Item_price' in df_featured.columns:
            df_featured['price_bucket'] = pd.cut(
                df_featured['Item_price'], 
                bins=[0, 1000, 5000, 15000, float('inf')], 
                labels=['Low', 'Medium', 'High', 'Premium']
            )
        
        # Create issue complexity flag
        if 'Customer_Remarks' in df_featured.columns:
            df_featured['has_remarks'] = df_featured['Customer_Remarks'].notna().astype(int)
            
            # Sentiment analysis (simple keyword-based)
            positive_words = ['good', 'excellent', 'great', 'satisfied', 'happy', 'perfect']
            negative_words = ['bad', 'terrible', 'awful', 'disappointed', 'angry', 'worst']
            
            df_featured['sentiment_score'] = 0
            for word in positive_words:
                df_featured['sentiment_score'] += df_featured['Customer_Remarks'].str.lower().str.contains(word, na=False).astype(int)
            
            for word in negative_words:
                df_featured['sentiment_score'] -= df_featured['Customer_Remarks'].str.lower().str.contains(word, na=False).astype(int)
        
        return df_featured
    
    def get_summary_statistics(self, df):
        """Get summary statistics for the dataset"""
        summary = {}
        
        # Basic statistics
        summary['total_records'] = len(df)
        summary['total_columns'] = len(df.columns)
        summary['missing_values'] = df.isnull().sum().sum()
        summary['duplicate_records'] = df.duplicated().sum()
        
        # CSAT statistics
        if 'CSAT_Score' in df.columns:
            summary['avg_csat'] = df['CSAT_Score'].mean()
            summary['csat_std'] = df['CSAT_Score'].std()
            summary['satisfaction_rate'] = (df['CSAT_Score'] >= 4).mean()
        
        # Channel statistics
        if 'channel_name' in df.columns:
            summary['unique_channels'] = df['channel_name'].nunique()
            summary['channel_distribution'] = df['channel_name'].value_counts().to_dict()
        
        # Category statistics
        if 'category' in df.columns:
            summary['unique_categories'] = df['category'].nunique()
            summary['category_distribution'] = df['category'].value_counts().to_dict()
        
        # Agent statistics
        if 'Agent_name' in df.columns:
            summary['unique_agents'] = df['Agent_name'].nunique()
        
        return summary
    
    def filter_data(self, df, filters):
        """Apply filters to the dataset"""
        filtered_df = df.copy()
        
        for column, values in filters.items():
            if column in filtered_df.columns:
                if isinstance(values, list):
                    filtered_df = filtered_df[filtered_df[column].isin(values)]
                else:
                    filtered_df = filtered_df[filtered_df[column] == values]
        
        return filtered_df
    
    def aggregate_data(self, df, group_by, metrics):
        """Aggregate data by specified columns"""
        try:
            agg_dict = {}
            
            for metric in metrics:
                if metric in df.columns:
                    if df[metric].dtype in ['int64', 'float64']:
                        agg_dict[metric] = ['mean', 'count', 'std']
                    else:
                        agg_dict[metric] = ['count', 'nunique']
            
            aggregated = df.groupby(group_by).agg(agg_dict)
            return aggregated
            
        except Exception as e:
            print(f"Error in data aggregation: {str(e)}")
            return pd.DataFrame()
