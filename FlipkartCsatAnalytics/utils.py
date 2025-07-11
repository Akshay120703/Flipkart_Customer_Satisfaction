import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import io
import base64

class Utils:
    """Utility functions for the dashboard"""
    
    @staticmethod
    def format_number(number):
        """Format numbers for display"""
        if pd.isna(number):
            return "N/A"
        
        if isinstance(number, (int, float)):
            if number >= 1000000:
                return f"{number/1000000:.1f}M"
            elif number >= 1000:
                return f"{number/1000:.1f}K"
            else:
                return f"{number:.0f}"
        
        return str(number)
    
    @staticmethod
    def format_percentage(value):
        """Format percentage values"""
        if pd.isna(value):
            return "N/A"
        return f"{value:.1f}%"
    
    @staticmethod
    def format_currency(value):
        """Format currency values"""
        if pd.isna(value):
            return "N/A"
        return f"₹{value:,.2f}"
    
    @staticmethod
    def calculate_satisfaction_rate(csat_scores):
        """Calculate satisfaction rate (CSAT >= 4)"""
        if len(csat_scores) == 0:
            return 0
        return (csat_scores >= 4).mean() * 100
    
    @staticmethod
    def get_satisfaction_category(score):
        """Get satisfaction category based on score"""
        if pd.isna(score):
            return "Unknown"
        elif score >= 4:
            return "Satisfied"
        elif score >= 3:
            return "Neutral"
        else:
            return "Dissatisfied"
    
    @staticmethod
    def calculate_performance_metrics(df, group_by_column):
        """Calculate performance metrics for a grouping column"""
        if group_by_column not in df.columns:
            return pd.DataFrame()
        
        metrics = df.groupby(group_by_column).agg({
            'CSAT Score': ['mean', 'count', 'std'],
            'connected_handling_time': ['mean', 'median'],
            'Unique id': 'count'
        }).round(2)
        
        # Flatten column names
        metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]
        
        # Calculate satisfaction rate
        satisfaction_rate = df.groupby(group_by_column)['CSAT Score'].apply(
            lambda x: (x >= 4).mean() * 100
        ).round(1)
        
        metrics['satisfaction_rate'] = satisfaction_rate
        
        return metrics
    
    @staticmethod
    def create_data_summary(df):
        """Create a comprehensive data summary"""
        summary = {
            'total_records': len(df),
            'date_range': Utils.get_date_range(df),
            'channels': df['channel_name'].unique().tolist() if 'channel_name' in df.columns else [],
            'categories': df['category'].unique().tolist() if 'category' in df.columns else [],
            'avg_csat': df['CSAT Score'].mean() if 'CSAT Score' in df.columns else None,
            'satisfaction_rate': Utils.calculate_satisfaction_rate(df['CSAT Score']) if 'CSAT Score' in df.columns else None,
            'unique_agents': df['Agent_name'].nunique() if 'Agent_name' in df.columns else None,
            'unique_managers': df['Manager'].nunique() if 'Manager' in df.columns else None,
            'missing_data': df.isnull().sum().sum(),
            'data_quality_score': Utils.calculate_data_quality_score(df)
        }
        
        return summary
    
    @staticmethod
    def get_date_range(df):
        """Get date range from dataset"""
        date_columns = ['Issue_reported_at', 'issue_responded', 'Survey_response_Date']
        
        for col in date_columns:
            if col in df.columns:
                try:
                    dates = pd.to_datetime(df[col], errors='coerce').dropna()
                    if len(dates) > 0:
                        return {
                            'start': dates.min().strftime('%Y-%m-%d'),
                            'end': dates.max().strftime('%Y-%m-%d'),
                            'days': (dates.max() - dates.min()).days
                        }
                except:
                    continue
        
        return None
    
    @staticmethod
    def calculate_data_quality_score(df):
        """Calculate data quality score (0-100)"""
        try:
            # Calculate completeness
            completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
            
            # Calculate uniqueness (check for duplicates)
            uniqueness = 1 - (df.duplicated().sum() / len(df))
            
            # Calculate validity (for key columns)
            validity_score = 1.0
            
            # Check CSAT Score validity
            if 'CSAT Score' in df.columns:
                valid_csat = df['CSAT Score'].between(1, 5, inclusive='both').sum()
                validity_score *= valid_csat / len(df)
            
            # Overall score
            quality_score = (completeness * 0.4 + uniqueness * 0.3 + validity_score * 0.3) * 100
            
            return round(quality_score, 1)
        
        except Exception as e:
            return 0.0
    
    @staticmethod
    def generate_insights(df):
        """Generate automated insights from the data"""
        insights = []
        
        try:
            # CSAT insights
            if 'CSAT Score' in df.columns:
                avg_csat = df['CSAT Score'].mean()
                satisfaction_rate = Utils.calculate_satisfaction_rate(df['CSAT Score'])
                
                insights.append({
                    'type': 'metric',
                    'title': 'Overall Performance',
                    'content': f"Average CSAT score is {avg_csat:.2f} with {satisfaction_rate:.1f}% satisfaction rate",
                    'impact': 'high' if satisfaction_rate >= 80 else 'medium' if satisfaction_rate >= 60 else 'low'
                })
            
            # Channel insights
            if 'channel_name' in df.columns:
                channel_performance = df.groupby('channel_name')['CSAT Score'].mean()
                best_channel = channel_performance.idxmax()
                worst_channel = channel_performance.idxmin()
                
                insights.append({
                    'type': 'comparison',
                    'title': 'Channel Performance',
                    'content': f"{best_channel} performs best ({channel_performance[best_channel]:.2f}), while {worst_channel} needs improvement ({channel_performance[worst_channel]:.2f})",
                    'impact': 'medium'
                })
            
            # Category insights
            if 'category' in df.columns:
                category_volumes = df['category'].value_counts()
                top_category = category_volumes.index[0]
                
                insights.append({
                    'type': 'trend',
                    'title': 'Support Categories',
                    'content': f"{top_category} is the most common support category ({category_volumes[top_category]} cases, {category_volumes[top_category]/len(df)*100:.1f}%)",
                    'impact': 'medium'
                })
            
            # Agent performance insights
            if 'Agent_name' in df.columns:
                agent_performance = df.groupby('Agent_name')['CSAT Score'].agg(['mean', 'count'])
                qualified_agents = agent_performance[agent_performance['count'] >= 10]
                
                if len(qualified_agents) > 0:
                    top_agent_score = qualified_agents['mean'].max()
                    performance_std = qualified_agents['mean'].std()
                    
                    insights.append({
                        'type': 'performance',
                        'title': 'Agent Performance',
                        'content': f"Top agent achieves {top_agent_score:.2f} CSAT. Performance variation is {'high' if performance_std > 0.5 else 'moderate' if performance_std > 0.3 else 'low'} (std: {performance_std:.2f})",
                        'impact': 'high' if performance_std > 0.5 else 'medium'
                    })
            
            # Time-based insights
            if 'Issue_reported_at' in df.columns:
                try:
                    df['Issue_reported_at'] = pd.to_datetime(df['Issue_reported_at'], errors='coerce')
                    df['hour'] = df['Issue_reported_at'].dt.hour
                    
                    hourly_volume = df.groupby('hour').size()
                    peak_hour = hourly_volume.idxmax()
                    
                    insights.append({
                        'type': 'pattern',
                        'title': 'Peak Hours',
                        'content': f"Peak support hour is {peak_hour}:00 with {hourly_volume[peak_hour]} interactions",
                        'impact': 'medium'
                    })
                except:
                    pass
            
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
        
        return insights
    
    @staticmethod
    def export_data_to_excel(df, filename="customer_satisfaction_data.xlsx"):
        """Export data to Excel format"""
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Main data
                df.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Summary statistics
                if 'CSAT Score' in df.columns:
                    summary_stats = df.groupby('channel_name')['CSAT Score'].agg(['mean', 'count', 'std']).round(2)
                    summary_stats.to_excel(writer, sheet_name='Channel Summary')
                
                # Agent performance
                if 'Agent_name' in df.columns:
                    agent_stats = df.groupby('Agent_name')['CSAT Score'].agg(['mean', 'count']).round(2)
                    agent_stats.to_excel(writer, sheet_name='Agent Performance')
            
            output.seek(0)
            return output.getvalue()
        
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")
            return None
    
    @staticmethod
    def create_download_link(data, filename, link_text):
        """Create download link for data"""
        try:
            b64 = base64.b64encode(data).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
            return href
        except Exception as e:
            return f"Error creating download link: {str(e)}"
    
    @staticmethod
    def validate_data_upload(df):
        """Validate uploaded data"""
        required_columns = ['CSAT Score', 'channel_name', 'category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        validation_result = {
            'is_valid': len(missing_columns) == 0,
            'missing_columns': missing_columns,
            'total_records': len(df),
            'total_columns': len(df.columns),
            'warnings': []
        }
        
        # Check for data quality issues
        if 'CSAT Score' in df.columns:
            invalid_csat = df[(df['CSAT Score'] < 1) | (df['CSAT Score'] > 5)]
            if len(invalid_csat) > 0:
                validation_result['warnings'].append(f"{len(invalid_csat)} records have invalid CSAT scores")
        
        # Check for excessive missing data
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 50:
            validation_result['warnings'].append(f"High missing data percentage: {missing_percentage:.1f}%")
        
        return validation_result
    
    @staticmethod
    def get_color_palette():
        """Get consistent color palette for visualizations"""
        return {
            'primary': '#047BD6',
            'secondary': '#FB641B',
            'success': '#228B22',
            'warning': '#FF6B35',
            'danger': '#DC3545',
            'info': '#17A2B8',
            'light': '#F8F9FA',
            'dark': '#212121',
            'background': '#F1F3F6'
        }
    
    @staticmethod
    def calculate_business_impact(df):
        """Calculate potential business impact metrics"""
        try:
            impact_metrics = {}
            
            if 'CSAT Score' in df.columns:
                current_satisfaction = Utils.calculate_satisfaction_rate(df['CSAT Score'])
                
                # Potential improvement scenarios
                impact_metrics['current_satisfaction_rate'] = current_satisfaction
                impact_metrics['improvement_potential'] = max(0, 90 - current_satisfaction)
                
                # Estimate impact of 10% improvement
                impact_metrics['retention_impact'] = "5-10% increase in customer retention"
                impact_metrics['revenue_impact'] = "15-25% increase in customer lifetime value"
                
                # Calculate dissatisfied customers
                dissatisfied_count = len(df[df['CSAT Score'] < 4])
                impact_metrics['dissatisfied_customers'] = dissatisfied_count
                impact_metrics['at_risk_percentage'] = (dissatisfied_count / len(df)) * 100
            
            return impact_metrics
        
        except Exception as e:
            return {}
    
    @staticmethod
    def generate_recommendations(df):
        """Generate actionable recommendations based on data analysis"""
        recommendations = []
        
        try:
            # Channel recommendations
            if 'channel_name' in df.columns and 'CSAT Score' in df.columns:
                channel_performance = df.groupby('channel_name')['CSAT Score'].mean()
                worst_channel = channel_performance.idxmin()
                best_channel = channel_performance.idxmax()
                
                recommendations.append({
                    'category': 'Channel Optimization',
                    'priority': 'High',
                    'recommendation': f"Improve {worst_channel} channel performance by implementing best practices from {best_channel}",
                    'expected_impact': 'Increase overall satisfaction by 5-10%'
                })
            
            # Category recommendations
            if 'category' in df.columns:
                category_volumes = df['category'].value_counts()
                top_category = category_volumes.index[0]
                
                recommendations.append({
                    'category': 'Process Improvement',
                    'priority': 'Medium',
                    'recommendation': f"Focus on streamlining {top_category} processes as they represent {category_volumes[top_category]/len(df)*100:.1f}% of all interactions",
                    'expected_impact': 'Reduce handling time by 15-20%'
                })
            
            # Agent training recommendations
            if 'Agent_name' in df.columns and 'CSAT Score' in df.columns:
                agent_performance = df.groupby('Agent_name')['CSAT Score'].agg(['mean', 'count'])
                qualified_agents = agent_performance[agent_performance['count'] >= 10]
                
                if len(qualified_agents) > 0:
                    low_performers = qualified_agents[qualified_agents['mean'] < 3.5]
                    
                    if len(low_performers) > 0:
                        recommendations.append({
                            'category': 'Training & Development',
                            'priority': 'High',
                            'recommendation': f"Provide targeted training for {len(low_performers)} agents with CSAT below 3.5",
                            'expected_impact': 'Improve team average CSAT by 0.2-0.5 points'
                        })
            
            # Time-based recommendations
            if 'Issue_reported_at' in df.columns:
                try:
                    df['Issue_reported_at'] = pd.to_datetime(df['Issue_reported_at'], errors='coerce')
                    df['hour'] = df['Issue_reported_at'].dt.hour
                    
                    hourly_volume = df.groupby('hour').size()
                    peak_hours = hourly_volume.nlargest(3).index.tolist()
                    
                    recommendations.append({
                        'category': 'Resource Planning',
                        'priority': 'Medium',
                        'recommendation': f"Optimize staffing during peak hours: {', '.join(map(str, peak_hours))}",
                        'expected_impact': 'Reduce customer wait times by 20-30%'
                    })
                except:
                    pass
            
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations
