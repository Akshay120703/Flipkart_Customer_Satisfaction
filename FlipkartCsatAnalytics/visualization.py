import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class Visualizer:
    """Visualization utilities for customer satisfaction analysis"""
    
    def __init__(self):
        # Set style and color palette
        plt.style.use('default')
        self.colors = {
            'primary': '#047BD6',
            'secondary': '#FB641B',
            'success': '#228B22',
            'warning': '#FF6B35',
            'background': '#F1F3F6',
            'text': '#212121'
        }
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def create_csat_distribution_plot(self, df, figsize=(12, 8)):
        """Create CSAT score distribution plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot
        csat_counts = df['CSAT Score'].value_counts().sort_index()
        colors = ['#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#047BD6']
        
        bars = ax1.bar(csat_counts.index, csat_counts.values, 
                      color=colors[:len(csat_counts)], alpha=0.8)
        ax1.set_title('CSAT Score Distribution', fontweight='bold')
        ax1.set_xlabel('CSAT Score')
        ax1.set_ylabel('Count')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(csat_counts.values, labels=csat_counts.index, autopct='%1.1f%%',
                colors=colors[:len(csat_counts)], startangle=90)
        ax2.set_title('CSAT Score Percentage Distribution', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_channel_analysis_plot(self, df, figsize=(15, 10)):
        """Create comprehensive channel analysis plot"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Channel volume
        channel_counts = df['channel_name'].value_counts()
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['success']]
        
        bars = ax1.bar(channel_counts.index, channel_counts.values, 
                      color=colors[:len(channel_counts)])
        ax1.set_title('Support Volume by Channel', fontweight='bold')
        ax1.set_ylabel('Number of Interactions')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Channel CSAT
        channel_csat = df.groupby('channel_name')['CSAT Score'].mean()
        bars = ax2.bar(channel_csat.index, channel_csat.values, 
                      color=colors[:len(channel_csat)])
        ax2.set_title('Average CSAT by Channel', fontweight='bold')
        ax2.set_ylabel('Average CSAT Score')
        ax2.set_ylim(0, 5)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Channel satisfaction rate
        channel_satisfaction = df.groupby('channel_name')['CSAT Score'].apply(
            lambda x: (x >= 4).mean() * 100
        )
        bars = ax3.bar(channel_satisfaction.index, channel_satisfaction.values,
                      color=colors[:len(channel_satisfaction)])
        ax3.set_title('Satisfaction Rate by Channel (%)', fontweight='bold')
        ax3.set_ylabel('Satisfaction Rate (%)')
        ax3.set_ylim(0, 100)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Channel handling time
        if 'connected_handling_time' in df.columns:
            channel_time = df.groupby('channel_name')['connected_handling_time'].mean()
            bars = ax4.bar(channel_time.index, channel_time.values,
                          color=colors[:len(channel_time)])
            ax4.set_title('Average Handling Time by Channel', fontweight='bold')
            ax4.set_ylabel('Average Handling Time (min)')
            
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_category_analysis_plot(self, df, figsize=(15, 10)):
        """Create category analysis visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Category distribution
        category_counts = df['category'].value_counts()
        bars = ax1.barh(category_counts.index, category_counts.values, 
                       color=self.colors['primary'])
        ax1.set_title('Support Volume by Category', fontweight='bold')
        ax1.set_xlabel('Number of Interactions')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center')
        
        # Category CSAT
        category_csat = df.groupby('category')['CSAT Score'].mean().sort_values(ascending=False)
        bars = ax2.barh(category_csat.index, category_csat.values, 
                       color=self.colors['secondary'])
        ax2.set_title('Average CSAT by Category', fontweight='bold')
        ax2.set_xlabel('Average CSAT Score')
        ax2.set_xlim(0, 5)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}', ha='left', va='center')
        
        # Category satisfaction distribution
        category_satisfaction = df.groupby('category')['CSAT Score'].apply(
            lambda x: (x >= 4).mean() * 100
        ).sort_values(ascending=False)
        bars = ax3.barh(category_satisfaction.index, category_satisfaction.values,
                       color=self.colors['success'])
        ax3.set_title('Satisfaction Rate by Category (%)', fontweight='bold')
        ax3.set_xlabel('Satisfaction Rate (%)')
        ax3.set_xlim(0, 100)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}%', ha='left', va='center')
        
        # Category-Channel heatmap
        pivot_table = df.pivot_table(values='CSAT Score', 
                                   index='category', 
                                   columns='channel_name', 
                                   aggfunc='mean')
        
        sns.heatmap(pivot_table, annot=True, cmap='RdYlBu_r', center=3,
                   ax=ax4, cbar_kws={'label': 'Average CSAT Score'})
        ax4.set_title('CSAT Heatmap: Category vs Channel', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_team_performance_plot(self, df, figsize=(15, 12)):
        """Create team performance analysis visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Manager performance
        manager_performance = df.groupby('Manager')['CSAT Score'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        bars = ax1.bar(range(len(manager_performance)), manager_performance['mean'], 
                      color=self.colors['primary'], alpha=0.8)
        ax1.set_title('Manager Performance (Average CSAT)', fontweight='bold')
        ax1.set_xlabel('Manager')
        ax1.set_ylabel('Average CSAT Score')
        ax1.set_xticks(range(len(manager_performance)))
        ax1.set_xticklabels(manager_performance.index, rotation=45, ha='right')
        ax1.set_ylim(0, 5)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            count = manager_performance.iloc[i]['count']
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}\n(n={count})', ha='center', va='bottom')
        
        # Top agents
        agent_performance = df.groupby('Agent_name')['CSAT Score'].agg(['mean', 'count'])
        top_agents = agent_performance[agent_performance['count'] >= 10].sort_values('mean', ascending=False).head(10)
        
        bars = ax2.bar(range(len(top_agents)), top_agents['mean'], 
                      color=self.colors['secondary'], alpha=0.8)
        ax2.set_title('Top 10 Agents by CSAT Score', fontweight='bold')
        ax2.set_xlabel('Agent Rank')
        ax2.set_ylabel('Average CSAT Score')
        ax2.set_xticks(range(len(top_agents)))
        ax2.set_xticklabels([f"#{i+1}" for i in range(len(top_agents))])
        ax2.set_ylim(0, 5)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            count = top_agents.iloc[i]['count']
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}\n({count})', ha='center', va='bottom')
        
        # Tenure analysis
        tenure_performance = df.groupby('Tenure Bucket')['CSAT Score'].mean().sort_values(ascending=False)
        bars = ax3.bar(tenure_performance.index, tenure_performance.values, 
                      color=self.colors['success'], alpha=0.8)
        ax3.set_title('Performance by Agent Tenure', fontweight='bold')
        ax3.set_xlabel('Tenure Bucket')
        ax3.set_ylabel('Average CSAT Score')
        ax3.set_ylim(0, 5)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Shift performance
        shift_performance = df.groupby('Agent Shift')['CSAT Score'].mean().sort_values(ascending=False)
        bars = ax4.bar(shift_performance.index, shift_performance.values, 
                      color=self.colors['warning'], alpha=0.8)
        ax4.set_title('Performance by Agent Shift', fontweight='bold')
        ax4.set_xlabel('Agent Shift')
        ax4.set_ylabel('Average CSAT Score')
        ax4.set_ylim(0, 5)
        
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_correlation_heatmap(self, df, figsize=(12, 10)):
        """Create correlation heatmap for numeric variables"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
        ax.set_title('Correlation Matrix of Numeric Variables', fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def create_time_series_analysis(self, df, figsize=(15, 10)):
        """Create time series analysis plots"""
        if 'Issue_reported_at' not in df.columns:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Convert to datetime
        df['Issue_reported_at'] = pd.to_datetime(df['Issue_reported_at'], errors='coerce')
        df['date'] = df['Issue_reported_at'].dt.date
        df['hour'] = df['Issue_reported_at'].dt.hour
        df['day_of_week'] = df['Issue_reported_at'].dt.dayofweek
        
        # Daily trend
        daily_csat = df.groupby('date')['CSAT Score'].mean()
        ax1.plot(daily_csat.index, daily_csat.values, color=self.colors['primary'], linewidth=2)
        ax1.set_title('Daily CSAT Score Trend', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Average CSAT Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Hourly pattern
        hourly_csat = df.groupby('hour')['CSAT Score'].mean()
        bars = ax2.bar(hourly_csat.index, hourly_csat.values, 
                      color=self.colors['secondary'], alpha=0.8)
        ax2.set_title('CSAT Score by Hour of Day', fontweight='bold')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average CSAT Score')
        ax2.set_ylim(0, 5)
        
        # Weekly pattern
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_csat = df.groupby('day_of_week')['CSAT Score'].mean()
        bars = ax3.bar(range(len(weekly_csat)), weekly_csat.values, 
                      color=self.colors['success'], alpha=0.8)
        ax3.set_title('CSAT Score by Day of Week', fontweight='bold')
        ax3.set_xlabel('Day of Week')
        ax3.set_ylabel('Average CSAT Score')
        ax3.set_xticks(range(len(weekly_csat)))
        ax3.set_xticklabels(day_names)
        ax3.set_ylim(0, 5)
        
        # Volume pattern
        hourly_volume = df.groupby('hour').size()
        bars = ax4.bar(hourly_volume.index, hourly_volume.values, 
                      color=self.colors['warning'], alpha=0.8)
        ax4.set_title('Support Volume by Hour of Day', fontweight='bold')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Number of Interactions')
        
        plt.tight_layout()
        return fig
    
    def create_satisfaction_drivers_plot(self, feature_importance, figsize=(12, 8)):
        """Create feature importance visualization"""
        if feature_importance is None:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort features by importance
        sorted_features = dict(sorted(feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        # Take top 15 features
        top_features = dict(list(sorted_features.items())[:15])
        
        features = list(top_features.keys())
        importance_values = list(top_features.values())
        
        # Create horizontal bar plot
        bars = ax.barh(features, importance_values, color=self.colors['primary'], alpha=0.8)
        ax.set_title('Top 15 Customer Satisfaction Drivers', fontweight='bold', pad=20)
        ax.set_xlabel('Feature Importance')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def create_business_metrics_dashboard(self, df, figsize=(16, 12)):
        """Create comprehensive business metrics dashboard"""
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('Business Metrics Dashboard', fontsize=16, fontweight='bold')
        
        # Satisfaction rate trend
        if 'Issue_reported_at' in df.columns:
            df['Issue_reported_at'] = pd.to_datetime(df['Issue_reported_at'], errors='coerce')
            df['date'] = df['Issue_reported_at'].dt.date
            daily_satisfaction = df.groupby('date')['CSAT Score'].apply(
                lambda x: (x >= 4).mean() * 100
            )
            axes[0, 0].plot(daily_satisfaction.index, daily_satisfaction.values, 
                           color=self.colors['success'], linewidth=2)
            axes[0, 0].set_title('Daily Satisfaction Rate (%)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Channel performance comparison
        channel_metrics = df.groupby('channel_name').agg({
            'CSAT Score': ['mean', 'count']
        }).round(2)
        channel_metrics.columns = ['avg_csat', 'volume']
        
        scatter = axes[0, 1].scatter(channel_metrics['volume'], channel_metrics['avg_csat'], 
                                   s=100, alpha=0.7, color=self.colors['primary'])
        axes[0, 1].set_title('Channel Performance: Volume vs CSAT')
        axes[0, 1].set_xlabel('Volume')
        axes[0, 1].set_ylabel('Average CSAT')
        
        # Add labels
        for i, txt in enumerate(channel_metrics.index):
            axes[0, 1].annotate(txt, (channel_metrics.iloc[i]['volume'], 
                                     channel_metrics.iloc[i]['avg_csat']))
        
        # Category satisfaction distribution
        category_satisfaction = df.groupby('category')['CSAT Score'].apply(
            lambda x: (x >= 4).mean() * 100
        ).sort_values(ascending=False)
        
        bars = axes[0, 2].bar(range(len(category_satisfaction)), category_satisfaction.values,
                             color=self.colors['secondary'], alpha=0.8)
        axes[0, 2].set_title('Satisfaction Rate by Category')
        axes[0, 2].set_ylabel('Satisfaction Rate (%)')
        axes[0, 2].set_xticks(range(len(category_satisfaction)))
        axes[0, 2].set_xticklabels(category_satisfaction.index, rotation=45, ha='right')
        
        # Agent performance distribution
        agent_performance = df.groupby('Agent_name')['CSAT Score'].agg(['mean', 'count'])
        qualified_agents = agent_performance[agent_performance['count'] >= 5]
        
        axes[1, 0].hist(qualified_agents['mean'], bins=20, color=self.colors['warning'], alpha=0.7)
        axes[1, 0].set_title('Agent Performance Distribution')
        axes[1, 0].set_xlabel('Average CSAT Score')
        axes[1, 0].set_ylabel('Number of Agents')
        axes[1, 0].axvline(qualified_agents['mean'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {qualified_agents["mean"].mean():.2f}')
        axes[1, 0].legend()
        
        # Response time analysis
        if 'connected_handling_time' in df.columns:
            # Remove outliers for better visualization
            q99 = df['connected_handling_time'].quantile(0.99)
            filtered_time = df[df['connected_handling_time'] <= q99]['connected_handling_time']
            
            axes[1, 1].hist(filtered_time, bins=30, color=self.colors['success'], alpha=0.7)
            axes[1, 1].set_title('Response Time Distribution')
            axes[1, 1].set_xlabel('Handling Time (minutes)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].axvline(filtered_time.mean(), color='red', linestyle='--', 
                              label=f'Mean: {filtered_time.mean():.1f}min')
            axes[1, 1].legend()
        
        # Satisfaction score distribution
        csat_dist = df['CSAT Score'].value_counts().sort_index()
        colors = ['#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#047BD6']
        
        axes[1, 2].pie(csat_dist.values, labels=csat_dist.index, autopct='%1.1f%%',
                      colors=colors[:len(csat_dist)], startangle=90)
        axes[1, 2].set_title('Overall CSAT Distribution')
        
        # Manager team performance
        manager_team_size = df.groupby('Manager')['Agent_name'].nunique()
        manager_performance = df.groupby('Manager')['CSAT Score'].mean()
        
        scatter = axes[2, 0].scatter(manager_team_size, manager_performance, 
                                   s=100, alpha=0.7, color=self.colors['primary'])
        axes[2, 0].set_title('Manager Performance: Team Size vs CSAT')
        axes[2, 0].set_xlabel('Team Size')
        axes[2, 0].set_ylabel('Average CSAT')
        
        # Tenure impact
        tenure_order = ['0-30', '31-60', '61-90', '>90', 'On Job Training']
        tenure_available = [t for t in tenure_order if t in df['Tenure Bucket'].unique()]
        
        if tenure_available:
            tenure_csat = df[df['Tenure Bucket'].isin(tenure_available)].groupby('Tenure Bucket')['CSAT Score'].mean()
            tenure_csat = tenure_csat.reindex(tenure_available)
            
            bars = axes[2, 1].bar(range(len(tenure_csat)), tenure_csat.values, 
                                 color=self.colors['success'], alpha=0.8)
            axes[2, 1].set_title('Performance by Agent Tenure')
            axes[2, 1].set_xlabel('Tenure Bucket')
            axes[2, 1].set_ylabel('Average CSAT')
            axes[2, 1].set_xticks(range(len(tenure_csat)))
            axes[2, 1].set_xticklabels(tenure_csat.index, rotation=45, ha='right')
        
        # Shift workload analysis
        shift_volume = df.groupby('Agent Shift').size()
        shift_csat = df.groupby('Agent Shift')['CSAT Score'].mean()
        
        ax_twin = axes[2, 2].twinx()
        bars1 = axes[2, 2].bar(shift_volume.index, shift_volume.values, 
                              alpha=0.7, color=self.colors['primary'], label='Volume')
        line = ax_twin.plot(shift_csat.index, shift_csat.values, 
                           color=self.colors['secondary'], marker='o', linewidth=2, 
                           markersize=8, label='CSAT')
        
        axes[2, 2].set_title('Shift Analysis: Volume vs CSAT')
        axes[2, 2].set_xlabel('Agent Shift')
        axes[2, 2].set_ylabel('Volume', color=self.colors['primary'])
        ax_twin.set_ylabel('Average CSAT', color=self.colors['secondary'])
        
        plt.tight_layout()
        return fig
    
    def save_plot(self, fig, filename, dpi=300):
        """Save plot to file"""
        try:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            return True
        except Exception as e:
            print(f"Error saving plot: {str(e)}")
            return False
