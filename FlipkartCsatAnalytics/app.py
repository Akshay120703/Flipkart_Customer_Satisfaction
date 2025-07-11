import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import DataProcessor
from ml_models import MLModels
from visualization import Visualizer
from utils import Utils

# Set page configuration
st.set_page_config(
    page_title="Flipkart Customer Satisfaction Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional CSS Styling
st.markdown("""
<style>
    /* Main container and layout */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #047BD6 0%, #FB641B 50%, #228B22 100%);
        color: white;
        text-align: center;
        padding: 3rem 2rem;
        margin: -2rem -2rem 3rem -2rem;
        border-radius: 0 0 30px 30px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="white" stroke-width="0.1" opacity="0.3"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
        opacity: 0.1;
    }
    
    .main-header h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        position: relative;
        z-index: 1;
        margin: 1rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Enhanced metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 2rem;
        border: none;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 5px solid #047BD6;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: linear-gradient(135deg, #047BD6, #FB641B);
        opacity: 0.05;
        border-radius: 50%;
        transform: translate(30px, -30px);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    }
    
    /* Enhanced insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #E8F4FD 0%, #F0F8FF 100%);
        border: none;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 6px 20px rgba(4, 123, 214, 0.1);
        border-left: 6px solid #047BD6;
        position: relative;
        overflow: hidden;
    }
    
    .insight-box::before {
        content: '💡';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 2rem;
        opacity: 0.3;
    }
    
    /* Navigation styling */
    .nav-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #E8F4FD;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #047BD6, #228B22);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 2rem;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #E8F4FD;
    }
    
    /* Enhanced data displays */
    .dataframe-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
        border: 1px solid #f0f0f0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #F8F9FA 0%, #E8F4FD 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #047BD6 0%, #228B22 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(4, 123, 214, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(4, 123, 214, 0.4);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: white;
        border-radius: 10px;
        border: 2px solid #E8F4FD;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #047BD6;
        box-shadow: 0 0 0 3px rgba(4, 123, 214, 0.1);
    }
    
    /* Metric value styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border-left: 5px solid #047BD6;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
        transition: all 0.3s ease;
    }
    
    /* Chart container */
    .plot-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
        border: 1px solid #f0f0f0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        border-radius: 15px;
        padding: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #F1F3F6 0%, #E8F4FD 100%);
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #047BD6 0%, #228B22 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(4, 123, 214, 0.3);
    }
    
    /* Loading spinner */
    .stSpinner {
        color: #047BD6 !important;
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #fce8b2 100%);
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    .status-info {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        color: #0c5460;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class CustomerSatisfactionDashboard:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.ml_models = MLModels()
        self.visualizer = Visualizer()
        self.utils = Utils()
        
    def get_column_name(self, df, original_name):
        """Get the correct column name after processing"""
        mapping = {
            'CSAT Score': 'CSAT_Score',
            'Sub-category': 'Sub_category',
            'Tenure Bucket': 'Tenure_Bucket',
            'Agent Shift': 'Agent_Shift',
            'Issue_reported at': 'Issue_reported_at'
        }
        
        # Check if cleaned name exists
        if original_name in mapping and mapping[original_name] in df.columns:
            return mapping[original_name]
        # Otherwise return original name
        return original_name
        
    def load_data(self):
        """Load and cache the dataset"""
        try:
            # Try to load from uploaded file first
            if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
                df = pd.read_csv(st.session_state.uploaded_file)
            else:
                # Try to load from local file
                df = pd.read_csv('attached_assets/Customer_support_data_1752131883923.csv')
            
            # Clean column names by removing extra spaces
            df.columns = df.columns.str.strip()
            
            # Debug: print column names
            print(f"Loaded columns: {df.columns.tolist()}")
            
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def display_header(self):
        """Display main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🛒 Flipkart Customer Satisfaction Analytics</h1>
            <p>Comprehensive insights into customer satisfaction, team performance, and business intelligence</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_overview_metrics(self, df):
        """Display key metrics overview"""
        st.markdown("## 📊 Key Performance Indicators")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_interactions = len(df)
            st.metric("Total Interactions", f"{total_interactions:,}")
        
        with col2:
            csat_col = self.get_column_name(df, 'CSAT Score')
            avg_csat = df[csat_col].mean()
            st.metric("Average CSAT Score", f"{avg_csat:.2f}")
        
        with col3:
            satisfaction_rate = (df[csat_col] >= 4).mean() * 100
            st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
        
        with col4:
            unique_agents = df['Agent_name'].nunique()
            st.metric("Active Agents", f"{unique_agents:,}")
        
        with col5:
            avg_handling_time = df['connected_handling_time'].dropna().mean()
            if not pd.isna(avg_handling_time):
                st.metric("Avg Handling Time", f"{avg_handling_time:.1f} min")
            else:
                st.metric("Avg Handling Time", "N/A")
    
    def exploratory_data_analysis(self, df):
        """Comprehensive EDA section"""
        st.markdown("## 🔍 Exploratory Data Analysis")
        
        # Data overview
        with st.expander("📋 Dataset Overview", expanded=True):
            st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write("**Column Information:**")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info)
        
        # Create tabs for different analysis types
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Univariate Analysis", "🔗 Bivariate Analysis", "🎯 Multivariate Analysis", "🧹 Data Quality"])
        
        with tab1:
            self.univariate_analysis(df)
        
        with tab2:
            self.bivariate_analysis(df)
        
        with tab3:
            self.multivariate_analysis(df)
        
        with tab4:
            self.data_quality_analysis(df)
    
    def univariate_analysis(self, df):
        """Univariate analysis with visualizations"""
        st.markdown("### 📊 Univariate Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSAT Score Distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            csat_col = self.get_column_name(df, 'CSAT Score')
            csat_counts = df[csat_col].value_counts().sort_index()
            colors = ['#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#047BD6']
            bars = ax.bar(csat_counts.index, csat_counts.values, color=colors[:len(csat_counts)])
            ax.set_title('CSAT Score Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('CSAT Score')
            ax.set_ylabel('Count')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Insight
            st.markdown("""
            <div class="insight-box">
            <strong>📈 Insight:</strong> Most customers rate their satisfaction highly (4-5), indicating generally positive service quality. 
            However, the presence of lower scores (1-3) represents opportunities for improvement.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Channel Distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            channel_counts = df['channel_name'].value_counts()
            colors = ['#047BD6', '#FB641B', '#228B22']
            wedges, texts, autotexts = ax.pie(channel_counts.values, labels=channel_counts.index, 
                                            autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Support Channel Distribution', fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            <div class="insight-box">
            <strong>📞 Insight:</strong> Channel preference analysis helps optimize resource allocation 
            and identify the most effective communication channels for customer support.
            </div>
            """, unsafe_allow_html=True)
        
        # Additional univariate analyses
        col3, col4 = st.columns(2)
        
        with col3:
            # Category Analysis
            fig, ax = plt.subplots(figsize=(12, 8))
            category_counts = df['category'].value_counts()
            bars = ax.barh(category_counts.index, category_counts.values, color='#047BD6')
            ax.set_title('Support Category Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Count')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{int(width)}', ha='left', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            <div class="insight-box">
            <strong>🎯 Insight:</strong> Returns and Order Related issues dominate support requests, 
            indicating potential areas for process improvement and proactive customer communication.
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Agent Shift Analysis
            fig, ax = plt.subplots(figsize=(10, 6))
            shift_col = self.get_column_name(df, 'Agent Shift')
            shift_counts = df[shift_col].value_counts()
            colors = ['#047BD6', '#FB641B', '#228B22', '#9B59B6']
            bars = ax.bar(shift_counts.index, shift_counts.values, color=colors[:len(shift_counts)])
            ax.set_title('Agent Shift Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Shift')
            ax.set_ylabel('Count')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            <div class="insight-box">
            <strong>⏰ Insight:</strong> Shift distribution analysis helps in workforce planning 
            and identifying peak support hours for better resource allocation.
            </div>
            """, unsafe_allow_html=True)
    
    def bivariate_analysis(self, df):
        """Bivariate analysis with visualizations"""
        st.markdown("### 🔗 Bivariate Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSAT by Channel
            fig, ax = plt.subplots(figsize=(12, 8))
            csat_col = self.get_column_name(df, 'CSAT Score')
            channel_csat = df.groupby('channel_name')[csat_col].mean().sort_values(ascending=False)
            bars = ax.bar(channel_csat.index, channel_csat.values, color=['#047BD6', '#FB641B', '#228B22'])
            ax.set_title('Average CSAT Score by Channel', fontsize=16, fontweight='bold')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Average CSAT Score')
            ax.set_ylim(0, 5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            <div class="insight-box">
            <strong>📊 Insight:</strong> Different channels show varying satisfaction levels. 
            This analysis helps identify which channels provide better customer experience.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # CSAT by Category
            fig, ax = plt.subplots(figsize=(12, 8))
            csat_col = self.get_column_name(df, 'CSAT Score')
            category_csat = df.groupby('category')[csat_col].mean().sort_values(ascending=False)
            bars = ax.barh(category_csat.index, category_csat.values, color='#FB641B')
            ax.set_title('Average CSAT Score by Category', fontsize=16, fontweight='bold')
            ax.set_xlabel('Average CSAT Score')
            ax.set_xlim(0, 5)
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.2f}', ha='left', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            <div class="insight-box">
            <strong>🎯 Insight:</strong> Category-wise satisfaction reveals which service areas 
            need attention and which are performing well.
            </div>
            """, unsafe_allow_html=True)
        
        # Additional bivariate analyses
        col3, col4 = st.columns(2)
        
        with col3:
            # CSAT by Tenure Bucket
            fig, ax = plt.subplots(figsize=(10, 6))
            tenure_col = self.get_column_name(df, 'Tenure Bucket')
            csat_col = self.get_column_name(df, 'CSAT Score')
            tenure_csat = df.groupby(tenure_col)[csat_col].mean().sort_values(ascending=False)
            bars = ax.bar(tenure_csat.index, tenure_csat.values, color='#228B22')
            ax.set_title('Average CSAT Score by Agent Tenure', fontsize=16, fontweight='bold')
            ax.set_xlabel('Tenure Bucket')
            ax.set_ylabel('Average CSAT Score')
            ax.set_ylim(0, 5)
            plt.xticks(rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            <div class="insight-box">
            <strong>👥 Insight:</strong> Agent experience correlates with customer satisfaction. 
            This helps in training and development planning.
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # CSAT by Shift
            fig, ax = plt.subplots(figsize=(10, 6))
            shift_col = self.get_column_name(df, 'Agent Shift')
            csat_col = self.get_column_name(df, 'CSAT Score')
            shift_csat = df.groupby(shift_col)[csat_col].mean().sort_values(ascending=False)
            bars = ax.bar(shift_csat.index, shift_csat.values, color='#9B59B6')
            ax.set_title('Average CSAT Score by Agent Shift', fontsize=16, fontweight='bold')
            ax.set_xlabel('Agent Shift')
            ax.set_ylabel('Average CSAT Score')
            ax.set_ylim(0, 5)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            <div class="insight-box">
            <strong>🕐 Insight:</strong> Shift timing impacts satisfaction levels, possibly due to 
            workload variations or agent performance across different times.
            </div>
            """, unsafe_allow_html=True)
    
    def multivariate_analysis(self, df):
        """Multivariate analysis with complex visualizations"""
        st.markdown("### 🎯 Multivariate Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation Heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Select numeric columns for correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = df[numeric_cols].corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Correlation Matrix of Numeric Variables', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            <div class="insight-box">
            <strong>🔗 Insight:</strong> Correlation analysis reveals relationships between variables 
            that can inform feature selection for predictive modeling.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # CSAT Distribution by Channel and Category
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create a pivot table for the heatmap
            csat_col = self.get_column_name(df, 'CSAT Score')
            pivot_table = df.pivot_table(values=csat_col, 
                                       index='category', 
                                       columns='channel_name', 
                                       aggfunc='mean')
            
            sns.heatmap(pivot_table, annot=True, cmap='RdYlBu_r', center=3,
                       square=False, ax=ax, cbar_kws={'label': 'Average CSAT Score'})
            ax.set_title('CSAT Score Heatmap: Category vs Channel', fontsize=16, fontweight='bold')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Category')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            <div class="insight-box">
            <strong>🎯 Insight:</strong> This cross-analysis identifies which category-channel 
            combinations perform best or worst, guiding targeted improvements.
            </div>
            """, unsafe_allow_html=True)
        
        # Manager vs Agent Performance Analysis
        st.markdown("#### 👥 Team Performance Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Manager Performance
            fig, ax = plt.subplots(figsize=(12, 8))
            csat_col = self.get_column_name(df, 'CSAT Score')
            manager_performance = df.groupby('Manager')[csat_col].agg(['mean', 'count']).sort_values('mean', ascending=False)
            
            bars = ax.bar(range(len(manager_performance)), manager_performance['mean'], 
                         color='#047BD6', alpha=0.7)
            ax.set_title('Average CSAT Score by Manager', fontsize=16, fontweight='bold')
            ax.set_xlabel('Manager')
            ax.set_ylabel('Average CSAT Score')
            ax.set_xticks(range(len(manager_performance)))
            ax.set_xticklabels(manager_performance.index, rotation=45, ha='right')
            ax.set_ylim(0, 5)
            
            # Add count labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                count = manager_performance.iloc[i]['count']
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}\n(n={count})', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col4:
            # Top 10 Agents Performance
            fig, ax = plt.subplots(figsize=(12, 8))
            csat_col = self.get_column_name(df, 'CSAT Score')
            agent_performance = df.groupby('Agent_name')[csat_col].agg(['mean', 'count'])
            top_agents = agent_performance[agent_performance['count'] >= 10].sort_values('mean', ascending=False).head(10)
            
            bars = ax.bar(range(len(top_agents)), top_agents['mean'], color='#FB641B', alpha=0.7)
            ax.set_title('Top 10 Agents by Average CSAT Score', fontsize=16, fontweight='bold')
            ax.set_xlabel('Agent')
            ax.set_ylabel('Average CSAT Score')
            ax.set_xticks(range(len(top_agents)))
            ax.set_xticklabels(top_agents.index, rotation=45, ha='right')
            ax.set_ylim(0, 5)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                count = top_agents.iloc[i]['count']
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}\n(n={count})', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("""
        <div class="insight-box">
        <strong>🏆 Insight:</strong> Team performance analysis helps identify top performers 
        and areas needing management attention, enabling targeted coaching and recognition.
        </div>
        """, unsafe_allow_html=True)
    
    def data_quality_analysis(self, df):
        """Data quality analysis"""
        st.markdown("### 🧹 Data Quality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing Values Analysis
            fig, ax = plt.subplots(figsize=(12, 8))
            missing_data = df.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) > 0:
                bars = ax.bar(range(len(missing_data)), missing_data.values, color='#FF6B6B')
                ax.set_title('Missing Values by Column', fontsize=16, fontweight='bold')
                ax.set_xlabel('Column')
                ax.set_ylabel('Missing Count')
                ax.set_xticks(range(len(missing_data)))
                ax.set_xticklabels(missing_data.index, rotation=45, ha='right')
                
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    percentage = (height / len(df)) * 100
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom')
            else:
                ax.text(0.5, 0.5, 'No Missing Values Found!', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=20, color='green')
                ax.set_title('Missing Values Analysis', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Data Type Distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            dtype_counts = df.dtypes.value_counts()
            colors = ['#047BD6', '#FB641B', '#228B22', '#9B59B6']
            
            wedges, texts, autotexts = ax.pie(dtype_counts.values, labels=dtype_counts.index, 
                                            autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Data Type Distribution', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Data Quality Summary
        st.markdown("#### 📊 Data Quality Summary")
        
        quality_metrics = {
            'Total Records': len(df),
            'Total Columns': len(df.columns),
            'Missing Values': df.isnull().sum().sum(),
            'Duplicate Records': df.duplicated().sum(),
            'Numeric Columns': len(df.select_dtypes(include=[np.number]).columns),
            'Categorical Columns': len(df.select_dtypes(include=['object']).columns),
            'Data Completeness': f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%"
        }
        
        col1, col2, col3, col4 = st.columns(4)
        metrics_items = list(quality_metrics.items())
        
        for i, (key, value) in enumerate(metrics_items):
            col = [col1, col2, col3, col4][i % 4]
            with col:
                st.metric(key, value)
        
        st.markdown("""
        <div class="insight-box">
        <strong>✅ Data Quality Insight:</strong> Understanding data quality is crucial for 
        reliable analysis. Missing values and duplicates can impact model performance and insights.
        </div>
        """, unsafe_allow_html=True)
    
    def machine_learning_analysis(self, df):
        """Machine learning prediction models"""
        st.markdown("## 🤖 Machine Learning Analysis")
        
        # Prepare data for ML
        ml_data = self.ml_models.prepare_data(df)
        
        if ml_data is not None:
            X, y = ml_data
            
            # Train and evaluate models
            results = self.ml_models.train_models(X, y)
            
            # Display results
            st.markdown("### 📈 Model Performance Comparison")
            
            # Create results DataFrame
            results_df = pd.DataFrame(results).T
            st.dataframe(results_df.style.highlight_max(axis=0))
            
            # Visualize model comparison
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(12, 8))
                models = list(results.keys())
                accuracy_scores = [results[model]['accuracy'] for model in models]
                
                bars = ax.bar(models, accuracy_scores, color=['#047BD6', '#FB641B', '#228B22', '#9B59B6'])
                ax.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
                ax.set_xlabel('Model')
                ax.set_ylabel('Accuracy')
                ax.set_ylim(0, 1)
                
                for bar, score in zip(bars, accuracy_scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.3f}', ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(12, 8))
                f1_scores = [results[model]['f1_score'] for model in models]
                
                bars = ax.bar(models, f1_scores, color=['#047BD6', '#FB641B', '#228B22', '#9B59B6'])
                ax.set_title('Model F1-Score Comparison', fontsize=16, fontweight='bold')
                ax.set_xlabel('Model')
                ax.set_ylabel('F1-Score')
                ax.set_ylim(0, 1)
                
                for bar, score in zip(bars, f1_scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.3f}', ha='center', va='bottom')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            # Feature importance
            st.markdown("### 🎯 Feature Importance Analysis")
            
            # Get feature importance from Random Forest
            feature_importance = self.ml_models.get_feature_importance(X, y)
            
            if feature_importance is not None:
                fig, ax = plt.subplots(figsize=(12, 8))
                features = list(feature_importance.keys())
                importance_values = list(feature_importance.values())
                
                bars = ax.barh(features, importance_values, color='#047BD6')
                ax.set_title('Feature Importance (Random Forest)', fontsize=16, fontweight='bold')
                ax.set_xlabel('Importance')
                
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2.,
                           f'{width:.3f}', ha='left', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                <div class="insight-box">
                <strong>🔍 ML Insight:</strong> Feature importance reveals which factors most 
                significantly impact customer satisfaction, enabling targeted improvements.
                </div>
                """, unsafe_allow_html=True)
    
    def team_performance_analysis(self, df):
        """Detailed team performance analysis"""
        st.markdown("## 👥 Team Performance Analysis")
        
        # Agent Performance Dashboard
        st.markdown("### 🏆 Agent Performance Dashboard")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_manager = st.selectbox(
                "Select Manager",
                options=['All'] + list(df['Manager'].unique())
            )
        
        with col2:
            selected_shift = st.selectbox(
                "Select Shift",
                options=['All'] + list(df['Agent Shift'].unique())
            )
        
        with col3:
            min_interactions = st.slider(
                "Minimum Interactions",
                min_value=1,
                max_value=100,
                value=10
            )
        
        # Filter data
        filtered_df = df.copy()
        if selected_manager != 'All':
            filtered_df = filtered_df[filtered_df['Manager'] == selected_manager]
        if selected_shift != 'All':
            filtered_df = filtered_df[filtered_df['Agent Shift'] == selected_shift]
        
        # Calculate agent metrics
        agent_metrics = filtered_df.groupby('Agent_name').agg({
            'CSAT Score': ['mean', 'count'],
            'connected_handling_time': 'mean'
        }).round(2)
        
        agent_metrics.columns = ['Avg_CSAT', 'Total_Interactions', 'Avg_Handling_Time']
        agent_metrics = agent_metrics[agent_metrics['Total_Interactions'] >= min_interactions]
        agent_metrics = agent_metrics.sort_values('Avg_CSAT', ascending=False)
        
        # Display top performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🥇 Top 10 Performers")
            top_performers = agent_metrics.head(10)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.bar(range(len(top_performers)), top_performers['Avg_CSAT'], 
                         color='#047BD6', alpha=0.8)
            ax.set_title('Top 10 Agents by CSAT Score', fontsize=16, fontweight='bold')
            ax.set_xlabel('Agent Rank')
            ax.set_ylabel('Average CSAT Score')
            ax.set_xticks(range(len(top_performers)))
            ax.set_xticklabels([f"#{i+1}" for i in range(len(top_performers))])
            ax.set_ylim(0, 5)
            
            for i, bar in enumerate(bars):
                height = bar.get_height()
                interactions = top_performers.iloc[i]['Total_Interactions']
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}\n({interactions} calls)', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 📊 Performance Distribution")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(agent_metrics['Avg_CSAT'], bins=20, color='#FB641B', alpha=0.7, edgecolor='black')
            ax.set_title('Agent CSAT Score Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Average CSAT Score')
            ax.set_ylabel('Number of Agents')
            ax.axvline(agent_metrics['Avg_CSAT'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {agent_metrics["Avg_CSAT"].mean():.2f}')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Detailed agent table
        st.markdown("#### 📋 Detailed Agent Performance")
        
        # Add agent details
        agent_details = filtered_df.groupby('Agent_name').agg({
            'Manager': 'first',
            'Supervisor': 'first',
            'Agent Shift': 'first',
            'Tenure Bucket': 'first'
        })
        
        final_metrics = agent_metrics.join(agent_details)
        final_metrics = final_metrics.reset_index()
        
        # Style the dataframe
        styled_df = final_metrics.style.format({
            'Avg_CSAT': '{:.2f}',
            'Avg_Handling_Time': '{:.1f}'
        }).background_gradient(subset=['Avg_CSAT'], cmap='RdYlGn')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Team insights
        st.markdown("### 💡 Team Performance Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_team_csat = agent_metrics['Avg_CSAT'].mean()
            st.metric("Team Average CSAT", f"{avg_team_csat:.2f}")
        
        with col2:
            top_performer_csat = agent_metrics['Avg_CSAT'].max()
            st.metric("Top Performer CSAT", f"{top_performer_csat:.2f}")
        
        with col3:
            performance_std = agent_metrics['Avg_CSAT'].std()
            st.metric("Performance Variability", f"{performance_std:.2f}")
        
        # Manager comparison
        st.markdown("### 👨‍💼 Manager Performance Comparison")
        
        csat_col = self.get_column_name(df, 'CSAT Score')
        manager_performance = df.groupby('Manager').agg({
            csat_col: ['mean', 'count'],
            'Agent_name': 'nunique'
        }).round(2)
        
        manager_performance.columns = ['Avg_CSAT', 'Total_Interactions', 'Team_Size']
        manager_performance = manager_performance.sort_values('Avg_CSAT', ascending=False)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.bar(range(len(manager_performance)), manager_performance['Avg_CSAT'], 
                     color='#228B22', alpha=0.8)
        ax.set_title('Manager Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Manager')
        ax.set_ylabel('Average Team CSAT Score')
        ax.set_xticks(range(len(manager_performance)))
        ax.set_xticklabels(manager_performance.index, rotation=45, ha='right')
        ax.set_ylim(0, 5)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            team_size = manager_performance.iloc[i]['Team_Size']
            interactions = manager_performance.iloc[i]['Total_Interactions']
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}\n({team_size} agents)\n({interactions} calls)', 
                   ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def channel_analysis(self, df):
        """Support channel analysis"""
        st.markdown("## 📞 Support Channel Analysis")
        
        # Channel performance metrics
        csat_col = self.get_column_name(df, 'CSAT Score')
        channel_metrics = df.groupby('channel_name').agg({
            csat_col: ['mean', 'count'],
            'connected_handling_time': 'mean'
        }).round(2)
        
        channel_metrics.columns = ['Avg_CSAT', 'Total_Interactions', 'Avg_Handling_Time']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Email CSAT", f"{channel_metrics.loc['Email', 'Avg_CSAT']:.2f}")
        
        with col2:
            st.metric("Inbound CSAT", f"{channel_metrics.loc['Inbound', 'Avg_CSAT']:.2f}")
        
        with col3:
            st.metric("Outcall CSAT", f"{channel_metrics.loc['Outcall', 'Avg_CSAT']:.2f}")
        
        # Channel analysis visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Channel volume vs satisfaction
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            channels = channel_metrics.index
            volumes = channel_metrics['Total_Interactions']
            csat_scores = channel_metrics['Avg_CSAT']
            
            color = '#047BD6'
            ax1.set_xlabel('Channel')
            ax1.set_ylabel('Total Interactions', color=color)
            bars1 = ax1.bar(channels, volumes, color=color, alpha=0.7)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = '#FB641B'
            ax2.set_ylabel('Average CSAT Score', color=color)
            line = ax2.plot(channels, csat_scores, color=color, marker='o', linewidth=3, markersize=8)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0, 5)
            
            plt.title('Channel Volume vs CSAT Score', fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Category distribution by channel
            fig, ax = plt.subplots(figsize=(12, 8))
            
            channel_category = pd.crosstab(df['channel_name'], df['category'], normalize='index') * 100
            channel_category.plot(kind='bar', stacked=True, ax=ax, 
                                color=['#047BD6', '#FB641B', '#228B22', '#9B59B6', '#FF6B6B'])
            ax.set_title('Category Distribution by Channel (%)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Percentage')
            ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Channel insights
        st.markdown("### 💡 Channel Performance Insights")
        
        best_channel = channel_metrics.loc[channel_metrics['Avg_CSAT'].idxmax()]
        worst_channel = channel_metrics.loc[channel_metrics['Avg_CSAT'].idxmin()]
        
        st.markdown(f"""
        <div class="insight-box">
        <strong>🏆 Best Performing Channel:</strong> {channel_metrics['Avg_CSAT'].idxmax()} 
        (CSAT: {best_channel['Avg_CSAT']:.2f}, Volume: {best_channel['Total_Interactions']:,})
        <br><br>
        <strong>⚠️ Needs Attention:</strong> {channel_metrics['Avg_CSAT'].idxmin()} 
        (CSAT: {worst_channel['Avg_CSAT']:.2f}, Volume: {worst_channel['Total_Interactions']:,})
        </div>
        """, unsafe_allow_html=True)
    
    def business_insights(self, df):
        """Business insights and recommendations"""
        st.markdown("## 💼 Business Insights & Recommendations")
        
        # Key findings
        st.markdown("### 🔍 Key Findings")
        
        # Calculate key metrics
        csat_col = self.get_column_name(df, 'CSAT Score')
        overall_satisfaction = (df[csat_col] >= 4).mean() * 100
        avg_csat = df[csat_col].mean()
        total_interactions = len(df)
        
        # Category performance
        category_performance = df.groupby('category')[csat_col].mean().sort_values(ascending=False)
        best_category = category_performance.index[0]
        worst_category = category_performance.index[-1]
        
        # Agent performance
        agent_performance = df.groupby('Agent_name')[csat_col].agg(['mean', 'count'])
        qualified_agents = agent_performance[agent_performance['count'] >= 10]
        top_agent_csat = qualified_agents['mean'].max()
        
        findings = [
            f"📊 **Overall Satisfaction Rate**: {overall_satisfaction:.1f}% of customers rate service as satisfactory (4+ stars)",
            f"⭐ **Average CSAT Score**: {avg_csat:.2f} out of 5.0",
            f"📈 **Total Interactions Analyzed**: {total_interactions:,} customer interactions",
            f"🏆 **Best Performing Category**: {best_category} ({category_performance[best_category]:.2f} CSAT)",
            f"⚠️ **Category Needing Attention**: {worst_category} ({category_performance[worst_category]:.2f} CSAT)",
            f"🥇 **Top Agent Performance**: {top_agent_csat:.2f} CSAT score achieved by best performer"
        ]
        
        for finding in findings:
            st.markdown(finding)
        
        # Recommendations
        st.markdown("### 🎯 Strategic Recommendations")
        
        recommendations = [
            {
                "category": "🔄 Process Improvement",
                "items": [
                    "Focus on improving Returns and Order Related processes as they dominate support volume",
                    "Implement proactive communication for order delays to reduce incoming queries",
                    "Streamline return pickup processes to improve customer experience"
                ]
            },
            {
                "category": "👥 Team Development",
                "items": [
                    "Implement mentorship programs pairing experienced agents with newcomers",
                    "Provide targeted training for agents in low-performing categories",
                    "Recognize and reward top performers to maintain motivation"
                ]
            },
            {
                "category": "📞 Channel Optimization",
                "items": [
                    "Optimize channel allocation based on customer preferences and satisfaction scores",
                    "Implement channel-specific training for agents",
                    "Consider automating simple queries to reduce agent workload"
                ]
            },
            {
                "category": "📊 Data-Driven Decisions",
                "items": [
                    "Regular monitoring of CSAT trends and early warning systems",
                    "Implement real-time feedback collection for immediate issue resolution",
                    "Use predictive analytics to identify at-risk customer interactions"
                ]
            }
        ]
        
        for rec in recommendations:
            st.markdown(f"#### {rec['category']}")
            for item in rec['items']:
                st.markdown(f"• {item}")
        
        # Expected business impact
        st.markdown("### 📈 Expected Business Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 Financial Impact")
            st.markdown("""
            - **Customer Retention**: 5-10% improvement in CSAT can lead to 25-50% increase in customer lifetime value
            - **Support Cost Reduction**: Process improvements can reduce average handling time by 15-20%
            - **Brand Loyalty**: Higher satisfaction correlates with increased customer advocacy and referrals
            """)
        
        with col2:
            st.markdown("#### 🎯 Operational Impact")
            st.markdown("""
            - **Agent Productivity**: Targeted training can improve resolution rates by 20-30%
            - **First Call Resolution**: Better processes can increase FCR by 15-25%
            - **Customer Effort Score**: Streamlined processes reduce customer effort and increase satisfaction
            """)
        
        # Implementation roadmap
        st.markdown("### 🗺️ Implementation Roadmap")
        
        roadmap = [
            ("Phase 1 (0-30 days)", "Immediate Actions", [
                "Implement agent performance dashboards",
                "Start weekly CSAT monitoring",
                "Identify and address critical pain points"
            ]),
            ("Phase 2 (30-90 days)", "Process Improvements", [
                "Launch agent training programs",
                "Optimize channel allocation",
                "Implement process improvements for top issue categories"
            ]),
            ("Phase 3 (90-180 days)", "Advanced Analytics", [
                "Deploy predictive models for customer satisfaction",
                "Implement real-time feedback systems",
                "Advanced team performance analytics"
            ])
        ]
        
        for phase, title, actions in roadmap:
            st.markdown(f"#### {phase}: {title}")
            for action in actions:
                st.markdown(f"• {action}")
    
    def run(self):
        """Main application runner"""
        # Display header
        self.display_header()
        
        # File upload option
        uploaded_file = st.file_uploader("Upload Customer Support Data (CSV)", type=['csv'])
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
        
        # Load data
        df = self.load_data()
        
        if df is not None:
            # Process data
            df = self.data_processor.clean_data(df)
            
            # Display overview metrics
            self.display_overview_metrics(df)
            
            # Create navigation
            st.sidebar.markdown("## 🧭 Navigation")
            page = st.sidebar.radio("Select Analysis", [
                "📊 Overview",
                "🔍 Exploratory Data Analysis",
                "🤖 Machine Learning",
                "👥 Team Performance",
                "📞 Channel Analysis",
                "💼 Business Insights"
            ])
            
            # Display selected page
            if page == "📊 Overview":
                st.markdown("### 📊 Dashboard Overview")
                st.markdown("""
                This comprehensive dashboard provides insights into customer satisfaction across 
                Flipkart's support channels. Use the navigation menu to explore different analyses.
                """)
                
                # Quick stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**🎯 Key Metrics**")
                    st.markdown(f"• Total Records: {len(df):,}")
                    csat_col = self.get_column_name(df, 'CSAT Score')
                    st.markdown(f"• Avg CSAT: {df[csat_col].mean():.2f}")
                    st.markdown(f"• Satisfaction Rate: {(df[csat_col] >= 4).mean()*100:.1f}%")
                
                with col2:
                    st.markdown("**📞 Channels**")
                    for channel, count in df['channel_name'].value_counts().items():
                        st.markdown(f"• {channel}: {count:,}")
                
                with col3:
                    st.markdown("**👥 Team**")
                    st.markdown(f"• Agents: {df['Agent_name'].nunique()}")
                    st.markdown(f"• Managers: {df['Manager'].nunique()}")
                    st.markdown(f"• Supervisors: {df['Supervisor'].nunique()}")
            
            elif page == "🔍 Exploratory Data Analysis":
                self.exploratory_data_analysis(df)
            
            elif page == "🤖 Machine Learning":
                self.machine_learning_analysis(df)
            
            elif page == "👥 Team Performance":
                self.team_performance_analysis(df)
            
            elif page == "📞 Channel Analysis":
                self.channel_analysis(df)
            
            elif page == "💼 Business Insights":
                self.business_insights(df)
        
        else:
            st.error("Please upload a CSV file or ensure the data file is available.")
            st.markdown("""
            ### 📋 Expected Data Format
            
            The application expects a CSV file with the following columns:
            - Unique id
            - channel_name (Inbound, Outcall, Email)
            - category (Order Related, Returns, etc.)
            - CSAT Score (1-5)
            - Agent_name
            - Manager
            - Supervisor
            - Tenure Bucket
            - Agent Shift
            - And other relevant columns...
            """)

# Main application
if __name__ == "__main__":
    app = CustomerSatisfactionDashboard()
    app.run()
