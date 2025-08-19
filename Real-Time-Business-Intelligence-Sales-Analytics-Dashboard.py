# ========================================================
# CHOCOLATE SALES DASHBOARD - FIXED VERSION
# Save as: chocolate_dashboard_fixed.py
# Run: python -m streamlit run chocolate_dashboard_fixed.py
# ========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Check if required libraries are installed
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not installed. Prediction features will be limited.")

# ========================================================
# PAGE CONFIGURATION
# ========================================================

st.set_page_config(
    page_title="Chocolate Sales Analytics",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================
# DATA LOADING
# ========================================================

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        df = pd.read_csv('Chocolate Sales.csv')
        
        # Debug: Show raw data structure
        st.sidebar.text(f"Data loaded: {len(df)} rows")
        
        # Clean data
        df['Revenue'] = df['Amount'].str.replace('[$,\s]', '', regex=True).astype(float)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
        df['Price_per_Box'] = df['Revenue'] / df['Boxes Shipped']
        df['Month'] = df['Date'].dt.month
        df['Month_Name'] = df['Date'].dt.month_name()
        df['Quarter'] = df['Date'].dt.quarter
        df['Year_Month'] = df['Date'].dt.to_period('M').astype(str)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Load data
df = load_data()

# ========================================================
# SIDEBAR FILTERS
# ========================================================

st.sidebar.title("🍫 Filters")
st.sidebar.markdown("---")

# Date filter
st.sidebar.subheader("📅 Date Range")
date_range = st.sidebar.date_input(
    "Select dates",
    value=(df['Date'].min(), df['Date'].max()),
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

# Apply date filter
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[(df['Date'] >= pd.Timestamp(start_date)) & 
                     (df['Date'] <= pd.Timestamp(end_date))]
else:
    filtered_df = df.copy()

# Country filter
st.sidebar.subheader("🌍 Country")
all_countries = df['Country'].unique().tolist()
selected_countries = st.sidebar.multiselect(
    "Select countries",
    options=all_countries,
    default=all_countries
)
if selected_countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

# Product filter
st.sidebar.subheader("📦 Products")
all_products = df['Product'].unique().tolist()
selected_products = st.sidebar.multiselect(
    "Select products",
    options=all_products,
    default=all_products[:10] if len(all_products) > 10 else all_products
)
if selected_products:
    filtered_df = filtered_df[filtered_df['Product'].isin(selected_products)]

# ========================================================
# MAIN DASHBOARD
# ========================================================

st.title("🍫 Chocolate Sales Analytics Dashboard")
st.markdown("Real-time insights for strategic decision-making")

# Check if we have data after filtering
if filtered_df.empty:
    st.warning("No data available for selected filters. Please adjust your filters.")
    st.stop()

# ========================================================
# KPI CARDS
# ========================================================

st.markdown("### 📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_revenue = filtered_df['Revenue'].sum()
    st.metric("Total Revenue", f"${total_revenue:,.0f}")

with col2:
    total_orders = len(filtered_df)
    st.metric("Total Orders", f"{total_orders:,}")

with col3:
    avg_order = filtered_df['Revenue'].mean() if not filtered_df.empty else 0
    st.metric("Avg Order Value", f"${avg_order:,.0f}")

with col4:
    total_boxes = filtered_df['Boxes Shipped'].sum()
    st.metric("Boxes Shipped", f"{total_boxes:,}")

with col5:
    avg_price_box = filtered_df['Price_per_Box'].mean() if not filtered_df.empty else 0
    st.metric("Avg Price/Box", f"${avg_price_box:.2f}")

st.markdown("---")

# ========================================================
# MAIN VISUALIZATIONS
# ========================================================

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🌍 Geographic", "📦 Products", "👥 Sales Team"])

# ========================================================
# TAB 1: TRENDS
# ========================================================

with tab1:
    st.subheader("Revenue Trends")
    
    # Daily revenue trend
    daily_revenue = filtered_df.groupby('Date')['Revenue'].sum().reset_index()
    
    if not daily_revenue.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_revenue['Date'], 
            y=daily_revenue['Revenue'],
            mode='lines+markers',
            name='Daily Revenue',
            line=dict(color='#3498db', width=2)
        ))
        
        # Add 7-day moving average if we have enough data
        if len(daily_revenue) > 7:
            daily_revenue['MA7'] = daily_revenue['Revenue'].rolling(window=7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=daily_revenue['Date'],
                y=daily_revenue['MA7'],
                mode='lines',
                name='7-Day Average',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Daily Revenue Trend",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected period")
    
    # Monthly revenue
    st.subheader("Monthly Performance")
    monthly_revenue = filtered_df.groupby('Year_Month')['Revenue'].sum().reset_index()
    
    if not monthly_revenue.empty:
        fig = px.bar(
            monthly_revenue, 
            x='Year_Month', 
            y='Revenue',
            title='Monthly Revenue',
            text='Revenue'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Quarterly performance - FIXED
    st.subheader("Quarterly Performance")
    
    if not filtered_df.empty:
        quarterly_data = filtered_df.groupby('Quarter').agg({
            'Revenue': 'sum',
            'Boxes Shipped': 'sum'
        }).round(0)
        
        if not quarterly_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Create pie chart with actual quarters
                quarter_names = [f'Q{q}' for q in quarterly_data.index]
                fig = px.pie(
                    values=quarterly_data['Revenue'].values,
                    names=quarter_names,
                    title='Revenue by Quarter'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display quarterly metrics
                st.dataframe(
                    quarterly_data.style.format({
                        'Revenue': '${:,.0f}',
                        'Boxes Shipped': '{:,.0f}'
                    }),
                    use_container_width=True
                )

# ========================================================
# TAB 2: GEOGRAPHIC ANALYSIS
# ========================================================

with tab2:
    st.subheader("Geographic Performance")
    
    # Country performance
    country_data = filtered_df.groupby('Country').agg({
        'Revenue': 'sum',
        'Boxes Shipped': 'sum'
    }).sort_values('Revenue', ascending=False)
    
    if not country_data.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                country_data.reset_index(),
                x='Revenue',
                y='Country',
                orientation='h',
                title='Revenue by Country',
                text='Revenue'
            )
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            best_country = country_data['Revenue'].idxmax()
            st.success(f"**Top Market:** {best_country}")
            st.info(f"**Revenue:** ${country_data.loc[best_country, 'Revenue']:,.0f}")
            
            # Country table
            st.dataframe(
                country_data.style.format({
                    'Revenue': '${:,.0f}',
                    'Boxes Shipped': '{:,.0f}'
                }),
                use_container_width=True
            )
    
    # Product-Country Heatmap
    st.subheader("Product Performance by Country")
    
    if not filtered_df.empty:
        heatmap_data = filtered_df.pivot_table(
            values='Revenue',
            index='Product',
            columns='Country',
            aggfunc='sum',
            fill_value=0
        )
        
        if not heatmap_data.empty:
            fig = px.imshow(
                heatmap_data.values,
                labels=dict(x="Country", y="Product", color="Revenue ($)"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                title="Revenue Heatmap: Products × Countries",
                color_continuous_scale='RdYlGn',
                aspect='auto'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

# ========================================================
# TAB 3: PRODUCT ANALYSIS
# ========================================================

with tab3:
    st.subheader("Product Performance")
    
    # Product metrics
    product_data = filtered_df.groupby('Product').agg({
        'Revenue': 'sum',
        'Boxes Shipped': 'sum',
        'Price_per_Box': 'mean'
    }).sort_values('Revenue', ascending=False)
    
    if not product_data.empty:
        # Top 10 products chart
        top_products = product_data.head(10)
        
        fig = px.bar(
            top_products.reset_index(),
            x='Revenue',
            y='Product',
            orientation='h',
            title='Top 10 Products by Revenue',
            text='Revenue'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Product table
        st.subheader("Product Details")
        st.dataframe(
            product_data.head(15).style.format({
                'Revenue': '${:,.0f}',
                'Boxes Shipped': '{:,.0f}',
                'Price_per_Box': '${:.2f}'
            }).background_gradient(subset=['Revenue'], cmap='Greens'),
            use_container_width=True
        )
        
        # Pareto Analysis
        col1, col2 = st.columns(2)
        with col1:
            cumsum = product_data['Revenue'].cumsum()
            cutoff_80 = cumsum[cumsum <= product_data['Revenue'].sum() * 0.8]
            st.info(f"""
            **Pareto Analysis (80/20 Rule)**
            - Products for 80% revenue: {len(cutoff_80)} of {len(product_data)}
            - Focus on top products for maximum impact
            """)
        
        with col2:
            best_product = product_data.index[0] if not product_data.empty else "N/A"
            best_revenue = product_data['Revenue'].iloc[0] if not product_data.empty else 0
            st.success(f"""
            **Best Product:** {best_product}
            - Revenue: ${best_revenue:,.0f}
            - Market Share: {(best_revenue/product_data['Revenue'].sum())*100:.1f}%
            """)

# ========================================================
# TAB 4: SALES TEAM
# ========================================================

with tab4:
    st.subheader("Sales Team Performance")
    
    # Sales person metrics
    sales_data = filtered_df.groupby('Sales Person').agg({
        'Revenue': 'sum',
        'Boxes Shipped': 'sum',
        'Date': 'count'
    }).rename(columns={'Date': 'Orders'})
    
    if not sales_data.empty:
        sales_data['Efficiency'] = sales_data['Revenue'] / sales_data['Orders']
        sales_data = sales_data.sort_values('Revenue', ascending=False)
        
        # Top performers chart
        top_sales = sales_data.head(10)
        
        fig = px.bar(
            top_sales.reset_index(),
            x='Revenue',
            y='Sales Person',
            orientation='h',
            title='Top 10 Sales Performers',
            text='Revenue',
            color='Efficiency',
            color_continuous_scale='RdYlGn',
            labels={'Efficiency': 'Avg Sale ($)'}
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            top_performer = sales_data.index[0] if not sales_data.empty else "N/A"
            top_revenue = sales_data['Revenue'].iloc[0] if not sales_data.empty else 0
            st.success(f"""
            **Top Performer:** {top_performer}
            - Revenue: ${top_revenue:,.0f}
            - Orders: {sales_data['Orders'].iloc[0] if not sales_data.empty else 0:,.0f}
            """)
        
        with col2:
            if not sales_data.empty and 'Efficiency' in sales_data.columns:
                most_efficient = sales_data['Efficiency'].idxmax()
                efficiency_value = sales_data.loc[most_efficient, 'Efficiency']
                st.info(f"""
                **Most Efficient:** {most_efficient}
                - Avg Sale: ${efficiency_value:,.0f}
                - High value per transaction
                """)
        
        # Team performance table
        st.subheader("Team Details")
        display_cols = ['Revenue', 'Orders', 'Boxes Shipped', 'Efficiency']
        st.dataframe(
            sales_data[display_cols].head(15).style.format({
                'Revenue': '${:,.0f}',
                'Orders': '{:,.0f}',
                'Boxes Shipped': '{:,.0f}',
                'Efficiency': '${:,.0f}'
            }).background_gradient(subset=['Revenue'], cmap='Blues'),
            use_container_width=True
        )

# ========================================================
# FOOTER
# ========================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🍫 Chocolate Sales Analytics Dashboard</p>
    <p>Built with Streamlit & Plotly | Data Analysis Project</p>
</div>
""", unsafe_allow_html=True)

# ========================================================
# SIDEBAR - SUMMARY STATS
# ========================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Summary Stats")
st.sidebar.info(f"""
**Filtered Data:**
- Records: {len(filtered_df):,}
- Revenue: ${filtered_df['Revenue'].sum():,.0f}
- Date Range: {filtered_df['Date'].min().date()} to {filtered_df['Date'].max().date()}
""")

# Download button for filtered data
@st.cache_data
def convert_df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(filtered_df)
st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=csv,
    file_name=f'chocolate_data_{datetime.now().strftime("%Y%m%d")}.csv',
    mime='text/csv'
)
