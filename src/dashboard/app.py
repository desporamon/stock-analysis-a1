"""
CSIS 4260 — Assignment 1: Stock Price Analysis Dashboard
Author: Desmond Chua

This Streamlit dashboard provides an interactive interface for exploring:
1. Storage benchmarking results (CSV vs Parquet)
2. Pandas vs Polars performance comparison
3. Stock price predictions with technical indicators

Run with: streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Stock Price Analysis — CSIS 4260",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# DATA LOADING (cached so it only loads once)
# ============================================================
@st.cache_data
def load_data():
    """Load all pre-computed data files from Part 1 and Part 2."""
    # Find the data directory (works from any working directory)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', '..', 'data')
    
    # Load the enhanced dataset with technical indicators
    stocks = pd.read_csv(os.path.join(data_dir, 'stocks_with_features.csv'))
    stocks['date'] = pd.to_datetime(stocks['date'])
    
    # Load test predictions
    predictions = pd.read_csv(os.path.join(data_dir, 'test_predictions.csv'))
    predictions['date'] = pd.to_datetime(predictions['date'])
    
    # Load model metrics
    metrics = pd.read_csv(os.path.join(data_dir, 'model_metrics.csv'))
    
    return stocks, predictions, metrics

# Load data
stocks, predictions, metrics = load_data()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["🏠 Overview", "💾 Benchmark Results", "📈 Stock Prediction", "📊 Technical Indicators"]
)

# Get list of all company tickers for the dropdown
all_tickers = sorted(stocks['name'].unique().tolist())

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "🏠 Overview":
    st.title("📊 S&P 500 Stock Price Analysis & Prediction")
    st.markdown("**CSIS 4260 — Assignment 1 | Author: Desmond Chua**")
    
    st.markdown("---")
    
    # Key dataset statistics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(stocks):,}")
    col2.metric("Companies", f"{stocks['name'].nunique()}")
    col3.metric("Date Range", f"{stocks['date'].min().strftime('%Y-%m-%d')}")
    col4.metric("To", f"{stocks['date'].max().strftime('%Y-%m-%d')}")
    
    st.markdown("---")
    
    st.markdown("""
    ### About This Dashboard
    This interactive dashboard presents the results of a comprehensive stock price analysis project covering:
    
    - **💾 Benchmark Results** — CSV vs Parquet storage performance at 1x, 10x, and 100x scales, plus Pandas vs Polars library comparison
    - **📈 Stock Prediction** — Next-day closing price predictions using Linear Regression and Random Forest models
    - **📊 Technical Indicators** — SMA, RSI, Volatility, and other indicators visualized per company
    
    Use the sidebar to navigate between pages. Select different stock tickers to see company-specific analysis.
    """)
    
    # Show sample of the data
    st.subheader("Dataset Sample")
    st.dataframe(stocks.head(10), use_container_width=True)

# ============================================================
# PAGE 2: BENCHMARK RESULTS
# ============================================================
elif page == "💾 Benchmark Results":
    st.title("💾 Storage & Library Benchmarking Results")
    
    # --- Section 1: CSV vs Parquet ---
    st.header("1. CSV vs Parquet Performance")
    st.markdown("""
    We benchmarked CSV against Parquet (with Snappy, Gzip, Brotli, and no compression) 
    at three data scales: **1x** (619K rows), **10x** (6.2M rows), and **100x** (62M rows).
    """)
    
    # Hardcoded benchmark results from Part 1 (actual results from student's machine)
    benchmark_data = {
        'Scale': ['1x','1x','1x','1x','1x','10x','10x','10x','10x','10x','100x','100x','100x','100x','100x'],
        'Format': ['CSV','Parquet (No Compress)','Parquet (Snappy)','Parquet (Gzip)','Parquet (Brotli)'] * 3,
        'Write Time (s)': [6.979,1.716,0.409,10.497,3.385, 49.761,6.150,6.565,121.673,56.126, 993.224,152.367,523.101,1926.712,1139.907],
        'Read Time (s)': [1.390,2.335,0.128,0.118,0.145, 7.595,1.657,1.841,2.129,2.344, 235.104,69.844,45.018,38.395,50.606],
        'File Size (MB)': [28.80,12.54,10.03,8.27,7.89, 288.01,116.91,94.92,78.74,75.17, 2880.05,1166.98,947.92,785.89,750.27]
    }
    df_bench = pd.DataFrame(benchmark_data)
    
    # Scale selector
    selected_scale = st.selectbox("Select data scale:", ['1x', '10x', '100x'])
    scale_data = df_bench[df_bench['Scale'] == selected_scale]
    
    # Three charts side by side
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.bar(scale_data, x='Format', y='Read Time (s)', 
                     title=f'Read Time at {selected_scale}',
                     color='Format', color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(scale_data, x='Format', y='Write Time (s)', 
                     title=f'Write Time at {selected_scale}',
                     color='Format', color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.bar(scale_data, x='Format', y='File Size (MB)', 
                     title=f'File Size at {selected_scale}',
                     color='Format', color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    st.subheader("Full Results Table")
    st.dataframe(scale_data.reset_index(drop=True), use_container_width=True)
    
    # Recommendation
    st.markdown(f"""
    ### Recommendation at {selected_scale} Scale
    {"**CSV is acceptable** for simplicity at this small scale. Performance differences are minimal." if selected_scale == '1x' else ""}
    {"**Parquet with Snappy** offers the best balance of read speed, write speed, and file size." if selected_scale == '10x' else ""}
    {"**Parquet with Snappy is strongly recommended.** CSV takes over 4 minutes to read, while Snappy takes under 1 minute." if selected_scale == '100x' else ""}
    """)
    
    st.markdown("---")
    
    # --- Section 2: Pandas vs Polars ---
    st.header("2. Pandas vs Polars Performance")
    st.markdown("We compared the two most popular Python dataframe libraries on common data operations.")
    
    # Hardcoded Pandas vs Polars results
    lib_data = {
        'Operation': ['CSV Read', 'Filter (close > 100)', 'GroupBy Mean', 'Sort (name, date)', 'Rolling Mean (20-day)'],
        'Pandas (s)': [1.6422, 0.0453, 0.1395, 0.3608, 0.7966],
        'Polars (s)': [0.2822, 0.0996, 0.1629, 0.4977, 0.3574],
        'Speedup': ['5.8x', '0.5x', '0.9x', '0.7x', '2.2x']
    }
    df_lib = pd.DataFrame(lib_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Pandas', x=df_lib['Operation'], y=df_lib['Pandas (s)'], marker_color='#3498db'))
    fig.add_trace(go.Bar(name='Polars', x=df_lib['Operation'], y=df_lib['Polars (s)'], marker_color='#e74c3c'))
    fig.update_layout(
        title='Pandas vs Polars: Execution Time Comparison',
        yaxis_title='Time (seconds)',
        barmode='group',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(df_lib, use_container_width=True)
    
    st.markdown("""
    **Conclusion:** Polars is faster for CSV reading (5.8x) and rolling calculations (2.2x), 
    but Pandas was chosen for the modeling pipeline due to better integration with Scikit-learn and Streamlit.
    """)

# ============================================================
# PAGE 3: STOCK PREDICTION
# ============================================================
elif page == "📈 Stock Prediction":
    st.title("📈 Stock Price Prediction")
    
    # Company selector with search
    selected_ticker = st.selectbox(
        "Select a company ticker:",
        all_tickers,
        index=all_tickers.index('AAPL') if 'AAPL' in all_tickers else 0
    )
    
    st.markdown("---")
    
    # --- Model Metrics ---
    st.subheader("Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    # Linear Regression metrics
    lr_row = metrics[metrics['Model'] == 'Linear Regression'].iloc[0]
    with col1:
        st.markdown("#### Linear Regression")
        st.metric("RMSE", f"${lr_row['RMSE']:.4f}")
        st.metric("MAE", f"${lr_row['MAE']:.4f}")
        st.metric("R² Score", f"{lr_row['R² Score']:.4f}")
    
    # Random Forest metrics
    rf_row = metrics[metrics['Model'] == 'Random Forest'].iloc[0]
    with col2:
        st.markdown("#### Random Forest")
        st.metric("RMSE", f"${rf_row['RMSE']:.4f}")
        st.metric("MAE", f"${rf_row['MAE']:.4f}")
        st.metric("R² Score", f"{rf_row['R² Score']:.4f}")
    
    st.markdown("*Lower RMSE/MAE = better. Higher R² = better. Linear Regression outperforms Random Forest for this task.*")
    
    st.markdown("---")
    
    # --- Actual vs Predicted Chart ---
    st.subheader(f"Actual vs Predicted Price — {selected_ticker}")
    
    # Filter predictions for selected company
    ticker_preds = predictions[predictions['name'] == selected_ticker].sort_values('date')
    
    if len(ticker_preds) > 0:
        fig = go.Figure()
        
        # Actual price
        fig.add_trace(go.Scatter(
            x=ticker_preds['date'], y=ticker_preds['next_close'],
            name='Actual Price', line=dict(color='black', width=2)
        ))
        
        # Linear Regression prediction
        fig.add_trace(go.Scatter(
            x=ticker_preds['date'], y=ticker_preds['lr_predicted'],
            name='Linear Regression', line=dict(color='#3498db', width=1.5, dash='dot')
        ))
        
        # Random Forest prediction
        fig.add_trace(go.Scatter(
            x=ticker_preds['date'], y=ticker_preds['rf_predicted'],
            name='Random Forest', line=dict(color='#e74c3c', width=1.5, dash='dot')
        ))
        
        fig.update_layout(
            title=f'{selected_ticker} — Actual vs Predicted Closing Price (Test Set)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No prediction data available for {selected_ticker} in the test set.")
    
    # --- Prediction Error Chart ---
    if len(ticker_preds) > 0:
        st.subheader(f"Prediction Error — {selected_ticker}")
        
        ticker_preds_copy = ticker_preds.copy()
        ticker_preds_copy['LR Error'] = ticker_preds_copy['lr_predicted'] - ticker_preds_copy['next_close']
        ticker_preds_copy['RF Error'] = ticker_preds_copy['rf_predicted'] - ticker_preds_copy['next_close']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ticker_preds_copy['date'], y=ticker_preds_copy['LR Error'],
            name='LR Error', line=dict(color='#3498db', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=ticker_preds_copy['date'], y=ticker_preds_copy['RF Error'],
            name='RF Error', line=dict(color='#e74c3c', width=1)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title=f'{selected_ticker} — Prediction Error Over Time',
            xaxis_title='Date',
            yaxis_title='Error ($)',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 4: TECHNICAL INDICATORS
# ============================================================
elif page == "📊 Technical Indicators":
    st.title("📊 Technical Indicators Explorer")
    
    # Company selector
    selected_ticker = st.selectbox(
        "Select a company ticker:",
        all_tickers,
        index=all_tickers.index('AAPL') if 'AAPL' in all_tickers else 0,
        key='tech_ticker'
    )
    
    # Filter data for selected company
    ticker_data = stocks[stocks['name'] == selected_ticker].sort_values('date')
    
    # Indicator toggles
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Indicator Toggles")
    show_sma20 = st.sidebar.checkbox("Show SMA 20", value=True)
    show_sma50 = st.sidebar.checkbox("Show SMA 50", value=True)
    show_rsi = st.sidebar.checkbox("Show RSI", value=True)
    show_volatility = st.sidebar.checkbox("Show Volatility", value=True)
    
    st.markdown("---")
    
    # --- Price Chart with SMA overlays ---
    st.subheader(f"{selected_ticker} — Price Chart with Moving Averages")
    
    fig = go.Figure()
    
    # Close price (always shown)
    fig.add_trace(go.Scatter(
        x=ticker_data['date'], y=ticker_data['close'],
        name='Close Price', line=dict(color='black', width=1.5)
    ))
    
    # SMA 20 (toggle)
    if show_sma20:
        fig.add_trace(go.Scatter(
            x=ticker_data['date'], y=ticker_data['sma_20'],
            name='SMA 20', line=dict(color='#3498db', width=1, dash='dash')
        ))
    
    # SMA 50 (toggle)
    if show_sma50:
        fig.add_trace(go.Scatter(
            x=ticker_data['date'], y=ticker_data['sma_50'],
            name='SMA 50', line=dict(color='#e74c3c', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title=f'{selected_ticker} — Closing Price with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=450,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Volume Chart ---
    st.subheader(f"{selected_ticker} — Trading Volume")
    fig_vol = px.bar(
        ticker_data, x='date', y='volume',
        title=f'{selected_ticker} — Daily Trading Volume',
        color_discrete_sequence=['#95a5a6']
    )
    fig_vol.update_layout(height=300, xaxis_title='Date', yaxis_title='Volume')
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # --- RSI Chart ---
    if show_rsi:
        st.subheader(f"{selected_ticker} — Relative Strength Index (RSI)")
        
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=ticker_data['date'], y=ticker_data['rsi_14'],
            name='RSI (14)', line=dict(color='#9b59b6', width=1.5)
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        
        fig_rsi.update_layout(
            title=f'{selected_ticker} — RSI (14-Day)',
            xaxis_title='Date',
            yaxis_title='RSI',
            yaxis_range=[0, 100],
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    # --- Volatility Chart ---
    if show_volatility:
        st.subheader(f"{selected_ticker} — 20-Day Volatility")
        
        fig_vol2 = go.Figure()
        fig_vol2.add_trace(go.Scatter(
            x=ticker_data['date'], y=ticker_data['volatility'],
            name='Volatility', line=dict(color='#f39c12', width=1.5),
            fill='tozeroy', fillcolor='rgba(243, 156, 18, 0.1)'
        ))
        
        fig_vol2.update_layout(
            title=f'{selected_ticker} — 20-Day Price Volatility',
            xaxis_title='Date',
            yaxis_title='Volatility ($)',
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig_vol2, use_container_width=True)
    
    # --- Summary Statistics ---
    st.subheader(f"{selected_ticker} — Summary Statistics")
    summary = ticker_data[['close', 'volume', 'sma_20', 'rsi_14', 'volatility', 'daily_return']].describe().round(2)
    st.dataframe(summary, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("**CSIS 4260 — Assignment 1**")
st.sidebar.markdown("Desmond Chua")
