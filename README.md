# CSIS 4260 — Assignment 1: Stock Price Analysis & Prediction

**Author:** Desmond Chua  
**Course:** CSIS 4260 — Special Topics in Data Analytics  
**Date:** February 2026

**Live Dashboard:** https://sp500-desmond-stockprediction.streamlit.app

---

## Project Structure

```
stock-analysis-a1/
├── data/                          # Dataset and generated files
│   ├── all_stocks_5yr.csv         # Original dataset (not in repo)
│   ├── stocks_with_features.csv   # Enhanced dataset with technical indicators
│   ├── test_predictions.csv       # Model predictions on test set
│   └── model_metrics.csv          # Model evaluation metrics
├── notebooks/
│   ├── part1_storage_benchmarking.ipynb   # Part 1: CSV vs Parquet benchmarking
│   └── part2_analysis_modeling.ipynb      # Part 2: Analysis & prediction models
├── src/
│   └── dashboard/
│       └── app.py                 # Part 3: Streamlit dashboard
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How to Run

1. Clone the repo and create a virtual environment:
```bash
git clone https://github.com/desporamon/stock-analysis-a1.git
cd stock-analysis-a1
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

2. Place `all_stocks_5yr.csv` in the `data/` folder.

3. Run the notebooks in order:
   - `part1_storage_benchmarking.ipynb` (note: 100x benchmarks take 30+ minutes)
   - `part2_analysis_modeling.ipynb` (generates the CSV files the dashboard needs)

4. Run the dashboard:
```bash
streamlit run src/dashboard/app.py
```

---

## Part 1: Storing and Retrieving Data

I compared CSV against Parquet with four compression options (no compression, Snappy, Gzip, Brotli) at 1x, 10x, and 100x data scales.

**Why Parquet?** CSV is a row-based text format — to read it, Python has to parse every line character by character. Parquet is a binary columnar format, meaning it stores data by column and can skip columns you don't need. It also supports built-in compression to reduce file sizes.

**Compression schemes I tested:**
- Snappy — fast compression/decompression, moderate file size reduction
- Gzip — better compression ratio but much slower to write
- Brotli — best compression ratio but slowest overall
- No compression — tests the benefit of columnar format alone

**Key results at 100x scale (most important since that's where differences really show):**

| Format | Read Time | Write Time | File Size |
|--------|-----------|------------|-----------|
| CSV | 235s (3.9 min) | 993s (16.5 min) | 2,880 MB |
| Parquet Snappy | 45s | 523s (8.7 min) | 948 MB |
| Parquet Gzip | 38s | 1,927s (32 min) | 786 MB |
| Parquet Brotli | 51s | 1,140s (19 min) | 750 MB |

**My recommendations:**
- **1x scale:** CSV is fine. Everything runs in under 2 seconds so there's no real benefit to switching formats.
- **10x scale:** Parquet with Snappy. It reads 4.1x faster and produces 3x smaller files. The performance gap starts to matter here.
- **100x scale:** Parquet with Snappy is the clear winner. CSV takes almost 4 minutes just to read the file — that's too slow for iterative analysis. Snappy reads in 45 seconds with 3x smaller files. Gzip has slightly better compression but its 32-minute write time makes it impractical unless you're archiving data you rarely update.

---

## Part 2: Data Manipulation, Analysis & Prediction

### Pandas vs Polars

I benchmarked five common operations to compare the two libraries:

| Operation | Pandas | Polars | Winner |
|-----------|--------|--------|--------|
| CSV Read | 1.64s | 0.28s | Polars (5.8x faster) |
| Filter | 0.05s | 0.10s | Pandas |
| GroupBy Mean | 0.14s | 0.16s | Roughly equal |
| Sort | 0.36s | 0.50s | Pandas |
| Rolling Mean | 0.80s | 0.36s | Polars (2.2x faster) |

Polars is faster for CSV reading and rolling calculations, but I chose **Pandas** for the rest of the project because:
- Scikit-learn expects Pandas DataFrames for model training
- Streamlit works natively with Pandas for charts and tables
- For 619K rows, Pandas is fast enough — every operation completes in under 2 seconds
- Polars would make more sense for datasets above 10M rows where the speed difference becomes significant

### Technical Indicators

I added 6 features to help the prediction models. All of them were calculated **per company** using `groupby('name')` — this is critical because without it, the rolling window calculations would mix data from different companies (e.g., the tail end of AAL data bleeding into the start of AAPL).

- **SMA 20 and SMA 50** — Simple Moving Averages over 20 and 50 days. These smooth out daily noise and show the short-term and medium-term trend direction.
- **RSI 14** — Relative Strength Index over 14 days. Measures momentum on a 0-100 scale. Above 70 suggests the stock may be overbought, below 30 suggests oversold. I implemented this with the proper gain/loss separation formula rather than just averaging price changes.
- **Volatility** — 20-day rolling standard deviation of closing prices. Higher values mean the stock price is swinging more — harder to predict.
- **Daily Return** — Percentage change from the previous day's close.
- **Price Momentum** — Close price minus SMA 20. Positive means the price is above its recent average (uptrend), negative means below (downtrend).

### Prediction Models

I trained two models to predict the next day's closing price:

**Features (inputs):** open, high, low, close, volume, sma_20, sma_50, rsi_14, volatility, price_momentum

**Target:** next day's closing price (created using `shift(-1)` grouped by company)

**Train-test split:** Time-based 80/20 — the first 80% of data chronologically goes to training, the last 20% to testing. I did NOT use a random split because that would let the model see future prices during training (data leakage), which would produce misleadingly high accuracy scores.

**Results:**

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | $2.23 | $0.91 | 0.9997 |
| Random Forest | $10.19 | $1.36 | 0.9936 |

Linear Regression outperformed Random Forest across all metrics. This makes sense because stock prices are highly linear day-to-day — tomorrow's price is usually very close to today's. Linear Regression captures this relationship precisely, while Random Forest's decision tree splits introduce more variance in predictions.

Both models identified `close` as the most important feature by far (LR coefficient: 0.65, RF importance: 0.87). This confirms the models are essentially learning "tomorrow's price ≈ today's price + small adjustment."

The R² scores above 0.99 look impressive but they're expected for this type of task — stock prices don't change dramatically day-to-day, so predicting "about the same as yesterday" already gets you most of the way there. For practical trading, predicting the direction of change would be more useful than the exact price.

---

## Part 3: Dashboard

I chose **Streamlit** over Dash and Reflex because:
- It's pure Python — no need to write HTML/CSS/JavaScript
- It has native support for Pandas DataFrames and Plotly charts
- Streamlit Cloud provides free deployment directly from GitHub
- Widgets like dropdowns and checkboxes are single function calls

The dashboard has 4 pages:

1. **Overview** — Dataset stats (619,040 rows, 505 companies, date range), project description, and a sample of the data.

2. **Benchmark Results** — CSV vs Parquet bar charts with a dropdown to switch between 1x, 10x, and 100x scales. Also includes the Pandas vs Polars comparison chart and table.

3. **Stock Prediction** — Select any company from a dropdown. Shows model metrics (RMSE, MAE, R²), an actual vs predicted price chart with both models overlaid, and a prediction error chart.

4. **Technical Indicators** — Select any company and toggle SMA, RSI, and Volatility on/off via checkboxes. Shows price chart with moving averages, trading volume, RSI with overbought/oversold lines, volatility, and summary statistics.

All charts are interactive (hover, zoom, pan) and update dynamically when a different ticker is selected.
