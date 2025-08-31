# 📊 PCA Stock Analysis Dashboard

An advanced financial analytics dashboard built with Python, featuring Principal Component Analysis (PCA), GARCH volatility modeling, and LSTM predictions for comprehensive stock market analysis.

## 🚀 Features

### 📈 Core Analytics
- **Principal Component Analysis (PCA)** - Dimensionality reduction for portfolio analysis
- **GARCH Volatility Modeling** - Advanced volatility forecasting and risk assessment
- **LSTM Stock Predictions** - Neural network-based price forecasting
- **Time Series Analysis** - Statistical analysis of market trends
- **Portfolio Optimization** - Data-driven portfolio construction

### 📊 Interactive Visualizations
- Real-time stock data integration via Yahoo Finance
- Dynamic charts and graphs with Plotly
- 2D/3D PCA scatter plots
- Correlation heatmaps
- Volatility forecasting charts
- LSTM prediction visualizations

### 🎯 Stock Analysis Tools
- Predefined sector groups (Tech, Banking, Healthcare, Energy, etc.)
- Custom stock ticker input
- Risk metrics and statistical tests
- Economic indicator integration
- Portfolio performance tracking

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/PCA_Dashboard.git
   cd PCA_Dashboard
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv INV_PCA_DASH_env
   
   # On macOS/Linux:
   source INV_PCA_DASH_env/bin/activate
   
   # On Windows:
   INV_PCA_DASH_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the dashboard**
   ```bash
   python main.py
   ```

5. **Access the application**
   - Open your web browser
   - Navigate to `http://localhost:8050` (or the port shown in terminal)
   - The dashboard will automatically find an available port between 8050-8060

## 📱 Usage

1. **Select Stock Group** - Choose from predefined sectors or enter custom tickers
2. **Configure Parameters** - Set analysis period, PCA components, model settings
3. **Run Analysis** - Generate comprehensive analytics reports
4. **Explore Results** - Navigate through different analysis tabs
5. **Export Data** - Download results and visualizations

### Sample Stock Groups
- **Tech Giants**: AAPL, MSFT, GOOGL, AMZN, META
- **Banking**: JPM, BAC, WFC, C, GS
- **Healthcare**: JNJ, PFE, UNH, MRK, ABBV
- **Energy**: XOM, CVX, COP, SLB, EOG

## 📁 Project Structure

```
PCA_Dashboard/
├── main.py                    # Main dashboard application
├── time_series_analysis.py    # Time series analytics engine
├── GARCH_update.py           # GARCH volatility modeling
├── LSTM_Pred.py              # LSTM prediction models
├── demos/                    # Demo scripts and screenshots
│   ├── sample_usage.py       # Usage examples
│   └── README.md            # Demo documentation
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🖼️ Screenshots & Demo

Check the [`demos/`](./demos/) folder for:
- Sample usage scripts
- Dashboard screenshots
- Feature demonstrations
- Analysis examples


## 📄 License

This project is proprietary software owned by Investalogical. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For support and questions, please open an issue in the GitHub repository.

---

*Built by Investalogical - Advanced Financial Analytics Solutions* 