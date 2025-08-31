# PCA_Dashboard
Investalogical's Newest Software

## How to Startup

### Prerequisites
- Python 3.11 or higher
- Git

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
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
   pip install -r requirement.txt
   ```

### Running the Application

1. **Start the dashboard**
   ```bash
   python main.py
   ```

2. **Access the application**
   - Open your web browser
   - Navigate to `http://localhost:8050` (or the port shown in terminal)
   - The dashboard will automatically find an available port between 8050-8060

### Features Available
- Advanced PCA Analysis
- Enhanced Volatility Modeling (GARCH)
- LSTM Stock Predictions
- Real-time Stock Data (Yahoo Finance)
- Interactive Visualizations
- Portfolio Optimization
- Risk Assessment Tools

### Demo Usage
Check the `demos/` folder for sample usage examples and screenshots of the dashboard capabilities.

### Structure 
## Analyzers 