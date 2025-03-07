# Rossmann Sales Prediction

This project aims to forecast sales for Rossmann Pharmaceuticals across their stores six weeks ahead of time. The project includes comprehensive data analysis, machine learning modeling, and a production-ready API for serving predictions.

## Project Structure

```
├── src/                  # Source code
│   ├── data/            # Data processing scripts
│   ├── features/        # Feature engineering
│   ├── models/          # ML models implementation
│   ├── api/             # FastAPI implementation
│   └── utils/           # Utility functions
├── notebooks/           # Jupyter notebooks for analysis
├── data/                # Data directory
│   ├── raw/            # Original data
│   ├── processed/      # Cleaned data
│   └── features/       # Feature engineered data
├── models/             # Saved model artifacts
├── config/             # Configuration files
├── docs/              # Documentation
├── tests/             # Unit tests
└── logs/              # Application logs
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rossmann-sales-prediction.git
cd rossmann-sales-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Exploration
- Run notebooks in the `notebooks/` directory sequentially
- Start with `01_data_exploration.ipynb`

### Training Models
```bash
python src/models/train_model.py
```

### Starting the API
```bash
uvicorn src.api.app:app --reload
```

## API Endpoints

- `POST /predict`: Get sales predictions for stores
  - Input: JSON with store features
  - Output: Predicted sales and confidence intervals

## Project Tasks

1. **Data Exploration**
   - Customer purchasing behavior analysis
   - Holiday impact analysis
   - Promotion effectiveness
   - Competition analysis

2. **Machine Learning Models**
   - Random Forest implementation
   - Feature importance analysis
   - Confidence interval estimation

3. **Deep Learning**
   - LSTM implementation
   - Time series analysis
   - Sales forecasting

4. **API Development**
   - RESTful API implementation
   - Model serving
   - Real-time predictions

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details 