# Rossmann Sales Prediction

## Project Overview
This project analyzes sales data from Rossmann Pharmaceuticals to help the finance team forecast sales across their stores. As a Machine Learning Engineer at Rossmann, we're developing a comprehensive analysis and prediction system.

## Current Progress: Exploratory Data Analysis

### Completed Analysis
1. **Data Loading and Preprocessing**
   - Implemented robust data loading system
   - Handled missing values
   - Created date-based features
   - Merged store metadata with sales data

2. **Promotion Analysis**
   - Analyzed promotion distribution in training and test sets
   - Evaluated promotion effectiveness on sales
   - Studied customer behavior during promotions
   - Identified optimal promotion strategies

3. **Sales Patterns**
   - Analyzed seasonal trends
   - Studied holiday impact on sales
   - Investigated correlation between sales and customer numbers
   - Examined store-specific patterns

4. **Competition Analysis**
   - Evaluated impact of competitor distance
   - Analyzed effects of new competitor openings
   - Studied sales patterns in different competitive environments

### Project Structure
```
├── src/                  # Source code
│   └── data/            
│       ├── data_loader.py    # Data loading and preprocessing
│       └── analysis_utils.py # Analysis utilities
├── notebooks/           
│   └── 01_data_exploration.ipynb  # EDA notebook
├── data/                
│   ├── raw/            # Original data
│   └── processed/      # Cleaned data
├── config/             
│   ├── model_config.yaml    # Analysis configuration
│   └── logging_config.yaml  # Logging settings
└── logs/               # Application logs
```

### Key Findings
1. **Promotion Impact**
   - Detailed analysis of promotion effectiveness
   - Distribution comparison between training and test sets
   - Impact on customer numbers and sales

2. **Seasonal Patterns**
   - Holiday sales behavior
   - Weekly and monthly trends
   - Special event impacts

3. **Store Performance**
   - Analysis by store type
   - Impact of competition
   - Store-specific characteristics

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/Selam46/rossmann-sales-prediction.git
cd Rossmann-Sales-Prediction
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
Run the exploratory analysis notebook:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Next Steps
- Implement machine learning models for sales prediction
- Develop deep learning approach using LSTM
- Create API for serving predictions
- Deploy the solution


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details 