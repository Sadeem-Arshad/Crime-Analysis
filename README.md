#  Crime Rate Analysis & Prediction System

A full-stack data analysis project that ingests, processes, and models 
U.S. crime data from multiple sources to uncover patterns, forecast 
trends, and classify incident severity.

##  Datasets
- **Chicago Crime Data** — fetched live from the City of Chicago 
  Open Data API (10,000+ records, 2023–present)
- **Gun Violence Incident Dataset** — multi-state incident records 
  with victim, suspect, and location data

##  Features
- **Data Collection**: Live API ingestion (Socrata/Chicago Data Portal) 
  + BeautifulSoup web scraping
- **EDA & Visualization**: Interactive charts with Plotly — bar, 
  histogram, line, pie, 3D scatter, and correlation heatmaps
- **Geospatial Clustering**: DBSCAN on gun-related incidents using 
  latitude/longitude (Haversine metric) visualized with Folium
- **Time-Series Forecasting**: ARIMA and Facebook Prophet models 
  for predicting daily incident counts by city
- **Severity Classification**: Logistic Regression, Decision Tree, 
  and SVM compared via accuracy and classification reports
- **Feature Engineering**: OneHotEncoding, LabelEncoding, 
  ColumnTransformer pipeline, train/test split (70/30)

##  Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Statsmodels (ARIMA), 
Prophet, Plotly, Folium, BeautifulSoup, Requests

##  How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook Crime_Rate_Analysis.ipynb`

##  Key Results
- DBSCAN identified high-density gun crime hotspots across Chicago
- ARIMA & Prophet forecasted daily incident trends per city
- Three classifiers benchmarked for incident severity prediction
