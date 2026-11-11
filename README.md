# Zomato Delivery Time Predictor
AUPP Data Science Club final project - Option 3: Machine Learning (Python).

Predicts food delivery time (minutes) from distance, traffic, weather, courier and order details.

## Run
1. Download the dataset from Kaggle (`saurabhbadole/zomato-delivery`) into `data/raw/zomato_delivery.csv`.
2. `pip install -r requirements.txt`
3. Run `notebooks/01_cleaning_eda.ipynb`, then `notebooks/02_modeling.ipynb` (creates `models/`).
4. `streamlit run app/app.py`

## Structure
- `notebooks/` cleaning + EDA, modeling
- `models/` saved pipeline + app dropdown options
- `app/` Streamlit app
- `figures/` charts for the slides
