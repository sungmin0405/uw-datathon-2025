import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import sys

def loading_animation(text, duration=2):
    """Simple loading animation"""
    for _ in range(duration):
        for frame in "|/-\\":
            sys.stdout.write(f'\r{text} {frame}')
            sys.stdout.flush()
            time.sleep(0.2)
    print("\r" + " " * (len(text) + 2), end="\r")  

def main():
    n_months = int(input("Enter number of months to predict: "))

    print("Loading and preprocessing data...")
    file_path = '/Users/leonidastaliadouros/Documents/Datathon/enriched_data.csv'
    df = pd.read_csv(file_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['YearMonthDay'] = df['InvoiceDate'].dt.to_period('D').dt.to_timestamp()
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    country_counts = df['Country'].value_counts()
    large_countries = country_counts[country_counts >= 800].index
    df['Country'] = df['Country'].apply(lambda x: x if x in large_countries else 'Smaller_countries')

    grouped2 = df.groupby(['Country', 'YearMonthDay']).agg({'Quantity': 'sum', 'Revenue': 'sum'}).reset_index()

    special_days = ['2011-12-09', '2011-01-18']
    grouped2['day_of_week'] = grouped2['YearMonthDay'].dt.dayofweek
    grouped2['month'] = grouped2['YearMonthDay'].dt.month
    grouped2['day_of_month'] = grouped2['YearMonthDay'].dt.day
    grouped2['is_special_day'] = grouped2['YearMonthDay'].isin(pd.to_datetime(special_days)).astype(int)

    price_variation = df.groupby(['Country', 'YearMonthDay']).agg({'UnitPrice': ['mean', 'std']}).reset_index()
    price_variation.columns = ['Country', 'YearMonthDay', 'UnitPrice_Mean', 'UnitPrice_Std']
    price_variation['price_spike_flag'] = (price_variation['UnitPrice_Std'] > 30.0).astype(int)

    grouped2 = grouped2.merge(price_variation[['Country', 'YearMonthDay', 'price_spike_flag']],
                               on=['Country', 'YearMonthDay'], how='left')
    grouped2['price_spike_flag'] = grouped2['price_spike_flag'].fillna(0)

    le = LabelEncoder()
    grouped2['Country_encoded'] = le.fit_transform(grouped2['Country'])

    features = ['Country_encoded', 'day_of_week', 'month', 'day_of_month', 'is_special_day', 'price_spike_flag']
    X = grouped2[features]
    y = grouped2['Quantity']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training models (this may take a few minutes)...")
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    n_models = 3  
    models = []
    for i in tqdm(range(n_models), desc="Training models"):
        model = xgb.XGBRegressor(**params, seed=i*10)
        model.fit(X_train, y_train)
        models.append(model)

    print("Generating predictions...")
    loading_animation("Generating predictions", duration=5)

    sys.stdout.flush()
    time.sleep(0.5)

    start_date = grouped2['YearMonthDay'].max() + pd.Timedelta(days=1)
    future_month_starts = pd.date_range(start=start_date, periods=n_months, freq='MS')

    countries = grouped2['Country'].dropna().unique()
    future_list = []

    for month_start in future_month_starts:
        month_days = pd.date_range(start=month_start, end=month_start + pd.offsets.MonthEnd(0))
        for country in countries:
            temp = pd.DataFrame({
                'Country': [country] * len(month_days),
                'YearMonthDay': month_days
            })
            future_list.append(temp)

    future = pd.concat(future_list).reset_index(drop=True)

    future['day_of_week'] = future['YearMonthDay'].dt.dayofweek
    future['month'] = future['YearMonthDay'].dt.month
    future['day_of_month'] = future['YearMonthDay'].dt.day
    future['is_special_day'] = future['YearMonthDay'].isin(pd.to_datetime(special_days)).astype(int)
    future = future.merge(price_variation[['Country', 'YearMonthDay', 'price_spike_flag']],
                          on=['Country', 'YearMonthDay'], how='left')
    future['price_spike_flag'] = future['price_spike_flag'].fillna(0)
    future['Country_encoded'] = le.transform(future['Country'])

    X_future = future[features]

    predictions = np.array([model.predict(X_future) for model in models])
    pred_mean = predictions.mean(axis=0)
    pred_std = predictions.std(axis=0)

    pred_lower = pred_mean - 1.64 * pred_std
    pred_upper = pred_mean + 1.64 * pred_std

    future['Predicted_Quantity'] = np.clip(pred_mean, 0, None)
    future['Predicted_Lower'] = np.clip(pred_lower, 0, None)
    future['Predicted_Upper'] = np.clip(pred_upper, 0, None)

    avg_price_per_country = df.groupby('Country').apply(lambda x: (x['Revenue'].sum() / x['Quantity'].sum())).to_dict()
    future['avg_price'] = future['Country'].map(avg_price_per_country)

    future['Predicted_Revenue'] = future['Predicted_Quantity'] * future['avg_price']
    future['Predicted_Revenue_Lower'] = future['Predicted_Lower'] * future['avg_price']
    future['Predicted_Revenue_Upper'] = future['Predicted_Upper'] * future['avg_price']

    daily_forecast = future.groupby('YearMonthDay').agg({
        'Predicted_Quantity': 'sum',
        'Predicted_Lower': 'sum',
        'Predicted_Upper': 'sum',
        'Predicted_Revenue': 'sum',
        'Predicted_Revenue_Lower': 'sum',
        'Predicted_Revenue_Upper': 'sum'
    }).reset_index()

    daily_forecast['YearMonth'] = daily_forecast['YearMonthDay'].dt.to_period('M')
    monthly_forecast = daily_forecast.groupby('YearMonth').agg({
        'Predicted_Quantity': 'sum',
        'Predicted_Revenue': 'sum'
    }).reset_index()

    print("Plotting results...")

    plt.figure(figsize=(14,6))
    plt.plot(daily_forecast['YearMonthDay'], daily_forecast['Predicted_Quantity'], label='Mean Predicted Quantity')
    plt.fill_between(daily_forecast['YearMonthDay'],
                     daily_forecast['Predicted_Lower'],
                     daily_forecast['Predicted_Upper'],
                     color='lightgray', alpha=0.5, label='Quantity Uncertainty Interval (90%)')
    plt.title('Day by Day: Forecasted Quantity')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(14,6))
    plt.plot(daily_forecast['YearMonthDay'], daily_forecast['Predicted_Revenue'], label='Mean Predicted Revenue')
    plt.fill_between(daily_forecast['YearMonthDay'],
                     daily_forecast['Predicted_Revenue_Lower'],
                     daily_forecast['Predicted_Revenue_Upper'],
                     color='lightgray', alpha=0.5, label='Revenue Uncertainty Interval (90%)')
    plt.title('Day by Day: Forecasted Revenue')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10,6))
    plt.bar(monthly_forecast['YearMonth'].astype(str), monthly_forecast['Predicted_Quantity'], color='skyblue')
    plt.title('Aggregate: Total Predicted Quantity per Month')
    plt.xlabel('Month')
    plt.ylabel('Quantity')
    plt.grid(axis='y')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.bar(monthly_forecast['YearMonth'].astype(str), monthly_forecast['Predicted_Revenue'], color='lightgreen')
    plt.title('Aggregate: Total Predicted Revenue per Month')
    plt.xlabel('Month')
    plt.ylabel('Revenue')
    plt.grid(axis='y')
    plt.show()

    for idx, row in monthly_forecast.iterrows():
        print(f"Month: {row['YearMonth']}, Quantity: {row['Predicted_Quantity']:.0f}, Revenue: {row['Predicted_Revenue']:.2f}")

    print("\nDone!")

if __name__ == "__main__":
    main()
