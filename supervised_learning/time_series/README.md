# Bitcoin Time Series Forecasting

## Description

This project uses a Recurrent Neural Network (LSTM) implemented with
TensorFlow/Keras to forecast the Bitcoin closing price using the previous
24 hours of market data.

The project includes:

- Data preprocessing
- Feature normalization
- Sequence generation
- TensorFlow Dataset pipeline
- LSTM model training
- Model evaluation
- Model saving

---

## Files

### preprocess_data.py

This script:

- Loads the Coinbase and Bitstamp datasets
- Merges the datasets
- Removes duplicate timestamps
- Cleans missing values
- Normalizes numerical features using MinMaxScaler
- Saves the processed dataset
- Saves the fitted scaler

Output:

- `btc_processed.csv`
- `scaler.pkl`

---

### forecast_btc.py

This script:

- Loads the processed dataset
- Creates training sequences
- Splits the data into training, validation, and testing sets
- Builds an LSTM model
- Trains using TensorFlow's `tf.data.Dataset`
- Evaluates the model using Mean Squared Error
- Saves the trained model

Output:

- `btc_forecast.keras`

---

## Features Used

The following features are used as model inputs:

- Open
- High
- Low
- Close
- Volume (BTC)
- Volume (Currency)
- Weighted Price

The Timestamp column is used only for sorting and is not used as an input
feature.

---

## Model Architecture

- LSTM (64 units)
- Dropout (0.2)
- LSTM (32 units)
- Dense (16, ReLU)
- Dense (1)

Loss Function:

- Mean Squared Error (MSE)

Optimizer:

- Adam

Metric:

- Mean Absolute Error (MAE)

---

## Dataset Pipeline

TensorFlow's `tf.data.Dataset` is used to:

- Shuffle the training data
- Batch the samples
- Prefetch batches for improved performance

---

## Running the Project

Preprocess the data:

```bash
./preprocess_data.py
```

Train the model:

```bash
./forecast_btc.py
```

---

## Requirements

- Python 3.9
- TensorFlow 2.15
- NumPy
- Pandas
- scikit-learn
- Joblib

---

## Author

Holberton School Machine Learning Project
