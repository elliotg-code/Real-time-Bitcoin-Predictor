import yfinance as yf
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

TICKER = "BTC-USD"
WINDOW_SIZE = 10  # last 10 minutes used to predict

def fetch_data():
    data = yf.download(ticker=TICKER, period="1d", interval="1m", progress=False)
    return data["Close"].values.reshape(-1, 1)

def train_and_predict():
    prices = fetch_data()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled)):
        X.append(scaled[i-WINDOW_SIZE:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)

    model = LinearRegression()
    model.fit(X, y)

    last_window = scaled[-WINDOW_SIZE:].reshape(1, -1)
    predicted_scaled = model.predict(last_window)
    predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
    return prices[-1][0], predicted_price[0][0]

def main():
    print("Starting Bitcoin 1-Minute Prediction...\n")
    while True:
        try:
            current, predicted = train_and_predict()
            print(f"Current BTC: ${current:.2f} | Predicted Next Minute: ${predicted:.2f}")
            print("-" * 40)
            time.sleep(60)
        except Exception as e:
            print("Error:", e)
            time.sleep(60)

if __name__ == "__main__":
    main()
