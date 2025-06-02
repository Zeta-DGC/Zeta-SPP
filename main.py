import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd
from rich.prompt import Prompt

id = Prompt.ask("ID of Stock")
start = datetime(int(Prompt.ask("Start Year")), int(Prompt.ask("Start Month")), int(Prompt.ask("Start Day(s)")))
end = datetime.now()
df = yf.download(id, start=start, end=end)

df.reset_index(inplace=True)
df = df[['Date', 'Close']]
df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

model = Prophet()
model.fit(df)

predictdays = int(Prompt.ask("Count of Days to Predict"))

future = model.make_future_dataframe(periods=predictdays)
forecast = model.predict(future)

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df['ds'], df['y'], label='Historical Data', color='blue')
ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2)
ax.set_title(f'{id} Stock Price Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
plt.grid(True)
plt.show()
