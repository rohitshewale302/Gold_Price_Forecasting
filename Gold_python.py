import pandas as pd
import streamlit as st
from pickle import load
import statsmodels.api as sm
import matplotlib.pyplot as plt

#import datetime
#from datetime import datetime


df_price = load(open('df_price.sav','rb'))

st.title('Model Deployment: Gold price Forecasting')



periods = st.number_input('Number of Days',min_value=1)

datetime = pd.DataFrame(pd.date_range('2021-12-22', periods=periods,freq='B'), columns = ['date'])



final_arima_model = sm.tsa.ARIMA(df_price['price'],order = (5,1,5))
arima_fit_final = final_arima_model.fit()
forecast = arima_fit_final.predict(len(df_price), len(df_price)+periods-1)
forecast_df = pd.DataFrame(forecast)
forecast_df.columns = ['price']

data_forecast = forecast_df.set_index(datetime.date)
st.write(data_forecast)



fig,ax = plt.subplots()
ax.plot(df_price['price'], label = 'price')
ax.plot(data_forecast, label = 'Forecast')
ax.set_title('Gold Price Forecast')
ax.set_xlabel('Year')
ax.set_ylabel('Gold Price')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True)
st.pyplot(fig)