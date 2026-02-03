# Devloped by: Ramasri K
# Reg No: 212224040267
# Date: 3/02/2026

# Ex.No: 1B              CONVERSION OF NON STATIONARY TO STATIONARY DATA

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data
### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.
### SOFTWARE REQUIRED:
google colab
### DATASET:
Silver Price Analysis & ML Forecasting
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
data=pd.read_csv('/silver_prices_data.csv')
data.head()
data=pd.read_csv('/silver_prices_data.csv')
data['Date']=pd.to_datetime(data['Date'])
data.set_index('Date',inplace=True)
data_aggregated = data.groupby(data.index)['Volume'].mean().to_frame()
data_aggregated['Volume_diff']=data_aggregated['Volume']-data_aggregated['Volume'].shift(1)
data_aggregated['Volume_log'] = np.log(data_aggregated['Volume'].replace(0, 1e-9))
data_aggregated['Volume_log_diff']=data_aggregated['Volume_log']-data_aggregated['Volume_log'].shift(1)
result = seasonal_decompose(data_aggregated['Volume_log_diff'].dropna(), model='additive', period=12)
data_aggregated['Volume_log_seasonal_diff']=result.resid
plt.figure(figsize=(20, 24))
plt.subplot(2, 1, 1) 
plt.plot(data_aggregated['Volume'], label='Original') 
plt.legend(loc='best')
plt.title('Original Data')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.subplot(2, 1, 1)
plt.plot(data_aggregated['Volume_diff'], label='Regular Difference')
plt.legend(loc='best')
plt.title('Regular Differencing')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.subplot(1, 1, 1) 
plt.plot(data_aggregated['Volume_log_seasonal_diff'], label='Log Transformation, Regular and Seasonal Differencing') 
plt.legend(loc='best')
plt.title('Seasonally Adjusted Data (Log Transformed and Differenced)') 
plt.xlabel('Date')
plt.ylabel('SDiff(RDiff(Log(Volume)))')
plt.subplot(2, 1, 1) 
plt.plot(data_aggregated['Volume_log'], label='Log Transformation') 
plt.legend(loc='best')
plt.title('Log Transformation')
plt.xlabel('Date')
plt.ylabel('Log(Volume)')
plt.subplot(2, 1, 1) 
plt.plot(data_aggregated['Volume_log'], label='Log Transformation') 
plt.legend(loc='best')
plt.title('Log Transformation')
plt.xlabel('Date')
plt.ylabel('Log(Volume)')
plt.subplot(1,1,1) 
plt.plot(data_aggregated['Volume_log_seasonal_diff'], label='Log Transformation and Regular Differencing and Seasonal Differencing')
plt.legend(loc='best')
plt.title('Seasonally Adjusted Data (Log Transformed and Differenced)') 
plt.xlabel('Date')
plt.ylabel('SDiff(RDiff(Log(Volume)))') 
plt.tight_layout()
plt.show()
```
### OUTPUT:
<img width="1686" height="495" alt="image" src="https://github.com/user-attachments/assets/a0c34580-76b4-44b2-9da2-3a459a5caecd" />
<img width="1669" height="467" alt="image" src="https://github.com/user-attachments/assets/c1722757-d71b-4eae-a741-16c45f1318f8" />

ORIGINAL DATA:
<img width="1612" height="509" alt="image" src="https://github.com/user-attachments/assets/a606859d-6ad9-47c8-aa95-562469b58e36" />


REGULAR DIFFERENCING:
<img width="1773" height="514" alt="image" src="https://github.com/user-attachments/assets/730b1736-5f1c-4564-a5a4-a1a167c43a59" />


SEASONAL ADJUSTMENT:
<img width="1696" height="744" alt="image" src="https://github.com/user-attachments/assets/2afa1de0-08f7-433a-91fb-95acc771e983" />

LOG TRANSFORMATION:
<img width="1726" height="514" alt="image" src="https://github.com/user-attachments/assets/c81a7002-0177-4af4-b4ec-68833072ea40" />
<img width="1658" height="767" alt="image" src="https://github.com/user-attachments/assets/61197ab5-3109-4f89-b93f-5b0c0e64eae8" />
<img width="1692" height="798" alt="image" src="https://github.com/user-attachments/assets/20951a87-fd8a-4d62-a466-a6df5c0e9aa3" />

### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
