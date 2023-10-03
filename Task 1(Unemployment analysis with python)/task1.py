import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Sample data for demonstration 
data = {
    'Year': [2018, 2019, 2020, 2021, 2022],
    'Unemployed': [5.2, 4.8, 9.2, 6.7, 5.0],
    'LaborForce': [100000, 102000, 105000, 110000, 108000]
}

#DataFrame from the data
df = pd.DataFrame(data)

#Calculate the unemployment rate
df['UnemploymentRate'] = (df['Unemployed'] / df['LaborForce']) * 100

#Plotting the unemployment rate trend
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['UnemploymentRate'], marker='o', linestyle='-', color='b')
plt.title('Unemployment Rate Trend (2018-2022)')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.show()
