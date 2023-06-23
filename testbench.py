import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sadia_data.csv', header=None, names=['Wavenumber', 'Intensity'], delimiter=' ')

x_data = df['Wavenumber']
y_data = df['Intensity']

# Plot the original data
plt.scatter(x_data, y_data, label='Data')
plt.xlabel('Wavenumber')
plt.ylabel('Intensity')
plt.legend()
plt.show()