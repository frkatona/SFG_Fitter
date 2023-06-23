import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

def create_model(num_peaks):
    model = None
    for i in range(num_peaks):
        prefix = f"g{i}_"
        if model is None:
            model = Model(gaussian, prefix=prefix)
        else:
            model += Model(gaussian, prefix=prefix)
    return model

color_list = ['#00296b', '#2b9348', '#4cc9f0', '#ffa62b', '#902923', '#3f7d20', '#d9c5b2', '#c879ff']

## LOAD DATA ##
df_data = pd.read_csv('SFG_data.csv', header=None, names=['Wavenumber', 'Intensity'], delimiter=' ')
x_data = df_data['Wavenumber']
y_data = df_data['Intensity']

df_params = pd.read_csv('SFG_parameters_test.csv', header=None, names=['amp', 'cen', 'wid'])

## USER INPUT ##
num_peaks = df_params.shape[0]

## GENERATE MODEL AND PARAMETERS ##
model = create_model(num_peaks)
params = model.make_params()

# Try to load parameters from csv
try:
    for index, row in df_params.iterrows():
        peak = index
        params[f'g{peak}_amp'].set(value=row['amp'], min=0)
        params[f'g{peak}_cen'].set(value=row['cen'], min=x_data.min(), max=x_data.max())
        params[f'g{peak}_wid'].set(value=row['wid'], min=0)
except FileNotFoundError:
    # if the file is not found, use generic parameters
    for i in range(num_peaks):
        params[f'g{i}_amp'].set(value=1, min=0)
        params[f'g{i}_cen'].set(value=x_data.mean(), min=x_data.min(), max=x_data.max())
        params[f'g{i}_wid'].set(value=x_data.std(), min=0)

## FIT MODEL ##
result = model.fit(y_data, params, x=x_data)

## PLOT ##
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, result.best_fit, 'r-', label='Fit')

comps = result.eval_components()
for i in range(num_peaks):
    plt.plot(x_data, comps[f'g{i}_'], '--', label=f'Peak {i+1}', color=color_list[i])
    plt.axvline(result.params[f'g{i}_cen'].value, linestyle=':', color=color_list[i])

## DISPLAY FIT STATISTICS ##
print(result.fit_report())
fit_stats = f"chi-square: {result.chisqr:.3e}\nredchi: {result.redchi:.3e}\naic: {result.aic:.0f}\nbic: {result.bic:.0f}\nr-squared: {result.rsquared:.2f}"
plt.text(0.3, 0.95, fit_stats, transform=plt.gca().transAxes, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel('cm-1')
plt.ylabel('Intensity')
plt.legend()
plt.show()