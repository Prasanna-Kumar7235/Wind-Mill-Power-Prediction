"""Create and save a demo scikit-learn model compatible with `app.py`.
This creates a simple LinearRegression that expects two features: [theoretical_power, windspeed].
Saved file: power_prediction.sav (joblib format)
"""
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# create synthetic training data (1000 samples)
rng = np.random.RandomState(0)
theo = rng.uniform(0, 100, size=1000)          # theoretical power input
wind = rng.uniform(0, 15, size=1000)           # wind speed
# target: roughly linear combination + noise
y = 0.6 * theo + 2.0 * wind + rng.normal(scale=5.0, size=1000)
X = np.column_stack([theo, wind])

model = LinearRegression()
model.fit(X, y)

# save model (joblib recommended for sklearn objects)
joblib.dump(model, 'power_prediction.sav')
print('Demo model saved to power_prediction.sav')