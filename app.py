from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import pickle
import joblib

app = Flask(__name__)

# load trained model (saved in workspace)
MODEL_PATH = os.path.join(app.root_path, 'power_prediction.sav')
print('DEBUG: app.root_path =', app.root_path)
print('DEBUG: MODEL_PATH =', MODEL_PATH, 'exists=', os.path.exists(MODEL_PATH))
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as _f:
            _head = _f.read(16)
            print('DEBUG: model-file header bytes:', _head)
            try:
                print('DEBUG: model-file header hex:', _head.hex())
            except Exception:
                pass
    except Exception as _err:
        print('DEBUG: could not read model-file header:', _err)
model = None
# prefer joblib.load for sklearn artifacts (fall back to pickle)
try:
    model = joblib.load(MODEL_PATH)
    print('DEBUG: model loaded with joblib from', MODEL_PATH)
except Exception as joblib_err:
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            print('DEBUG: model loaded with pickle from', MODEL_PATH)
    except Exception as pickle_err:
        model = None
        print('Could not load model with joblib or pickle:', joblib_err, '/', pickle_err)

# mocked weather data (sample values shown in screenshots)
SAMPLE_WEATHER = {
    'Agartala': { 'temp_c': 28.22, 'humidity': 78, 'pressure_mm': 1008, 'wind_mps': 2.6 },
    'Delhi':    { 'temp_c': 30.1,  'humidity': 45, 'pressure_mm': 1005, 'wind_mps': 3.2 },
    'Mumbai':   { 'temp_c': 30.8,  'humidity': 70, 'pressure_mm': 1007, 'wind_mps': 4.1 },
    'Kolkata':  { 'temp_c': 29.5,  'humidity': 82, 'pressure_mm': 1006, 'wind_mps': 3.5 },
    'Chennai':  { 'temp_c': 31.0,  'humidity': 73, 'pressure_mm': 1009, 'wind_mps': 4.0 }
}

@app.route('/')
def index():
    # prefer `turbine.png` first, then `wind_turbine.jpg`, otherwise use bundled SVG
    png = os.path.join(app.root_path, 'static', 'img', 'turbine.png')
    jpg = os.path.join(app.root_path, 'static', 'img', 'wind_turbine.jpg')
    if os.path.exists(png):
        hero_image = 'img/turbine.png'
    elif os.path.exists(jpg):
        hero_image = 'img/wind_turbine.jpg'
    else:
        hero_image = 'img/wind.svg'
    return render_template('index.html', hero_image=hero_image)

@app.route('/y_predict', methods=['GET', 'POST'])
def y_predict():
    prediction = None
    error = None

    if request.method == 'POST':
        # read form values
        try:
            theo = float(request.form.get('theoretical_power', '0'))
            wind = float(request.form.get('windspeed', '0'))

            if model is None:
                error = 'Model not found — place `power_prediction.sav` in the project root.'
            else:
                # model expects a 2D array-like input; adjust if your model uses more features
                X = [[theo, wind]]
                pred = model.predict(X)
                prediction = round(float(pred[0]), 2)
        except Exception as e:
            error = f'Invalid input or prediction error: {e}'

    cities = sorted(SAMPLE_WEATHER.keys())
    return render_template('y_predict.html', cities=cities, prediction=prediction, error=error)

@app.route('/weather')
def weather():
    city = request.args.get('city', '')
    data = SAMPLE_WEATHER.get(city)
    if not data:
        return jsonify({'error': 'city not found', 'city': city}), 404
    return jsonify(data)

if __name__ == '__main__':
    # disable the auto-reloader to prevent repeated restarts caused by
    # joblib/loky temporary activity (watchdog was detecting site-packages)
    app.run(debug=True, use_reloader=False)
