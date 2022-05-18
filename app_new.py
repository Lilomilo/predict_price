from flask import Flask, request
import joblib
import numpy

RF_MODEL_PATH = 'mlmodels/rf_model.pkl'
CB_MODEL_PATH = 'mlmodels/cb_model.pkl'
SCALER_X_PATH = 'mlmodels/scaler_x.pkl'
SCALER_Y_PATH = 'mlmodels/scaler_y.pkl'

rf_model = joblib.load(RF_MODEL_PATH)
cb_model = joblib.load(CB_MODEL_PATH)
sc_x = joblib.load(SCALER_X_PATH)
sc_y = joblib.load(SCALER_Y_PATH)

app = Flask(__name__)

@app.route("/predict_price", methods = ['GET'])

def predict():
    args = request.args

    model_version = args.get('model_version', default=-1, type = int)
    open_plan = args.get('open_plan', default=-1, type = int)
    rooms = args.get('rooms', default=-1, type = int)
    area = args.get('area', default=-1, type = float)
    kitchen_area = args.get('kitchen_area', default=-1, type = float)
    living_area = args.get('living_area', default=-1, type = float)
    renovation = args.get('renovation', default=-1, type=int)

    params = [open_plan, rooms, area, kitchen_area, living_area, renovation]

    x = numpy.array(params).reshape(1, -1)
    x = sc_x.transform(x)

    if model_version == 1:
        result = rf_model.predict(x)
        result = sc_y.inverse_transform(result.reshape(1, -1))
    elif model_version == 2:
        result = cb_model.predict(x)
        result = sc_y.inverse_transform(result.reshape(1, -1))

    if any([i == -1 for i in params]):
        return 'ex.500 Internal server error', 500
    else:
        return str(result[0][0])

if __name__ == '__main__':
    app.run(debug=True, port=6262, host='0.0.0.0')
