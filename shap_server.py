from shapash.explainer.smart_explainer import SmartExplainer
import joblib

y_pred = joblib.load('./y_pred.joblib')
Xtest = joblib.load('./Xtest.joblib')
regressor = joblib.load('./regressor.joblib')
encoder = joblib.load('./encoder.joblib')
house_dict = joblib.load('./house_dict.joblib')

xpl = SmartExplainer(features_dict=house_dict) # optional parameter, specifies label for features name
xpl.compile(
    x=Xtest,
    model=regressor,
    preprocessing=encoder, # Optional: compile step can use inverse_transform method
    y_pred=y_pred # Optional
)

app = xpl.run_app()

try:
    while True:
        pass
except KeyboardInterrupt:
    app.kill()

