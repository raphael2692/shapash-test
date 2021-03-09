
import pandas as pd
from category_encoders import OrdinalEncoder # https://contrib.scikit-learn.org/category_encoders/
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from shapash.data.data_loader import data_loading

import joblib

house_df, house_dict = data_loading('house_prices')


y_df=house_df['SalePrice'].to_frame()
X_df=house_df[house_df.columns.difference(['SalePrice'])]

categorical_features = [col for col in X_df.columns if X_df[col].dtype == 'object']
encoder = OrdinalEncoder(
    cols=categorical_features,
    handle_unknown='ignore',
    return_df=True).fit(X_df)

X_df=encoder.transform(X_df)

Xtrain, Xtest, ytrain, ytest = train_test_split(X_df, y_df, train_size=0.75, random_state=1)

regressor = LGBMRegressor(n_estimators=200).fit(Xtrain,ytrain)

y_pred = pd.DataFrame(regressor.predict(Xtest),columns=['pred'],index=Xtest.index)

# Salva
joblib.dump(y_pred, './y_pred.joblib')
joblib.dump(Xtest, './Xtest.joblib')
joblib.dump(regressor, './regressor.joblib')
joblib.dump(encoder, './encoder.joblib')
joblib.dump(house_dict, './house_dict.joblib')
