import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import ParameterGrid
from xgboost import XGBRegressor


# INTERPOLATION PIPLINE !!!
def interpolation_model(df, model, param_distributions, target='cnt'):

    # Split features en target
    X = df.drop(columns=[target])
    y = df[target]

    # Handmatig split voor time series data
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # TimeSeriesSplit voor hyperparameter tuning
    tscv = TimeSeriesSplit(n_splits=5)

    # Pipeline om data leakage te voorkomen
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])

    # Random search voor efficiencie
    rand = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
    )

    # Fitten van het beste model uit hyperparameter tuning
    rand.fit(X_train, y_train)
    best_pipe = rand.best_estimator_

    # Predicties
    y_train_pred = best_pipe.predict(X_train)
    y_test_pred = best_pipe.predict(X_test)

    # Score metrics
    print("Train RMSE:", mean_squared_error(y_train, y_train_pred), "R2:", r2_score(y_train, y_train_pred))
    print("Test  RMSE:", mean_squared_error(y_test, y_test_pred), "R2:", r2_score(y_test, y_test_pred))

    # Plotting resultaten
    plt.figure(figsize=(15, 5))
    plt.plot(y_test.values, label="Original")
    plt.plot(y_test_pred, label="Predicted")
    plt.legend()
    plt.title("Original VS Predicted")
    plt.tight_layout()
    plt.show()

    return best_pipe, rand.best_params_

# EXTRAPOLATION MODELS!!!
def sarimax_extrapolation(df, order, seasonal_order, target, exogenous, split=0.8):
    # Handmatig split voor time series data
    y = df[target]
    X = df[exogenous]
    split_idx = int(len(df) * split)
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_train, X_test = X[:split_idx], X[split_idx:]

    # SARIMAX model
    model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order)
    res = model.fit(disp=False)

    # Predicties
    y_train_pred = res.predict(start=y_train.index[0], end=y_train.index[-1], exog=X_train)
    y_test_pred = res.get_forecast(steps=len(y_test), exog=X_test).predicted_mean

    # Score metrics
    print("Train RMSE:", mean_squared_error(y_train, y_train_pred), "R2:", r2_score(y_train, y_train_pred))
    print("Test  RMSE:", mean_squared_error(y_test, y_test_pred), "R2:", r2_score(y_test, y_test_pred))

    # Plotting resultaten
    plt.figure(figsize=(15, 5))
    plt.plot(y_test.values, label="Original")
    plt.plot(y_test_pred.values, label="Forecast")
    plt.legend()
    plt.show()

    return res, {"order": order, "seasonal_order": seasonal_order}

def extrapolation_arma(df, order_grid, target='cnt', split=0.8):
    # Handmatig split voor time series data
    y = df[target]
    split_idx = int(len(df) * split)
    y_train, y_test = y[:split_idx], y[split_idx:]

    best_rmse = 1e18
    best_model = None
    best_order = None

    # Prompt 1: Parameter grid explanation for time series
    for params in ParameterGrid(order_grid):
        order = params['order']
        model = SARIMAX(y_train, order=order, seasonal_order=(0,0,0,0))
        res = model.fit(disp=False)

        preds = res.get_forecast(steps=len(y_test)).predicted_mean
        rmse = mean_squared_error(y_test, preds)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = res
            best_order = order

    # Predicties
    y_train_pred = best_model.fittedvalues
    y_test_pred = best_model.get_forecast(steps=len(y_test)).predicted_mean

    # Score metrics
    print("Best order:", best_order)
    print("Train RMSE:", mean_squared_error(y_train, y_train_pred))
    print("Test RMSE:", mean_squared_error(y_test, y_test_pred))

    # Plotting resultaten
    plt.figure(figsize=(15, 5))
    plt.plot(y_test.values, label="Original")
    plt.plot(y_test_pred.values, label="Forecast")
    plt.legend()
    plt.show()

    return best_model, {"best_order": best_order}

# HYBRID MODEL !!!
def hybrid_model(df, target='cnt', split=0.8):
    y = df[target]
    X = df.drop(columns=[target])

    split_idx = int(len(df) * split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Trend model (lr)
    trend = LinearRegression()
    trend.fit(X_train, y_train)
    trend_train_pred = trend.predict(X_train)
    trend_test_pred = trend.predict(X_test)

    # Seasonality + Residual model (xgb)
    residuals_train = y_train - trend_train_pred
    xgb = XGBRegressor()
    xgb.fit(X_train, residuals_train)

    # Predict residuals en seasonality
    residuals_train_pred = xgb.predict(X_train)
    residuals_test_pred = xgb.predict(X_test)

    # Predictions
    train_pred = trend_train_pred + residuals_train_pred
    test_pred = trend_test_pred + residuals_test_pred

    # Metrics
    print("Train RMSE:", mean_squared_error(y_train, train_pred))
    print("Test RMSE:", mean_squared_error(y_test, test_pred))

    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(y_test.values, label='Actual')
    plt.plot(test_pred, label='Prediction')
    plt.legend()
    plt.show()

    return trend, xgb