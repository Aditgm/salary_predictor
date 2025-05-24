from salary_predictor.core import LinearRegressor

def test_prediction():
    model = LinearRegressor(weights=1.5, bias=10)
    assert model.predict(2) == 13  # 1.5*2 + 10
