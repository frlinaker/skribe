import pandas as pd
from promptlearn import PromptRegressor


def test_zero_row_regressor_runs():
    X = pd.DataFrame(columns=["length"])
    y = pd.Series(name="mass", dtype=float)

    reg = PromptRegressor(verbose=False)
    reg.fit(X, y)

    result = reg.predict(pd.DataFrame([{"length": 2.5}]))
    assert isinstance(result[0], float)
