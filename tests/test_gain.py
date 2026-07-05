from gouda import GAIN
from sklearn.utils.estimator_checks import check_estimator
import numpy as np

def test_nans_simple():
    data = np.random.rand(500, 5)
    data[data < 0.48] = np.nan
    imputed = GAIN().fit(data).transform(data)
    print("data:\n", data)
    print(f"imputed:\n{imputed}")
    assert not np.isnan(imputed).any(), "Imputed still has missing values"

def test_checksklearn_simple():
    check_estimator(GAIN(encoding="label"))
