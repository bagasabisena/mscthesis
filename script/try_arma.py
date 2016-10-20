# -*- coding: utf-8 -*-
import numpy as np
from statsmodels import api as sm

x = np.linspace(-2, 2)
y = np.sin(x)

model_arma = sm.tsa.ARMA(y, (2, 0))
fit = model_arma.fit()
y_, err, cov = fit.forecast(1)