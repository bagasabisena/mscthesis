from unittest import TestCase
import numpy as np
from thesis.data import HorizonWindowData
from thesis.data import create_window


class TestHorizonWindowData(TestCase):
    def test_split(self):
        x = np.linspace(0, 1, 10)
        y = (np.arange(10) + 1) * 100

        windows = create_window(x, y, 3, 2)
        x_train, y_train, x_test, y_test = windows[0]

        # lag=1, h=1
        hwd = HorizonWindowData(y_train, y_test, 1, 1)

        np.testing.assert_allclose(hwd.x_train.flatten(),
                                   np.array([200, 300, 400, 500, 600]))

        np.testing.assert_allclose(hwd.y_train.flatten(),
                                   np.array([300, 400, 500, 600, 700]))

        np.testing.assert_allclose(hwd.x_test.flatten(),
                                   np.array([700]))

        np.testing.assert_allclose(hwd.y_test.flatten(),
                                   np.array([800]))

        # lag=1, h=3
        hwd = HorizonWindowData(y_train, y_test, 3, 1)

        np.testing.assert_allclose(hwd.x_train.flatten(),
                                   np.array([200, 300, 400]))

        np.testing.assert_allclose(hwd.y_train.flatten(),
                                   np.array([500, 600, 700]))

        np.testing.assert_allclose(hwd.x_test.flatten(),
                                   np.array([700]))

        np.testing.assert_allclose(hwd.y_test.flatten(),
                                   np.array([1000]))

        # lag=3, h=2
        hwd = HorizonWindowData(y_train, y_test, 2, 3)

        np.testing.assert_allclose(hwd.x_train,
                                   np.array([[200, 300, 400],
                                             [300, 400, 500]]))

        np.testing.assert_allclose(hwd.y_train.flatten(),
                                   np.array([600, 700]))

        np.testing.assert_allclose(hwd.x_test,
                                   np.array([[500, 600, 700]]))

        np.testing.assert_allclose(hwd.y_test.flatten(),
                                   np.array([900]))
