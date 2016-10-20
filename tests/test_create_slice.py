from thesis import data
import numpy as np


class TestCreateSlice:


    def test_create_slice(self):
        x = np.linspace(0, 1, 10)
        y = (np.arange(10) + 1) * 100

        # first case
        max_h = 3
        num_window = 2
        training_ratio = 1
        fold_gap = 1
        windows = data.create_window(x, y, max_h,
                                     num_window,
                                     training_ratio,
                                     fold_gap)

        window1, window2 = windows

        np.testing.assert_allclose(window1[0],
                                   np.array([0.11111,
                                             0.22222,
                                             0.33333,
                                             0.44444,
                                             0.55555,
                                             0.66667]),
                                   rtol=1e-2)
        np.testing.assert_allclose(window1[1], np.array([200, 300, 400, 500, 600, 700]))
        np.testing.assert_allclose(window1[2], np.array([0.77777, 0.8888, 1]),
                                   rtol=1e-2)
        np.testing.assert_allclose(window1[3], np.array([800, 900, 1000]))

        np.testing.assert_allclose(window2[0],
                                   np.array([0,
                                             0.11111,
                                             0.22222,
                                             0.33333,
                                             0.44444,
                                             0.55555]),
                                   rtol=1e-2)
        np.testing.assert_allclose(window2[1], np.array([100, 200, 300, 400, 500, 600]))
        np.testing.assert_allclose(window2[2], np.array([0.66667, 0.77777, 0.8888]),
                                   rtol=1e-2)
        np.testing.assert_allclose(window2[3], np.array([700, 800, 900]))


    def test_create_slice_with_fold_gap(self):
        x = np.linspace(0, 1, 10)
        y = (np.arange(10) + 1) * 100
        # second case
        fold_gap = 2
        max_h = 2
        num_window = 2
        training_ratio = 1
        windows = data.create_window(x, y, max_h,
                                     num_window,
                                     training_ratio,
                                     fold_gap)

        window1, window2 = windows

        np.testing.assert_allclose(window1[0],
                                   np.array([0.22222,
                                             0.33333,
                                             0.44444,
                                             0.55555,
                                             0.66667,
                                             0.77777]),
                                   rtol=1e-2)
        np.testing.assert_allclose(window1[1], np.array([300, 400, 500, 600, 700, 800]))
        np.testing.assert_allclose(window1[2], np.array([0.8888, 1]),
                                   rtol=1e-2)
        np.testing.assert_allclose(window1[3], np.array([900, 1000]))

        np.testing.assert_allclose(window2[0],
                                   np.array([0,
                                             0.11111,
                                             0.22222,
                                             0.33333,
                                             0.44444,
                                             0.55555]),
                                   rtol=1e-2)
        np.testing.assert_allclose(window2[1], np.array([100, 200, 300, 400, 500, 600]))
        np.testing.assert_allclose(window2[2], np.array([0.66667, 0.77777]),
                                   rtol=1e-2)
        np.testing.assert_allclose(window2[3], np.array([700, 800]))


    def test_create_slice_with_training_ratio(self):
        x = np.linspace(0, 1, 10)
        y = (np.arange(10) + 1) * 100
        # second case
        fold_gap = 1
        max_h = 4
        num_window = 2
        training_ratio = 0.75
        windows = data.create_window(x, y, max_h,
                                     num_window,
                                     training_ratio,
                                     fold_gap)

        window1, window2 = windows

        np.testing.assert_allclose(window1[0],
                                   np.array([0.22222,
                                             0.33333,
                                             0.44444,
                                             0.55555]),
                                   rtol=1e-2)
        np.testing.assert_allclose(window1[1], np.array([300, 400, 500, 600]))
        np.testing.assert_allclose(window1[2], np.array([0.66667, 0.7777,
                                                         0.8888, 1]),
                                   rtol=1e-2)
        np.testing.assert_allclose(window1[3], np.array([700, 800, 900, 1000]))

        np.testing.assert_allclose(window2[0],
                                   np.array([0.11111,
                                             0.22222,
                                             0.33333,
                                             0.44444]),
                                   rtol=1e-2)
        np.testing.assert_allclose(window2[1], np.array([200, 300, 400, 500]))
        np.testing.assert_allclose(window2[2], np.array([0.55555, 0.66667, 0.7777,
                                                         0.8888]),
                                   rtol=1e-2)
        np.testing.assert_allclose(window2[3], np.array([600, 700, 800, 900]))
