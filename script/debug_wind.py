from thesis import config
import numpy as np
from thesis.result.wind import WindResult

if __name__ == '__main__':
    output_folder = config.output + 'experiment/wind30/winter4/result/'
    horizons = np.array([3, 12, 24, 48])
    result = WindResult(horizons, 'MAE')
    result.read_lagmodel(output_folder+ 'narwind_blr*.npy', 'BLR', 48)