import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

import customHammingKernelSVM as chksvm

if __name__ == '__main__':
    chksvm.execute()
