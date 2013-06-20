import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)

import weightedHammingKernel as whksvm

if __name__ == '__main__':
    whksvm.execute()
