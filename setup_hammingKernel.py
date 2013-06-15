from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

pyxname = 'hammingKernelSVM'

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension(pyxname, [pyxname+".pyx"])]
)
