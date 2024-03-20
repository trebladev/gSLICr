
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import glob
os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sources = glob.glob('gSLICr_Lib/engines/*.cpp')+glob.glob('gSLICr_Lib/engines/*.cu') + glob.glob('gSLICr_Lib/objects/*.cpp') + glob.glob('ext.cpp')
include_dirs = [os.path.join(ROOT_DIR, "gSLICr_Lib")]

setup(
    name="gslic",
    ext_modules=[
        CUDAExtension(
            name="gslic",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']},
    )],
    cmdclass={
        'build_ext': BuildExtension
    }
)
