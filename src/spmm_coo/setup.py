from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='spmm_coo',
    version='1.0',
    description='Pytorch Extension Library of Optimized Sparse Matrix multiplication',
    author='Berke Kisin',
    author_email="kisinberke@gmail.com",
    python_requires='>=3.8',
    ext_modules=[
          CUDAExtension('spmm_coo', [
                'spmm_coo.cpp',
                'cuda/spmm_coo_cuda.cu'
          ])

    ],
    cmdclass={
          'build_ext': BuildExtension
    }
)