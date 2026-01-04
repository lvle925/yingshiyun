from setuptools import setup, Extension
from Cython.Build import cythonize

# 定义需要被编译的模块
# "prompt_logic" 是编译后模块的名字，其他文件可以通过 from prompt_logic import ... 来使用它
# ["prompt_logic.pyx"] 是源文件列表
extensions = [
    Extension("prompt_logic", ["prompt_logic.pyx"])
]

setup(
    ext_modules=cythonize(extensions)
)
