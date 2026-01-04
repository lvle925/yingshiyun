# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# 定义要编译的模块
# 我们加密核心逻辑，但保留 API 入口和配置为 .py 文件
modules_to_compile = [
    "app/processing.py",
    "app/utils.py",
]

extensions = [
    Extension(
        name=module.replace(os.path.sep, ".").replace(".py", ""), # e.g., "app.processing"
        sources=[module],
    )
    for module in modules_to_compile
]

setup(
    name="ZhanXingApp Core",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)