import os
import multiprocessing
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

modules_to_compile = [
    "models",
    "utils",
    "ziwei_ai_function",
    "database.db_manager",
    "services.ziwei_analyzer",
    "clients.vllm_client"
]

extensions = []
for module_path in modules_to_compile:
    filepath = module_path.replace('.', os.sep) + '.py'
    if not os.path.exists(filepath):  # 跳过不存在的文件（避免编译报错+耗时）
        continue
    ext = Extension(
        name=module_path,
        sources=[filepath],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],  # 最高级优化，适配本地CPU
        extra_link_args=["-O3"],
        language="c"
    )
    extensions.append(ext)

n_cpu = multiprocessing.cpu_count()  # 自动获取CPU核心数

setup(
    name="ZiweiAIServiceCore",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'infer_types': True,
            'boundscheck': False,  # 关闭边界检查（提速，非安全敏感场景可用）
            'wraparound': False,   # 关闭负索引检查（提速）
            'initializedcheck': False  # 关闭初始化检查（提速）
        },
        exclude=[
            "api_main.py", "config.py", "setup.py", "services/chat_processor.py"
        ],
        nthreads=n_cpu  # 并行编译核心数（关键：多核同时编译）
    ),
    zip_safe=False,
    setup_requires=[],
    install_requires=[],
)
