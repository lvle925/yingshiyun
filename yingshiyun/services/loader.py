from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Iterable, Optional


@contextmanager
def _temporary_sys_path(paths: Iterable[Path]):
    original_sys_path = list(sys.path)
    for path in reversed(list(paths)):
        sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path = original_sys_path


def load_module(module_name: str, module_path: Path, extra_paths: Optional[Iterable[Path]] = None) -> ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]

    if extra_paths is None:
        extra_paths = []

    with _temporary_sys_path(extra_paths):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


@dataclass(frozen=True)
class ServiceModule:
    name: str
    module_name: str
    module_path: Path
    extra_paths: tuple[Path, ...] = ()
    app_attr: str = "app"
    startup_attr: str = "startup_event"
    shutdown_attr: str = "shutdown_event"

    def load(self) -> ModuleType:
        return load_module(self.module_name, self.module_path, self.extra_paths)
