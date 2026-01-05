from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from .loader import ServiceModule


@dataclass(frozen=True)
class ServiceDefinition:
    name: str
    module: ServiceModule

    def load_app(self):
        loaded = self.module.load()
        return getattr(loaded, self.module.app_attr)

    def load_startup(self) -> Optional[Callable]:
        loaded = self.module.load()
        return getattr(loaded, self.module.startup_attr, None)

    def load_shutdown(self) -> Optional[Callable]:
        loaded = self.module.load()
        return getattr(loaded, self.module.shutdown_attr, None)


BASE_DIR = Path(__file__).resolve().parents[2]
SERVICES_DIR = BASE_DIR / "yingshiyun" / "services"


SERVICE_DEFINITIONS: tuple[ServiceDefinition, ...] = (
    ServiceDefinition(
        name="leinuo_yunshi_day",
        module=ServiceModule(
            name="leinuo_yunshi_day",
            module_name="yingshiyun_ext.leinuo_yunshi_day.api_main",
            module_path=SERVICES_DIR / "10_10_leinuo_yunshi_day" / "api_main.py",
            extra_paths=(SERVICES_DIR / "10_10_leinuo_yunshi_day",),
        ),
    ),
    ServiceDefinition(
        name="yunshi_year_score",
        module=ServiceModule(
            name="yunshi_year_score",
            module_name="yingshiyun_ext.yunshi_year_score.app.api",
            module_path=SERVICES_DIR / "11_5_yingshi_yunshi_year_score_ali" / "app" / "api.py",
            extra_paths=(SERVICES_DIR / "11_5_yingshi_yunshi_year_score_ali",),
        ),
    ),
    ServiceDefinition(
        name="qimen_day",
        module=ServiceModule(
            name="qimen_day",
            module_name="yingshiyun_ext.qimen_day.main",
            module_path=SERVICES_DIR / "12_11_yingshi_yunshi_day_new_qimen" / "main.py",
            extra_paths=(SERVICES_DIR / "12_11_yingshi_yunshi_day_new_qimen",),
        ),
    ),
    ServiceDefinition(
        name="llm_leinuo",
        module=ServiceModule(
            name="llm_leinuo",
            module_name="yingshiyun_ext.llm_leinuo.main",
            module_path=SERVICES_DIR / "12_12_llm_yingshi_leinuo_ali" / "main.py",
            extra_paths=(SERVICES_DIR / "12_12_llm_yingshi_leinuo_ali",),
        ),
    ),
    ServiceDefinition(
        name="llm_qimen",
        module=ServiceModule(
            name="llm_qimen",
            module_name="yingshiyun_ext.llm_qimen.main",
            module_path=SERVICES_DIR / "12_12_llm_yingshi_qimen_ali" / "main.py",
            extra_paths=(SERVICES_DIR / "12_12_llm_yingshi_qimen_ali",),
        ),
    ),
    ServiceDefinition(
        name="llm_summary",
        module=ServiceModule(
            name="llm_summary",
            module_name="yingshiyun_ext.llm_summary.app.main",
            module_path=SERVICES_DIR / "12_12_llm_yingshi_summary_ali" / "app" / "main.py",
            extra_paths=(SERVICES_DIR / "12_12_llm_yingshi_summary_ali",),
        ),
    ),
    ServiceDefinition(
        name="llm_ziwei",
        module=ServiceModule(
            name="llm_ziwei",
            module_name="yingshiyun_ext.llm_ziwei.api_main",
            module_path=SERVICES_DIR / "12_12_llm_yingshi_ziwei_ali" / "api_main.py",
            extra_paths=(SERVICES_DIR / "12_12_llm_yingshi_ziwei_ali",),
        ),
    ),
    ServiceDefinition(
        name="report_prediction",
        module=ServiceModule(
            name="report_prediction",
            module_name="yingshiyun_ext.report_prediction.api_main",
            module_path=SERVICES_DIR / "report_prediction_bendi_test_12_8" / "api_main.py",
            extra_paths=(SERVICES_DIR / "report_prediction_bendi_test_12_8",),
        ),
    ),
    ServiceDefinition(
        name="ziwei_report_year",
        module=ServiceModule(
            name="ziwei_report_year",
            module_name="yingshiyun_ext.ziwei_report_year.api_main",
            module_path=SERVICES_DIR / "ziwei_report_year_aliyun_11_12" / "api_main.py",
            extra_paths=(SERVICES_DIR / "ziwei_report_year_aliyun_11_12",),
        ),
    ),
)


def iter_service_definitions() -> Iterable[ServiceDefinition]:
    return SERVICE_DEFINITIONS


def get_service_definition(name: str) -> ServiceDefinition:
    for service in SERVICE_DEFINITIONS:
        if service.name == name:
            return service
    raise KeyError(f"Unknown service name: {name}")
