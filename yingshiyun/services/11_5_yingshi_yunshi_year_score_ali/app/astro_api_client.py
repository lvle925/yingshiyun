import json
import asyncio
import requests
import aiohttp
from datetime import datetime
from zhdate import ZhDate


class AstroAPIClient:
    """
    封装了占星API调用、日期转换和结果处理的客户端。
    """

    def __init__(self, api_url, birth_info, astro_type, query_year=2025, save_results=True,
                 output_file="lunar_astro_results.json"):
        """
        初始化客户端并设置所有必要的配置。

        Args:
            api_url (str): 占星API的地址。
            birth_info (dict): 包含出生日期、类型、时辰、性别的字典。
            astro_type (str): 占星类型（例如："heaven"）。
            query_year (int): 默认查询年份。
            save_results (bool): 是否将结果保存到文件。
            output_file (str): 结果保存的文件名。
        """
        self.api_url = api_url
        self.birth_info = birth_info
        self.astro_type = astro_type
        self.query_year = query_year
        self.save_results = save_results
        self.output_file = output_file

    def get_lunar_first_days(self, year):
        """
        获取指定年份每个月农历初一的公历日期。

        Args:
            year (int): 查询年份。

        Returns:
            list: 包含 (农历月份, 公历日期对象) 元组的列表。
        """
        lunar_first_days = []
        for month in range(1, 13):
            try:
                lunar_date = ZhDate(year, month, 1)
                gregorian_date = lunar_date.to_datetime()
                lunar_first_days.append((month, gregorian_date))
            except Exception as e:
                print(f"警告：无法获取农历{year}年{month}月初一的公历日期：{e}")
                lunar_first_days.append((month, None))
        return lunar_first_days

    def _build_payload(self, horoscope_date):
        payload = {
            "dateStr": self.birth_info["dateStr"],
            "type": self.birth_info["type"],
            "timeIndex": self.birth_info["timeIndex"],
            "gender": self.birth_info["gender"],
            "horoscopeDate": horoscope_date,
            "astroType": self.astro_type
        }
        return payload

    def _call_api(self, horoscope_date):
        """
        调用占星API的核心方法。

        Args:
            horoscope_date (str): 运限日期。

        Returns:
            dict: API响应结果。
        """
        payload = self._build_payload(horoscope_date)

        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            return {
                "success": True,
                "data": response.json(),
                "status_code": response.status_code
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }

    async def _call_api_async(self, session, horoscope_date, request_timeout=None):
        """
        异步调用占星API的核心方法，使用已有的aiohttp ClientSession。
        """
        payload = self._build_payload(horoscope_date)

        try:
            async with session.post(self.api_url, json=payload, timeout=request_timeout) as response:
                response.raise_for_status()
                data = await response.json()
                return {
                    "success": True,
                    "data": data,
                    "status_code": response.status
                }
        except aiohttp.ClientResponseError as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": e.status
            }
        except aiohttp.ClientError as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": None
            }

    def run_query(self, year):
        """
        执行完整的查询流程，包括日期转换和API调用。

        Args:
            year (int): 查询年份。

        Returns:
            list: 包含每个月查询结果的列表。
        """
        all_results = []
        lunar_first_days = self.get_lunar_first_days(year)

        for month, date in lunar_first_days:
            horoscope_date = None
            api_result = {"success": False, "error": "无法获取公历日期"}

            if date is not None:
                horoscope_date = self._format_horoscope_date(date)
                api_result = self._call_api(horoscope_date)

            result_entry = {
                "lunar_month": month,
                "gregorian_date": date.strftime('%Y-%m-%d %H:%M:%S') if date else None,
                "horoscope_date": horoscope_date,
                "api_result": api_result
            }
            all_results.append(result_entry)

        return all_results

    async def run_query_async(self, session, year=None, concurrency: int = 5, request_timeout=None):
        """
        异步执行查询流程，可并发调用API。

        Args:
            session (aiohttp.ClientSession): 已初始化的aiohttp客户端。
            year (int, optional): 查询年份，默认为实例的 query_year。
            concurrency (int): 并发上限。
            request_timeout (float | ClientTimeout, optional): aiohttp请求超时时间。

        Returns:
            list: 包含每个月查询结果的列表，与 run_query 相同格式。
        """
        if year is None:
            year = self.query_year

        lunar_first_days = self.get_lunar_first_days(year)
        results = [None] * len(lunar_first_days)
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def process_month(idx, month, date):
            horoscope_date = None
            api_result = {"success": False, "error": "无法获取公历日期"}

            if date is not None:
                horoscope_date = self._format_horoscope_date(date)
                async with semaphore:
                    api_result = await self._call_api_async(session, horoscope_date, request_timeout)

            results[idx] = {
                "lunar_month": month,
                "gregorian_date": date.strftime('%Y-%m-%d %H:%M:%S') if date else None,
                "horoscope_date": horoscope_date,
                "api_result": api_result
            }

        await asyncio.gather(*[
            process_month(idx, month, date)
            for idx, (month, date) in enumerate(lunar_first_days)
        ])

        return results

    def _format_date_output(self, month, date):
        """格式化日期输出字符串。"""
        if date is None:
            return f"农历{month}月初一：无法获取对应的公历日期"
        formatted_date = date.strftime('%Y年%m月%d日 %H:%M')
        return f"农历{month:2d}月初一：{formatted_date}"

    def _format_horoscope_date(self, date):
        """格式化API需要的日期字符串。"""
        if date is None:
            return None
        return f"{date.year}-{date.month:02d}-{date.day:02d} {date.hour:02d}:{date.minute:02d}:00"