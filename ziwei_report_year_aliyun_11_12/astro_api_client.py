import json
import requests
import os
from zhdate import ZhDate
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

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

    def _call_api(self, horoscope_date):
        """
        调用占星API的核心方法。

        Args:
            horoscope_date (str): 运限日期。

        Returns:
            dict: API响应结果。
        """
        payload = {
            "dateStr": self.birth_info["dateStr"],
            "type": self.birth_info["type"],
            "timeIndex": self.birth_info["timeIndex"],
            "gender": self.birth_info["gender"],
            "horoscopeDate": horoscope_date,
            "astroType": self.astro_type
        }

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
        return f"{date.year}-{date.month}-{date.day} {date.hour}:{date.minute}"


if __name__ == '__main__':
    # 使用示例，直接定义配置参数
    API_URL = os.getenv("ZIWEI_API_URL", "http://43.242.97.25:3000/astro_with_option")
    BIRTH_INFO = {
        "dateStr": "2023-11-22",
        "type": "solar",
        "timeIndex": 2,
        "gender": "male"
    }
    ASTRO_TYPE = "heaven"
    QUERY_YEAR = 2025
    SAVE_RESULTS = True
    OUTPUT_FILE = "lunar_astro_results.json"

    try:
        # 1. 使用参数直接创建客户端实例
        client = AstroAPIClient(
            api_url=API_URL,
            birth_info=BIRTH_INFO,
            astro_type=ASTRO_TYPE,
            query_year=QUERY_YEAR,
            save_results=SAVE_RESULTS,
            output_file=OUTPUT_FILE
        )

        # 2. 运行完整的查询流程
        print(f"正在为 {client.query_year} 年的农历日期执行查询...")
        results = client.run_query(client.query_year)

        # 3. 打印或处理结果
        success_count = 0
        for result in results:
            print("-" * 40)
            print(f"【农历{result['lunar_month']}月初一】")
            print(f"公历日期：{result['gregorian_date']}")
            if result['api_result']['success']:
                print(f"✓ API调用成功！")
                # 在这里可以进一步处理 result['api_result']['data']
                success_count += 1
            else:
                print(f"✗ API调用失败：{result['api_result']['error']}")

        print("-" * 40)
        print(f"\n程序执行完成！共处理了 {len(results)} 个查询，其中 {success_count} 个成功。")

        # 保存结果到文件
        if client.save_results:
            final_result = {
                "query_year": client.query_year,
                "birth_info": client.birth_info,
                "astro_type": client.astro_type,
                "query_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "results": results
            }
            try:
                with open(client.output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_result, f, ensure_ascii=False, indent=2)
                print(f"\n结果已保存到: {client.output_file}")
            except Exception as e:
                print(f"保存结果文件时出错: {e}")

    except Exception as e:
        print(f"程序执行时发生未知错误：{e}")
        import traceback

        traceback.print_exc()
