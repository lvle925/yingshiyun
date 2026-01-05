import os
import time
import hmac
import hashlib
import requests
import concurrent.futures
import re
import pandas as pd
from datetime import datetime

# ===== 配置 =====
APPID = "yingshi_appid"
APP_SECRET = "zhongzhoullm"
API_URL = "https://ai.mianxiaoxian.com/chat_yingshis_V10_23"
MAX_WORKERS = 20

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
input_path = os.path.join(desktop, "紫薇拒绝回答.txt")
output_excel = os.path.join(desktop, f"紫薇拒绝回答评估_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")


def generate_signature(params, app_secret):
    filtered = {k: str(v) for k, v in params.items() if k not in ['sign', 'skip_intent_check', 'weather_info', 'detailed_intent', 'skip_intent_check', 'is_knowledge_query']}
    sorted_params = dict(sorted(filtered.items()))
    string_to_sign = "".join(f"{k}{v}" for k, v in sorted_params.items())
    return hmac.new(
        app_secret.encode('utf-8'),
        string_to_sign.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def send_single_request(prompt, index):
    session_id = f"test_session_{index}"
    # ftime = int(time.time())  
    timestamp = int(time.time())  

    # params = {
    #     "appid": APPID,
    #     "prompt": prompt.strip(),
    #     "format": "json",
    #     "ftime": ftime,
    #     "session_id": session_id
    # }

    params = {
        "appid": APPID,
        "query": prompt.strip(),     
        "timestamp": str(timestamp),    
        "session_id": session_id,
        # 可按需开启的可选字段：
        # "skip_intent_check": False,
        # "is_knowledge_query": False,
        # "weather_info": None,
    }
    params["sign"] = generate_signature(params, APP_SECRET)

    start = time.time()
    try:
        resp = requests.post(API_URL, json=params, stream=True, timeout=30)
        status = resp.status_code
        full_response = ""
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                full_response += chunk
        duration = time.time() - start
        return index, prompt.strip(), full_response, status, duration
    except Exception as e:
        duration = time.time() - start
        error_msg = f"[请求异常] {str(e)}"
        return index, prompt.strip(), error_msg, 0, duration


def analyze_response(prompt, response, status, duration):
    """
    分析响应，返回评估字典
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 判断是否回复
    is_replied = "是" if (status == 200 and "[请求异常]" not in response and len(response.strip()) > 10) else "否"

    if is_replied == "否":
        return {
            "时间": now,
            "序号": None,
            "问题": prompt,
            "是否回复": "否",
            "收到的响应": response[:1000],
            "回复是否合理": "否",
            "逻辑是否清晰": "否",
            "建议的时间点有几条": 0,
            "响应时间（秒）": round(duration, 2),
            "基础分": 100,
            "最终得分": 0,
            "扣分原因": "未成功返回有效内容"
        }

    # 尝试提取 JSON 中的 data（如果格式为 {"code":200,"data":"..."}）
    content = response
    if '"data"' in response:
        try:
            import json
            parsed = json.loads(response)
            if "data" in parsed:
                content = str(parsed["data"])
        except:
            pass  # 保持原 response

    # 判断合理性（简单规则）
    unreasonable_keywords = ["错误", "失败", "异常", "不支持", "无法", "error", "Error"]
    is_reasonable = "否"
    for kw in unreasonable_keywords:
        if kw in content:
            is_reasonable = "否"
            break
    else:
        is_reasonable = "是" if len(content) > 20 else "否"

    # 判断逻辑清晰（是否有明确建议、结构化语言）
    clear_indicators = ["适合", "建议", "推荐", "最佳", "可选", "如下", "吉日", "时间"]
    is_clear = "是" if any(kw in content for kw in clear_indicators) and len(content.split("。")) >= 2 else "否"

    # 提取“建议的时间点数量”
    time_count = 0
    # 匹配数字 + 个/条/天/日 等
    patterns = [
        r"(\d+)个.*?时间",
        r"(\d+)条.*?建议",
        r"(\d+)个.*?日期",
        r"(\d+)个.*?吉日",
        r"(\d+)个.*?日子",
        r"(\d+)个.*?时间点",
        r"如下.*?(\d+)个",
        r"有.*?(\d+)个",
    ]
    for pat in patterns:
        match = re.search(pat, content)
        if match:
            time_count = int(match.group(1))
            break

    # 如果没匹配到，但内容中有多个日期（如 3月5日、3月12日），可估算
    if time_count == 0:
        # 粗略匹配日期模式：X月X日 或 X/X
        dates = re.findall(r"\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}", content)
        if dates:
            time_count = len(set(dates))  # 去重

    # 评分逻辑
    base_score = 100
    deduction = 0
    reasons = []

    if is_reasonable == "否":
        deduction += 40
        reasons.append("回复不合理")
    if is_clear == "否":
        deduction += 30
        reasons.append("逻辑不清晰")
    if time_count == 0 and ("什么时间" in prompt or "适合" in prompt):
        deduction += 30
        reasons.append("未给出具体时间建议")

    final_score = max(0, base_score - deduction)
    reason_str = "; ".join(reasons) if reasons else "无"

    return {
        "时间": now,
        "序号": None,  # 后续填充
        "问题": prompt,
        "是否回复": "是",
        "收到的响应": content[:1000],  # 截断避免 Excel 卡顿
        "回复是否合理": is_reasonable,
        "逻辑是否清晰": is_clear,
        "建议的时间点有几条": time_count,
        "响应时间（秒）": round(duration, 2),
        "基础分": base_score,
        "最终得分": final_score,
        "扣分原因": reason_str
    }


def main():
    if not os.path.exists(input_path):
        print(f"❌ 错误：未找到输入文件\n{input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        print("⚠️ 输入文件为空！")
        return

    print(f"🚀 开始并发测试，共 {len(prompts)} 个问题，最大并发数: {MAX_WORKERS}")

    results = [None] * len(prompts)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(send_single_request, prompt, i)
            for i, prompt in enumerate(prompts)
        ]

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            idx, prompt, response, status, duration = future.result()
            analysis = analyze_response(prompt, response, status, duration)
            analysis["序号"] = idx + 1
            results[idx] = analysis
            completed += 1
            print(f"✅ 已完成 {completed}/{len(prompts)} | 问题 {idx + 1} | 得分: {analysis['最终得分']}")

    # 转为 DataFrame 并保存 Excel
    df = pd.DataFrame(results)
    # 按序号排序（确保顺序）
    df = df.sort_values("序号").reset_index(drop=True)
    df.to_excel(output_excel, index=False, engine="openpyxl")

    print(f"\n🎉 评估完成！结果已保存至：\n{output_excel}")


if __name__ == "__main__":
    main()