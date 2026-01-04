import asyncio
import json
import logging
import requests
from app.config import VLLM_API_BASE_URL, VLLM_MODEL_NAME,API_KEY
from app.prompt_manager import get_prompt_manager

logger = logging.getLogger(__name__)

# 初始化提示词管理器
prompt_manager = get_prompt_manager()


def clean_titles(data):
    """
    遍历字典，清洗所有包含 '标题' 的键的值，去除值中冒号及其之前的部分。
    支持中文冒号（：）和英文冒号（:）。
    """
    cleaned_data = {}
    for key, value in data.items():
        if "标题" in key and isinstance(value, str):
            # 查找中英文冒号的位置，取最后一个冒号后面的部分
            
            # 1. 查找最后一个中文冒号
            last_zh_colon = value.rfind('：')
            # 2. 查找最后一个英文冒号
            last_en_colon = value.rfind(':')
            
            # 找出最靠后的冒号位置
            last_colon_index = max(last_zh_colon, last_en_colon)
            
            if last_colon_index != -1:
                # 截取冒号后面的内容，并去除前后的空格
                cleaned_value = value[last_colon_index + 1:].strip()
                cleaned_data[key] = cleaned_value
            else:
                # 如果没有冒号，则保持原样
                cleaned_data[key] = value
        else:
            # 非标题键或非字符串值保持原样
            cleaned_data[key] = value
            
    return cleaned_data



def get_natural_conversation_role():
    """获取自然对话角色提示词（支持热更新）"""
    return prompt_manager.get("natural_conversation_role", "")


def _validate_llm_advice_format(advice: dict) -> bool:
    """
    一个严格的验证函数，用于检查LLM返回的运势建议JSON的格式。
    """
    # 规则 1: 检查 "运势判断"
    judgement = advice.get("运势判断")
    if not isinstance(judgement, str) or len(judgement) < 10:
        logger.warning(f"验证失败: '运势判断' 内容不符合要求 (必须是20-50字的字符串)。内容: '{judgement}'")
        return False

    # 新增规则: 检查 "运势判断详情"
    judgement_detail = advice.get("运势判断详情")
    if not isinstance(judgement_detail, str) or len(judgement_detail) < 120:
        logger.warning(f"验证失败: '运势判断详情' 内容不符合要求 (必须是不少于120字的字符串)。内容长度: {len(judgement_detail)}")
        return False

    # 规则 2 & 3: 检查 "建议"
    suggestions = advice.get("建议")

    for item in suggestions:
        if not isinstance(item, str) or len(item) > 5:
            logger.warning(f"验证失败: '建议' 中的项目 '{item}' 必须是不超过5个字的字符串。")
            return False

    # 规则 4 & 5: 检查 "避免"
    avoidances = advice.get("避免")

    for item in avoidances:
        if not isinstance(item, str) or len(item) > 5:
            logger.warning(f"验证失败: '避免' 中的项目 '{item}' 必须是不超过5个字的字符串。")
            return False

    return True


async def get_llm_daily_advice(palace_details_str: str, composite_score: float, 
                                llm_advice_guanlu: str = "", llm_advice_caibo: str = "",
                                llm_advice_fuqi: str = "", llm_advice_jie: str = "",
                                llm_advice_qianyi: str = "", llm_advice_puyi: str = "") -> dict:
    """
    调用LLM获取基于宫位详情和综合评分的今日运势建议，包含简短判断和详细解读。
    """

    api_url = f"{VLLM_API_BASE_URL}/chat/completions"
    headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

    # --- 【核心修改】: 更新默认响应，增加 "运势判断详情" ---
    default_response = {
        "运势判断": "今日运势稳中有升，事业与财运皆有良好进展，情感上多些耐心与理解，整体积极向好。",
        "运势判断详情": "今天，在事业的舞台上，你可能会发现之前的努力开始显现成果，或许是一个项目的顺利推进，或许是同事或上级的认可。这不仅是外界的肯定，更是你内在力量的体现。财务方面，机遇如同含苞待放的花朵，需要你用智慧和耐心去浇灌。保持敏锐的觉察力，但避免冲动决策。而在人际与情感的互动中，今天的主题是‘倾听’。花些时间真正去理解对方的感受，你会发现关系因此变得更加和谐与深入。这是一个适合自我沉淀、感受内心力量的日子，请善待自己，相信每一步都走在正确的道路上。",
        "今日关键词": "平顺",
        "运势概况": "运势呈现积蓄力量的态势。", # 新增的运势概况 15字以内
        "建议": ["保持耐心", "寻求支持"],
        "避免": ["过度忧虑", "冲动行事"]
    }


    # --- 【修改后的System Prompt】 ---
    try:
        system_prompt = prompt_manager.format_prompt(
            "daily_advice_system_prompt_template",
            palace_details_str=palace_details_str,
            composite_score=composite_score,
            llm_advice_guanlu=llm_advice_guanlu,
            llm_advice_caibo=llm_advice_caibo,
            llm_advice_fuqi=llm_advice_fuqi,
            llm_advice_jie=llm_advice_jie,
            llm_advice_qianyi=llm_advice_qianyi,
            llm_advice_puyi=llm_advice_puyi
        )
        if not system_prompt:
            logger.error("提示词模板为空，使用默认响应")
            return default_response
    except Exception as e:
        logger.error(f"格式化提示词失败: {e}", exc_info=True)
        return default_response


    user_prompt = (
        "<input_data>\n"
        f"  <palace_details>{palace_details_str}</palace_details>\n"
        f"  <composite_score>{composite_score:.1f}/100</composite_score>\n"
        "</input_data>"
    )

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.75,
        "max_tokens": 500,
        "response_format": {"type": "json_object"}
    }

    max_retries = 3
    last_exception = None

    for attempt in range(max_retries):
        try:
            logger.info(f"正在为综合评分 {composite_score:.1f} 请求LLM解读... (尝试 {attempt + 1}/{max_retries})")
            response = await asyncio.to_thread(
                requests.post, api_url, headers=headers, json=payload,timeout=6000
            )
            response.raise_for_status()
            content_str = response.json().get("choices", [{}])[0].get("message", {}).get("content")

            if not content_str:
                raise ValueError("LLM返回的content为空")

            advice_json = json.loads(content_str)
            logger.info(f"LLM在尝试 {attempt + 1} 时返回: {advice_json}")

            if _validate_llm_advice_format(advice_json):
                logger.info("LLM解读成功获取，且格式验证通过！")
                return advice_json
            else:
                raise ValueError("LLM返回的JSON内容格式不符合业务规则")

        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"调用LLM服务时在第 {attempt + 1} 次尝试中失败: {e}")
            last_exception = e
        except Exception as e:
            logger.critical(f"调用LLM服务时发生未预料的严重错误: {e}", exc_info=True)
            last_exception = e
            break

        if attempt < max_retries - 1:
            await asyncio.sleep(1.5)

    logger.error(f"在 {max_retries} 次尝试后，仍无法从LLM获取有效响应。最后一次错误: {last_exception}")
    return default_response



async def get_llm_palace_narrative(palace_name: str, palace_details_str: str, score_100: float) -> dict:
    """
    为单个宫位生成包含标题、详细讲解和引导性问题的JSON。
    """
    api_url = f"{VLLM_API_BASE_URL}/chat/completions"
    headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

    # --- 【核心修改】: 定义宫位到输出键的映射 ---
    palace_map = {
        "夫妻宫": {"theme": "感情", "title_key": "感情标题"},
        "财帛宫": {"theme": "财富", "title_key": "财富标题"},
        "事业宫": {"theme": "事业", "title_key": "事业标题"},
        "官禄宫": {"theme": "事业", "title_key": "事业标题"},
        "迁移宫": {"theme": "出行", "title_key": "出行标题"},
        "疾厄宫": {"theme": "健康", "title_key": "健康标题"},
        "交友宫": {"theme": "人际", "title_key": "人际标题"},
        "仆役宫": {"theme": "人际", "title_key": "人际标题"},
    }
    palace_info = palace_map.get(palace_name, {"theme": "运势", "title_key": "运势标题"})
    theme_name = palace_info["theme"]
    title_key = palace_info["title_key"]

    # --- 【核心修改】: 更新默认响应 ---
    default_response = {
        title_key: f"{theme_name}:平稳安定的一天",
        "运势详解": f"今日{theme_name}的能量较为平稳。建议您保持观察，静待时机，关注当下的每一个瞬间，内在的平和便是最好的指引。",
        "引导问题": [
            f"我今天的{theme_name}运势如何？",
            "在哪些方面我需要特别注意？",
            "今天适合做重要的决定吗？",
            "如何提升我的人际关系？",
            "我应该如何调整自己的心态？",
            "今天有什么潜在的机会吗？",
            "如何避免可能的风险？",
            "今天的幸运色或数字是什么？",
            "我应该主动出击还是静观其变？",
            "今天的整体能量是怎样的？"
        ]
    }

    natural_conversation_role = get_natural_conversation_role()
    
    # 获取宫位叙述的提示词模板
    try:
        system_prompt_template = prompt_manager.format_prompt(
            "palace_narrative_system_prompt_template",
            theme_name=theme_name,
            palace_details_str=palace_details_str,
            score_100=score_100,
            title_key=title_key
        )
        if not system_prompt_template:
            logger.error("提示词模板为空，使用默认响应")
            return default_response
        system_prompt = natural_conversation_role + "\n\n" + system_prompt_template
    except Exception as e:
        logger.error(f"格式化提示词失败: {e}", exc_info=True)
        return default_response

    user_prompt = (
        f"请严格按照JSON格式和范例风格，为【{theme_name}】创作运势详解。\n\n"
        f"【{theme_name}分数 (决定故事基调)】: {score_100:.1f} / 100\n\n"
        f"【{palace_name}详情 (作为创作素材)】:\n{palace_details_str}"
    )

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": 0.7,
        "max_tokens": 600,
        "response_format": {"type": "json_object"}
    }

    # print("payload",payload)

    max_retries = 5
    last_exception = None

    for attempt in range(max_retries):
        try:
            logger.info(f"正在为【{palace_name}】生成详细讲解... (尝试 {attempt + 1}/{max_retries})")
            response = await asyncio.to_thread(
                requests.post, api_url, headers=headers, json=payload, timeout=6000
            )
            response.raise_for_status()
            content_str = response.json().get("choices", [{}])[0].get("message", {}).get("content")

            if not content_str:
                raise ValueError("LLM返回的content为空")

            narrative_json = json.loads(content_str)
            narrative_json = clean_titles(narrative_json)
            print("narrative_json",narrative_json)
            
            # --- 【核心修改】: 增强验证逻辑 ---
            if (title_key in narrative_json and
                    isinstance(narrative_json[title_key], str) and
                    "运势详解" in narrative_json and
                    isinstance(narrative_json["运势详解"], str) and
                    "引导问题" in narrative_json and
                    isinstance(narrative_json["引导问题"], list) and
                    len(narrative_json["引导问题"]) == 10):
                logger.info(f"【{palace_name}】详细讲解生成成功。")
                return narrative_json
            else:
                raise ValueError("返回的JSON结构不完整或类型不正确")

        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"为【{palace_name}】生成讲解时在第 {attempt + 1} 次尝试中失败: {e}")
            last_exception = e
        except Exception as e:
            logger.critical(f"为【{palace_name}】生成讲解时发生未预料的严重错误: {e}", exc_info=True)
            last_exception = e
            break

        if attempt < max_retries - 1:
            await asyncio.sleep(1.5)

    logger.error(f"在 {max_retries} 次尝试后，仍无法为【{palace_name}】获取有效讲解。最后一次错误: {last_exception}")
    return default_response



async def get_llm_period_summary(
    full_text: str,
    topic: str,
    min_words: int,
    is_advice: bool = False
) -> dict:
    """
    调用LLM对一段时间内的多份报告文本进行总结，并返回一个包含总结的JSON对象。
    内置健壮的、最多5次的重试机制。
    【核心更新】：强制要求LLM在输出中使用“本周”作为时间定语。
    """
    api_url = f"{VLLM_API_BASE_URL}/chat/completions"
    headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
    
    default_response = {
        "summary": f"未能为【{topic}】生成有效的周期性总结，AI解读服务可能暂不可用。"
    }

    if not full_text or not full_text.strip():
        return {"summary": f"未能提供有效的【{topic}】内容进行总结。"}

    # --- 根据模式选择不同的系统提示词 ---
    if is_advice:
        system_prompt = f"""
        你是一位顶级的策略分析师，擅长从分散的信息中提炼出核心的、可执行的行动指南。你的任务是整合以下多天内所有的“建议”和“避免”事项，归纳出一个综合性的周期核心行动指南，并以严格的JSON格式输出。

        【核心指令】
        1. **结构化输出**：你的总结必须分为【核心建议】和【主要提醒】两个清晰的部分。
        2. **语言风格**：语言必须精练、有力，直指要点，避免空话套话。
        3. **【强制】时间定语**：在你的总结中，所有指代这个时间周期的词语，都必须强制使用“**本周**”二字。例如，你应该说“**本周**的核心建议是...”，而不是“这个周期的核心建议是...”。这是一个硬性要求。
        4. **【最重要】输出格式**：你的回答必须且只能是一个JSON对象，结构如下：
        
        【输出JSON格式要求】
        {{
          "summary": "【核心建议】\\n1. [提炼出的第一条核心建议]\\n2. [提炼出的第二条核心建议]\\n\\n【主要提醒】\\n1. [提炼出的第一条主要提醒]\\n2. [提炼出的第二条主要提醒]"
        }}
        """
        user_prompt_content = f"请根据以下每日建议与禁忌，严格遵守所有指令（特别是时间定语和JSON格式），提炼出**本周**的核心行动指南：\n\n{full_text}"

    else: # 普通文本总结模式
        system_prompt = f"""
        你是一位专业的首席报告撰写师，拥有卓越的归纳、分析和文字组织能力。你的任务是阅读并深度整合以下关于【{topic}】的多份每日报告，然后撰写一份逻辑连贯、观点深刻的周期性总结报告，并以严格的JSON格式输出。
        
        【核心指令】
        1. **深度与洞察力**：你的总结不能是简单的内容拼接，必须体现出对整体趋势、核心变化和潜在机遇/挑战的深刻洞察。直接指出关键点。
        2. **字数要求**：总结的内容正文，**必须不少于 {min_words} 字**。这是一个硬性要求。
        3. **语言风格**：行文流畅，专业且富有说服力，直接阐述。
        4. **【强制】时间定语**：在你的总结中，所有指代这个时间周期的词语（如“这段时间”、“在此期间”、“这个周期内”等），都**必须强制使用“本周”**。这是一个严格的指令，直接影响输出质量。例如，你应该写“**本周**你的整体趋势是...”，而不能写成“这段时间你的整体趋势是...”。
        5. **【最重要】输出格式**: 你的回答必须且只能是一个JSON对象，结构如下：

        【输出JSON格式要求】
        {{
          "summary": "这里是您撰写的、关于「{topic}」的、不少于{min_words}字的深度周期性总结报告..."
        }}
        """
        user_prompt_content = f"请基于以下每日【{topic}】的详细内容，严格遵守所有指令（特别是时间定语和JSON格式），撰写一份关于**本周**的、不少于{min_words}字的周期性总结报告：\n\n{full_text}"

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_content}],
        "temperature": 0.7,
        "max_tokens"              : 1200,
        "response_format": {"type": "json_object"}
    }
    
    # --- 重试循环部分保持不变，它已经足够健壮 ---
    max_retries = 5
    last_exception = None

    for attempt in range(max_retries):
        try:
            logger.info(f"正在为【{topic}】生成周期性总结... (尝试次数: {attempt + 1}/{max_retries})")
            
            response = await asyncio.to_thread(
                requests.post, api_url, headers=headers, json=payload, timeout=6000
            )
            response.raise_for_status()
            
            llm_response_data = response.json()
            content_str = llm_response_data.get("choices", [{}])[0].get("message", {}).get("content")

            if not content_str:
                raise ValueError("LLM返回的content为空")

            summary_json = json.loads(content_str)
            
            if "summary" in summary_json and isinstance(summary_json["summary"], str) and summary_json["summary"].strip():
                logger.info(f"【{topic}】周期性总结生成成功。")
                return summary_json
            else:
                raise ValueError("返回的JSON中缺少'summary'键、其值不是字符串或内容为空")

        except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"为【{topic}】生成总结时在第 {attempt + 1} 次尝试中失败: {e}")
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2.0)
            continue
        except Exception as e:
            logger.critical(f"为【{topic}】生成总结时发生未预料的严重错误: {e}", exc_info=True)
            last_exception = e
            break

    logger.error(f"在 {max_retries} 次尝试后，仍无法为【{topic}】获取有效的周期性总结。最后一次错误: {last_exception}")
    return default_response