# prompt_logic.py
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Optional, Dict
from jinja2 import Template

logger = logging.getLogger(__name__)

# ===== 提示词配置文件管理（热更新支持）=====
# 全局变量存储提示词模板
_prompt_templates: Dict[str, str] = {}
_prompt_templates_lock = Lock()
_prompts_dir = Path("prompts")

def load_xml_prompt(filename: str) -> Optional[str]:
    """
    从 XML 文件加载提示词模板
    
    Args:
        filename: XML 文件名（如 "ossp_xml_template_str.xml"）
    
    Returns:
        提示词模板字符串，如果加载失败返回 None
    """
    try:
        prompt_file = _prompts_dir / filename
        logger.debug(f"尝试加载提示词文件: {prompt_file} (存在: {prompt_file.exists()})")
        
        if not prompt_file.exists():
            logger.warning(f"提示词文件不存在: {prompt_file}")
            return None
        
        logger.debug(f"开始读取文件: {filename}")
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"成功加载提示词文件: {filename} (大小: {len(content)} 字符)")
        return content
    except Exception as e:
        logger.error(f"加载提示词文件 {filename} 失败: {e}", exc_info=True)
        return None

def reload_all_prompts(use_lock: bool = True):
    """
    重新加载所有提示词模板（支持热更新）
    
    Args:
        use_lock: 是否使用锁（启动时可以不使用，热更新时使用）
    """
    global _prompt_templates
    
    def _do_reload():
        logger.info("开始重新加载提示词模板...")
        logger.info(f"提示词目录: {_prompts_dir.absolute()} (存在: {_prompts_dir.exists()})")
        
        # 加载紫微运势生成提示词
        logger.info("正在加载 OSSP 提示词...")
        ossp = load_xml_prompt("ossp_xml_template_str.xml")
        if ossp:
            _prompt_templates["ossp"] = ossp
            logger.info("✓ OSSP 提示词加载成功")
        else:
            logger.warning("⚠️ OSSP 提示词加载失败或文件不存在")
        
        # 加载出行建议提示词
        logger.info("正在加载出行建议提示词...")
        travel = load_xml_prompt("travel_advice_prompt_template.xml")
        if travel:
            _prompt_templates["travel"] = travel
            logger.info("✓ 出行建议提示词加载成功")
        else:
            logger.warning("⚠️ 出行建议提示词加载失败或文件不存在")
        
        # 加载意图提取提示词
        logger.info("正在加载意图提取提示词...")
        query_intent = load_xml_prompt("query_intent_extraction_prompt_template.xml")
        if query_intent:
            _prompt_templates["query_intent"] = query_intent
            logger.info("✓ 意图提取提示词加载成功")
        else:
            logger.warning("⚠️ 意图提取提示词加载失败或文件不存在")
        
        # 加载多大运 Jinja2 模板
        logger.info("正在加载多大运分析提示词...")
        multi_decade = load_xml_prompt("multi_decade_analysis.j2")
        if multi_decade:
            _prompt_templates["multi_decade"] = multi_decade
            logger.info("✓ 多大运分析提示词加载成功")
        else:
            logger.warning("⚠️ 多大运分析提示词加载失败或文件不存在")
        
        # 加载多流年 Jinja2 模板
        logger.info("正在加载多流年分析提示词...")
        multi_yearly = load_xml_prompt("multi_yearly_analysis.j2")
        if multi_yearly:
            _prompt_templates["multi_yearly"] = multi_yearly
            logger.info("✓ 多流年分析提示词加载成功")
        else:
            logger.warning("⚠️ 多流年分析提示词加载失败或文件不存在")
        
        # 加载多时间分析判断提示词
        logger.info("正在加载多时间分析判断提示词...")
        multi_time_analysis = load_xml_prompt("multi_time_analysis.xml")
        if multi_time_analysis:
            _prompt_templates["multi_time_analysis"] = multi_time_analysis
            logger.info("✓ 多时间分析判断提示词加载成功")
        else:
            logger.warning("⚠️ 多时间分析判断提示词加载失败或文件不存在")
        
        # 加载查询类型分类提示词
        logger.info("正在加载查询类型分类提示词...")
        query_type_classification = load_xml_prompt("query_type_classification.xml")
        if query_type_classification:
            _prompt_templates["query_type_classification"] = query_type_classification
            logger.info("✓ 查询类型分类提示词加载成功")
        else:
            logger.warning("⚠️ 查询类型分类提示词加载失败或文件不存在")
        
        # 加载详细意图提取提示词
        logger.info("正在加载详细意图提取提示词...")
        detailed_intent_extraction = load_xml_prompt("detailed_intent_extraction.xml")
        if detailed_intent_extraction:
            _prompt_templates["detailed_intent_extraction"] = detailed_intent_extraction
            logger.info("✓ 详细意图提取提示词加载成功")
        else:
            logger.warning("⚠️ 详细意图提取提示词加载失败或文件不存在")
        
        logger.info(f"提示词模板加载完成，共加载 {len(_prompt_templates)} 个模板")
    
    if use_lock:
        with _prompt_templates_lock:
            _do_reload()
    else:
        # 启动时不需要锁（没有并发访问）
        _do_reload()

def get_prompt_template(template_name: str) -> Optional[str]:
    """
    获取提示词模板
    
    Args:
        template_name: 模板名称 ("ossp", "travel", "query_intent", "multi_decade", "multi_yearly")
    
    Returns:
        提示词模板字符串
    """
    # 如果缓存为空，先加载（启动时不需要锁）
    if not _prompt_templates:
        logger.info("提示词模板缓存为空，首次加载...")
        reload_all_prompts(use_lock=False)  # 启动时不需要锁
    
    # 读取时使用锁（虽然启动时不需要，但为了一致性还是使用）
    with _prompt_templates_lock:
        result = _prompt_templates.get(template_name)
        if not result:
            logger.warning(f"⚠️ 模板 '{template_name}' 在缓存中不存在。当前缓存中的模板: {list(_prompt_templates.keys())}")
        return result

def render_jinja_template(template_name: str, **context) -> Optional[str]:
    """
    渲染 Jinja2 模板
    
    Args:
        template_name: 模板名称 ("multi_decade", "multi_yearly")
        **context: 模板上下文变量
    
    Returns:
        渲染后的提示词字符串
    """
    try:
        template_str = get_prompt_template(template_name)
        if not template_str:
            logger.error(f"无法获取模板: {template_name}")
            return None
        
        template = Template(template_str)
        rendered = template.render(**context)
        logger.debug(f"成功渲染模板: {template_name}")
        return rendered
    
    except Exception as e:
        logger.error(f"渲染模板 {template_name} 失败: {e}", exc_info=True)
        return None

# ===== 原有代码继续 =====

# --- 出生信息提取Prompt ---
BIRTH_INFO_EXTRACTION_PROMPT_TEMPLATE = (
        """<|im_start|>system
你是一个专门用于解析用户出生信息的AI助手，角色是【信息提取器】。
你的唯一任务是分析下面<user_input>标签中的文本，并严格按照<output_format>中定义的JSON结构和规则，提取并生成一个JSON对象。

**【核心指令】**
1.  **绝对专注**: 只关注明确的、与个人出生相关的年月日时和性别信息。完全忽略任何与“运势查询”、“未来时间”等相关的词语，如“明天”、“2025年运势”。
2.  **严格格式**: 你的输出必须且只能是一个JSON对象。在JSON对象生成完毕后，立即停止，不得添加任何解释、注释或额外文本。
3.  **数值转换**: `year`, `month`, `day`, `hour`, `minute` 字段必须是整数（integer）。将所有口语化时间（如“晚上9点半”）精确转换为24小时制的数字。
4.  **空值处理**: 如果任何信息未被提供，对应的字段值必须为`null`。如果未提及分钟，`minute`默认为`0`。
5.  **时辰优先级**: 只有当用户明确使用“子时”、“丑时”等十二地支时辰词语时，才填充`traditional_hour_branch`字段（例如'丑时'），并同时将`hour`字段设为`null`。这是唯一`hour`可以为`null`而`traditional_hour_branch`有值的场景。

**【输出格式定义】**
<output_format>
{
  "year": "integer | null - 出生公历年份",
  "month": "integer | null - 出生月份 (1-12)",
  "day": "integer | null - 出生日期 (1-31)",
  "hour": "integer | null - 24小时制出生小时 (0-23)",
  "minute": "integer | null - 出生分钟 (0-59)",
  "gender": "string | null - '男' 或 '女'",
  "is_lunar": "boolean - 是否为农历，默认为false",
  "traditional_hour_branch": "string | null - 传统时辰，如 '子时', '丑时'等"
}
</output_format>

**【处理示例】**
这里有一些你必须严格遵循的输入输出范例。

<example>
  <user_input>我的阳历出生日期是1990年1月1日早上8点，我是女性</user_input>
  <assistant_output>
    {"year": 1990, "month": 1, "day": 1, "hour": 8, "minute": 0, "gender": "女", "is_lunar": false, "traditional_hour_branch": null}
  </assistant_output>
</example>
<example>
  <user_input>农历1985年冬月十五晚上9点半，女</user_input>
  <assistant_output>
    {"year": 1985, "month": 11, "day": 15, "hour": 21, "minute": 30, "gender": "女", "is_lunar": true, "traditional_hour_branch": null}
  </assistant_output>
</example>
<example>
  <user_input>我是2000年3月3日丑时出生的女性</user_input>
  <assistant_output>
    {"year": 2000, "month": 3, "day": 3, "hour": null, "minute": 0, "gender": "女", "is_lunar": false, "traditional_hour_branch": "丑时"}
  </assistant_output>
</example>
<example>
  <user_input>我是个女孩，生日是2001年5月20号</user_input>
  <assistant_output>
    {"year": 2001, "month": 5, "day": 20, "hour": null, "minute": 0, "gender": "女", "is_lunar": false, "traditional_hour_branch": null}
  </assistant_output>
</example>
<example>
  <user_input>1990年生的，我想问问12月12日运势如何</user_input>
  <assistant_output>
    {"year": 1990, "month": null, "day": null, "hour": null, "minute": 0, "gender": null, "is_lunar": false, "traditional_hour_branch": null}
  </assistant_output>
</example>
<example>
  <user_input>我的命宫怎么样？</user_input>
  <assistant_output>
    {"year": null, "month": null, "day": null, "hour": null, "minute": 0, "gender": null, "is_lunar": false, "traditional_hour_branch": null}
  </assistant_output>
</example>
<|im_end|>""")

# --- 查询意图提取Prompt ---
# 【改进】从 XML 文件动态加载提示词模板，支持热更新
# 注意：改为函数形式延迟加载，避免在模块导入时立即读取文件（可能文件还未挂载）
def get_query_intent_prompt():
    """延迟加载查询意图提取提示词"""
    return get_prompt_template("query_intent") or ""

def get_travel_advice_prompt():
    """延迟加载出行建议提示词"""
    return get_prompt_template("travel") or ""

def get_ossp_xml_template():
    """延迟加载紫微运势生成提示词"""
    return get_prompt_template("ossp") or ""

# 保持向后兼容：提供模块级变量（延迟加载，在启动时初始化）
# 初始化为空字符串，避免在模块导入时立即读取文件（可能文件还未挂载）
QUERY_INTENT_EXTRACTION_PROMPT_TEMPLATE = ""
TRAVEL_ADVICE_PROMPT_TEMPLATE = ""
OSSP_XML_TEMPLATE_STR = ""
# GENERAL_QNA_XML_TEMPLATE_STR 在 vllm_client.py 中定义，这里只提供占位符（向后兼容）
GENERAL_QNA_XML_TEMPLATE_STR = ""

def _init_prompt_templates():
    """初始化提示词模板（延迟加载，在应用启动时调用）"""
    global QUERY_INTENT_EXTRACTION_PROMPT_TEMPLATE, TRAVEL_ADVICE_PROMPT_TEMPLATE, OSSP_XML_TEMPLATE_STR
    try:
        QUERY_INTENT_EXTRACTION_PROMPT_TEMPLATE = get_query_intent_prompt()
        TRAVEL_ADVICE_PROMPT_TEMPLATE = get_travel_advice_prompt()
        OSSP_XML_TEMPLATE_STR = get_ossp_xml_template()
        logger.info("提示词模板初始化完成")
    except Exception as e:
        logger.error(f"初始化提示词模板失败: {e}", exc_info=True)

user_prompt_xml = """
<context>
    <current_date>{current_dt.strftime('%Y-%m-%d')}</current_date>
</context>
<user_input>
{user_input}
</user_input>
"""

response_string_capability = """
我将运用易经理论在内的综合知识，辅助您从思辨角度审视不同时间维度下，在工作、事业、财富、婚恋、人际关系、出行等方方面面可能面临的决策。通过深入剖析，将为您提供富有洞察力的参考，助您做出明智选择，享受精彩人生
"""

def get_multi_period_analysis_prompt(
    birth_info: dict,
    chart_data: dict,
    user_question: str,
    analysis_scope_str: str,
    analysis_level: str,  # 'decadal' 或 'yearly'
    relevant_palaces: list = None  # 【新增】相关宫位列表
) -> str:
    """
    多时段分析（多大运/多流年）专用prompt
    
    Args:
        birth_info: 出生信息
        chart_data: 合并后的多个周期命盘数据
        user_question: 用户问题
        analysis_scope_str: 分析范围描述，如"2026-2035年（3个大运）"
        analysis_level: 分析级别
        relevant_palaces: 相关宫位列表，如["命宫", "财帛宫"]
    """
    
    # 【修复】添加类型检查，防止传入字符串
    if not isinstance(chart_data, dict):
        raise ValueError(f"chart_data必须是字典类型，但收到了: {type(chart_data)}")
    
    if not chart_data:
        raise ValueError("chart_data不能为空字典")
    
    period_name = "大运" if analysis_level == "decadal" else "流年"
    
    # 构建多周期数据说明
    period_list = []
    if analysis_level == "decadal":
        for palace, palace_data in chart_data.items():
            for period_key in palace_data.keys():
                if "大运盘" in period_key:
                    period_list.append(period_key.replace("盘", ""))
    else:  # yearly
        for palace, palace_data in chart_data.items():
            for period_key in palace_data.keys():
                if "流年盘" in period_key:
                    period_list.append(period_key.replace("盘", ""))
    
    periods_str = "、".join(list(dict.fromkeys(period_list)))  # 去重
    
    # 【新增】根据relevant_palaces动态生成宫位分析部分
    # 宫位到分析标题的映射
    palace_analysis_map = {
        "命宫": ("核心要素", "命主的核心状态、整体运势和人生格局"),
        "财帛宫": ("财富状况", "财运、收入状况和理财能力"),
        "事业宫": ("事业发展", "事业状况、工作环境和职业发展"),
        "官禄宫": ("事业发展", "事业状况、工作环境和职业发展"),  # 兼容别名
        "夫妻宫": ("感情婚姻", "感情状态、婚姻关系和伴侣互动"),
        "疾厄宫": ("健康状况", "健康情况、体质状况和疾病预防"),
        "交友宫": ("人际关系", "人际状况、贵人运和社交圈"),
        "迁移宫": ("外出迁移", "出行运、变动机会和外地发展"),
        "田宅宫": ("家宅财产", "家庭环境、不动产和居住状况"),
        "福德宫": ("精神享受", "内心状态、精神享受和生活品质"),
        "父母宫": ("长辈关系", "与父母长辈的关系、贵人助力"),
        "子女宫": ("子女缘分", "子女关系、桃花运和创造力"),
        "兄弟宫": ("手足情谊", "兄弟姐妹关系、同辈互动")
    }
    
    # 【新增】判断是否为"直接问大运/流年"还是"问具体方面"
    # 如果relevant_palaces为空或只有命宫，说明用户没有问具体方面，应该输出全部宫位
    # 如果relevant_palaces有多个且包含特定宫位，说明LLM识别到了具体问题
    has_specific_question = relevant_palaces and len(relevant_palaces) >= 2 and any(p != "命宫" for p in relevant_palaces)
    
    if not relevant_palaces or not has_specific_question:
        # 用户没有问具体方面，输出核心6宫位（使用事业宫而非官禄宫）
        relevant_palaces = ["命宫", "事业宫", "财帛宫", "夫妻宫", "疾厄宫", "交友宫"]
        logger.info(f"[多时段分析] 未检测到具体问题焦点，将输出核心6宫位: {relevant_palaces}")
    else:
        # 用户问了具体方面，只输出相关宫位
        # 【修复】将官禄宫统一转换为事业宫
        relevant_palaces = [宫.replace("官禄宫", "事业宫") for 宫 in relevant_palaces]
        # 确保命宫始终在列表中（如果不在的话添加到开头）
        if "命宫" not in relevant_palaces:
            relevant_palaces = ["命宫"] + relevant_palaces
        logger.info(f"[多时段分析] 检测到具体问题焦点，只输出相关宫位: {relevant_palaces}")
    
    # 构建宫位分析部分
    palace_analysis_sections = []
    idx = 1  # 手动控制序号，确保连续
    for palace in relevant_palaces:
        if palace in palace_analysis_map:
            title, description = palace_analysis_map[palace]
            palace_analysis_sections.append(f"""{idx}. **{title}**（{palace}）
   - 推论：在[第一个{period_name}名称]期间，[{description}的具体分析]；在[第二个{period_name}名称]期间，[{description}的变化]；在[第三个{period_name}名称]期间，[{description}的趋势]。
   - 综合来看，整个时期的{title}呈现[总体趋势判断]。""")
            idx += 1  # 只有成功添加后才递增
    
    palace_analysis_text = "\n\n".join(palace_analysis_sections)
    
    # 构建其他宫位补充说明
    analyzed_palace_names = "、".join([palace_analysis_map.get(p, (p, ""))[0] for p in relevant_palaces if p in palace_analysis_map])
    
    # 【改进】使用 Jinja2 模板渲染，支持热更新
    template_name = "multi_decade" if analysis_level == "decadal" else "multi_yearly"
    
    # 准备模板上下文
    context = {
        "birth_info": birth_info,
        "chart_data_json": json.dumps(chart_data, ensure_ascii=False, indent=2),
        "user_question": user_question,
        "analysis_scope_str": analysis_scope_str,
        "periods_str": periods_str,
        "analyzed_palace_names": analyzed_palace_names,
        "palace_analysis_text": palace_analysis_text
    }
    
    # 渲染模板
    prompt = render_jinja_template(template_name, **context)
    
    if not prompt:
        raise ValueError(
            f"无法渲染模板 '{template_name}'，请检查 prompts/{template_name}_analysis.j2 文件是否存在且格式正确"
        )
    
    return prompt
