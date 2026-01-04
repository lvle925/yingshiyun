# models.py

from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Dict, Any, List, Literal, Optional

# --- API 请求/响应模型 ---
class SignableAPIRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="会话ID")
    query: str = Field(..., description="用户的查询内容")
    appid: str = Field(..., description="应用ID")
    timestamp: str = Field(..., description="请求时间戳 (秒)")
    sign: str = Field(..., description="HMAC-SHA256 签名")
    # 【核心新增】: 添加可选的天气信息字段
    weather_info: Optional[str] = Field(None, description="可选的天气信息，期望是一个JSON格式的字符串")
    # 问题7&10: 标记是否为知识类查询
    is_knowledge_query: Optional[bool] = Field(False, description="标记是否为知识类查询")
    # 控制是否跳过意图识别
    skip_intent_check: Optional[bool] = Field(False, description="是否跳过意图识别，直接进行分析")
    # 【新增】从Summary传来的意图信息（用于混合传递模式）
    summary_intent_type: Optional[str] = Field(None, description="Summary服务识别的意图类型（如general_long_term、knowledge_question等）")
    summary_intent_reason: Optional[str] = Field(None, description="Summary服务的分类理由")

# --- 内部数据结构模型 ---
class BirthInfo(BaseModel):
    year: Optional[int] = Field(None, description="出生年份，例如 1990")
    month: Optional[int] = Field(None, description="出生月份，1到12")
    day: Optional[int] = Field(None, description="出生日期，1到31")
    hour: Optional[int] = Field(0, description="出生小时，0到23，24小时制。例如，下午1点是13。如果用户未明确提及小时，请默认为0。")
    minute: Optional[int] = Field(0, description="出生分钟，0到59。如果用户未明确提及分钟，请默认为0。")
    gender: Optional[Literal["男", "女"]] = Field(None, description="性别，男或女")
    is_lunar: Optional[bool] = Field(False, description="是否农历，true表示农历，false表示公历。如果用户未明确指出，请默认为公历（false）。")
    traditional_hour_branch: Optional[str] = Field(None, description="如果用户提供了传统时辰（如子、丑、寅等），请提取此字段。")

    @field_validator('minute', mode='before')
    @classmethod
    def set_minute_default(cls, v):
        if v is None:
            return 0
        return v

class LLMExtractionResult(BaseModel):
    """LLM从用户输入中提取的原始信息"""
    time_expression: Optional[str] = Field(None, description="用户提到的原始时间描述，如 '今天', '明年', '2025年3月15日', '下一个大限', '这辈子'。若未提及时间，则为null。")
    relative_time_indicator: Optional[str] = Field(None, description="提取用户问题中的时间指示器，如：'今后十年'、'后10年'、'未来五年'、'下一个大运'等。若无明确时间指示器，则为null。")
    decadal_query_type: Optional[Literal["single_next_decadal", "multi_decadal", "comparison"]] = Field(None, description="判断大运查询类型：'single_next_decadal'=单个下一个大运（如'下一个大运'、'当前大运'），'multi_decadal'=多大运（如'未来10年大运'），'comparison'=多大运比较（如'当前大运和下个大运哪个好'），null=不是大运查询。")
    multi_decadal_span: Optional[int] = Field(None, description="当decadal_query_type为'multi_decadal'时，解析出的时间跨度年数。例如：'未来10年大运'返回10。")
    multi_yearly_query: bool = Field(False, description="判断是否为多流年查询（多个年份且不含'大运'关键词）。例如：'26 27 28年'、'未来5年运势'返回true。单个年份如'明年'返回false。")
    multi_yearly_years: Optional[List[int]] = Field(None, description="当multi_yearly_query为true时，解析出的具体年份列表。例如：'未来5年'返回[2024, 2025, 2026, 2027, 2028]。")
    explicit_decadal_ganzhi: Optional[str] = Field(
        None,
        description="如果用户在问题中明确提到某个大运的干支名称（如'壬戌大运'），这里只填干支本身（如'壬戌'）；否则为null。"
    )
    is_monthly_query: bool = Field(False, description="是否为流月/月份级别的问题，如'这个月'、'下个月'、'未来3个月'。")
    is_daily_query: bool = Field(False, description="是否为流日/天级别的问题，如'今天'、'明天'、'5月3日'。")
    topics: List[str] = Field(default_factory=list, description="查询涉及的核心主题，如 '财运', '事业', '婚姻', '健康', '整体运势', '学业'。")
    is_about_other: bool = Field(False, description="查询是否明确关于他人，如 '我丈夫', '我孩子'。")
    relationship: Optional[str] = Field(None, description="如果is_about_other为true，这里是关系描述，如 '丈夫', '孩子', '父母'。")
    is_sensitive_topic: bool = Field(False, description="查询是否涉及政治、军事、战争等不应讨论的敏感话题。")
    # 【已移除】is_irrelevant_topic 字段 - 理由：重复判断已在 classify_query_intent_with_llm 中完成
    is_naming_question: bool = Field(False, description="查询是否是关于给某人或某物取名。")  # 新增字段
    is_capability_inquiry: bool = Field(False, description="判断用户是否在询问AI自身的功能、能力或身份。")  # 新增字段
    is_knowledge_question: bool = Field(False, description="判断用户是否在询问命理专业术语、概念解释、理论知识。") # 新增字段
    relevant_palaces: List[Literal[
        "命宫", "福德宫", "事业宫", "财帛宫", "迁移宫",
        "田宅宫", "夫妻宫", "子女宫", "兄弟宫", "父母宫",
        "交友宫", "疾厄宫"
    ]] = Field(
        default_factory=list,
        description="根据用户问题判定的宫位列表。如果无法确定，请返回全部十二宫。"
    )

class MultiTimeAnalysisResult(BaseModel):
    """多时间分析判断结果"""
    query_type: Literal["multi_yearly_analysis", "multi_decadal_analysis", "multi_monthly_analysis", "none"] = Field(
        ..., description="查询类型：多流年分析、多大运分析、多流月分析，或不属于以上类型"
    )
    multi_yearly_years: Optional[List[int]] = Field(None, description="当query_type为multi_yearly_analysis时，解析出的具体年份列表")
    multi_decadal_span: Optional[int] = Field(None, description="当query_type为multi_decadal_analysis时，解析出的时间跨度年数或大运数量（负数表示大运数量）")
    multi_month_span: Optional[int] = Field(None, description="当query_type为multi_monthly_analysis时，解析出的月份数")
    relative_time_indicator: Optional[str] = Field(None, description="提取的原始时间指示器")
    # 【新增】大模型生成的选项列表，用于"是否想问"提示
    suggestion_options: Optional[List[str]] = Field(None, description="当query_type为multi_yearly_analysis、multi_decadal_analysis或multi_monthly_analysis时，大模型生成的选项列表，格式如：['2025年的这几年财运如何', '2026年的这几年财运如何', '2027年的这几年财运如何']。每个选项应该是一个完整的问句，包含年份/月份/大运信息和核心问题。最多生成3个选项。")


class QueryTypeClassificationResult(BaseModel):
    """查询类型分类结果（第一步判断）"""
    query_time_type: Literal["single_yearly", "single_decadal", "single_monthly", "single_daily", "lifetime"] = Field(
        ..., description="查询的时间粒度类型：单流年、单大运、单流月、单流日、一生查询"
    )
    time_expression: Optional[str] = Field(None, description="提取的时间表达式，用于后续处理")


class DetailedIntentExtractionResult(BaseModel):
    """详细意图提取结果（第二步提取）"""
    topics: List[str] = Field(default_factory=list, description="查询涉及的核心主题")
    is_about_other: bool = Field(False, description="查询是否明确关于他人")
    relationship: Optional[str] = Field(None, description="如果is_about_other为true，这里是关系描述")
    relevant_palaces: List[Literal[
        "命宫", "福德宫", "事业宫", "财帛宫", "迁移宫",
        "田宅宫", "夫妻宫", "子女宫", "兄弟宫", "父母宫",
        "交友宫", "疾厄宫"
    ]] = Field(default_factory=list, description="根据用户问题判定的宫位列表")
    # 单大运特有字段
    explicit_decadal_ganzhi: Optional[str] = Field(None, description="如果用户在问题中明确提到某个大运的干支名称，只填干支本身；否则为null")
    is_next_decadal_query: Optional[bool] = Field(None, description="判断是否询问下一个大运（仅当query_time_type为single_decadal时有效）")
    # 单流年/单流月/单流日/一生查询的通用字段
    relative_time_indicator: Optional[str] = Field(None, description="提取的原始时间指示器")
    # 时间解析字段（仅当query_time_type为single_yearly/single_monthly/single_daily时有效）
    target_year: Optional[int] = Field(None, description="解析后的目标年份，如2025。根据relative_time_indicator和当前日期计算得出")
    target_month: Optional[int] = Field(None, description="解析后的目标月份，1-12。根据relative_time_indicator和当前日期计算得出")
    target_day: Optional[int] = Field(None, description="解析后的目标日期，1-31。根据relative_time_indicator和当前日期计算得出")
    target_hour: Optional[int] = Field(None, description="解析后的目标小时，0-23。如果未指定，默认12")
    target_minute: Optional[int] = Field(None, description="解析后的目标分钟，0-59。如果未指定，默认0")
    resolved_horoscope_date: Optional[str] = Field(None, description="解析后的完整日期时间字符串，格式：YYYY-MM-DD HH:MM:SS。根据target_year、target_month、target_day、target_hour、target_minute构建")
    analysis_level: Optional[Literal["yearly", "monthly", "daily"]] = Field(None, description="分析级别：yearly(流年)、monthly(流月)、daily(流日)。根据query_time_type确定")