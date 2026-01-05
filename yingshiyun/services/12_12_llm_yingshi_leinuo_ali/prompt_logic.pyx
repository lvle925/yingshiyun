import pandas as pd
import random
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from threading import Lock

# 获取一个logger实例，让这个模块也能打印日志
logger = logging.getLogger(__name__)

# ===== 新增：提示词配置文件管理 =====
# 全局变量存储提示词模板
_prompt_templates: Dict[str, str] = {}
_prompt_templates_lock = Lock()
_prompts_dir = Path("prompts")

def load_xml_prompt(filename: str) -> Optional[str]:
    """
    从 XML 文件加载提示词模板

    Args:
        filename: XML 文件名（如 "non_choice_prompt.xml"）

    Returns:
        提示词模板字符串，如果加载失败返回 None
    """
    try:
        prompt_file = _prompts_dir / filename
        if not prompt_file.exists():
            logger.error(f"提示词文件不存在: {prompt_file}")
            return None

        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        logger.info(f"成功加载提示词文件: {filename}")
        return content
    except Exception as e:
        logger.error(f"加载提示词文件 {filename} 失败: {e}", exc_info=True)
        return None

def reload_all_prompts():
    """重新加载所有提示词模板（支持热更新）"""
    global _prompt_templates

    with _prompt_templates_lock:
        logger.info("开始重新加载提示词模板...")

        # 加载非选择类提示词
        non_choice = load_xml_prompt("non_choice_prompt.xml")
        if non_choice:
            _prompt_templates["non_choice"] = non_choice
            logger.info("✓ 非选择类提示词加载成功")

        # 加载选择类提示词
        choice = load_xml_prompt("choice_prompt.xml")
        if choice:
            _prompt_templates["choice"] = choice
            logger.info("✓ 选择类提示词加载成功")

        logger.info(f"提示词模板加载完成，共加载 {len(_prompt_templates)} 个模板")

def get_prompt_template(template_name: str) -> Optional[str]:
    """
    获取提示词模板

    Args:
        template_name: 模板名称 ("non_choice" 或 "choice")

    Returns:
        提示词模板字符串
    """
    with _prompt_templates_lock:
        # 如果缓存为空，先加载
        if not _prompt_templates:
            logger.info("提示词模板缓存为空，首次加载...")
            reload_all_prompts()

        return _prompt_templates.get(template_name)

def start_prompt_file_watcher():
    """
    启动提示词文件监控（支持热更新）
    注意：需要 watchdog 库支持
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class PromptFileHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.src_path.endswith('.xml'):
                    logger.info(f"检测到提示词文件变化: {event.src_path}")
                    reload_all_prompts()

        observer = Observer()
        event_handler = PromptFileHandler()
        observer.schedule(event_handler, path=str(_prompts_dir), recursive=False)
        observer.start()
        logger.info(f"✓ 提示词文件监控已启动，监控目录: {_prompts_dir.absolute()}")
        return observer
    except ImportError:
        logger.warning("未安装 watchdog 库，热更新功能不可用。修改提示词需要重启服务。")
        return None
    except Exception as e:
        logger.error(f"启动提示词文件监控失败: {e}", exc_info=True)
        return None

# 初始化：在模块加载时预加载提示词
reload_all_prompts()

# ===== 以上为新增内容 =====

# 步骤2：把被依赖的 _draw_and_get_card_data 函数也移到这里
def _draw_and_get_card_data(
        cards_df: pd.DataFrame,
        meanings_df: pd.DataFrame,
        num_cards: int,
        card_number_pool: Optional[List[int]] = None
) -> tuple[Optional[tuple[str, ...]], Optional[tuple[int, ...]], Optional[tuple[str, ...]]]:
    """
    内部辅助函数：执行抽牌、获取牌名和牌意。
    """
    print("card_number_pool", card_number_pool)
    if cards_df is None or meanings_df is None:
        logger.error("错误: CSV 数据未加载。无法抽牌。")
        return None, None, None

    try:
        sampling_pool = cards_df["牌号"].tolist()
        if card_number_pool and len(card_number_pool) >= num_cards:
            all_valid_card_numbers_in_df = set(cards_df["牌号"].tolist())
            filtered_pool = [num for num in card_number_pool if num in all_valid_card_numbers_in_df]
            if len(filtered_pool) >= num_cards:
                sampling_pool = filtered_pool
                logger.info(f"使用提供的有效卡牌池进行抽样，池大小: {len(sampling_pool)}")

        if len(sampling_pool) < num_cards:
            logger.error(f"错误: 有效卡牌池不足 {num_cards} 张。")
            return None, None, None

        if len(sampling_pool) == num_cards:
            # 如果池中牌的数量刚好等于抽取数量，则按原顺序选择，不进行随机抽取
            selected_card_numbers = sampling_pool
            print("卡牌池数量与抽取数量一致，按原顺序选择，不进行随机抽样。")
        else:
            # 否则，进行随机抽样
            selected_card_numbers = random.sample(sampling_pool, num_cards)
            print("进行随机抽样。")

        selected_cards_df = cards_df[cards_df["牌号"].isin(selected_card_numbers)].copy()
        if len(selected_cards_df) < num_cards:
            return None, None, None

        selected_cards_df["牌号_ordered"] = pd.Categorical(selected_cards_df["牌号"], categories=selected_card_numbers,
                                                           ordered=True)
        selected_cards_df = selected_cards_df.sort_values("牌号_ordered")
        selected_cards_data = selected_cards_df.to_dict(orient="records")
        print("selected_cards_data", selected_cards_data)

        card_numbers = tuple(card['牌号'] for card in selected_cards_data)
        print("card_numbers", card_numbers)
        card_names = tuple(card['卡牌'] for card in selected_cards_data)
        print("card_names", card_names)

        task_texts = []
        for card_name in card_names:
            meaning_row_df = meanings_df.loc[meanings_df['卡牌1'] == card_name]
            if not meaning_row_df.empty:
                meaning_columns = [col for col in meaning_row_df.columns if col.startswith('卡牌1的汇总')]
                valid_meanings = [str(meaning_row_df.iloc[0][col]) for col in meaning_columns if
                                  pd.notna(meaning_row_df.iloc[0][col])]
                task_texts.append(", ".join(valid_meanings) if valid_meanings else "N/A")
            else:
                task_texts.append("N/A")

        return card_names, card_numbers, tuple(task_texts)

    except Exception as e:
        logger.error(f"抽牌并获取数据时发生错误: {e}", exc_info=True)
        return None, None, None

def generate_non_choice_prompt(
        cards_df: pd.DataFrame,
        meanings_df: pd.DataFrame,
        user_question: str,
        card_number_pool: Optional[List[int]] = None
) -> tuple[Optional[str], Optional[tuple[str, ...]], Optional[tuple[int, ...]]]:
    """
    为非选择类问题生成雷诺曼占卜提示词。

    【改进】现在从 XML 配置文件加载提示词模板，支持外部修改和热更新。
    """
    # 复用核心抽牌逻辑，固定抽3张
    card_data = _draw_and_get_card_data(cards_df, meanings_df, 3, card_number_pool)
    if not all(card_data):
        logger.error("无法为非选择类问题获取卡牌数据。")
        return None, None, None

    card_names, card_numbers, task_texts = card_data
    logger.info(f"非选择类问题: 选定卡牌 {card_names} (编号: {card_numbers})")

    # 【改进】从配置文件加载提示词模板
    template = get_prompt_template("non_choice")
    if not template:
        logger.error("无法获取非选择类提示词模板，请检查 prompts/non_choice_prompt.xml 文件")
        return None, None, None

    # 使用字符串替换方法替换占位符（因为 .format() 不支持方括号）
    try:
        llm_prompt = template

        # 替换用户问题
        llm_prompt = llm_prompt.replace("{user_question}", user_question)

        # 替换卡牌名称和含义
        for i in range(len(card_names)):
            llm_prompt = llm_prompt.replace(f"{{card_names[{i}]}}", card_names[i])
        for i in range(len(task_texts)):
            llm_prompt = llm_prompt.replace(f"{{task_texts[{i}]}}", task_texts[i])

    except Exception as e:
        logger.error(f"格式化提示词模板失败: {e}", exc_info=True)
        return None, None, None

    return llm_prompt, card_names, card_numbers

# --- 新增：为选择类问题生成对比提示词的函数 ---
def generate_choice_prompt(
        user_question: str,
        choices_with_cards: Dict[str, Dict[str, Any]]
) -> Optional[str]:
    """
    为选择类问题生成一个复杂的、要求对比分析的提示词。

    【改进】现在从 XML 配置文件加载提示词模板，支持外部修改和热更新。
    """
    try:
        # 【改进】从配置文件加载提示词模板
        template = get_prompt_template("choice")
        if not template:
            logger.error("无法获取选择类提示词模板，请检查 prompts/choice_prompt.xml 文件")
            return None

        # 构建选项的 XML 数据
        options_xml = ""
        for option_name, card_data in choices_with_cards.items():
            card_names = card_data['names']
            task_texts = card_data['texts']
            # 根据选项包含的牌数动态生成牌阵
            cards_xml = ""
            for i in range(len(card_names)):
                cards_xml += f"""
                    <card position="{i + 1}" name="{card_names[i]}">
                        <interpretation>{task_texts[i]}</interpretation>
                    </card>"""

            options_xml += f"""
            <option name="{option_name}">
                <card_spread>{cards_xml}
                </card_spread>
            </option>"""

        # 使用字符串替换方法替换占位符
        llm_prompt = template
        llm_prompt = llm_prompt.replace("{user_question}", user_question)
        llm_prompt = llm_prompt.replace("{options_xml}", options_xml)

        return llm_prompt
    except Exception as e:
        logger.error(f"生成选择类问题提示词时出错: {e}", exc_info=True)
        return None