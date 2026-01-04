# 奇门遁甲LLM服务

基于FastAPI的奇门遁甲分析服务，支持用户问题意图识别、数据库查询和LLM生成分析结果。

## 项目结构

```
12_12_llm_yingshi_qimen_bendi/
├── app/
│   ├── __init__.py
│   └── monitor.py          # 监控日志模块
├── prompts/
│   └── final_response_prompt.xml  # 最终响应提示词模板
├── logs/                    # 日志目录
├── main.py                  # 主入口文件
├── config.py                # 配置文件
├── user_info_extractor.py   # 用户信息提取模块
├── query_intent.py          # 意图识别模块（第一层和第二层）
├── db_query.py              # 数据库查询模块
├── llm_response.py          # LLM响应生成模块
├── validation_rules.py      # 验证规则模块
├── session_manager.py       # 会话管理模块
└── requirements.txt         # 依赖包列表
```

## 功能说明

### 1. 用户信息提取
- 从用户输入中提取出生年月日、性别等信息
- 将身份信息和问题意图剥离

### 2. 第一层意图识别
判断是否为奇门问题，三类问题归类为奇门：
- 具体时间点(精确到时辰)做具体事件是否合适
- 什么时间做具体事件
- 具体时间点做什么事件

### 3. 第二层意图识别
判断奇门问题的具体类型并提取：
- 具体时间点或时间范围
- 具体事件标签（匹配数据库标签）

### 4. 数据库查询
根据不同类型进行数据库查询：
- 类型1：查询具体时间点和事件的匹配数据
- 类型2：查询时间范围内适合做某事件的时间
- 类型3：查询具体时间点适合做什么事件

#### 数据查询到 LLM 的当前流程（重点）
- 输入参数：`qimen_type`、`specific_event`、`time_range_start`、`time_range_end`、`jixiong_preference`（可选："吉"|"凶"|"吉凶"）。
- 数据拉取：  
  - type1 先取最多 200 条；type2/3 先取最多 400 条，限定 7-23 点。  
  - 时间范围由用户提供；type2 无范围时默认“现在~一年后”。
- 是否执行吉凶筛选：仅当 `jixiong_preference` 明确为“吉”或“凶”且查询为时间范围（type1/3 需 start!=end，type2 天然是范围）时执行；否则只做时间+分数排序去重。
  - 吉凶评估（仅在需要筛选时启用）：  
    - 直接查看数据行的 `吉凶` 字段，仅识别“吉”/“凶”，其余视为未知；不再计算“有宜进行/不宜进行”。
- 筛选与排序：  
  - 有吉凶筛选时：保留符合偏好的记录，排序优先级 吉 > 平/未知 > 凶，同级按 `total_score` 降序、`date_str` 升序。  
  - 无吉凶筛选时：按 `date_str` 升序、`total_score` 降序排序。  
  - 去重：同一 `date_str` 只保留排序后的第一条，最终截断最多 10 条。
- 输出给 LLM：筛选后的记录列表（≤10 条）连同用户问题进入 `generate_final_response`，生成最终回答。

### 5. 最终响应生成
使用LLM生成专业的奇门分析结果

## 配置说明

在`.env`文件中配置以下环境变量：

```env
# Redis配置
REDIS_URL=redis://:password@host:port/db

# VLLM配置
VLLM_API_BASE_URL=http://192.168.1.101:6002/v1
VLLM_MODEL_NAME=Qwen3-30B-A3B-Instruct-2507
API_KEY=your_api_key

# 数据库配置
DB_USER=proxysql
DB_PASSWORD=your_password
DB_HOST=192.168.1.101
DB_PORT=6033
DB_DATABASE=yingshi
```

## API接口

### POST /chat_qimen

请求参数：
- `appid`: 应用ID
- `prompt`: 用户问题（包含出生信息）
- `ftime`: 时间戳
- `sign`: 请求签名
- `session_id`: 会话ID（可选）
- `skip_intent_check`: 是否跳过意图识别（0=不跳过，1=跳过）

响应：流式返回分析结果

## 注意事项

1. 用户信息格式固定，不需要验证格式
2. 提示词支持热更新，修改`prompts/final_response_prompt.xml`后自动生效
3. 如果是从上层来的请求（skip_intent_check=1），会跳过第一层意图识别
4. 所有日志记录在`logs/api_monitor.log`中

