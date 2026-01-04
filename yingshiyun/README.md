# yingshiyun

统一将 9 个现有项目整合为 Router-Service-Schema 分层的单体 FastAPI 应用。

## 目标架构
```
yingshiyun/
  app/
    main.py                # FastAPI 入口，统一挂载各业务路由
    core/                  # 基础设施：配置、DB、日志
      config.py
      db.py
    routers/               # 路由层：只做请求绑定/鉴权/参数校验
      lenormand_daily.py   # 10_10_leinuo_yunshi_day
      lenormand_llm.py     # 12_12_llm_yingshi_leinuo_ali
      ziwei_year_score.py  # 11_5_yingshi_yunshi_year_score_ali
      qimen_day.py         # 12_11_yingshi_yunshi_day_new_qimen
      qimen_llm.py         # 12_12_llm_yingshi_qimen_ali
      ziwei_llm.py         # 12_12_llm_yingshi_ziwei_ali
      summary_llm.py       # 12_12_llm_yingshi_summary_ali
      report_prediction.py # report_prediction_bendi_test_12_8
      ziwei_report_year.py # ziwei_report_year_aliyun_11_12
    schemas/               # Pydantic 入参/出参模型
      lenormand.py
      ziwei.py
      qimen.py
      summary.py
    services/              # 业务层，封装逻辑 & 依赖
      lenormand_daily.py
      lenormand_llm.py
      ziwei_year_score.py
      qimen_day.py
      qimen_llm.py
      ziwei_llm.py
      summary_llm.py
      report_prediction.py
      ziwei_report_year.py
    clients/               # 外部接口/LLM/DB 等
    shared/                # 通用工具（翻译 key、签名、监控等）
  requirements.txt         # 统一依赖
  .env.example             # 环境变量示例
```

## 迁移映射
- 10_10_leinuo_yunshi_day → routers/services/schemas/lenormand_daily
- 11_5_yingshi_yunshi_year_score_ali → routers/services/schemas/ziwei_year_score
- 12_11_yingshi_yunshi_day_new_qimen → routers/services/schemas/qimen_day
- 12_12_llm_yingshi_leinuo_ali → routers/services/schemas/lenormand_llm
- 12_12_llm_yingshi_qimen_ali → routers/services/schemas/qimen_llm
- 12_12_llm_yingshi_summary_ali → routers/services/schemas/summary_llm
- 12_12_llm_yingshi_ziwei_ali → routers/services/schemas/ziwei_llm
- report_prediction_bendi_test_12_8 → routers/services/schemas/report_prediction
- ziwei_report_year_aliyun_11_12 → routers/services/schemas/ziwei_report_year

## 合并思路
1) **路由层瘦身**：仅做路径、Tag、依赖注入、参数校验，直接调用对应 Service。
2) **业务层收口**：将原各项目核心函数搬到 services/*，保持协程接口，内部再调用 clients/shared 工具。
3) **Schema 统一**：按各项目现有 Pydantic 模型迁移到 schemas/*，输出响应模型可逐步补齐。
4) **共享依赖**：
   - DB 连接池、HTTP 客户端、LLM 客户端放到 core/clients。
   - 通用工具（键翻译、签名、监控、日志）放 shared/。
5) **配置**：core/config.py 读取 .env，避免散落硬编码。
6) **启动/关闭**：main.py 统一初始化资源（DB/HTTP/LLM），在 lifespan 中分发给 service。

## 下一步迁移建议
- 按映射逐个把原项目的模型/服务搬入对应 schemas/services，保持函数签名不变，路由调用即可替换。
- 将原 assets/CSV 放到 yingshiyun/assets 下或通过绝对路径配置。
- 合并 requirements：取各项目 superset，优先 fastapi/pydantic/aiomysql/aiohttp/langchain/openai/pandas。
- 增加统一的日志与监控中间件（可复用 12_12_llm_yingshi_ziwei_ali 的 StepMonitor）。

> 当前提交只是骨架，后续可逐步搬运各项目实现。
