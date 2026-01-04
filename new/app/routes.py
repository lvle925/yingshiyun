from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas import YearlyAnalysisRequest, ClientRequest
from app import ziwei_service, lenormand_service

router = APIRouter()

# --- 1. 紫微斗数接口 ---
@router.post("/yearFortuneScore", summary="生成年度运势评分")
async def generate_yearly_scores(request: YearlyAnalysisRequest):
    """
    紫微斗数年运评分
    """
    return await ziwei_service.process_yearly_fortune(request)

# --- 2. 雷诺曼 LLM 接口 ---
@router.post("/chat_daily_leipai", summary="雷诺曼每日占卜")
async def chat_daily_leipai(client_request: ClientRequest, request: Request):
    """
    雷诺曼 LLM 流式对话
    """
    # 从 app.state 获取数据库连接池 (在 main.py 中初始化)
    db_pool = getattr(request.app.state, 'db_pool', None)
    
    try:
        # 调用 Service 层获取生成器
        generator_func = await lenormand_service.handle_chat_request(client_request, db_pool)
        
        return StreamingResponse(
            generator_func(), 
            media_type="text/plain; charset=utf-8"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))