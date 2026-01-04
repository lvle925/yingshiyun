# app/ziwei_service.py (简略版，确保包含 process_yearly_fortune)
from app.processing import get_yearly_scores_only, get_monthly_scores
from app.utils import convert_time_to_time_index
# ... (其他 import 和常量定义) ...

async def process_yearly_fortune(request):
    # ... (原有 api.py 中的 calculate 逻辑) ...
    pass