import math
from fastapi import APIRouter

router = APIRouter(prefix='/api/logs', tags=['logs'])


def _sanitize(val, default=0.0):
    return default if math.isnan(val) or math.isinf(val) else val


@router.get('/today')
async def get_today():
    from dex import DEX
    if DEX is None:
        return {'summary': 'DEX not initialized'}
    return DEX.logger.get_today_summary()


@router.get('/history')
async def get_history(days: int = 7):
    from dex import DEX
    if DEX is None:
        return {'history': []}
    return {'history': DEX.logger.get_history(days)}


@router.get('/metrics')
async def get_metrics():
    from dex import DEX
    if DEX is None:
        return {'fitness': [], 'errors': []}
    return {
        'fitness': [_sanitize(f) for f in DEX.pipeline.fitness_log[-200:]],
        'errors': [_sanitize(e) for e in DEX.pipeline.error_log[-200:]],
        'total_steps': DEX.pipeline.total_steps,
    }
