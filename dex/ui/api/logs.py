from fastapi import APIRouter

router = APIRouter(prefix='/api/logs', tags=['logs'])


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
        'fitness': DEX.pipeline.fitness_log[-200:],
        'errors': DEX.pipeline.error_log[-200:],
        'total_steps': DEX.pipeline.total_steps,
    }
