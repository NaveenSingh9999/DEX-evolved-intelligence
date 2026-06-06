from fastapi import APIRouter

router = APIRouter(prefix='/api/skills', tags=['skills'])


@router.get('/tree')
async def get_skills_tree():
    from dex import DEX
    if DEX is None:
        return {'skills': []}
    skills = DEX.skill_discoverer.skills
    return {'skills': [s.to_dict() for s in skills.values()]}


@router.get('/emergence')
async def get_emergence_count():
    from dex import DEX
    if DEX is None:
        return {'count': 0}
    return {'count': DEX.skill_discoverer.emergence_counter}
