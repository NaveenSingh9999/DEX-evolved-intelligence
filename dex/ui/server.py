import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .api import brain, skills, chat, logs

app = FastAPI(title='DEX Dashboard', version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(brain.router)
app.include_router(skills.router)
app.include_router(chat.router)
app.include_router(logs.router)

app.mount('/', StaticFiles(directory='dex/ui/static', html=True), name='static')


def run(host: str = '0.0.0.0', port: int = 3000):
    uvicorn.run(app, host=host, port=port, log_level='info')
