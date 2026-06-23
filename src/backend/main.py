"""
FastAPI 主入口.

启动: uvicorn src.backend.main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from .routes.auth_routes import router as auth_router
from .routes.detect import router as detect_router

# 前端模板目录
FRONTEND_DIR = Path(__file__).resolve().parent.parent / 'frontend'

app = FastAPI(title="危险标志识别系统", version="1.0")

# 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由
app.include_router(auth_router)
app.include_router(detect_router)

# 前端页面
from fastapi import Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

_templates = Environment(loader=FileSystemLoader(str(FRONTEND_DIR / 'templates')))

@app.get("/")
async def root():
    """重定向到登录页."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/login")


@app.get("/login")
async def login_page(request: Request):
    template = _templates.get_template("login.html")
    return HTMLResponse(template.render(request=request))


@app.get("/detect")
async def detect_page(request: Request):
    template = _templates.get_template("detect.html")
    return HTMLResponse(template.render(request=request))
