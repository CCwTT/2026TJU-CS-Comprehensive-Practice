"""认证路由: /api/register, /api/login"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..auth import register, login, reset_password

router = APIRouter(prefix="/api", tags=["auth"])


class AuthRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    ok: bool
    message: str = ""
    token: str = ""


@router.post("/register", response_model=AuthResponse)
def api_register(req: AuthRequest):
    if len(req.username) < 3:
        raise HTTPException(400, "用户名至少 3 个字符")
    if len(req.password) < 4:
        raise HTTPException(400, "密码至少 4 个字符")
    if register(req.username, req.password):
        return AuthResponse(ok=True, message="注册成功")
    else:
        raise HTTPException(400, "用户名已存在")


@router.post("/login", response_model=AuthResponse)
def api_login(req: AuthRequest):
    token = login(req.username, req.password)
    if token:
        return AuthResponse(ok=True, message="登录成功", token=token)
    else:
        raise HTTPException(401, "用户名或密码错误")


@router.post("/reset-password", response_model=AuthResponse)
def api_reset_password(req: AuthRequest):
    """重置密码 (演示版, 跳过验证码)."""
    if len(req.password) < 4:
        raise HTTPException(400, "新密码至少 4 个字符")
    if reset_password(req.username, req.password):
        return AuthResponse(ok=True, message="密码重置成功, 请登录")
    else:
        raise HTTPException(404, "用户不存在")
