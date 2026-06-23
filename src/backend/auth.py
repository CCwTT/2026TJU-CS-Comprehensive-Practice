"""
认证模块 — 文件持久化 + 内存 Session.

用户数据存储在 src/backend/users.json (JSON 文件).
"""
import hashlib
import secrets
import json
from pathlib import Path

# 用户数据文件 (与 auth.py 同目录)
_USERS_FILE = Path(__file__).resolve().parent / 'users.json'

# 内存 Session {token: username}
_sessions = {}


def _load_users() -> dict:
    """从文件加载用户数据."""
    if _USERS_FILE.exists():
        with open(_USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_users(users: dict):
    """保存用户数据到文件."""
    with open(_USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """SHA-256 加盐哈希."""
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.sha256((password + salt).encode()).hexdigest()
    return h, salt


# ============================================================
# 对外接口
# ============================================================

def register(username: str, password: str) -> bool:
    """注册, 用户名已存在返回 False."""
    users = _load_users()
    if username in users:
        return False
    h, salt = _hash_password(password)
    users[username] = {'password_hash': h, 'salt': salt}
    _save_users(users)
    return True


def login(username: str, password: str) -> str | None:
    """登录成功返回 token, 失败返回 None."""
    users = _load_users()
    user = users.get(username)
    if user is None:
        return None
    h, _ = _hash_password(password, user['salt'])
    if h != user['password_hash']:
        return None
    token = secrets.token_hex(32)
    _sessions[token] = username
    return token


def reset_password(username: str, new_password: str) -> bool:
    """重置密码. 用户名不存在返回 False."""
    users = _load_users()
    if username not in users:
        return False
    h, salt = _hash_password(new_password)
    users[username] = {'password_hash': h, 'salt': salt}
    _save_users(users)
    return True


def get_username(token: str) -> str | None:
    """根据 token 获取用户名."""
    return _sessions.get(token)


def is_authenticated(token: str) -> bool:
    """检查 token 是否有效."""
    return token in _sessions


def list_users() -> list[str]:
    """列出所有已注册的用户名."""
    return list(_load_users().keys())