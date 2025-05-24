"""
Authentication module for the embedding service.
Handles JWT token-based authentication and authorization.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, List, Union
import os
import json
import logging
from pathlib import Path
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Authentication settings
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "false").lower() == "true"
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "supersecretkey123456789")  # Change in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token URL
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# User roles
ROLE_USER = "user"
ROLE_ADMIN = "admin"
ROLE_API = "api"

# Store users in a JSON file (in production, use a database)
USERS_FILE = Path(__file__).parent / "users.json"


class UserRole(BaseModel):
    name: str
    can_use_models: List[str] = []
    rate_limit: int = 100  # requests per minute
    can_admin: bool = False


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    hashed_password: str
    role: str = ROLE_USER
    api_keys: List[str] = []


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str
    role: str


# Role definitions
ROLES = {
    ROLE_USER: UserRole(
        name=ROLE_USER,
        can_use_models=["minilm", "bge-small"],
        rate_limit=100,
        can_admin=False
    ),
    ROLE_ADMIN: UserRole(
        name=ROLE_ADMIN,
        can_use_models=["*"],  # All models
        rate_limit=1000,
        can_admin=True
    ),
    ROLE_API: UserRole(
        name=ROLE_API,
        can_use_models=["*"],
        rate_limit=5000,
        can_admin=False
    )
}


def load_users() -> Dict[str, User]:
    """Load users from the JSON file."""
    if not USERS_FILE.exists():
        # Create default admin user if no users exist
        users = {
            "admin": User(
                username="admin",
                email="admin@example.com",
                full_name="Administrator",
                disabled=False,
                hashed_password=get_password_hash("admin"),  # Change in production!
                role=ROLE_ADMIN
            )
        }
        save_users(users)
        return users

    try:
        with open(USERS_FILE, "r") as f:
            users_dict = json.load(f)
        
        # Convert dict to User objects
        return {
            username: User(**user_data)
            for username, user_data in users_dict.items()
        }
    except Exception as e:
        logger.error(f"Error loading users: {e}")
        return {}


def save_users(users: Dict[str, User]) -> None:
    """Save users to the JSON file."""
    try:
        # Convert User objects to dict
        users_dict = {
            username: user.dict()
            for username, user in users.items()
        }
        
        with open(USERS_FILE, "w") as f:
            json.dump(users_dict, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving users: {e}")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate a user by username and password."""
    users = load_users()
    user = users.get(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Optional[User]:
    """Get the current user from a JWT token."""
    if not AUTH_ENABLED:
        # When auth is disabled, return a default admin user
        return User(
            username="default",
            full_name="Default User",
            email="default@example.com",
            hashed_password="",
            role=ROLE_ADMIN,
            disabled=False
        )
        
    if token is None:
        return None
        
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role", ROLE_USER)
        if username is None:
            return None
        token_data = TokenData(username=username, role=role)
    except JWTError:
        return None
        
    users = load_users()
    user = users.get(token_data.username)
    if user is None:
        return None
    if user.disabled:
        return None
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current active user."""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_auth(current_user: User = Depends(get_current_active_user)) -> User:
    """Require authentication for a route."""
    return current_user


def require_admin(current_user: User = Depends(get_current_active_user)) -> User:
    """Require admin role for a route."""
    if current_user.role != ROLE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized",
        )
    return current_user


def can_use_model(model_id: str, current_user: User = Depends(get_current_active_user)) -> bool:
    """Check if a user can use a specific model."""
    if not AUTH_ENABLED:
        return True
        
    role = ROLES.get(current_user.role, ROLES[ROLE_USER])
    if "*" in role.can_use_models:
        return True
    return model_id in role.can_use_models


def add_user(username: str, password: str, email: str = None, 
             full_name: str = None, role: str = ROLE_USER) -> User:
    """Add a new user."""
    users = load_users()
    if username in users:
        raise ValueError(f"User {username} already exists")
        
    new_user = User(
        username=username,
        email=email,
        full_name=full_name,
        disabled=False,
        hashed_password=get_password_hash(password),
        role=role
    )
    
    users[username] = new_user
    save_users(users)
    return new_user


def delete_user(username: str) -> bool:
    """Delete a user."""
    users = load_users()
    if username not in users:
        return False
        
    del users[username]
    save_users(users)
    return True


def generate_api_key(username: str) -> str:
    """Generate a new API key for a user."""
    import secrets
    
    users = load_users()
    if username not in users:
        raise ValueError(f"User {username} does not exist")
        
    api_key = f"vectron_{secrets.token_urlsafe(32)}"
    users[username].api_keys.append(api_key)
    save_users(users)
    return api_key


def authenticate_api_key(api_key: str) -> Optional[User]:
    """Authenticate using an API key."""
    if not api_key:
        return None
        
    users = load_users()
    for user in users.values():
        if api_key in user.api_keys:
            return user
            
    return None 