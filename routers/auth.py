# routers/auth.py - Authentication endpoints with bcrypt password hashing
import secrets
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from database import get_db, safe_commit
from db_models import User
from auth import (
    hash_password,
    verify_password,
    verify_legacy_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    get_current_user
)
from schemas import AuthResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


class SignUpRequest(BaseModel):
    email: str
    password: str
    name: str


class SignInRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/signup", response_model=AuthResponse)
async def sign_up(request: SignUpRequest, db: Session = Depends(get_db)):
    """Create a new account with email and password"""

    # Check if email already exists
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Validate password strength
    if len(request.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters"
        )
    if not any(c.isupper() for c in request.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one uppercase letter"
        )
    if not any(c.islower() for c in request.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one lowercase letter"
        )
    if not any(c.isdigit() for c in request.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must contain at least one number"
        )

    # Create new user with bcrypt hashed password
    user = User(
        google_id=f"local_{secrets.token_hex(8)}",  # Generate unique ID
        email=request.email,
        name=request.name,
        picture_url=None,
        password_hash=hash_password(request.password),  # Proper password storage
        last_login=datetime.utcnow()
    )

    db.add(user)
    safe_commit(db)
    db.refresh(user)

    # Generate tokens
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            picture_url=None,
            created_at=user.created_at
        )
    )


@router.post("/signin", response_model=AuthResponse)
async def sign_in(request: SignInRequest, db: Session = Depends(get_db)):
    """Sign in with email and password"""

    # Find user by email
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Check password - try new bcrypt hash first, then legacy migration
    if user.password_hash:
        # New bcrypt password
        if not verify_password(request.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
    elif user.picture_url and user.picture_url.startswith("pwd:"):
        # Legacy password stored in picture_url - migrate on successful login
        old_hash = user.picture_url[4:]  # Remove "pwd:" prefix
        if not verify_legacy_password(request.password, old_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        # Migrate to bcrypt
        user.password_hash = hash_password(request.password)
        user.picture_url = None  # Clear legacy storage
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Update last login
    user.last_login = datetime.utcnow()
    safe_commit(db)

    # Generate tokens
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    return AuthResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            picture_url=user.picture_url if user.picture_url and not user.picture_url.startswith("pwd:") else None,
            created_at=user.created_at
        )
    )


@router.post("/refresh")
async def refresh_token(request: RefreshRequest, db: Session = Depends(get_db)):
    """Refresh access token using refresh token"""

    user_id = verify_token(request.refresh_token, token_type="refresh")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )

    # Generate new tokens
    access_token = create_access_token(user.id)
    new_refresh_token = create_refresh_token(user.id)

    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        picture_url=current_user.picture_url if current_user.picture_url and not current_user.picture_url.startswith("pwd:") else None,
        created_at=current_user.created_at
    )


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Logout current user (client should discard tokens)"""
    return {"message": "Successfully logged out"}
