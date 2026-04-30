from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from typing import List, Optional
import os

# Security Config
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "SUPER_SECRET_PHARMA_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username: str
    role: str # Junior_Chemist, Senior_Scientist, Lab_Director

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class EnterpriseSSO:
    """
    Corporate Identity Management & Role-Based Access Control (RBAC).
    Bridges corporate SAML/OAuth2 with ZANE's internal execution engine.
    """

    @staticmethod
    def verify_role(required_roles: List[str]):
        """RBAC Dependency: Ensures the user has the required seniority."""
        async def role_checker(token: str = Depends(oauth2_scheme)):
            credentials_exception = HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                username: str = payload.get("sub")
                role: str = payload.get("role")
                if username is None or role is None:
                    raise credentials_exception
                token_data = TokenData(username=username, role=role)
            except JWTError:
                raise credentials_exception

            if token_data.role not in required_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required roles: {required_roles}. Current role: {token_data.role}"
                )
            return token_data
        
        return role_checker

    @staticmethod
    def get_current_user(token: str = Depends(oauth2_scheme)):
        """Basic authentication verification."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return User(username=payload.get("sub"), role=payload.get("role"))
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

# Usage Examples for Endpoints:
# @app.post("/export/dataset", dependencies=[Depends(EnterpriseSSO.verify_role(["Lab_Director"]))])
# async def export_dataset():
#    ...

# @app.post("/predict", dependencies=[Depends(EnterpriseSSO.verify_role(["Junior_Chemist", "Senior_Scientist", "Lab_Director"]))])
# async def run_prediction():
#    ...
