import json
import hashlib
import functools
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class AuditEntry(Base):
    """SQLAlchemy model for ALCOA+ audit trail."""
    __tablename__ = 'audit_ledger'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_id = Column(String(100), nullable=False)
    action = Column(String(255), nullable=False)
    parameters = Column(Text, nullable=False)
    previous_hash = Column(String(64), nullable=False)
    current_hash = Column(String(64), nullable=False, unique=True)
    metadata_json = Column(Text)  # For ALCOA+ extra context

class ALCOAAuditLedger:
    """
    Enforces Attributable, Legible, Contemporaneous, Original, and Accurate (ALCOA+) 
    data integrity as per 21 CFR Part 11 and EMA Annex 11.
    """
    
    def __init__(self, db_url: str = "sqlite:///compliance_audit.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def _get_last_hash(self) -> str:
        """Retrieve the hash of the latest entry to maintain the chain."""
        session = self.Session()
        last_entry = session.query(AuditEntry).order_by(AuditEntry.id.desc()).first()
        session.close()
        return last_entry.current_hash if last_entry else "0" * 64

    def log_immutable_event(self, user_id: str, function_name: str, params: Dict[str, Any], metadata: Optional[Dict] = None):
        """
        Logs an event with cryptographic chaining to prevent silent tampering.
        Ensures data is Attributable and Contemporaneous.
        """
        session = self.Session()
        try:
            previous_hash = self._get_last_hash()
            param_str = json.dumps(params, sort_keys=True)
            timestamp = datetime.utcnow()
            
            # Create payload for hashing (Chaining)
            payload = f"{previous_hash}|{timestamp.isoformat()}|{user_id}|{function_name}|{param_str}"
            current_hash = hashlib.sha256(payload.encode()).hexdigest()
            
            entry = AuditEntry(
                timestamp=timestamp,
                user_id=user_id,
                action=function_name,
                parameters=param_str,
                previous_hash=previous_hash,
                current_hash=current_hash,
                metadata_json=json.dumps(metadata) if metadata else None
            )
            
            session.add(entry)
            session.commit()
            return current_hash
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Audit logging failed: {e}")
        finally:
            session.close()

def requires_e_signature(func: Callable):
    """
    Decorator that forces re-authentication before executing critical functions.
    Compliance: 21 CFR Part 11.10(j) and 11.50.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # In a production environment, this would interface with an Identity Provider (IdP)
        # or a session manager to verify recent re-auth for 'Electronic Signature' intent.
        print(f"[E-SIGNATURE] Verification required for: {func.__name__}")
        
        # Simulated re-authentication check
        authenticated = kwargs.get('__authenticated_esign', False)
        if not authenticated:
            raise PermissionError(
                f"Electronic Signature verification failed for '{func.__name__}'. "
                "Active session re-authentication is required."
            )
        
        # If authenticated, execute and log
        user_id = kwargs.get('user_id', 'unknown_user')
        ledger = ALCOAAuditLedger()
        
        # Filter sensitive kwargs before logging
        log_params = {k: v for k, v in kwargs.items() if k != '__authenticated_esign'}
        
        result = func(*args, **kwargs)
        
        ledger.log_immutable_event(
            user_id=user_id,
            function_name=func.__name__,
            params=log_params,
            metadata={"esignature_verified": True}
        )
        
        return result
    return wrapper
