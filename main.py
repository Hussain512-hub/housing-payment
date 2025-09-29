from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, create_engine
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta, timezone
import os
import io
import csv

# ---------------------
# CONFIGURATION
# ---------------------
# database URL and engine (replace existing DATABASE_URL / engine lines with this)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./housing_payments.db")

# only set check_same_thread for sqlite (not for Postgres)
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "secretadmintoken")

app = FastAPI(title="Housing Project Payment Middleware")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------
# DATABASE MODELS
# ---------------------
class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: Optional[str] = None

class Plot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    plot_number: str
    owner_name: Optional[str] = None
    owner_contact: Optional[str] = None
    base_price: float = 0.0

class PaymentType(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: Optional[str] = None
    default_amount: Optional[float] = None

class Challan(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    challan_id: str = Field(index=True, unique=True)
    project_id: int = Field(foreign_key="project.id")
    plot_id: int = Field(foreign_key="plot.id")
    payment_type_id: int = Field(foreign_key="paymenttype.id")
    amount: float
    due_date: datetime
    penalty_amount: Optional[float] = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = Field(default="PENDING")  # PENDING / PAID / FAILED

class PaymentTransaction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    challan_id: str
    amount: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str  # SUCCESS / FAILED
    bank_reference: Optional[str] = None
    error_code: Optional[str] = None
    raw_payload: Optional[str] = None

# --- SECURITY SETUP ---
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if token != "mysecrettoken":  # yahan apna token daalna
        raise HTTPException(status_code=401, detail="Unauthorized (invalid token)")
    return True

# ---------------------
# PYDANTIC SCHEMAS
# ---------------------
class CreateProject(BaseModel):
    name: str
    description: Optional[str] = None

class CreatePlot(BaseModel):
    project_id: int
    plot_number: str
    owner_name: Optional[str] = None
    owner_contact: Optional[str] = None
    base_price: Optional[float] = 0.0

class CreatePaymentType(BaseModel):
    name: str
    description: Optional[str] = None
    default_amount: Optional[float] = None

class GenerateChallanRequest(BaseModel):
    project_id: int
    plot_id: int
    payment_type_id: int
    amount: Optional[float] = None
    due_in_days: Optional[int] = 7

class ConfirmPaymentRequest(BaseModel):
    challan_id: str
    amount: float
    bank_reference: Optional[str] = None

class BankCallbackPayload(BaseModel):
    challan_id: str
    amount: float
    bank_reference: str
    status: str  # SUCCESS / FAILED
    raw: Optional[dict] = None

# ---------------------
# HELPERS
# ---------------------
def get_session():
    with Session(engine) as session:
        yield session

def admin_auth(x_admin_token: Optional[str] = Header(None)):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized (invalid token)")
    return True

def ensure_tables():
    SQLModel.metadata.create_all(engine)

@app.on_event("startup")
def on_startup():
    ensure_tables()

# ---------------------
# ENDPOINTS
# ---------------------

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

# --- Projects ---
@app.post("/projects", dependencies=[Depends(admin_auth)])
def create_project(data: CreateProject, session: Session = Depends(get_session)):
    project = Project(name=data.name, description=data.description)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project

@app.get("/projects", response_model=List[Project])
def list_projects(session: Session = Depends(get_session)):
    return session.exec(select(Project)).all()

# --- Plots ---
@app.post("/plots", dependencies=[Depends(admin_auth)])
def create_plot(data: CreatePlot, session: Session = Depends(get_session)):
    plot = Plot(**data.dict())
    session.add(plot)
    session.commit()
    session.refresh(plot)
    return plot

@app.get("/plots", response_model=List[Plot])
def list_plots(session: Session = Depends(get_session)):
    return session.exec(select(Plot)).all()

# --- Payment Types ---
@app.post("/payment_types", dependencies=[Depends(admin_auth)])
def create_payment_type(data: CreatePaymentType, session: Session = Depends(get_session)):
    pt = PaymentType(**data.dict())
    session.add(pt)
    session.commit()
    session.refresh(pt)
    return pt

@app.get("/payment_types", response_model=List[PaymentType])
def list_payment_types(session: Session = Depends(get_session)):
    return session.exec(select(PaymentType)).all()

# --- Challans ---
@app.post("/challans", dependencies=[Depends(admin_auth)])
def generate_challan(data: GenerateChallanRequest, session: Session = Depends(get_session)):
    challan = Challan(
        challan_id=f"CH-{int(datetime.now().timestamp())}",
        project_id=data.project_id,
        plot_id=data.plot_id,
        payment_type_id=data.payment_type_id,
        amount=data.amount or 0.0,
        due_date=datetime.now(timezone.utc) + timedelta(days=data.due_in_days or 7),
    )
    session.add(challan)
    session.commit()
    session.refresh(challan)
    return challan

@app.get("/challans/{challan_id}")
def get_challan(challan_id: str, session: Session = Depends(get_session)):
    challan = session.exec(select(Challan).where(Challan.challan_id == challan_id)).first()
    if not challan:
        raise HTTPException(status_code=404, detail="Challan not found")
    return challan

# --- Payments ---
@app.post("/payments/confirm", dependencies=[Depends(admin_auth)])
def confirm_payment(data: ConfirmPaymentRequest, session: Session = Depends(get_session)):
    challan = session.exec(select(Challan).where(Challan.challan_id == data.challan_id)).first()
    if not challan:
        raise HTTPException(status_code=404, detail="Challan not found")

    challan.status = "PAID"
    txn = PaymentTransaction(
        challan_id=data.challan_id,
        amount=data.amount,
        status="SUCCESS",
        bank_reference=data.bank_reference,
    )
    session.add(txn)
    session.add(challan)
    session.commit()
    return {"message": "Payment confirmed", "transaction": txn}

@app.post("/bank/callback")
def bank_callback(payload: BankCallbackPayload, session: Session = Depends(get_session)):
    challan = session.exec(select(Challan).where(Challan.challan_id == payload.challan_id)).first()
    if not challan:
        raise HTTPException(status_code=404, detail="Challan not found")

    challan.status = payload.status
    txn = PaymentTransaction(
        challan_id=payload.challan_id,
        amount=payload.amount,
        status=payload.status,
        bank_reference=payload.bank_reference,
        raw_payload=str(payload.raw),
    )
    session.add(txn)
    session.add(challan)
    session.commit()
    return {"message": "Callback processed"}

# --- Reports ---
@app.get("/reports/payments")
def report_payments(download: bool = Query(False), session: Session = Depends(get_session)):
    txns = session.exec(select(PaymentTransaction)).all()

    if download:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "challan_id", "amount", "status", "bank_reference", "timestamp"])
        for t in txns:
            writer.writerow([t.id, t.challan_id, t.amount, t.status, t.bank_reference, t.timestamp])
        output.seek(0)
        return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=payments.csv"})
    return txns


@app.get("/reports/payments")
def get_payments(user=Depends(get_current_user)):
    return {"report": "All payment reports (protected)"}

@app.get("/reports/tenants")
def get_tenants(user=Depends(get_current_user)):
    return {"report": "All tenant reports (protected)"}
