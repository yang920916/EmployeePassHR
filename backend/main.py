import os
import hmac
import base64
from datetime import datetime, timedelta, timezone
from math import radians, sin, cos, asin, sqrt
from typing import Optional, List, Dict, Any
from collections import defaultdict
from io import BytesIO
from fastapi import Query
from fastapi.responses import Response, HTMLResponse
from fastapi import FastAPI, Depends, HTTPException, Header, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt, JWTError
from passlib.hash import pbkdf2_sha256
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# ================== 設定與資料庫 ==================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./employee_pass.db")
SECRET_KEY = os.getenv("SECRET_KEY", "change_this_to_a_long_random_string")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "120"))
QR_SHARED_SECRET = os.getenv("QR_SHARED_SECRET", "qr_demo_secret")

# 公司位置（預設台北 101 附近）
OFFICE_LAT = float(os.getenv("OFFICE_LAT", "24.17893"))
OFFICE_LNG = float(os.getenv("OFFICE_LNG", "120.64996"))
OFFICE_RADIUS_M = float(os.getenv("OFFICE_RADIUS_M", "200"))  # 打卡半徑 200 公尺內

connect_args: Dict[str, Any] = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args=connect_args,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

app = FastAPI(title="員工通 Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 開發階段全部允許
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== 初始化資料庫 ==================


def init_db():
    """建立資料表 & demo 測試資料（只有在空資料庫時塞一次）"""
    with engine.begin() as conn:
        # 使用者
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            dept TEXT,
            level TEXT,
            hourly_rate INTEGER DEFAULT 200,  -- 時薪（demo 預設 200）
            role TEXT NOT NULL  -- 'admin' or 'employee'
        );
        """))

        # ====== Schema migration: users.hourly_rate（時薪） ======
        # 注意：CREATE TABLE IF NOT EXISTS 不會替既有表補欄位，所以需要 migration
        try:
            if DATABASE_URL.startswith("sqlite"):
                cols = conn.execute(text("PRAGMA table_info(users)")).mappings().all()
                names = {c["name"] for c in cols}
                if "hourly_rate" not in names:
                    conn.execute(text("ALTER TABLE users ADD COLUMN hourly_rate INTEGER DEFAULT 200;"))
            else:
                cols = conn.execute(text("SHOW COLUMNS FROM users")).mappings().all()
                names = {c.get("Field") for c in cols}
                if "hourly_rate" not in names:
                    conn.execute(text("ALTER TABLE users ADD COLUMN hourly_rate INT DEFAULT 200;"))
        except Exception:
            # 若資料庫權限限制或其他原因導致失敗，先忽略（你也可在 DB 手動 ALTER TABLE）
            pass

        # 打卡紀錄
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS timelogs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ts TIMESTAMP NOT NULL,
            type TEXT NOT NULL,  -- 'In' or 'Out'
            lat REAL,
            lng REAL,
            qr_payload TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """))

        # 行事曆
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            start_at TIMESTAMP NOT NULL,
            end_at TIMESTAMP
        );
        """))

        # ---------- 預設帳號（只有第一次 DB 空時才建立） ----------
        count_users = conn.execute(text("SELECT COUNT(*) FROM users")).scalar_one()
        if count_users == 0:
            admin_pw = pbkdf2_sha256.hash("admin123")
            user1_pw = pbkdf2_sha256.hash("user1")
            user2_pw = pbkdf2_sha256.hash("user2")

            conn.execute(
                text("""
                INSERT INTO users (name, email, password_hash, dept, level, role)
                VALUES
                ('系統管理員', 'admin@example.com', :admin_pw, '人事部', '管理員', 'admin'),
                ('測試用帳號1', 'user1@test.com', :user1_pw, '研發部', '菜鳥', 'employee'),
                ('測試用帳號2', 'user2@test.com', :user2_pw, '人事部', '實習生', 'employee');
                """),
                {
                    "admin_pw": admin_pw,
                    "user1_pw": user1_pw,
                    "user2_pw": user2_pw,
                },
            )

        # ---------- 行事曆 demo ----------
        count_events = conn.execute(text("SELECT COUNT(*) FROM events")).scalar_one()
        if count_events == 0:
            now = datetime.now(timezone.utc)
            conn.execute(
                text("""
                INSERT INTO events (title, description, start_at, end_at)
                VALUES
                ('人事部週會', '討論本週 HR 專案與事項', :d1_start, :d1_end),
                ('員工大會', '公司營運狀況與策略說明', :d2_start, :d2_end),
                ('年度考核', '各部門年度績效考核', :d3_start, :d3_end)
                """),
                {
                    "d1_start": now + timedelta(days=1),
                    "d1_end":   now + timedelta(days=1, hours=1),
                    "d2_start": now + timedelta(days=7),
                    "d2_end":   now + timedelta(days=7, hours=2),
                    "d3_start": now + timedelta(days=30),
                    "d3_end":   now + timedelta(days=30, hours=8),
                }
            )

        # ---------- 打卡 demo（只在空的時候塞） ----------
        count_logs = conn.execute(text("SELECT COUNT(*) FROM timelogs")).scalar_one()
        if count_logs == 0:
            rows = conn.execute(
                text("SELECT id, email FROM users")
            ).mappings().all()
            id_map = {r["email"]: r["id"] for r in rows}

            now = datetime.now(timezone.utc)
            base_day = datetime(now.year, now.month, min(now.day, 15), tzinfo=timezone.utc)

            demo_logs: List[Dict[str, Any]] = []

            def add_workday(email: str, day_offset: int):
                uid = id_map[email]
                day = base_day - timedelta(days=day_offset)
                in_ts = day.replace(hour=9, minute=0, second=0, microsecond=0)
                out_ts = day.replace(hour=18, minute=0, second=0, microsecond=0)

                demo_logs.append({
                    "uid": uid,
                    "ts": in_ts,
                    "type": "In",
                    "lat": OFFICE_LAT,
                    "lng": OFFICE_LNG,
                    "qr": "DEMO",
                })
                demo_logs.append({
                    "uid": uid,
                    "ts": out_ts,
                    "type": "Out",
                    "lat": OFFICE_LAT,
                    "lng": OFFICE_LNG,
                    "qr": "DEMO",
                })

            # user1：最近三天
            add_workday("user1@test.com", 1)
            add_workday("user1@test.com", 2)
            add_workday("user1@test.com", 3)

            # user2：最近兩天
            add_workday("user2@test.com", 1)
            add_workday("user2@test.com", 3)

            conn.execute(
                text("""
                INSERT INTO timelogs (user_id, ts, type, lat, lng, qr_payload)
                VALUES (:uid, :ts, :type, :lat, :lng, :qr)
                """),
                demo_logs,
            )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    # 1. 開啟 SQLite WAL 模式 (大幅提升讀寫並發速度)
    if DATABASE_URL.startswith("sqlite"):
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL;"))
            conn.execute(text("PRAGMA synchronous=NORMAL;")) # 選用，可犧牲一點點安全性換取更高速度
            
    init_db()

# ================== Pydantic models ==================


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    dept: Optional[str] = None   # 部門
    level: Optional[str] = None  # 資歷
    role: str = "employee"       # 'admin' or 'employee'


class UserOut(BaseModel):
    id: int
    name: str
    email: str
    dept: Optional[str] = None
    level: Optional[str] = None
    role: str


class LoginResponse(BaseModel):
    token: str
    user: UserOut


class PunchRequest(BaseModel):
    qr_payload: str
    lat: float
    lng: float


class PunchResponse(BaseModel):
    status: str
    type: str
    ts: datetime
    distance_m: float
    message: str


class CalendarEventOut(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    start_at: datetime
    end_at: Optional[datetime] = None


class CreateEventRequest(BaseModel):
    title: str
    description: Optional[str] = None
    start_at: datetime
    end_at: Optional[datetime] = None


class TimelogOut(BaseModel):
    id: int
    ts: datetime
    type: str
    lat: Optional[float] = None
    lng: Optional[float] = None


class PayrollPreview(BaseModel):
    month: str
    minutes: int
    hours: float
    gross_demo: int
    
class AdminMonthlyPayrollRowOut(BaseModel):
    user_id: int
    name: str
    email: str
    dept: Optional[str] = None

    month: str                 # "YYYY-MM"
    hourly_rate: int

    worked_minutes: int
    worked_hours: float
    overtime_minutes: int
    overtime_hours: float

    base_pay: int
    overtime_pay: int
    gross_pay: int


# ================== Auth & Utils ==================


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def get_current_user(
    authorization: str = Header(..., alias="Authorization"),
    db=Depends(get_db),
) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid auth header")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        uid = int(data["sub"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    row = db.execute(
        text("SELECT id, name, email, dept, level, role FROM users WHERE id = :id"),
        {"id": uid},
    ).mappings().first()
    if not row:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(row)


def generate_qr_payload(minutes_valid: int = 5) -> dict:
    exp_ts = int((datetime.now(timezone.utc) + timedelta(minutes=minutes_valid)).timestamp())
    data_to_sign = f"EMPQR|{exp_ts}"
    sig = hmac.new(QR_SHARED_SECRET.encode("utf-8"), data_to_sign.encode("utf-8"), "sha256").digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")
    payload = f"{data_to_sign}|{sig_b64}"
    return {"payload": payload, "exp": exp_ts}


def verify_qr_payload(payload: str) -> int:
    parts = payload.split("|")
    if len(parts) != 3 or parts[0] != "EMPQR":
        raise HTTPException(status_code=400, detail="無效的 QR 格式")

    _, exp_str, sig_b64 = parts
    try:
        exp_ts = int(exp_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="無效的 QR 資料")

    data_to_sign = f"EMPQR|{exp_ts}"
    expected_sig = base64.urlsafe_b64encode(
        hmac.new(QR_SHARED_SECRET.encode("utf-8"), data_to_sign.encode("utf-8"), "sha256").digest()
    ).decode("utf-8").rstrip("=")

    if not hmac.compare_digest(expected_sig, sig_b64):
        raise HTTPException(status_code=400, detail="QR 簽章錯誤")

    if datetime.now(timezone.utc).timestamp() > exp_ts:
        raise HTTPException(status_code=400, detail="QR 已過期")

    return exp_ts


def distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lng2 - lng1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def to_datetime_utc(value) -> datetime:
    """把 SQLite 回傳的 ts 轉成 datetime（支援 datetime 或字串）"""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                dt = datetime.strptime(value, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

    raise ValueError(f"無法解析時間格式: {value!r}")

# ================== Routes ==================


@app.get("/")
def root():
    return {"status": "ok", "message": "EmployeePassHR backend running"}


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest, db=Depends(get_db)):
    row = db.execute(
        text("SELECT id, name, email, password_hash, dept, level, role FROM users WHERE email = :email"),
        {"email": req.email},
    ).mappings().first()

    if not row or not pbkdf2_sha256.verify(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")

    token = create_access_token(row["id"])
    user_out = UserOut(
        id=row["id"],
        name=row["name"],
        email=row["email"],
        dept=row["dept"],
        level=row["level"],
        role=row["role"],
    )
    return LoginResponse(token=token, user=user_out)


@app.post("/auth/register", response_model=LoginResponse)
def register(req: RegisterRequest, db=Depends(get_db)):
    existing = db.execute(
        text("SELECT id FROM users WHERE email = :email"),
        {"email": req.email},
    ).scalar_one_or_none()

    if existing is not None:
        raise HTTPException(status_code=400, detail="Email 已被使用")

    pw_hash = pbkdf2_sha256.hash(req.password)
    role = req.role if req.role in ("admin", "employee") else "employee"

    db.execute(
        text("""
            INSERT INTO users (name, email, password_hash, dept, level, role)
            VALUES (:name, :email, :pw, :dept, :level, :role)
        """),
        {
            "name": req.name,
            "email": req.email,
            "pw": pw_hash,
            "dept": req.dept,
            "level": req.level,
            "role": role,
        },
    )
    db.commit()

    row = db.execute(
        text("SELECT id, name, email, dept, level, role FROM users WHERE email = :email"),
        {"email": req.email},
    ).mappings().first()

    token = create_access_token(row["id"])
    user_out = UserOut(
        id=row["id"],
        name=row["name"],
        email=row["email"],
        dept=row["dept"],
        level=row["level"],
        role=row["role"],
    )
    return LoginResponse(token=token, user=user_out)


@app.get("/me", response_model=UserOut)
def get_me(user=Depends(get_current_user)):
    return UserOut(**user)


@app.get("/qr/issue")
def issue_qr(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="只有管理員可以產生 QR")
    return generate_qr_payload(minutes_valid=5)


@app.post("/punch", response_model=PunchResponse)
def punch(req: PunchRequest, user=Depends(get_current_user), db=Depends(get_db)):
    verify_qr_payload(req.qr_payload)

    dist = distance_m(req.lat, req.lng, OFFICE_LAT, OFFICE_LNG)
    if dist > OFFICE_RADIUS_M:
        raise HTTPException(status_code=400, detail=f"目前位置不在公司範圍內（距離約 {dist:.1f} 公尺）")

    now = datetime.now(timezone.utc)

    last = db.execute(
        text("SELECT type FROM timelogs WHERE user_id = :uid ORDER BY ts DESC LIMIT 1"),
        {"uid": user["id"]},
    ).mappings().first()

    next_type = "In"
    if last and last["type"] == "In":
        next_type = "Out"

    db.execute(
        text("""
            INSERT INTO timelogs (user_id, ts, type, lat, lng, qr_payload)
            VALUES (:uid, :ts, :type, :lat, :lng, :qr)
        """),
        {
            "uid": user["id"],
            "ts": now,
            "type": next_type,
            "lat": req.lat,
            "lng": req.lng,
            "qr": req.qr_payload,
        },
    )
    db.commit()

    return PunchResponse(
        status="ok",
        type=next_type,
        ts=now,
        distance_m=dist,
        message=f"{next_type} 打卡成功",
    )


@app.get("/calendar", response_model=List[CalendarEventOut])
def get_calendar(user=Depends(get_current_user), db=Depends(get_db)):
    rows = db.execute(
        text("SELECT id, title, description, start_at, end_at FROM events ORDER BY start_at ASC")
    ).mappings().all()

    return [
        CalendarEventOut(
            id=r["id"],
            title=r["title"],
            description=r["description"],
            start_at=r["start_at"],
            end_at=r["end_at"],
        )
        for r in rows
    ]


@app.post("/calendar", response_model=CalendarEventOut)
def create_calendar_event(
    req: CreateEventRequest,
    user=Depends(get_current_user),
    db=Depends(get_db),
):
    # 目前所有登入者都可新增，如要只給 admin 就在此檢查 role
    db.execute(
        text("""
            INSERT INTO events (title, description, start_at, end_at)
            VALUES (:title, :description, :start_at, :end_at)
        """),
        {
            "title": req.title,
            "description": req.description,
            "start_at": req.start_at,
            "end_at": req.end_at,
        },
    )
    db.commit()

    new_id = db.execute(text("SELECT last_insert_rowid()")).scalar_one()

    row = db.execute(
        text("SELECT id, title, description, start_at, end_at FROM events WHERE id = :id"),
        {"id": new_id},
    ).mappings().first()

    return CalendarEventOut(
        id=row["id"],
        title=row["title"],
        description=row["description"],
        start_at=row["start_at"],
        end_at=row["end_at"],
    )


@app.get("/timelogs/history", response_model=List[TimelogOut])
def history(limit: int = 50, user=Depends(get_current_user), db=Depends(get_db)):
    rows = db.execute(
        text("""
            SELECT id, ts, type, lat, lng
            FROM timelogs
            WHERE user_id = :uid
            ORDER BY ts DESC
            LIMIT :limit
        """),
        {"uid": user["id"], "limit": limit},
    ).mappings().all()

    return [
        TimelogOut(
            id=r["id"],
            ts=r["ts"],
            type=r["type"],
            lat=r["lat"],
            lng=r["lng"],
        )
        for r in rows
    ]


@app.get("/payroll/preview", response_model=PayrollPreview)
def payroll_preview(
    month: Optional[str] = None,
    user=Depends(get_current_user),
    db=Depends(get_db),
):
    now = datetime.now(timezone.utc)
    if month:
        year, mon = map(int, month.split("-"))
    else:
        year, mon = now.year, now.month

    start = datetime(year, mon, 1, tzinfo=timezone.utc)
    if mon == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, mon + 1, 1, tzinfo=timezone.utc)

    rows = db.execute(
        text("""
            SELECT ts, type
            FROM timelogs
            WHERE user_id = :uid
              AND ts >= :start AND ts < :end
            ORDER BY ts ASC
        """),
        {"uid": user["id"], "start": start, "end": end},
    ).mappings().all()

    minutes = 0
    last_in: Optional[datetime] = None

    for r in rows:
        ts = to_datetime_utc(r["ts"])
        if r["type"] == "In":
            last_in = ts
        elif r["type"] == "Out" and last_in:
            minutes += int((ts - last_in).total_seconds() // 60)
            last_in = None

    hours = round(minutes / 60.0, 2)
    hourly_rate = 200  # demo：時薪 200
    gross = int(hours * hourly_rate)

    return PayrollPreview(
        month=f"{year}-{mon:02d}",
        minutes=minutes,
        hours=hours,
        gross_demo=gross,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


# ================== Admin Web (HR Dashboard) ==================

TPE = timezone(timedelta(hours=8))
OVERTIME_MULTIPLIER = float(os.getenv("OVERTIME_MULTIPLIER", "1.33"))


def require_admin(user=Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="只有管理員可以使用此功能")
    return user




# ================== Admin API (Mobile / HR) ==================

class AdminUserOut(BaseModel):
    id: int
    name: str
    email: str
    dept: Optional[str] = None
    level: Optional[str] = None
    role: str
    hourly_rate: int


@app.get("/admin/users", response_model=List[AdminUserOut])
@app.get("/admin/users/all", response_model=List[AdminUserOut])
@app.get("/admin/users/list", response_model=List[AdminUserOut])
def admin_get_users(
    q: Optional[str] = None,  # 可用來搜尋姓名/email/部門
    role: Optional[str] = None,  # 可選：employee / admin
    admin=Depends(require_admin),
    db=Depends(get_db),
):
    where = []
    params = {}

    if q:
        where.append("(name LIKE :q OR email LIKE :q OR COALESCE(dept,'') LIKE :q)")
        params["q"] = f"%{q}%"

    if role:
        where.append("role = :role")
        params["role"] = role

    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    rows = db.execute(
        text(
            "SELECT id, name, email, dept, level, role, COALESCE(hourly_rate, 200) AS hourly_rate "
            "FROM users" + where_sql + " ORDER BY dept ASC, id ASC"
        ),
        params,
    ).mappings().all()

    return [
        AdminUserOut(
            id=r["id"],
            name=r["name"],
            email=r["email"],
            dept=r.get("dept"),
            level=r.get("level"),
            role=r["role"],
            hourly_rate=int(r.get("hourly_rate") or 200),
        )
        for r in rows
    ]


@app.get("/admin/users/{user_id}/payroll-preview", response_model=AdminMonthlyPayrollRowOut)
def admin_get_user_payroll_preview(
    user_id: int,
    month: Optional[str] = None,  # YYYY-MM（台北時間）
    admin=Depends(require_admin),
    db=Depends(get_db),
):
    # 直接重用整月匯總邏輯，然後挑出指定使用者
    rows = admin_payroll_monthly(month=month, admin=admin, db=db)
    for r in rows:
        if r.user_id == user_id:
            return r
    raise HTTPException(status_code=404, detail="User not found in payroll data")
def _calc_work_minutes_for_day(items: list[tuple[datetime, str]], day_start_utc: datetime, day_end_utc: datetime, now_utc: datetime) -> int:
    """items: [(ts_utc, type)] sorted by ts"""
    minutes = 0
    last_in: Optional[datetime] = None

    for ts, tp in items:
        if tp == "In":
            last_in = ts
        elif tp == "Out" and last_in:
            minutes += int((ts - last_in).total_seconds() // 60)
            last_in = None

    # 若最後還在上班（In 沒 Out）
    if last_in:
        end = now_utc if (day_start_utc.date() == now_utc.date()) else day_end_utc
        if end > last_in:
            minutes += int((end - last_in).total_seconds() // 60)

    return max(minutes, 0)
def admin_get_user_timelogs(
    user_id: int,
    limit: int = 50,
    admin=Depends(require_admin),
    db=Depends(get_db),
):
    # 查詢指定使用者的打卡紀錄
    rows = db.execute(
        text("""
            SELECT id, ts, type, lat, lng
            FROM timelogs
            WHERE user_id = :uid
            ORDER BY ts DESC
            LIMIT :limit
        """),
        {"uid": user_id, "limit": limit},
    ).mappings().all()

    return [
        TimelogOut(
            id=r["id"],
            ts=r["ts"],
            type=r["type"],
            lat=r["lat"],
            lng=r["lng"],
        )
        for r in rows
    ]


class UpdateHourlyRateIn(BaseModel):
    hourly_rate: int


@app.patch("/admin/users/{user_id}/hourly_rate")
def admin_update_hourly_rate(
    user_id: int,
    req: UpdateHourlyRateIn,
    admin=Depends(require_admin),
    db=Depends(get_db),
):
    if req.hourly_rate < 0 or req.hourly_rate > 100000:
        raise HTTPException(status_code=400, detail="hourly_rate 不合理")

    db.execute(
        text("UPDATE users SET hourly_rate = :r WHERE id = :id"),
        {"r": req.hourly_rate, "id": user_id},
    )
    db.commit()
    return {"ok": True}


class AdminDailyAttendanceRowOut(BaseModel):
    user_id: int
    name: str
    email: str
    dept: Optional[str] = None
    level: Optional[str] = None
    hourly_rate: int
    first_in: Optional[datetime] = None
    last_out: Optional[datetime] = None
    worked_minutes: int
    worked_hours: float
    overtime_minutes: int
    overtime_hours: float
    status: str
    late: bool


@app.get("/admin/attendance/daily", response_model=List[AdminDailyAttendanceRowOut])
def admin_attendance_daily(
    date: Optional[str] = None,  # YYYY-MM-DD（台北時間）
    admin=Depends(require_admin),
    db=Depends(get_db),
):
    now_tpe = datetime.now(TPE)

    if date:
        try:
            y, m, d = map(int, date.split("-"))
            day_start_tpe = datetime(y, m, d, 0, 0, 0, tzinfo=TPE)
        except Exception:
            raise HTTPException(status_code=400, detail="date 格式需為 YYYY-MM-DD")
    else:
        day_start_tpe = now_tpe.replace(hour=0, minute=0, second=0, microsecond=0)

    day_end_tpe = day_start_tpe + timedelta(days=1)
    day_start_utc = day_start_tpe.astimezone(timezone.utc)
    day_end_utc = day_end_tpe.astimezone(timezone.utc)
    now_utc = datetime.now(timezone.utc)

    users = db.execute(
        text(
            "SELECT id, name, email, dept, level, COALESCE(hourly_rate, 200) AS hourly_rate "
            "FROM users WHERE role = 'employee' "
            "ORDER BY dept ASC, id ASC"
        )
    ).mappings().all()

    logs = db.execute(
        text(
            "SELECT user_id, ts, type FROM timelogs "
            "WHERE ts >= :s AND ts < :e "
            "ORDER BY user_id ASC, ts ASC"
        ),
        {"s": day_start_utc, "e": day_end_utc},
    ).mappings().all()

    per_user = defaultdict(list)
    for r in logs:
        per_user[r["user_id"]].append((to_datetime_utc(r["ts"]), r["type"]))

    # 遲到門檻：09:10（台北時間）
    late_threshold_utc = day_start_tpe.replace(hour=9, minute=10).astimezone(timezone.utc)
    standard_minutes = 8 * 60

    out_rows: List[AdminDailyAttendanceRowOut] = []
    for u in users:
        items = per_user.get(u["id"], [])
        first_in = next((ts for ts, tp in items if tp == "In"), None)
        last_out = next((ts for ts, tp in reversed(items) if tp == "Out"), None)

        worked_minutes = _calc_work_minutes_for_day(items, day_start_utc, day_end_utc, now_utc)
        overtime_minutes = max(0, worked_minutes - standard_minutes)

        if not items:
            status = "未打卡"
        else:
            last_type = items[-1][1]
            status = "上班中（未下班）" if last_type == "In" else "已下班"
            if first_in is None and last_out is not None:
                status = "異常（只有下班卡）"

        late = bool(first_in and first_in > late_threshold_utc)

        out_rows.append(AdminDailyAttendanceRowOut(
            user_id=u["id"],
            name=u["name"],
            email=u["email"],
            dept=u.get("dept"),
            level=u.get("level"),
            hourly_rate=int(u["hourly_rate"] or 200),
            first_in=first_in,
            last_out=last_out,
            worked_minutes=worked_minutes,
            worked_hours=round(worked_minutes / 60.0, 2),
            overtime_minutes=overtime_minutes,
            overtime_hours=round(overtime_minutes / 60.0, 2),
            status=status,
            late=late,
        ))

    return out_rows


@app.get("/admin/attendance/export/csv")
def admin_attendance_export_csv(
    date: Optional[str] = None,
    admin=Depends(require_admin),
    db=Depends(get_db),
):
    rows = admin_attendance_daily(date=date, admin=admin, db=db)
    d = date or datetime.now(TPE).strftime("%Y-%m-%d")

    csv_lines = ["\ufeffuser_id,name,email,dept,level,hourly_rate,first_in,last_out,worked_hours,overtime_hours,status,late"]
    for r in rows:
        dept = (r.dept or "").replace(",", " ")
        level = (r.level or "").replace(",", " ")
        fi = r.first_in.isoformat() if r.first_in else ""
        lo = r.last_out.isoformat() if r.last_out else ""
        csv_lines.append(f"{r.user_id},{r.name},{r.email},{dept},{level},{r.hourly_rate},{fi},{lo},{r.worked_hours},{r.overtime_hours},{r.status},{int(r.late)}")

    csv_text = "\n".join(csv_lines)
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="attendance_{d}.csv"'},
    )


class AdminMonthlyPayrollRowOut(BaseModel):
    user_id: int
    name: str
    email: str
    dept: Optional[str] = None
    level: Optional[str] = None
    hourly_rate: int
    working_days: int
    worked_minutes: int
    worked_hours: float
    overtime_minutes: int
    overtime_hours: float
    regular_pay: int
    overtime_pay: int
    gross_pay: int


def _parse_month(month: Optional[str]) -> tuple[datetime, datetime, str]:
    now = datetime.now(TPE)
    if month:
        try:
            y, m = map(int, month.split("-"))
        except Exception:
            raise HTTPException(status_code=400, detail="month 格式需為 YYYY-MM")
    else:
        y, m = now.year, now.month

    start_tpe = datetime(y, m, 1, 0, 0, 0, tzinfo=TPE)
    if m == 12:
        end_tpe = datetime(y + 1, 1, 1, 0, 0, 0, tzinfo=TPE)
    else:
        end_tpe = datetime(y, m + 1, 1, 0, 0, 0, tzinfo=TPE)
    return start_tpe, end_tpe, f"{y}-{m:02d}"


@app.get("/admin/payroll/monthly", response_model=List[AdminMonthlyPayrollRowOut])
def admin_payroll_monthly(
    month: Optional[str] = None,  # YYYY-MM（台北時間）
    admin=Depends(require_admin),
    db=Depends(get_db),
):
    start_tpe, end_tpe, label = _parse_month(month)
    start_utc = start_tpe.astimezone(timezone.utc)
    end_utc = end_tpe.astimezone(timezone.utc)
    now_utc = datetime.now(timezone.utc)
    now_tpe = datetime.now(TPE)

    users = db.execute(
        text(
            "SELECT id, name, email, dept, level, COALESCE(hourly_rate, 200) AS hourly_rate "
            "FROM users WHERE role = 'employee' "
            "ORDER BY dept ASC, id ASC"
        )
    ).mappings().all()

    logs = db.execute(
        text(
            "SELECT user_id, ts, type FROM timelogs "
            "WHERE ts >= :s AND ts < :e "
            "ORDER BY user_id ASC, ts ASC"
        ),
        {"s": start_utc, "e": end_utc},
    ).mappings().all()

    # group by (user_id, tpe_date)
    per_user_day: dict[tuple[int, str], list[tuple[datetime, str]]] = defaultdict(list)
    for r in logs:
        ts_utc = to_datetime_utc(r["ts"])
        d = ts_utc.astimezone(TPE).date().isoformat()
        per_user_day[(r["user_id"], d)].append((ts_utc, r["type"]))

    standard_minutes = 8 * 60
    out_rows: list[AdminMonthlyPayrollRowOut] = []

    for u in users:
        uid = u["id"]
        hourly = int(u["hourly_rate"] or 200)

        worked_total = 0
        overtime_total = 0
        working_days = 0

        # iterate all days in month (only count days with punches as working_days)
        day = start_tpe
        while day < end_tpe:
            d_key = day.date().isoformat()
            items = per_user_day.get((uid, d_key), [])
            if items:
                working_days += 1
                # compute day boundaries (in UTC)
                day_start_utc = day.astimezone(timezone.utc)
                day_end_utc = (day + timedelta(days=1)).astimezone(timezone.utc)
                # if this is today (tpe), allow open session to now
                effective_now_utc = now_utc if day.date() == now_tpe.date() else day_end_utc
                mins = _calc_work_minutes_for_day(items, day_start_utc, day_end_utc, effective_now_utc)
                worked_total += mins
                overtime_total += max(0, mins - standard_minutes)
            day += timedelta(days=1)

        regular_minutes = max(0, worked_total - overtime_total)
        regular_pay = int(round((regular_minutes / 60.0) * hourly))
        overtime_pay = int(round((overtime_total / 60.0) * hourly * OVERTIME_MULTIPLIER))
        gross = regular_pay + overtime_pay

        out_rows.append(AdminMonthlyPayrollRowOut(
            user_id=uid,
            name=u["name"],
            email=u["email"],
            dept=u.get("dept"),
            level=u.get("level"),
            hourly_rate=hourly,
            working_days=working_days,
            worked_minutes=worked_total,
            worked_hours=round(worked_total / 60.0, 2),
            overtime_minutes=overtime_total,
            overtime_hours=round(overtime_total / 60.0, 2),
            regular_pay=regular_pay,
            overtime_pay=overtime_pay,
            gross_pay=gross,
        ))

    return out_rows


@app.get("/admin/payroll/export/csv")
def admin_payroll_export_csv(
    month: Optional[str] = None,
    admin=Depends(require_admin),
    db=Depends(get_db),
):
    rows = admin_payroll_monthly(month=month, admin=admin, db=db)
    _, _, label = _parse_month(month)

    csv_lines = ["\ufeffuser_id,name,email,dept,level,hourly_rate,working_days,worked_hours,overtime_hours,regular_pay,overtime_pay,gross_pay"]
    for r in rows:
        dept = (r.dept or "").replace(",", " ")
        level = (r.level or "").replace(",", " ")
        csv_lines.append(
            f"{r.user_id},{r.name},{r.email},{dept},{level},{r.hourly_rate},{r.working_days},{r.worked_hours},{r.overtime_hours},{r.regular_pay},{r.overtime_pay},{r.gross_pay}"
        )
    csv_text = "\n".join(csv_lines)
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="payroll_{label}.csv"'},
    )


# -------- Web pages (inline) --------

ADMIN_CSS = """    :root{
  --glass: rgba(255,255,255,0.06);
  --stroke: rgba(255,255,255,0.14);
}
body{ background:#0b0f14; }
.glass-card{
  background: var(--glass) !important;
  border: 1px solid var(--stroke) !important;
  border-radius: 18px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}
.table-dark{
  --bs-table-bg: rgba(255,255,255,0.03);
  --bs-table-striped-bg: rgba(255,255,255,0.05);
  --bs-table-hover-bg: rgba(255,255,255,0.06);
}
.badge-soft-danger{ background: rgba(255,0,0,0.18); border: 1px solid rgba(255,0,0,0.28); }
.badge-soft-warn{ background: rgba(255,193,7,0.18); border: 1px solid rgba(255,193,7,0.28); }
.badge-soft-ok{ background: rgba(25,135,84,0.18); border: 1px solid rgba(25,135,84,0.28); }
.wage-input{ max-width: 110px; }
"""


ADMIN_JS = r"""// ===== Shared helpers =====
const TOKEN_KEY = "hr_admin_token";
const USER_KEY  = "hr_admin_user";
function $(id){ return document.getElementById(id); }
function getToken(){ return localStorage.getItem(TOKEN_KEY); }
function setToken(t){ localStorage.setItem(TOKEN_KEY, t); }
function clearToken(){ localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(USER_KEY); }
function setUser(u){ localStorage.setItem(USER_KEY, JSON.stringify(u)); }
function getUser(){ try { return JSON.parse(localStorage.getItem(USER_KEY) || "null"); } catch { return null; } }
function showErr(msg){
  const el = $("err");
  if(!el) return;
  el.textContent = msg;
  el.classList.remove("d-none");
}
async function apiFetch(path, opts={}){
  const token = getToken();
  const headers = opts.headers || {};
  headers["Content-Type"] = headers["Content-Type"] || "application/json";
  if(token) headers["Authorization"] = `Bearer ${token}`;
  opts.headers = headers;
  const resp = await fetch(path, opts);
  if(resp.status === 401 || resp.status === 403){
    clearToken();
    location.href = "/admin-web/login";
    throw new Error("Unauthorized");
  }
  if(!resp.ok){
    let detail = `${resp.status}`;
    try{
      const j = await resp.json();
      detail = j.detail ? String(j.detail) : JSON.stringify(j);
    }catch{
      try{ detail = await resp.text(); }catch{}
    }
    throw new Error(detail);
  }
  return resp;
}
function fmtTime(iso){
  if(!iso) return "-";
  const d = new Date(iso);
  const pad = (n)=> String(n).padStart(2,"0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// ===== Login page =====
function initLogin(){
  const form = $("loginForm");
  if(!form) return;
  form.addEventListener("submit", async (e)=>{
    e.preventDefault();
    const email = $("email").value.trim();
    const password = $("password").value;
    try{
      const resp = await fetch("/auth/login", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({email, password})
      });
      if(!resp.ok){
        const j = await resp.json().catch(()=>({detail:"登入失敗"}));
        throw new Error(j.detail || "登入失敗");
      }
      const data = await resp.json();
      if(!data.token || !data.user) throw new Error("後端回傳格式不正確");
      if(data.user.role !== "admin") throw new Error("此帳號不是管理員，無法進入後台");
      setToken(data.token);
      setUser(data.user);
      location.href = "/admin-web/dashboard";
    }catch(err){
      showErr(err.message || String(err));
    }
  });
}

// ===== Daily attendance =====
let attendanceRows = [];
function computeCards(rows){
  const total = rows.length;
  const absent = rows.filter(r=> r.status === "未打卡").length;
  const missingOut = rows.filter(r=> (r.status || "").includes("未下班")).length;
  const late = rows.filter(r=> r.late).length;
  const otMinutes = rows.reduce((s,r)=> s + (r.overtime_minutes || 0), 0);
  const workedMinutes = rows.reduce((s,r)=> s + (r.worked_minutes || 0), 0);
  const avgHours = total ? (workedMinutes/60/total) : 0;
  $("c_total").textContent = total;
  $("c_absent").textContent = absent;
  $("c_missingOut").textContent = missingOut;
  $("c_late").textContent = late;
  $("c_ot").textContent = (otMinutes/60).toFixed(2);
  $("c_avg").textContent = avgHours.toFixed(2);
}
function badgeForStatus(status){
  if(status === "未打卡") return `<span class="badge badge-soft-danger text-danger fw-semibold">${status}</span>`;
  if(status.includes("未下班") || status.includes("異常")) return `<span class="badge badge-soft-warn text-warning fw-semibold">${status}</span>`;
  return `<span class="badge badge-soft-ok text-success fw-semibold">${status}</span>`;
}
function renderAttendance(rows){
  const q = ($("searchInput")?.value || "").trim().toLowerCase();
  const filtered = !q ? rows : rows.filter(r =>
    (r.name||"").toLowerCase().includes(q) ||
    (r.email||"").toLowerCase().includes(q) ||
    (r.dept||"").toLowerCase().includes(q)
  );
  const tbody = $("tbody");
  if(!tbody) return;
  if(filtered.length === 0){
    tbody.innerHTML = `<tr><td colspan="11" class="text-center text-secondary py-4">沒有資料</td></tr>`;
    return;
  }
  tbody.innerHTML = filtered.map(r=>{
    const dept = r.dept || "-";
    const level = r.level || "-";
    const wage = r.hourly_rate ?? 200;
    const rowClass = (r.status === "未打卡") ? "table-danger" :
                     ((r.status || "").includes("未下班") || (r.status || "").includes("異常")) ? "table-warning" : "";
    return `<tr class="${rowClass}">
      <td>${dept}</td>
      <td>${level}</td>
      <td class="fw-semibold">${r.name}</td>
      <td class="text-secondary">${r.email}</td>
      <td><input class="form-control form-control-sm wage-input" type="number" min="0" max="100000" value="${wage}" data-user-id="${r.user_id}"></td>
      <td>${fmtTime(r.first_in)}</td>
      <td>${fmtTime(r.last_out)}</td>
      <td class="text-end">${(r.worked_hours ?? 0).toFixed(2)}</td>
      <td class="text-end">${(r.overtime_hours ?? 0).toFixed(2)}</td>
      <td>${badgeForStatus(r.status)}</td>
      <td>${r.late ? `<span class="text-warning fw-semibold">是</span>` : `<span class="text-secondary">否</span>`}</td>
    </tr>`;
  }).join("");
  tbody.querySelectorAll(".wage-input").forEach(inp=>{
    inp.addEventListener("change", async (e)=>{
      const userId = e.target.dataset.userId;
      const hourly_rate = parseInt(e.target.value || "0", 10);
      try{
        await apiFetch(`/admin/users/${userId}/hourly_rate`, {
          method: "PATCH",
          body: JSON.stringify({hourly_rate})
        });
      }catch(err){
        showErr(`更新時薪失敗：${err.message || err}`);
      }
    });
  });
}
async function loadAttendance(){
  try{
    $("err")?.classList.add("d-none");
    const date = $("dateInput")?.value;
    const qs = date ? `?date=${encodeURIComponent(date)}` : "";
    const resp = await apiFetch(`/admin/attendance/daily${qs}`);
    attendanceRows = await resp.json();
    computeCards(attendanceRows);
    renderAttendance(attendanceRows);
  }catch(err){
    showErr(err.message || String(err));
  }
}
async function exportAttendanceCSV(){
  try{
    const date = $("dateInput")?.value;
    const qs = date ? `?date=${encodeURIComponent(date)}` : "";
    const resp = await apiFetch(`/admin/attendance/export/csv${qs}`, { method:"GET", headers:{"Content-Type":"text/plain"} });
    const text = await resp.text();
    const blob = new Blob([text], {type:"text/csv;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `attendance_${date || "today"}.csv`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }catch(err){
    showErr(`匯出失敗：${err.message || err}`);
  }
}

// ===== Monthly payroll =====
let payrollRows = [];
function renderPayroll(rows){
  const q = ($("searchInput")?.value || "").trim().toLowerCase();
  const filtered = !q ? rows : rows.filter(r =>
    (r.name||"").toLowerCase().includes(q) ||
    (r.email||"").toLowerCase().includes(q) ||
    (r.dept||"").toLowerCase().includes(q)
  );
  const tbody = $("tbodyPayroll");
  if(!tbody) return;
  if(filtered.length === 0){
    tbody.innerHTML = `<tr><td colspan="13" class="text-center text-secondary py-4">沒有資料</td></tr>`;
    return;
  }
  tbody.innerHTML = filtered.map(r=>{
    const dept = r.dept || "-";
    const level = r.level || "-";
    const wage = r.hourly_rate ?? 200;
    return `<tr>
      <td>${dept}</td>
      <td>${level}</td>
      <td class="fw-semibold">${r.name}</td>
      <td class="text-secondary">${r.email}</td>
      <td><input class="form-control form-control-sm wage-input" type="number" min="0" max="100000" value="${wage}" data-user-id="${r.user_id}"></td>
      <td class="text-end">${r.working_days}</td>
      <td class="text-end">${(r.worked_hours ?? 0).toFixed(2)}</td>
      <td class="text-end">${(r.overtime_hours ?? 0).toFixed(2)}</td>
      <td class="text-end">${r.regular_pay}</td>
      <td class="text-end">${r.overtime_pay}</td>
      <td class="text-end fw-semibold">${r.gross_pay}</td>
    </tr>`;
  }).join("");
  tbody.querySelectorAll(".wage-input").forEach(inp=>{
    inp.addEventListener("change", async (e)=>{
      const userId = e.target.dataset.userId;
      const hourly_rate = parseInt(e.target.value || "0", 10);
      try{
        await apiFetch(`/admin/users/${userId}/hourly_rate`, {
          method: "PATCH",
          body: JSON.stringify({hourly_rate})
        });
      }catch(err){
        showErr(`更新時薪失敗：${err.message || err}`);
      }
    });
  });
}
async function loadPayroll(){
  try{
    $("err")?.classList.add("d-none");
    const month = $("monthInput")?.value;
    const qs = month ? `?month=${encodeURIComponent(month)}` : "";
    const resp = await apiFetch(`/admin/payroll/monthly${qs}`);
    payrollRows = await resp.json();
    renderPayroll(payrollRows);
  }catch(err){
    showErr(err.message || String(err));
  }
}
async function exportPayrollCSV(){
  try{
    const month = $("monthInput")?.value;
    const qs = month ? `?month=${encodeURIComponent(month)}` : "";
    const resp = await apiFetch(`/admin/payroll/export/csv${qs}`, { method:"GET", headers:{"Content-Type":"text/plain"} });
    const text = await resp.text();
    const blob = new Blob([text], {type:"text/csv;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `payroll_${month || "this_month"}.csv`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }catch(err){
    showErr(`匯出失敗：${err.message || err}`);
  }
}

function initCommonNav(){
  const token = getToken();
  if(!token){
    // allow login page
    if(location.pathname.startsWith("/admin-web/login")) return;
    location.href = "/admin-web/login";
    return;
  }
  const u = getUser();
  if(u && $("who")) $("who").textContent = `${u.name}（${u.email}）`;
  $("logoutBtn")?.addEventListener("click", ()=>{ clearToken(); location.href="/admin-web/login"; });
}

function initDashboard(){
  if(!$("tbody")) return;
  initCommonNav();
  const di = $("dateInput");
  if(di && !di.value){
    const now = new Date();
    const pad = (n)=> String(n).padStart(2,"0");
    di.value = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}`;
  }
  $("refreshBtn")?.addEventListener("click", loadAttendance);
  $("exportBtn")?.addEventListener("click", exportAttendanceCSV);
  $("searchInput")?.addEventListener("input", ()=> renderAttendance(attendanceRows));
  $("dateInput")?.addEventListener("change", loadAttendance);
  loadAttendance();
}

function initPayroll(){
  if(!$("tbodyPayroll")) return;
  initCommonNav();
  const mi = $("monthInput");
  if(mi && !mi.value){
    const now = new Date();
    const pad = (n)=> String(n).padStart(2,"0");
    mi.value = `${now.getFullYear()}-${pad(now.getMonth()+1)}`;
  }
  $("refreshBtn")?.addEventListener("click", loadPayroll);
  $("exportBtn")?.addEventListener("click", exportPayrollCSV);
  $("searchInput")?.addEventListener("input", ()=> renderPayroll(payrollRows));
  $("monthInput")?.addEventListener("change", loadPayroll);
  loadPayroll();
}

initLogin();
initDashboard();
initPayroll();
"""  # end ADMIN_JS


@app.get("/admin-web/static/admin.css")
def admin_web_css():
    return Response(content=ADMIN_CSS, media_type="text/css; charset=utf-8")


@app.get("/admin-web/static/admin.js")
def admin_web_js():
    return Response(content=ADMIN_JS, media_type="application/javascript; charset=utf-8")


@app.get("/admin-web/login", response_class=HTMLResponse)
def admin_web_login():
    return """<!doctype html>
<html lang="zh-Hant"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>員工通｜後台登入</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="/admin-web/static/admin.css">
</head><body class="text-light">
  <div class="container py-5" style="max-width: 520px;">
    <div class="card glass-card p-4">
      <h3 class="mb-2 fw-bold">後台登入</h3>
      <p class="text-secondary mb-4">請使用「管理員」帳號登入（admin 才能看到資料）</p>
      <form id="loginForm">
        <div class="mb-3">
          <label class="form-label">Email</label>
          <input class="form-control form-control-lg" type="email" id="email" placeholder="admin@example.com" required>
        </div>
        <div class="mb-3">
          <label class="form-label">密碼</label>
          <input class="form-control form-control-lg" type="password" id="password" placeholder="admin123" required>
        </div>
        <button class="btn btn-warning w-100 btn-lg fw-semibold" type="submit">登入</button>
      </form>
      <div id="err" class="alert alert-danger mt-3 d-none"></div>
      <div class="mt-3 small text-secondary">入口：/admin-web/login</div>
    </div>
  </div>
  <script src="/admin-web/static/admin.js"></script>
</body></html>"""


@app.get("/admin-web/dashboard", response_class=HTMLResponse)
def admin_web_dashboard():
    return """<!doctype html>
<html lang="zh-Hant"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>員工通｜人資後台（每日出勤）</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="/admin-web/static/admin.css">
</head><body class="text-light">
  <nav class="navbar navbar-expand-lg navbar-dark border-bottom border-secondary">
    <div class="container-fluid px-3">
      <span class="navbar-brand fw-bold">員工通｜人資後台</span>
      <div class="ms-auto d-flex gap-2 align-items-center">
        <a class="btn btn-outline-light btn-sm" href="/admin-web/payroll">整月薪資</a>
        <span id="who" class="text-secondary small"></span>
        <button id="logoutBtn" class="btn btn-outline-light btn-sm">登出</button>
      </div>
    </div>
  </nav>
  <div class="container-fluid px-3 py-4">
    <div class="d-flex flex-wrap gap-2 align-items-end mb-3">
      <div>
        <label class="form-label text-secondary">日期（台北）</label>
        <input type="date" id="dateInput" class="form-control">
      </div>
      <div class="flex-grow-1">
        <label class="form-label text-secondary">搜尋</label>
        <input type="text" id="searchInput" class="form-control" placeholder="姓名 / Email / 部門">
      </div>
      <div>
        <label class="form-label text-secondary">快速操作</label>
        <div class="d-flex gap-2">
          <button id="refreshBtn" class="btn btn-warning fw-semibold">重新整理</button>
          <button id="exportBtn" class="btn btn-outline-light">匯出 CSV</button>
        </div>
      </div>
    </div>
    <div class="row g-3 mb-3">
      <div class="col-6 col-md-2"><div class="card glass-card p-3"><div class="text-secondary small">員工總數</div><div class="fs-4 fw-bold" id="c_total">-</div></div></div>
      <div class="col-6 col-md-2"><div class="card glass-card p-3"><div class="text-secondary small">未打卡</div><div class="fs-4 fw-bold text-danger" id="c_absent">-</div></div></div>
      <div class="col-6 col-md-2"><div class="card glass-card p-3"><div class="text-secondary small">缺下班卡</div><div class="fs-4 fw-bold text-warning" id="c_missingOut">-</div></div></div>
      <div class="col-6 col-md-2"><div class="card glass-card p-3"><div class="text-secondary small">遲到</div><div class="fs-4 fw-bold text-warning" id="c_late">-</div></div></div>
      <div class="col-6 col-md-2"><div class="card glass-card p-3"><div class="text-secondary small">今日總加班</div><div class="fs-4 fw-bold" id="c_ot">-</div></div></div>
      <div class="col-6 col-md-2"><div class="card glass-card p-3"><div class="text-secondary small">平均工時</div><div class="fs-4 fw-bold" id="c_avg">-</div></div></div>
    </div>
    <div class="card glass-card p-0 overflow-hidden">
      <div class="table-responsive">
        <table class="table table-dark table-hover align-middle mb-0">
          <thead><tr class="text-secondary">
            <th>部門</th><th>職等</th><th>姓名</th><th>Email</th><th style="min-width:130px;">時薪（可改）</th>
            <th>上班</th><th>下班</th><th class="text-end">工時</th><th class="text-end">加班</th><th>狀態</th><th>遲到</th>
          </tr></thead>
          <tbody id="tbody"><tr><td colspan="11" class="text-center text-secondary py-4">載入中…</td></tr></tbody>
        </table>
      </div>
    </div>
    <div id="err" class="alert alert-danger mt-3 d-none"></div>
  </div>
  <script src="/admin-web/static/admin.js"></script>
</body></html>"""


@app.get("/admin-web/payroll", response_class=HTMLResponse)
def admin_web_payroll():
    return """<!doctype html>
<html lang="zh-Hant"><head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>員工通｜人資後台（整月薪資）</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <link rel="stylesheet" href="/admin-web/static/admin.css">
</head><body class="text-light">
  <nav class="navbar navbar-expand-lg navbar-dark border-bottom border-secondary">
    <div class="container-fluid px-3">
      <span class="navbar-brand fw-bold">員工通｜人資後台</span>
      <div class="ms-auto d-flex gap-2 align-items-center">
        <a class="btn btn-outline-light btn-sm" href="/admin-web/dashboard">每日出勤</a>
        <span id="who" class="text-secondary small"></span>
        <button id="logoutBtn" class="btn btn-outline-light btn-sm">登出</button>
      </div>
    </div>
  </nav>
  <div class="container-fluid px-3 py-4">
    <div class="d-flex flex-wrap gap-2 align-items-end mb-3">
      <div>
        <label class="form-label text-secondary">月份（台北）</label>
        <input type="month" id="monthInput" class="form-control">
      </div>
      <div class="flex-grow-1">
        <label class="form-label text-secondary">搜尋</label>
        <input type="text" id="searchInput" class="form-control" placeholder="姓名 / Email / 部門">
      </div>
      <div>
        <label class="form-label text-secondary">快速操作</label>
        <div class="d-flex gap-2">
          <button id="refreshBtn" class="btn btn-warning fw-semibold">重新整理</button>
          <button id="exportBtn" class="btn btn-outline-light">匯出 CSV</button>
        </div>
      </div>
    </div>
    <div class="card glass-card p-0 overflow-hidden">
      <div class="table-responsive">
        <table class="table table-dark table-hover align-middle mb-0">
          <thead><tr class="text-secondary">
            <th>部門</th><th>職等</th><th>姓名</th><th>Email</th><th style="min-width:130px;">時薪（可改）</th>
            <th class="text-end">出勤日</th><th class="text-end">總工時</th><th class="text-end">加班時數</th>
            <th class="text-end">基本薪資</th><th class="text-end">加班薪資</th><th class="text-end">應發合計</th>
          </tr></thead>
          <tbody id="tbodyPayroll"><tr><td colspan="13" class="text-center text-secondary py-4">載入中…</td></tr></tbody>
        </table>
      </div>
  <script src="/admin-web/static/admin.js"></script>
</body></html>"""




# ================== Office QR Web Page ==================
# 目的：在公司桌機/電腦用瀏覽器顯示「動態打卡 QR Code」給員工掃描
#
# 你原本的 API：GET /qr/issue（admin 才能呼叫）會保留不變。
# 這裡額外提供「public 顯示頁」：
# - GET /qr-web            -> 顯示 QR 的網頁（可選 key）
# - GET /qr/public/issue   -> 提供 QR payload（可選 key）
#
# 安全（可選）：
# 若你不希望任何人都能打開 /qr-web，請設定環境變數：
#   export QR_WEB_KEY="your-secret"
# 並使用：/qr-web?key=your-secret

QR_WEB_KEY = os.getenv("QR_WEB_KEY", "").strip()

def _check_qr_web_key(key: str | None):
    if QR_WEB_KEY:
        if not key or key.strip() != QR_WEB_KEY:
            raise HTTPException(status_code=403, detail="QR Web key invalid")


try:
    import qrcode
except Exception:
    qrcode = None


@app.get("/qr/public/png")
def qr_public_png(key: str | None = Query(default=None)):
    # 若你有設定 QR_WEB_KEY，這裡會做權限檢查；沒設定就直接放行
    _check_qr_web_key(key)

    if qrcode is None:
        raise HTTPException(
            status_code=500,
            detail="qrcode 套件未安裝，請先執行：pip install qrcode[pil] pillow"
        )

    data = generate_qr_payload(minutes_valid=5)  # 產生 payload（5 分鐘有效）
    payload = data.get("payload")
    if not payload:
        raise HTTPException(status_code=500, detail="QR payload 產生失敗")

    img = qrcode.make(payload)
    buf = BytesIO()
    img.save(buf, format="PNG")

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )

@app.get("/qr/public/issue")
def qr_public_issue(key: str | None = Query(default=None)):
    _check_qr_web_key(key)
    # 這個 generate_qr_payload 你在 /qr/public/png 應該已經有用到
    return generate_qr_payload(minutes_valid=5)



@app.get("/qr-web", response_class=HTMLResponse)
def qr_web_page(key: str | None = Query(default=None)):
    _check_qr_web_key(key)

    html = """
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>員工通｜打卡 QR Code</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background:#0b1220; color:#e5e7eb; }
    .cardx { background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.14); border-radius:18px; }
    .muted { color: rgba(229,231,235,0.7); }
    #qrImg { width: 320px; height: 320px; object-fit: contain; }
  </style>
</head>
<body>
  <div class="container py-5" style="max-width: 900px;">
    <div class="d-flex align-items-center justify-content-between mb-3">
      <div>
        <h2 class="fw-bold m-0">員工通｜打卡 QR Code</h2>
        <div class="muted">請使用 App 掃描此 QR 進行打卡（會定期更新）</div>
      </div>
      <div class="text-end muted small">
        <div>Server: <span id="serverTime">-</span></div>
        <div>有效期限: <span id="expTime">-</span></div>
      </div>
    </div>

    <div class="row g-3">
      <div class="col-12 col-lg-6">
        <div class="cardx p-4 text-center">
          <img id="qrImg" alt="QR Code"/>
          <div class="mt-3 small muted">每 <span id="refreshSec">60</span> 秒自動刷新</div>
          <div id="err" class="alert alert-danger mt-3 d-none"></div>
        </div>
      </div>
      <div class="col-12 col-lg-6">
        <div class="cardx p-4">
          <h5 class="fw-semibold">注意事項</h5>
          <ul class="muted">
            <li>此頁建議在公司桌機/前台螢幕顯示。</li>
            <li>若手機掃不到：請確認 App 的 baseURL 指向此後端（例如 Mac 的區網 IP）。</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

    <script>
    const refreshEvery = 60;

    function buildUrl(path) {
      const params = new URLSearchParams(location.search);
      const key = params.get("key");
      if (key) return path + "?key=" + encodeURIComponent(key);
      return path;
    }

    function showErr(msg) {
      const el = document.getElementById("err");
      el.textContent = msg;
      el.classList.remove("d-none");
    }
    function clearErr() {
      document.getElementById("err").classList.add("d-none");
    }

    async function refreshQR() {
      clearErr();

      // ✅ 先更新圖片（不管 issue 成不成功，至少 QR 圖會出來）
      const pngBase = buildUrl("/qr/public/png");
      const joiner = pngBase.includes("?") ? "&" : "?";
      document.getElementById("qrImg").src = pngBase + joiner + "t=" + Date.now();

      // 再抓 JSON 來顯示 exp（失敗也不要影響顯示）
      try {
        const issueUrl = buildUrl("/qr/public/issue");
        const resp = await fetch(issueUrl);
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();

        document.getElementById("serverTime").textContent = new Date().toLocaleString();
        if (data.exp) {
          const exp = new Date(data.exp * 1000);
          document.getElementById("expTime").textContent = exp.toLocaleString();
        } else {
          document.getElementById("expTime").textContent = "-";
        }
      } catch (e) {
        // 只顯示警告，不阻擋 QR 圖
        showErr("提示：exp 資訊取得失敗（但 QR 仍可掃）：" + (e.message || String(e)));
      }
    }

    refreshQR();
    setInterval(refreshQR, refreshEvery * 1000);
  </script>

</body>
</html>
    """
    return HTMLResponse(html)



# 舊名稱相容（如果你以前用 /qr 顯示頁）
@app.get("/qr", response_class=HTMLResponse)
def qr_web_alias(key: str | None = Query(default=None)):
    return qr_web_page(key=key)
