"""
08_expense_tracker.py
=======================
Personal expense & income tracker backed by a local SQLite database.

This module is intentionally independent from the RAG pipeline (01-06) —
it does not touch documents, chunks, embeddings, or Chroma. It only
manages the user's own financial records: salary/income entries and
expense entries, plus simple summary queries used to power the
Streamlit UI (both a manual form and free-text chat parsing).

⚠️ MULTI-USER UPDATE: this module now requires real authentication
(username + password). Every income/expense row is tagged with the
user_id of whoever created it, and every read/write function requires
a user_id so one person can never see or edit another person's data.
See register_user() / authenticate_user() and the "Streamlit wiring"
notes at the bottom of this file for how to use this in the UI.

⚠️ Storage note: this uses a local SQLite file (EXPENSES_DB_PATH). On
Streamlit Cloud, the filesystem is ephemeral — data may be wiped on
redeploy/restart. Fine for personal/experimental use; if persistence
across deploys is ever needed later, swap this for a hosted DB
(e.g. Supabase/Postgres) behind the same function signatures below.
"""

import hashlib
import hmac
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime

EXPENSES_DB_PATH = "./expenses.db"

CATEGORIES = [
    "Food & Groceries", "Transportation", "Housing", "Utilities",
    "Entertainment", "Healthcare", "Shopping", "Debt Payment",
    "Savings", "Other",
]

# PBKDF2 settings for password hashing. 100k iterations is a reasonable
# default (OWASP recommends 100k+ for PBKDF2-SHA256 as of 2023).
_PBKDF2_ITERATIONS = 100_000
_PBKDF2_ALGO = "sha256"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

@contextmanager
def _get_connection():
    conn = sqlite3.connect(EXPENSES_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _column_exists(conn, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def init_db() -> None:
    """Create tables if they don't already exist. Safe to call every run.

    Also migrates older single-user databases: if income/expenses tables
    already exist WITHOUT a user_id column (from before this update), the
    column is added automatically. Any pre-existing rows from that old,
    unauthenticated version will have user_id = NULL, which means they
    won't show up for anyone anymore (since every query below filters by
    a real user_id) — this is intentional: that old data was never
    actually tied to a specific person, so it can't be safely attributed
    to any one account.
    """
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS income (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                amount REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'Salary',
                entry_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                note TEXT DEFAULT '',
                entry_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # --- migration for pre-existing single-user databases ---
        for table in ("income", "expenses"):
            if not _column_exists(conn, table, "user_id"):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_income_user ON income(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expenses_user ON expenses(user_id)")


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        _PBKDF2_ALGO, password.encode("utf-8"), salt.encode("utf-8"), _PBKDF2_ITERATIONS
    ).hex()


def register_user(username: str, password: str) -> int:
    """Create a new account. Raises ValueError on invalid input or a
    username that's already taken. Returns the new user's id."""
    username = username.strip()
    if not username:
        raise ValueError("اسم المستخدم مينفعش يبقى فاضي")
    if len(password) < 8:
        raise ValueError("الباسورد لازم يكون 8 حروف على الأقل")

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)

    with _get_connection() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if existing:
            raise ValueError("اسم المستخدم ده مستخدم بالفعل")

        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, salt, datetime.now().isoformat()),
        )
        return cursor.lastrowid


def authenticate_user(username: str, password: str) -> int | None:
    """Check username/password. Returns the user_id on success, or None
    on failure (wrong username OR wrong password — deliberately not
    distinguished, so a login form can't be used to enumerate which
    usernames exist)."""
    username = username.strip()
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT id, password_hash, salt FROM users WHERE username = ?", (username,)
        ).fetchone()

    if row is None:
        return None

    candidate_hash = _hash_password(password, row["salt"])
    # constant-time comparison to avoid leaking timing information
    if hmac.compare_digest(candidate_hash, row["password_hash"]):
        return row["id"]
    return None


# ---------------------------------------------------------------------------
# Writing records — every write is scoped to a user_id
# ---------------------------------------------------------------------------

def add_income(user_id: int, amount: float, source: str = "Salary",
               entry_date: str | None = None) -> int:
    entry_date = entry_date or date.today().isoformat()
    with _get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO income (user_id, amount, source, entry_date, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, amount, source, entry_date, datetime.now().isoformat()),
        )
        return cursor.lastrowid


def add_expense(user_id: int, amount: float, category: str = "Other", note: str = "",
                 entry_date: str | None = None) -> int:
    entry_date = entry_date or date.today().isoformat()
    category = category if category in CATEGORIES else "Other"
    with _get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO expenses (user_id, amount, category, note, entry_date, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, amount, category, note, entry_date, datetime.now().isoformat()),
        )
        return cursor.lastrowid


def delete_entry(user_id: int, table: str, entry_id: int) -> None:
    """Deletes entry_id from `table`, but ONLY if it belongs to user_id —
    this stops one user from deleting another user's row even if they
    somehow guess/enumerate the row id."""
    if table not in ("income", "expenses"):
        raise ValueError("table must be 'income' or 'expenses'")
    with _get_connection() as conn:
        conn.execute(
            f"DELETE FROM {table} WHERE id = ? AND user_id = ?", (entry_id, user_id)
        )


# ---------------------------------------------------------------------------
# Reading / summaries — every read is scoped to a user_id
# ---------------------------------------------------------------------------

def get_all_expenses(user_id: int, limit: int = 100):
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM expenses WHERE user_id = ? ORDER BY entry_date DESC, id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]


def get_all_income(user_id: int, limit: int = 100):
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM income WHERE user_id = ? ORDER BY entry_date DESC, id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]


def get_monthly_summary(user_id: int, year: int | None = None, month: int | None = None) -> dict:
    """Total income, total expenses, balance, and per-category breakdown
    for a given month (defaults to the current month) — for this user only."""
    today = date.today()
    year = year or today.year
    month = month or today.month
    month_prefix = f"{year:04d}-{month:02d}"

    with _get_connection() as conn:
        total_income = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS total FROM income WHERE user_id = ? AND entry_date LIKE ?",
            (user_id, f"{month_prefix}%"),
        ).fetchone()["total"]

        total_expenses = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS total FROM expenses WHERE user_id = ? AND entry_date LIKE ?",
            (user_id, f"{month_prefix}%"),
        ).fetchone()["total"]

        by_category = conn.execute(
            """SELECT category, COALESCE(SUM(amount), 0) AS total
               FROM expenses WHERE user_id = ? AND entry_date LIKE ?
               GROUP BY category ORDER BY total DESC""",
            (user_id, f"{month_prefix}%"),
        ).fetchall()

    return {
        "year": year, "month": month,
        "total_income": total_income, "total_expenses": total_expenses,
        "balance": total_income - total_expenses,
        "by_category": [dict(row) for row in by_category],
    }


# ---------------------------------------------------------------------------
# Free-text parsing (for chat-style input, e.g. "صرفت 200 جنيه أكل")
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS = {
    "Food & Groceries": ["أكل", "طعام", "سوبر ماركت", "بقالة", "مطعم", "food", "grocery", "groceries"],
    "Transportation": ["مواصلات", "بنزين", "تاكسي", "أوبر", "uber", "transport", "fuel", "taxi"],
    "Housing": ["إيجار", "شقة", "rent", "housing"],
    "Utilities": ["كهرباء", "مياه", "غاز", "فاتورة", "utilities", "bill"],
    "Entertainment": ["ترفيه", "سينما", "فيلم", "entertainment", "movie"],
    "Healthcare": ["دكتور", "دوا", "علاج", "صحة", "health", "medicine", "doctor"],
    "Shopping": ["تسوق", "شراء", "لبس", "shopping", "clothes"],
}

_INCOME_KEYWORDS = ["مرتب", "دخل", "salary", "income", "استلمت"]
_EXPENSE_KEYWORDS = ["صرفت", "دفعت", "اشتريت", "spent", "paid", "bought"]

_AMOUNT_PATTERN = re.compile(r"(\d+(?:[.,]\d+)?)")


def _guess_category(text: str) -> str:
    text_lower = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return "Other"


def parse_and_log_message(user_id: int, text: str) -> dict | None:
    """
    Try to interpret a free-text chat message as an income or expense entry
    for the given user_id. Returns a dict describing what was logged, or
    None if the message doesn't look like a financial log entry at all
    (caller should then fall through to the normal RAG chatbot flow).
    """
    amount_match = _AMOUNT_PATTERN.search(text.replace(",", ""))
    if not amount_match:
        return None

    amount = float(amount_match.group(1))
    text_lower = text.lower()

    is_income = any(keyword in text_lower for keyword in _INCOME_KEYWORDS)
    is_expense = any(keyword in text_lower for keyword in _EXPENSE_KEYWORDS)

    if is_income and not is_expense:
        entry_id = add_income(user_id, amount, source="Salary")
        return {"type": "income", "amount": amount, "id": entry_id}

    if is_expense:
        category = _guess_category(text)
        entry_id = add_expense(user_id, amount, category=category, note=text)
        return {"type": "expense", "amount": amount, "category": category, "id": entry_id}

    return None


def get_monthly_trend(user_id: int, num_months: int = 6) -> list:
    """
    Income vs. expenses totals for each of the last `num_months` months
    (oldest first), for this user only — for a trend chart. Months with
    no activity still appear in the result with zero totals, so the
    chart has no gaps.
    """
    today = date.today()
    months = []
    y, m = today.year, today.month
    for _ in range(num_months):
        months.append((y, m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    months.reverse()

    with _get_connection() as conn:
        rows = []
        for year, month in months:
            month_prefix = f"{year:04d}-{month:02d}"
            income_total = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) AS total FROM income WHERE user_id = ? AND entry_date LIKE ?",
                (user_id, f"{month_prefix}%"),
            ).fetchone()["total"]
            expense_total = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) AS total FROM expenses WHERE user_id = ? AND entry_date LIKE ?",
                (user_id, f"{month_prefix}%"),
            ).fetchone()["total"]
            rows.append({
                "month": month_prefix, "income": income_total, "expenses": expense_total,
            })
    return rows


# ---------------------------------------------------------------------------
# Plan-request detection (for chat-driven personalized savings/budget plans)
# ---------------------------------------------------------------------------

_PLAN_TRIGGER_WORDS = [
    "خطة", "بلان", "plan", "اعمل لي خطة", "اعملى خطة", "اعمللي خطة",
]
_GOAL_VERBS = ["احوش", "اوفر", "ادخر", "وفر", "اجمع", "هوفر", "هدخر"]
_DURATION_PATTERN = re.compile(
    r"(شهرين|شهور|أشهر|اشهر|شهر|اسبوعين|أسبوع|اسبوع|سنتين|سنة|يوم)"
)


def is_plan_request(text: str) -> bool:
    """
    Heuristic: an explicit plan/بلان word always counts. Otherwise, a
    savings/goal verb ("اوفر", "احوش"...) combined with a time reference
    ("شهرين", "خلال ٣ شهور"...) is treated as an implicit plan request —
    distinct from a generic how-to question like "إزاي أوفر فلوس؟" which
    has a goal verb but no timeframe, and should stay a normal RAG question.
    """
    text_lower = text.lower()
    if any(word in text_lower for word in _PLAN_TRIGGER_WORDS):
        return True
    has_goal_verb = any(verb in text_lower for verb in _GOAL_VERBS)
    has_duration = bool(_DURATION_PATTERN.search(text))
    return has_goal_verb and has_duration


# ---------------------------------------------------------------------------
# Streamlit wiring (reference — adapt inside your actual streamlit_app.py)
# ---------------------------------------------------------------------------
"""
Typical login gate at the top of streamlit_app.py:

    import streamlit as st
    from importlib import import_module
    tracker = import_module("08_expense_tracker")
    tracker.init_db()

    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    if st.session_state.user_id is None:
        tab_login, tab_register = st.tabs(["تسجيل الدخول", "حساب جديد"])

        with tab_login:
            username = st.text_input("اسم المستخدم", key="login_user")
            password = st.text_input("الباسورد", type="password", key="login_pass")
            if st.button("دخول"):
                user_id = tracker.authenticate_user(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("اسم المستخدم أو الباسورد غلط")

        with tab_register:
            new_username = st.text_input("اسم مستخدم جديد", key="reg_user")
            new_password = st.text_input("باسورد جديد", type="password", key="reg_pass")
            if st.button("إنشاء حساب"):
                try:
                    tracker.register_user(new_username, new_password)
                    st.success("اتعمل الحساب! دلوقتي سجّلي دخول.")
                except ValueError as e:
                    st.error(str(e))

        st.stop()  # nothing below this runs until user_id is set

    # --- from here on, every call passes st.session_state.user_id ---
    summary = tracker.get_monthly_summary(st.session_state.user_id)
    tracker.add_expense(st.session_state.user_id, amount=100, category="Food & Groceries")

    if st.button("تسجيل خروج"):
        st.session_state.user_id = None
        st.rerun()
"""


if __name__ == "__main__":
    init_db()
    print("Database initialized at", EXPENSES_DB_PATH)

    # --- demo: register two separate users and prove data isolation ---
    import contextlib

    for demo_user, demo_pass in [("nada_demo", "testpass123"), ("sara_demo", "testpass456")]:
        with contextlib.suppress(ValueError):
            register_user(demo_user, demo_pass)

    nada_id = authenticate_user("nada_demo", "testpass123")
    sara_id = authenticate_user("sara_demo", "testpass456")

    add_income(nada_id, 15000, source="Salary")
    add_expense(nada_id, 250, category="Food & Groceries", note="بقالة الأسبوع")

    add_income(sara_id, 9000, source="Salary")
    add_expense(sara_id, 700, category="Housing", note="إيجار")

    print("\nNada's summary (only her own data):")
    print(get_monthly_summary(nada_id))

    print("\nSara's summary (only her own data):")
    print(get_monthly_summary(sara_id))

    print("\nWrong password test:", authenticate_user("nada_demo", "wrongpassword"))
