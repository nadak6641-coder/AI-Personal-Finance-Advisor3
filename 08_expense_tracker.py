"""
07_expense_tracker.py
=======================
Personal expense & income tracker backed by a local SQLite database.

This module is intentionally independent from the RAG pipeline (01-06) —
it does not touch documents, chunks, embeddings, or Chroma. It only
manages the user's own financial records: salary/income entries and
expense entries, plus simple summary queries used to power the
Streamlit UI (both a manual form and free-text chat parsing).

⚠️ Storage note: this uses a local SQLite file (EXPENSES_DB_PATH). On
Streamlit Cloud, the filesystem is ephemeral — data may be wiped on
redeploy/restart. Fine for personal/experimental use; if persistence
across deploys is ever needed later, swap this for a hosted DB
(e.g. Supabase/Postgres) behind the same function signatures below.
"""

import re
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime

EXPENSES_DB_PATH = "./expenses.db"

CATEGORIES = [
    "Food & Groceries", "Transportation", "Housing", "Utilities",
    "Entertainment", "Healthcare", "Shopping", "Debt Payment",
    "Savings", "Other",
]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

@contextmanager
def _get_connection():
    conn = sqlite3.connect(EXPENSES_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't already exist. Safe to call every run."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS income (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                amount REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'Salary',
                entry_date TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                note TEXT DEFAULT '',
                entry_date TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


# ---------------------------------------------------------------------------
# Writing records
# ---------------------------------------------------------------------------

def add_income(amount: float, source: str = "Salary", entry_date: str | None = None) -> int:
    entry_date = entry_date or date.today().isoformat()
    with _get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO income (amount, source, entry_date, created_at) VALUES (?, ?, ?, ?)",
            (amount, source, entry_date, datetime.now().isoformat()),
        )
        return cursor.lastrowid


def add_expense(amount: float, category: str = "Other", note: str = "",
                 entry_date: str | None = None) -> int:
    entry_date = entry_date or date.today().isoformat()
    category = category if category in CATEGORIES else "Other"
    with _get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO expenses (amount, category, note, entry_date, created_at) VALUES (?, ?, ?, ?, ?)",
            (amount, category, note, entry_date, datetime.now().isoformat()),
        )
        return cursor.lastrowid


def delete_entry(table: str, entry_id: int) -> None:
    if table not in ("income", "expenses"):
        raise ValueError("table must be 'income' or 'expenses'")
    with _get_connection() as conn:
        conn.execute(f"DELETE FROM {table} WHERE id = ?", (entry_id,))


# ---------------------------------------------------------------------------
# Reading / summaries
# ---------------------------------------------------------------------------

def get_all_expenses(limit: int = 100):
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM expenses ORDER BY entry_date DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_all_income(limit: int = 100):
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM income ORDER BY entry_date DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_monthly_summary(year: int | None = None, month: int | None = None) -> dict:
    """Total income, total expenses, balance, and per-category breakdown
    for a given month (defaults to the current month)."""
    today = date.today()
    year = year or today.year
    month = month or today.month
    month_prefix = f"{year:04d}-{month:02d}"

    with _get_connection() as conn:
        total_income = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS total FROM income WHERE entry_date LIKE ?",
            (f"{month_prefix}%",),
        ).fetchone()["total"]

        total_expenses = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) AS total FROM expenses WHERE entry_date LIKE ?",
            (f"{month_prefix}%",),
        ).fetchone()["total"]

        by_category = conn.execute(
            """SELECT category, COALESCE(SUM(amount), 0) AS total
               FROM expenses WHERE entry_date LIKE ?
               GROUP BY category ORDER BY total DESC""",
            (f"{month_prefix}%",),
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


def parse_and_log_message(text: str) -> dict | None:
    """
    Try to interpret a free-text chat message as an income or expense entry.
    Returns a dict describing what was logged, or None if the message
    doesn't look like a financial log entry at all (caller should then
    fall through to the normal RAG chatbot flow).
    """
    amount_match = _AMOUNT_PATTERN.search(text.replace(",", ""))
    if not amount_match:
        return None

    amount = float(amount_match.group(1))
    text_lower = text.lower()

    is_income = any(keyword in text_lower for keyword in _INCOME_KEYWORDS)
    is_expense = any(keyword in text_lower for keyword in _EXPENSE_KEYWORDS)

    if is_income and not is_expense:
        entry_id = add_income(amount, source="Salary")
        return {"type": "income", "amount": amount, "id": entry_id}

    if is_expense:
        category = _guess_category(text)
        entry_id = add_expense(amount, category=category, note=text)
        return {"type": "expense", "amount": amount, "category": category, "id": entry_id}

    return None


def get_monthly_trend(num_months: int = 6) -> list:
    """
    Income vs. expenses totals for each of the last `num_months` months
    (oldest first), for a trend chart. Months with no activity still
    appear in the result with zero totals, so the chart has no gaps.
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
                "SELECT COALESCE(SUM(amount), 0) AS total FROM income WHERE entry_date LIKE ?",
                (f"{month_prefix}%",),
            ).fetchone()["total"]
            expense_total = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) AS total FROM expenses WHERE entry_date LIKE ?",
                (f"{month_prefix}%",),
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


if __name__ == "__main__":
    init_db()
    print("Database initialized at", EXPENSES_DB_PATH)

    add_income(15000, source="Salary")
    add_expense(250, category="Food & Groceries", note="بقالة الأسبوع")
    add_expense(100, category="Transportation", note="بنزين")

    summary = get_monthly_summary()
    print("\nThis month's summary:")
    print(f"  Income:   {summary['total_income']}")
    print(f"  Expenses: {summary['total_expenses']}")
    print(f"  Balance:  {summary['balance']}")
    print("  By category:")
    for row in summary["by_category"]:
        print(f"    {row['category']}: {row['total']}")

    print("\nParsing test messages:")
    for msg in ["صرفت 200 جنيه على الأكل", "استلمت مرتب 15000", "إزاي أعمل ميزانية؟"]:
        result = parse_and_log_message(msg)
        print(f"  '{msg}' -> {result}")
