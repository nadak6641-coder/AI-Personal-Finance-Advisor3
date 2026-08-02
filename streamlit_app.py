"""
streamlit_app.py
==================
The deployed assistant. Any user can type a personal finance question —
this is general knowledge retrieval, not tied to any individual's data.

Also includes a personal expense & income tracker (SQLite-backed, local
use, no login) — see 08_expense_tracker.py. Financial log entries can be
typed directly into the chat box (e.g. "صرفت 200 جنيه أكل") or entered
through the dedicated form in the "Expense Tracker" tab.

Run locally:
    streamlit run streamlit_app.py
(requires ./chroma_db to already exist — run 05_create_chroma_store.py first)
"""
from importlib import import_module
from datetime import date

import pandas as pd
import streamlit as st

retrieve_context = import_module("06_retrieve_context")
prompting = import_module("07_prompting")
expense_tracker = import_module("08_expense_tracker")

# --- API key resolution: environment variable first, then Streamlit secrets ---
try:
    if not prompting.OPENROUTER_API_KEY:
        prompting.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        prompting.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", prompting.OPENROUTER_MODEL)
except Exception:
    pass

# --- Make sure the expenses database exists before anything touches it ---
expense_tracker.init_db()

st.set_page_config(page_title="Personal Finance Assistant", page_icon="💰")
st.title("💰 Personal Finance Assistant")

chat_tab, tracker_tab = st.tabs(["🤖 المساعد", "📊 تتبع المصاريف"])

# ---------------------------------------------------------------------------
# Tab 1: Chat assistant (RAG) + free-text expense/income logging
# ---------------------------------------------------------------------------
with chat_tab:
    st.caption(
        "اسأل سؤال عن الميزانية، الكريدت، الادخار، أو الديون — أو اكتب مصروف/دخل "
        "مباشرة زي \"صرفت 200 جنيه أكل\" أو \"استلمت مرتب 15000\"."
    )

    query = st.text_input("سؤالك أو تسجيل مصروف/دخل", placeholder="e.g. صرفت 200 جنيه على الأكل")

    if st.button("Ask", type="primary") and query.strip():
        # 1) Try to interpret the message as a financial log entry first.
        log_result = expense_tracker.parse_and_log_message(query)

        if log_result is not None:
            if log_result["type"] == "income":
                st.success(f"✅ تم تسجيل دخل بقيمة {log_result['amount']:.0f} جنيه.")
            else:
                st.success(
                    f"✅ تم تسجيل مصروف بقيمة {log_result['amount']:.0f} جنيه "
                    f"تحت بند «{log_result['category']}»."
                )
            st.caption("تقدر تراجع أو تعدل التسجيل من تاب «تتبع المصاريف».")
        elif expense_tracker.is_plan_request(query):
            # 2) A personalized plan request — combine the user's actual
            #    tracked numbers with general RAG guidance.
            summary = expense_tracker.get_monthly_summary()
            with st.spinner("Retrieving relevant sources..."):
                package = retrieve_context.build_context_package(query)
            with st.spinner("Building your plan..."):
                plan = prompting.generate_plan(query, summary, package["context_text"])
            st.subheader("خطتك")
            st.write(plan)
            if package["num_sources"] > 0:
                with st.expander(f"View {package['num_sources']} cited source(s)"):
                    for row in package["selected"]:
                        status = "🟢 Current" if row["is_current"] else "🟡 Outdated"
                        st.markdown(f"**{row['title']}** — {row['effective_date']} — {status}")
                        st.write(row["chunk_text"])
                        st.divider()

        else:
            # 3) Not a log entry or plan request — the normal RAG chatbot flow.
            with st.spinner("Retrieving relevant sources..."):
                package = retrieve_context.build_context_package(query)

            if package["num_sources"] == 0:
                st.warning("No sufficiently relevant source was found in the knowledge base for this question.")
            else:
                with st.spinner("Generating answer..."):
                    answer = prompting.generate_answer(query, package["context_text"])
                st.subheader("Answer")
                st.write(answer)
                with st.expander(f"View {package['num_sources']} cited source(s)"):
                    for row in package["selected"]:
                        status = "🟢 Current" if row["is_current"] else "🟡 Outdated"
                        st.markdown(f"**{row['title']}** — {row['effective_date']} — {status}")
                        st.write(row["chunk_text"])
                        st.divider()

# ---------------------------------------------------------------------------
# Tab 2: Manual expense tracker (form + monthly summary)
# ---------------------------------------------------------------------------
with tracker_tab:
    st.subheader("إضافة قيد يدوي")

    entry_kind = st.radio("النوع", ["مصروف", "دخل"], horizontal=True)

    with st.form("manual_entry_form", clear_on_submit=True):
        amount = st.number_input("المبلغ (جنيه)", min_value=0.0, step=10.0)
        entry_date = st.date_input("التاريخ", value=date.today())

        if entry_kind == "مصروف":
            category = st.selectbox("البند", expense_tracker.CATEGORIES)
            note = st.text_input("ملاحظة (اختياري)")
        else:
            source = st.text_input("مصدر الدخل", value="Salary")

        submitted = st.form_submit_button("إضافة")

    if submitted and amount > 0:
        if entry_kind == "مصروف":
            expense_tracker.add_expense(amount, category=category, note=note, entry_date=entry_date.isoformat())
            st.success(f"✅ تم تسجيل مصروف {amount:.0f} جنيه تحت «{category}».")
        else:
            expense_tracker.add_income(amount, source=source, entry_date=entry_date.isoformat())
            st.success(f"✅ تم تسجيل دخل {amount:.0f} جنيه من «{source}».")
        st.rerun()

    st.divider()
    st.subheader("ملخص الشهر الحالي")

    summary = expense_tracker.get_monthly_summary()
    col1, col2, col3 = st.columns(3)
    col1.metric("الدخل", f"{summary['total_income']:.0f} ج.م")
    col2.metric("المصاريف", f"{summary['total_expenses']:.0f} ج.م")
    col3.metric("الرصيد", f"{summary['balance']:.0f} ج.م")

    if summary["by_category"]:
        st.markdown("**المصاريف حسب البند:**")
        for row in summary["by_category"]:
            st.write(f"- {row['category']}: {row['total']:.0f} ج.م")

        category_df = pd.DataFrame(summary["by_category"]).set_index("category")
        st.bar_chart(category_df["total"])

    st.divider()
    st.subheader("الاتجاه خلال آخر 6 شهور")

    trend = expense_tracker.get_monthly_trend(6)
    trend_df = pd.DataFrame(trend).set_index("month")
    trend_df = trend_df.rename(columns={"income": "الدخل", "expenses": "المصاريف"})
    st.line_chart(trend_df)

    st.divider()
    st.subheader("آخر القيود")

    recent_expenses = expense_tracker.get_all_expenses(limit=10)
    recent_income = expense_tracker.get_all_income(limit=10)

    with st.expander(f"آخر المصاريف ({len(recent_expenses)})"):
        for row in recent_expenses:
            st.write(f"{row['entry_date']} — {row['category']} — {row['amount']:.0f} ج.م — {row['note']}")

    with st.expander(f"آخر الدخل ({len(recent_income)})"):
        for row in recent_income:
            st.write(f"{row['entry_date']} — {row['source']} — {row['amount']:.0f} ج.م")
