"""
01_documents.py
================
Defines the raw document corpus for the personal finance RAG assistant.

This is general knowledge content (budgeting, credit, saving, debt, identity
theft, overdraft fees) — it is NOT tied to any specific user. Any person
asking a personal finance question can be served from this same corpus.

Two documents are intentionally outdated and conflict with a current
document on the same topic, to test that later pipeline stages correctly
prefer current information.
"""

DOCUMENTS = [
    {
        "document_id": 0, "title": "Making a Budget", "doc_type": "guide",
        "effective_date": "2023-03-13", "is_current": True,
        "text": (
            "Begin by recording every source of income you receive, including wages, "
            "freelance work, and any government benefits. Track your outgoing spending "
            "for at least one month, sorting it into categories such as housing, "
            "utilities, groceries, and entertainment. Map out bill due dates against "
            "your income schedule, since running short at month end is often a timing "
            "issue rather than a spending issue. Combine income, spending, and bill "
            "timing into one working budget worksheet."
        )
    },
    {
        "document_id": 1, "title": "Sticking to a Budget", "doc_type": "guide",
        "effective_date": "2023-03-13", "is_current": True,
        "text": (
            "Choose a simple tracking method you will actually maintain, such as a "
            "daily note or a weekly receipt review. Analyze spending habits regularly "
            "to identify categories that consistently run over, especially if impulse "
            "purchases are a known weakness. Set a concrete savings goal to stay "
            "motivated, and consider involving a trusted friend or family member for "
            "accountability, similar to a workout partner."
        )
    },
    {
        "document_id": 2, "title": "Needs Versus Wants", "doc_type": "guide",
        "effective_date": "2019-06-11", "is_current": True,
        "text": (
            "Obligations such as rent or mortgage payments, utilities, healthcare, and "
            "childcare are considered needs and generally cannot be reduced quickly. "
            "Everything outside of these core obligations falls under discretionary "
            "wants. When overall spending is too high, the wants category is usually "
            "the first place to cut back, since needs are far less flexible."
        )
    },
    {
        "document_id": 3, "title": "Tracking Daily Spending", "doc_type": "help page",
        "effective_date": "2023-03-13", "is_current": True,
        "text": (
            "Logging every purchase for a full month, even small ones like a coffee or "
            "a bus fare, reveals patterns that a single bank statement glance misses. "
            "If a full month feels overwhelming, start with just one week of receipts "
            "or bank transactions. Small recurring purchases are often the biggest "
            "hidden contributor to monthly overspending."
        )
    },
    {
        "document_id": 4, "title": "Automatic Transfers to Savings", "doc_type": "guide",
        "effective_date": "2024-01-01", "is_current": True,
        "text": (
            "Scheduling a recurring transfer from a checking account to a savings "
            "account, timed to coincide with payday, removes the temptation to spend "
            "funds meant for saving. Financial institutions often allow a low minimum "
            "transfer amount, which can be increased gradually. Confirm with the bank "
            "whether any fees apply to recurring transfers before enrolling."
        )
    },
    {
        "document_id": 5, "title": "Credit Score Basics", "doc_type": "FAQ",
        "effective_date": "2023-08-28", "is_current": True,
        "text": (
            "A credit score is a numerical prediction of how likely a borrower is to "
            "repay a loan on time, calculated from information in the credit report "
            "such as payment history, total debt, length of credit history, and recent "
            "credit applications. Different scoring models can produce different "
            "numbers for the same person, so an individual does not have just one score."
        )
    },
    {
        "document_id": 6, "title": "Disputing a Credit Report Error", "doc_type": "procedure",
        "effective_date": "2024-12-12", "is_current": True,
        "text": (
            "A consumer who finds incorrect information on a credit report should file "
            "a written dispute directly with the reporting company, attaching any "
            "supporting documents. The reporting company must investigate and forward "
            "the dispute to whoever supplied the information. If the investigation "
            "confirms an error, the information must be corrected or removed."
        )
    },
    {
        "document_id": 7, "title": "Obtaining Free Credit Reports", "doc_type": "procedure",
        "effective_date": "2025-09-08", "is_current": True,
        "text": (
            "Consumers are entitled to view their credit report from each of the three "
            "nationwide reporting companies at no charge through the authorized federal "
            "portal, with weekly access currently available online. Requests can also "
            "be made by phone or mail. Third-party sites advertising free reports "
            "should be treated with caution, as many require a paid subscription."
        )
    },
    {
        "document_id": 8, "title": "Building and Maintaining a Strong Credit Score",
        "doc_type": "guide", "effective_date": "2024-12-18", "is_current": True,
        "text": (
            "Consistently paying bills on time is treated as the largest single factor "
            "in most credit scoring models. Keeping revolving balances well below the "
            "total available limit, generally under thirty percent, also supports a "
            "healthy score. Carrying a balance is not required — paying in full each "
            "cycle both protects the score and avoids unnecessary interest charges."
        )
    },
    {
        "document_id": 9, "title": "Where Consumers Can Check Their Score",
        "doc_type": "FAQ", "effective_date": "2025-01-29", "is_current": True,
        "text": (
            "Many card issuers, lenders, and nonprofit counseling agencies provide a "
            "credit score at no cost. Some of these are labeled educational scores, "
            "which closely approximate but do not always exactly match the score a "
            "lender would use for a specific loan decision."
        )
    },
    {
        "document_id": 10, "title": "Starting an Emergency Fund", "doc_type": "guide",
        "effective_date": "2022-03-01", "is_current": True,
        "text": (
            "Setting aside even a small amount for unexpected costs, such as a car "
            "repair or a medical bill, reduces the need to rely on high-interest credit "
            "when something goes wrong. A common first target is enough to cover one "
            "month of essential expenses, built up gradually through automatic "
            "transfers rather than one large deposit."
        )
    },
    {
        "document_id": 11, "title": "Limits on Debt Collector Contact", "doc_type": "policy",
        "effective_date": "2021-11-30", "is_current": True,
        "text": (
            "Debt collectors are restricted from contacting a consumer before eight in "
            "the morning or after nine at night without permission, and must stop "
            "contacting an individual at their workplace if told the employer prohibits "
            "such calls. Consumers may request communication in writing only, and "
            "collectors must honor that request going forward."
        )
    },
    {
        "document_id": 12, "title": "Current Overdraft Fee Policy", "doc_type": "policy",
        "effective_date": "2025-02-01", "is_current": True,
        "text": (
            "Many financial institutions have reduced or eliminated standard overdraft "
            "fees in recent years, with a number of large banks now charging no fee at "
            "all for a covered overdraft, or capping the number of fees charged per "
            "day. Consumers should review their specific account's current disclosure, "
            "since practices vary significantly between institutions."
        )
    },
    {
        "document_id": 13, "title": "Recognizing Identity Theft", "doc_type": "procedure",
        "effective_date": "2024-05-10", "is_current": True,
        "text": (
            "Unexplained charges, unfamiliar accounts on a credit report, or bills for "
            "services never used can indicate that someone else is using a consumer's "
            "personal information without permission. Filing a report at the federal "
            "recovery portal generates a personal recovery plan and the documentation "
            "needed to dispute fraudulent accounts with creditors."
        )
    },
    {
        "document_id": 14, "title": "Aligning Bill Due Dates With Income", "doc_type": "guide",
        "effective_date": "2023-03-13", "is_current": True,
        "text": (
            "A household that consistently runs low on funds before the next payday "
            "may be experiencing a timing mismatch rather than a true shortfall. Listing "
            "every bill's due date alongside expected income dates can reveal specific "
            "weeks that require extra caution, even when the total monthly numbers "
            "appear balanced."
        )
    },
    {
        "document_id": 15, "title": "Cutting Food and Grocery Spending", "doc_type": "guide",
        "effective_date": "2024-06-01", "is_current": True,
        "text": (
            "Planning meals around a weekly list reduces impulse purchases at the "
            "grocery store, which is one of the most common sources of food overspending. "
            "Buying store-brand staples instead of name brands, cooking larger batches to "
            "reduce takeout frequency, and checking unit prices rather than package price "
            "can meaningfully lower a monthly food bill without a major lifestyle change."
        )
    },
    {
        "document_id": 16, "title": "Reducing Transportation and Commuting Costs",
        "doc_type": "guide", "effective_date": "2024-06-01", "is_current": True,
        "text": (
            "Comparing the true cost of driving — fuel, parking, and maintenance — "
            "against public transit or carpooling for a regular commute often reveals "
            "meaningful monthly savings. Combining errands into fewer trips and keeping "
            "a vehicle properly maintained also reduces fuel and repair costs over time."
        )
    },
    # --- Outdated documents that conflict with current ones ---
    {
        "document_id": 17, "title": "Former Annual Credit Report Access Rule",
        "doc_type": "old notice", "effective_date": "2019-01-01", "is_current": False,
        "text": (
            "Under the rule previously in effect, consumers could request a free copy "
            "of their credit report only once every twelve months from each reporting "
            "company, submitted by mail or through the annual request portal. This "
            "notice is retained for historical reference only and should not be used "
            "to answer current questions about free report access."
        )
    },
    {
        "document_id": 18, "title": "Old Standard Overdraft Fee Notice",
        "doc_type": "old notice", "effective_date": "2019-05-01", "is_current": False,
        "text": (
            "Prior to recent industry changes, a standard overdraft fee of thirty-five "
            "dollars per transaction was common across most major banks, often charged "
            "multiple times per day with no daily cap. This pricing information is "
            "outdated and should not be used to answer questions about current "
            "overdraft charges."
        )
    },
]


if __name__ == "__main__":
    print(f"Loaded {len(DOCUMENTS)} documents")
    print(f"Current: {sum(1 for d in DOCUMENTS if d['is_current'])} | "
          f"Outdated: {sum(1 for d in DOCUMENTS if not d['is_current'])}")
