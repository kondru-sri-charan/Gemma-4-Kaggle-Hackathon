"""Indian labor-law knowledge base for grounding Gemma 4's risk flags.

This module ships a small SQLite database of authoritative-ish labor-law
entries that Gemma 4 can consult via function calling. The entries are
researched from well-established statutes and common guidance, but they are
informational only -- they are not a substitute for actual legal advice, and
the UI surfaces this disclaimer everywhere the data is shown.

Schema
------
Each row has:
- topic: short slug keyed from the ``LaborLawTopic`` enum-ish constants
- state: optional state name ("Karnataka", "Maharashtra", ...) or "" for
  national rules
- title: short human label, e.g. "Notice period under Shops & Establishments"
- summary: worker-facing explanation in plain English
- statute_reference: which act / section the guidance comes from

Why SQLite
----------
We want a real, queryable backing store rather than a Python dict so the
function-calling loop feels substantive. SQLite is zero-config, shipped in
the Python stdlib, and deterministic across installs. The database is
built in-memory on first call and seeded from this module's constants, so
there is no external file to manage or commit.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from typing import Iterable


TOPIC_NOTICE_PERIOD = "notice_period"
TOPIC_BOND_CLAUSE = "bond_clause"
TOPIC_MINIMUM_WAGE = "minimum_wage"
TOPIC_OVERTIME = "overtime"
TOPIC_WORKING_HOURS = "working_hours"
TOPIC_PF = "pf_deductions"
TOPIC_ESI = "esi_deductions"
TOPIC_PENALTY = "penalty_or_liquidated_damages"
TOPIC_LEAVE = "leave_entitlement"
TOPIC_TERMINATION = "termination"

ALL_TOPICS = [
    TOPIC_NOTICE_PERIOD,
    TOPIC_BOND_CLAUSE,
    TOPIC_MINIMUM_WAGE,
    TOPIC_OVERTIME,
    TOPIC_WORKING_HOURS,
    TOPIC_PF,
    TOPIC_ESI,
    TOPIC_PENALTY,
    TOPIC_LEAVE,
    TOPIC_TERMINATION,
]


@dataclass(frozen=True)
class LaborLawEntry:
    topic: str
    state: str  # empty string for national
    title: str
    summary: str
    statute_reference: str

    def to_dict(self) -> dict[str, str]:
        return {
            "topic": self.topic,
            "state": self.state or "(national)",
            "title": self.title,
            "summary": self.summary,
            "statute_reference": self.statute_reference,
        }


# Note: all summaries are framed as general informational guidance, not legal
# advice. If you update an entry, keep the worker-facing tone.
_SEED_ENTRIES: list[LaborLawEntry] = [
    LaborLawEntry(
        topic=TOPIC_NOTICE_PERIOD,
        state="",
        title="Notice period under Industrial Employment (Standing Orders) Act",
        summary=(
            "For establishments covered by the Industrial Employment "
            "(Standing Orders) Act, 1946, the standard workman notice "
            "period is one month on either side for permanent workmen, "
            "and is typically shorter (14 days) during probation. Both "
            "parties are expected to give symmetric notice; grossly "
            "asymmetric notice (for example 60 days from employee, 15 "
            "from employer) is unusual and may be challenged before a "
            "labour court."
        ),
        statute_reference="Industrial Employment (Standing Orders) Act, 1946, model standing orders",
    ),
    LaborLawEntry(
        topic=TOPIC_NOTICE_PERIOD,
        state="Karnataka",
        title="Notice period under Karnataka Shops & Commercial Establishments Act",
        summary=(
            "Under the Karnataka Shops and Commercial Establishments "
            "Act, 1961, an employer who wishes to dispense with the "
            "services of an employee who has been in continuous "
            "employment for six months must give at least one month's "
            "notice in writing or pay wages in lieu. The same is "
            "generally expected from the employee side as a matter of "
            "practice, though some IT/ITeS contracts try to impose "
            "longer periods."
        ),
        statute_reference="Karnataka Shops & Commercial Establishments Act, 1961, Section 39",
    ),
    LaborLawEntry(
        topic=TOPIC_NOTICE_PERIOD,
        state="Maharashtra",
        title="Notice period under Maharashtra Shops & Establishments Act",
        summary=(
            "The Maharashtra Shops and Establishments (Regulation of "
            "Employment and Conditions of Service) Act, 2017 requires "
            "30 days' notice in writing from the employer for employees "
            "with three or more months of service. Equivalent notice "
            "is commonly asked of employees but is governed by the "
            "contract rather than statute."
        ),
        statute_reference="Maharashtra Shops & Establishments Act, 2017, Section 34",
    ),
    LaborLawEntry(
        topic=TOPIC_BOND_CLAUSE,
        state="",
        title="Enforceability of employment bonds in India",
        summary=(
            "Indian law (Indian Contract Act, 1872, Section 27) treats "
            "agreements in restraint of trade as void. Employment bonds "
            "are generally enforceable only to the extent of the "
            "employer's actual, provable financial loss -- typically the "
            "cost of specialised training given to the employee. A flat "
            "penalty amount unrelated to real loss (for example Rs "
            "2,50,000 regardless of when the employee leaves) is "
            "usually unenforceable. Courts have also required the bond "
            "amount to be reasonable relative to the employee's salary."
        ),
        statute_reference="Indian Contract Act, 1872, Section 27; Section 74 on penalties",
    ),
    LaborLawEntry(
        topic=TOPIC_PENALTY,
        state="",
        title="Penalties and liquidated damages in employment contracts",
        summary=(
            "Under the Indian Contract Act Section 74, a clause that "
            "imposes a fixed penalty for breach is enforceable only up "
            "to 'reasonable compensation' for the actual loss caused, "
            "regardless of the amount written in the contract. This "
            "means an employer cannot recover Rs 50,000 just because "
            "the contract says so; they would have to show they "
            "actually suffered that much loss. Wage deductions for "
            "non-statutory penalties are further restricted by the "
            "Payment of Wages Act, 1936."
        ),
        statute_reference="Indian Contract Act, 1872, Section 74; Payment of Wages Act, 1936, Section 7",
    ),
    LaborLawEntry(
        topic=TOPIC_MINIMUM_WAGE,
        state="",
        title="National minimum wage framework",
        summary=(
            "The Code on Wages, 2019 empowers the Central Government to "
            "fix a floor wage below which no state can set its minimum "
            "wage. As of the most recent revisions, the floor is set in "
            "the low hundreds of rupees per day. Actual enforceable "
            "minimum wages are set by each State Government for each "
            "scheduled employment (factory worker, domestic worker, "
            "retail, construction, etc.) and revised periodically."
        ),
        statute_reference="Code on Wages, 2019, Chapter II",
    ),
    LaborLawEntry(
        topic=TOPIC_MINIMUM_WAGE,
        state="Karnataka",
        title="Minimum wages in Karnataka",
        summary=(
            "Karnataka's Labour Department publishes scheduled-employment "
            "minimum wages that are revised every year in April. Rates "
            "vary by zone (Zone 1 -- Bengaluru, Zone 2, Zone 3) and by "
            "skill level (unskilled, semi-skilled, skilled, "
            "highly-skilled). A worker paid below the applicable "
            "minimum can file a complaint with the state Labour "
            "Commissioner."
        ),
        statute_reference="Karnataka Minimum Wages notifications under the Minimum Wages Act, 1948",
    ),
    LaborLawEntry(
        topic=TOPIC_OVERTIME,
        state="",
        title="Overtime pay under the Factories Act",
        summary=(
            "The Factories Act, 1948 (Section 59) requires that workers "
            "who work more than nine hours in any day or 48 hours in "
            "any week be paid at twice the ordinary rate of wages for "
            "the overtime hours. For shops and commercial "
            "establishments, similar double-rate overtime rules apply "
            "under state Shops & Establishments Acts. 'Overtime may be "
            "required' clauses without a specified overtime rate are a "
            "common red flag."
        ),
        statute_reference="Factories Act, 1948, Section 59; state Shops & Establishments Acts",
    ),
    LaborLawEntry(
        topic=TOPIC_WORKING_HOURS,
        state="",
        title="Maximum working hours",
        summary=(
            "The Factories Act, 1948 caps working hours at 9 per day "
            "and 48 per week for factory workers, with a compulsory "
            "rest interval of at least half an hour after five hours "
            "of work. State Shops & Establishments Acts set similar "
            "limits for retail and service workers, typically 48 hours "
            "per week. Contracts that require 12 hours a day, six "
            "days a week (72 hours a week) exceed these statutory "
            "limits and usually trigger overtime obligations."
        ),
        statute_reference="Factories Act, 1948, Sections 51-55; state Shops & Establishments Acts",
    ),
    LaborLawEntry(
        topic=TOPIC_PF,
        state="",
        title="Provident Fund deductions",
        summary=(
            "Under the Employees' Provident Funds and Miscellaneous "
            "Provisions Act, 1952, employees earning up to Rs 15,000 "
            "per month in a covered establishment (20+ employees) "
            "mandatorily contribute 12% of basic wages to PF, matched "
            "by the employer. Above the Rs 15,000 threshold, "
            "membership is voluntary for new members but typically "
            "continued for existing ones. PF is a savings for the "
            "worker, not a penalty, and appears as a standard line on "
            "the payslip."
        ),
        statute_reference="Employees' Provident Funds Act, 1952; EPF Scheme, 1952",
    ),
    LaborLawEntry(
        topic=TOPIC_ESI,
        state="",
        title="Employees' State Insurance deductions",
        summary=(
            "The Employees' State Insurance Act, 1948 covers workers "
            "earning up to Rs 21,000 per month (Rs 25,000 for persons "
            "with disabilities) in covered establishments. The "
            "employee contributes 0.75% of wages and the employer 3.25% "
            "as of the most recent revisions. ESI funds medical and "
            "disability benefits for the worker and dependents."
        ),
        statute_reference="Employees' State Insurance Act, 1948",
    ),
    LaborLawEntry(
        topic=TOPIC_LEAVE,
        state="",
        title="Leave entitlements -- baseline",
        summary=(
            "The Factories Act entitles a factory worker to one day of "
            "earned leave for every 20 days worked (roughly 15 days a "
            "year). State Shops & Establishments Acts prescribe "
            "comparable earned, sick, and casual leave. Offer letters "
            "that provide fewer than 12 paid leaves per year often sit "
            "below statutory entitlement, though the exact number "
            "depends on the Act the establishment is registered under."
        ),
        statute_reference="Factories Act, 1948, Section 79; state Shops & Establishments Acts",
    ),
    LaborLawEntry(
        topic=TOPIC_TERMINATION,
        state="",
        title="Termination and retrenchment protections",
        summary=(
            "Under the Industrial Disputes Act, 1947, a workman with "
            "more than one year of continuous service cannot be "
            "retrenched without one month's written notice (or wages in "
            "lieu) and retrenchment compensation of 15 days' average "
            "pay for each completed year of service. 'Termination' "
            "clauses that purport to waive these protections for "
            "covered workmen are unenforceable."
        ),
        statute_reference="Industrial Disputes Act, 1947, Sections 25F and 25N",
    ),
]


# The database is singleton per process. We guard with a lock so concurrent
# Streamlit requests don't race to initialize it.
_db_lock = threading.Lock()
_db: sqlite3.Connection | None = None


def _initialize_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute(
        "CREATE TABLE labor_law ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "  topic TEXT NOT NULL, "
        "  state TEXT NOT NULL DEFAULT '', "
        "  title TEXT NOT NULL, "
        "  summary TEXT NOT NULL, "
        "  statute_reference TEXT NOT NULL"
        ")"
    )
    conn.execute("CREATE INDEX idx_topic ON labor_law(topic)")
    conn.execute("CREATE INDEX idx_state ON labor_law(state)")
    conn.executemany(
        "INSERT INTO labor_law (topic, state, title, summary, statute_reference) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (e.topic, e.state, e.title, e.summary, e.statute_reference)
            for e in _SEED_ENTRIES
        ],
    )
    conn.commit()
    return conn


def _get_db() -> sqlite3.Connection:
    global _db
    with _db_lock:
        if _db is None:
            _db = _initialize_db()
        return _db


def reset_db_for_tests() -> None:
    """Test hook: clear the singleton so the next lookup re-seeds."""
    global _db
    with _db_lock:
        if _db is not None:
            _db.close()
        _db = None


def lookup_labor_law(topic: str, state: str | None = None) -> list[dict[str, str]]:
    """Fetch entries for a given topic, optionally filtered by state.

    Args:
        topic: One of ``ALL_TOPICS`` (free-form strings are tolerated but
            will probably return an empty list).
        state: Optional state name. If provided, we prefer state-specific
            rows and fall back to national rows. If omitted, only national
            rows are returned.

    Returns:
        A list of dicts ready to feed back to the model as tool output.
        Empty list if nothing matches.
    """
    normalized_topic = (topic or "").strip().lower()
    normalized_state = (state or "").strip().title()  # "karnataka" -> "Karnataka"

    conn = _get_db()
    cursor = conn.cursor()

    if normalized_state:
        cursor.execute(
            "SELECT topic, state, title, summary, statute_reference "
            "FROM labor_law "
            "WHERE topic = ? AND (state = ? OR state = '') "
            "ORDER BY CASE WHEN state = ? THEN 0 ELSE 1 END, id",
            (normalized_topic, normalized_state, normalized_state),
        )
    else:
        cursor.execute(
            "SELECT topic, state, title, summary, statute_reference "
            "FROM labor_law "
            "WHERE topic = ? AND state = '' "
            "ORDER BY id",
            (normalized_topic,),
        )

    rows = cursor.fetchall()
    return [
        {
            "topic": row[0],
            "state": row[1] or "(national)",
            "title": row[2],
            "summary": row[3],
            "statute_reference": row[4],
        }
        for row in rows
    ]


def list_all_topics() -> list[str]:
    """Expose the topic list so the tool schema can mention them to the model."""
    return list(ALL_TOPICS)


# === Tool schema for Ollama /api/chat ===

LOOKUP_TOOL_NAME = "lookup_labor_law"


def get_lookup_tool_schema() -> dict:
    """The JSON schema we ship to Ollama for the lookup_labor_law function."""
    return {
        "type": "function",
        "function": {
            "name": LOOKUP_TOOL_NAME,
            "description": (
                "Look up authoritative Indian labor-law guidance on a "
                "specific topic. You MUST call this before asserting any "
                "legal claim in a risk flag, so that the analysis is "
                "grounded in actual statutes rather than generic guidance. "
                "Call it once per topic you need to cite. Available topics: "
                + ", ".join(ALL_TOPICS)
                + "."
            ),
            "parameters": {
                "type": "object",
                "required": ["topic"],
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": (
                            "The topic slug to look up. Must be one of: "
                            + ", ".join(ALL_TOPICS)
                        ),
                        "enum": list(ALL_TOPICS),
                    },
                    "state": {
                        "type": "string",
                        "description": (
                            "Optional Indian state name (e.g. 'Karnataka', "
                            "'Maharashtra'). Include it when the worker is "
                            "known to be in a particular state; omit for "
                            "general national rules."
                        ),
                    },
                },
            },
        },
    }


def execute_tool_call(name: str, arguments: dict) -> list[dict[str, str]] | dict[str, str]:
    """Dispatch a tool call from the model to the matching Python function."""
    if name == LOOKUP_TOOL_NAME:
        topic = arguments.get("topic", "")
        state = arguments.get("state") or None
        rows = lookup_labor_law(topic, state)
        if not rows:
            return {
                "error": (
                    f"No labor-law entries found for topic={topic!r} "
                    f"state={state!r}. Known topics: {', '.join(ALL_TOPICS)}."
                )
            }
        return rows
    return {"error": f"Unknown tool: {name!r}"}


def format_tool_result_for_model(result: Iterable[dict] | dict) -> str:
    """Serialize a tool result to the short string Ollama expects as the
    ``content`` of a role=tool message."""
    import json

    return json.dumps(result, ensure_ascii=False)
