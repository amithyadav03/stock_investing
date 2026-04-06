"""
Idempotent migration script for the trading database.

This is the SINGLE SOURCE OF TRUTH for the database schema.

Behaviour:
- Fresh deployment: creates all tables with full schema from scratch.
- Existing deployment: safely adds only the missing columns (no data loss).

Run this ONCE after cloning the repo, and again after every schema change.
Usage: python scripts/migrate_db.py
"""

import sqlite3
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'db', 'trading.db'))

# ---------------------------------------------------------------------------
# SCHEMA DEFINITIONS — THE CANONICAL SOURCE OF TRUTH
# ---------------------------------------------------------------------------

# Full CREATE TABLE statement for trade_proposals.
# Always keep this 100% in sync with db/schema.py (the SQLAlchemy model).
CREATE_TRADE_PROPOSALS = """
CREATE TABLE IF NOT EXISTS trade_proposals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol            TEXT    NOT NULL,
    direction         TEXT    NOT NULL,
    timeframe         TEXT,

    -- Pricing at time of proposal
    proposed_price    REAL,
    stop_loss         REAL,
    take_profit       REAL,

    -- Dynamic portfolio sizing (added in Phase 1 upgrade)
    quantity          INTEGER NOT NULL DEFAULT 1,
    risk_percentage   REAL    NOT NULL DEFAULT 0.0,
    conviction_tier   TEXT    NOT NULL DEFAULT 'LOW',

    -- Human-readable rationale & lifecycle status
    rationale         TEXT,
    status            TEXT    NOT NULL DEFAULT 'PENDING',

    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at        DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_TRADE_EXECUTIONS = """
CREATE TABLE IF NOT EXISTS trade_executions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    proposal_id     INTEGER,
    symbol          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,

    entry_price     REAL,
    entry_time      DATETIME,

    exit_price      REAL,
    exit_time       DATETIME,
    exit_reason     TEXT,      -- 'STOP_LOSS', 'TAKE_PROFIT', 'EMERGENCY_EXIT'

    realized_pnl    REAL,

    FOREIGN KEY (proposal_id) REFERENCES trade_proposals(id)
);
"""

# ---------------------------------------------------------------------------
# INCREMENTAL COLUMN MIGRATIONS
# Defines EVERY column that should exist, with its DDL type string.
# The script will ALTER TABLE for any column missing from an existing table.
# ---------------------------------------------------------------------------

TRADE_PROPOSALS_COLUMNS = [
    ("id",               "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("symbol",           "TEXT NOT NULL"),
    ("direction",        "TEXT NOT NULL"),
    ("timeframe",        "TEXT"),
    ("proposed_price",   "REAL"),
    ("stop_loss",        "REAL"),
    ("take_profit",      "REAL"),
    ("quantity",         "INTEGER NOT NULL DEFAULT 1"),
    ("risk_percentage",  "REAL NOT NULL DEFAULT 0.0"),
    ("conviction_tier",  "TEXT NOT NULL DEFAULT 'LOW'"),
    ("rationale",        "TEXT"),
    ("status",           "TEXT NOT NULL DEFAULT 'PENDING'"),
    ("created_at",       "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ("updated_at",       "DATETIME DEFAULT CURRENT_TIMESTAMP"),
]

TRADE_EXECUTIONS_COLUMNS = [
    ("id",           "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("proposal_id",  "INTEGER"),
    ("symbol",       "TEXT NOT NULL"),
    ("direction",    "TEXT NOT NULL"),
    ("entry_price",  "REAL"),
    ("entry_time",   "DATETIME"),
    ("exit_price",   "REAL"),
    ("exit_time",    "DATETIME"),
    ("exit_reason",  "TEXT"),
    ("realized_pnl", "REAL"),
]


# ---------------------------------------------------------------------------
# MIGRATION RUNNER
# ---------------------------------------------------------------------------

def get_existing_columns(cur, table_name: str) -> list[str]:
    rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [row[1] for row in rows]


def table_exists(cur, table_name: str) -> bool:
    res = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    return res is not None


def migrate_table(cur, table_name: str, create_sql: str, expected_columns: list):
    """
    Create the table if it doesn't exist, otherwise add any missing columns.
    PRIMARY KEY columns cannot be added via ALTER TABLE — they must exist from
    creation, so this only soft-adds non-PK columns.
    """
    if not table_exists(cur, table_name):
        print(f"  [CREATE] Table '{table_name}' does not exist — creating with full schema.")
        cur.execute(create_sql)
        print(f"  ✅ '{table_name}' created.")
        return

    existing = get_existing_columns(cur, table_name)
    print(f"  [EXISTING] '{table_name}' found with columns: {existing}")

    added = []
    skipped = []
    for col_name, col_type in expected_columns:
        if col_name in existing:
            skipped.append(col_name)
            continue
        # PRIMARY KEY / AUTOINCREMENT columns cannot be added via ALTER TABLE
        if "PRIMARY KEY" in col_type.upper():
            skipped.append(col_name)
            continue
        try:
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
            added.append(col_name)
        except Exception as e:
            print(f"    ⚠️  Could not add column '{col_name}': {e}")

    if added:
        print(f"  ✅ Added columns to '{table_name}': {added}")
    else:
        print(f"  ✅ '{table_name}' schema is already up to date.")


def run_migration():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    print(f"\n{'=' * 55}")
    print(f"  DB Migration — {DB_PATH}")
    print(f"{'=' * 55}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")  # Better concurrency

    migrate_table(cur, "trade_proposals",  CREATE_TRADE_PROPOSALS,  TRADE_PROPOSALS_COLUMNS)
    print()
    migrate_table(cur, "trade_executions", CREATE_TRADE_EXECUTIONS, TRADE_EXECUTIONS_COLUMNS)

    conn.commit()
    conn.close()
    print(f"\n{'=' * 55}")
    print("  Migration complete. Database is fully up to date.")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    run_migration()
