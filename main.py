import os
import re
import json
import warnings
import logging
from typing import List, Dict, Any, Optional, Tuple
import yaml
import uuid
from datetime import datetime, timezone, timedelta
import math
from threading import Lock

warnings.filterwarnings("ignore", category=FutureWarning)

# =====================
# Logging Configuration
# =====================
# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Application starting...")

# Environment
from dotenv import load_dotenv
load_dotenv(override=True)

# DB
import pymysql

# Embeddings / Vector store
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# LLM + Agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_core.tools import Tool

from langgraph.prebuilt import create_react_agent as lg_create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage

# UI
import gradio as gr

# Import external modules
from styles.app_styles import get_css_for_density

# =====================
# Environment variables
# =====================
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "iics")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")

# Ensure data directory exists
os.makedirs("data", exist_ok=True)
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")

# =====================
# Embedding Config (YAML)
# =====================
# Ensure config directory exists
os.makedirs("config", exist_ok=True)
EMBED_CONFIG_PATH = os.getenv("EMBED_CONFIG_PATH", "config/embedding_config.yaml")
EMBED_CONFIG: Dict[str, Any] = {
    "descriptions": {},
    "column_descriptions": {},
    "embed_values": {},
    "skip_values": [],
    "skip_tables": [],
    "include_tables": [],
}

def load_embedding_config(path: str = EMBED_CONFIG_PATH) -> Dict[str, Any]:
    """Load embedding configuration from YAML file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            # Normalize keys
            data.setdefault("descriptions", {})
            data.setdefault("column_descriptions", {})
            data.setdefault("embed_values", {})
            data.setdefault("skip_values", [])
            data.setdefault("skip_tables", [])
            data.setdefault("include_tables", [])
            logger.info(f"Loaded embedding config from {path}")
            return data
    except FileNotFoundError:
        logger.warning(f"Embedding config file not found: {path}, using defaults")
        return {"descriptions": {}, "column_descriptions": {}, "embed_values": {}, "skip_values": [], "skip_tables": [], "include_tables": []}
    except Exception as e:
        logger.error(f"Failed to load embedding config: {e}")
        return {"descriptions": {}, "column_descriptions": {}, "embed_values": {}, "skip_values": [], "skip_tables": [], "include_tables": []}

EMBED_CONFIG = load_embedding_config()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

# =====================
# DB Connection Helpers
# =====================

def get_connection(autocommit: bool = True) -> pymysql.connections.Connection:
    """Create a MySQL database connection with proper error handling and logging."""
    try:
        logger.debug(f"Connecting to MySQL: host={MYSQL_HOST}, db={MYSQL_DATABASE}")
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            autocommit=autocommit,
            cursorclass=pymysql.cursors.Cursor,
        )
        return conn
    except Exception as e:
        logger.error(f"MySQL connection failed: {e}")
        raise

# ======================================
# ChromaDB and Embedding Model (persistent)
# ======================================
# Suppress progress bars from sentence transformers to reduce console noise
import os as _os
_os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model = SentenceTransformer("all-MiniLM-L6-v2")
# Override encode to disable progress bars
_original_encode = model.encode
def _encode_no_progress(*args, **kwargs):
    kwargs['show_progress_bar'] = False
    return _original_encode(*args, **kwargs)
model.encode = _encode_no_progress

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

# Collections
TABLES_COL = chroma_client.get_or_create_collection("mssql_table_embeddings")
COLS_COL = chroma_client.get_or_create_collection("mssql_column_embeddings")
VALS_COL = chroma_client.get_or_create_collection("mssql_value_embeddings")
RELS_COL = chroma_client.get_or_create_collection("mssql_relationship_embeddings")

# =====================
# Schema Introspection
# =====================

def fetch_schema() -> Dict[str, Any]:
    """Load tables, columns, types, and foreign keys from the configured MySQL database only."""
    logger.info(f"Fetching schema from MySQL database '{MYSQL_DATABASE}'")
    with get_connection() as conn:
        cur = conn.cursor()
        # Tables
        cur.execute(
            """
            SELECT TABLE_SCHEMA, TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE='BASE TABLE' AND TABLE_SCHEMA = %s
            ORDER BY TABLE_NAME
            """,
            (MYSQL_DATABASE,),
        )
        tables = [(row[0], row[1]) for row in cur.fetchall()]

        # Columns
        cur.execute(
            """
            SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """,
            (MYSQL_DATABASE,),
        )
        columns = [(r[0], r[1], r[2], r[3]) for r in cur.fetchall()]

        # Relationships (FKs)
        cur.execute(
            """
            SELECT 
                constraint_name,
                table_schema,
                table_name,
                column_name,
                referenced_table_schema,
                referenced_table_name,
                referenced_column_name
            FROM information_schema.key_column_usage
            WHERE referenced_table_name IS NOT NULL
              AND table_schema = %s
              AND referenced_table_schema = %s
            ORDER BY table_schema, table_name
            """,
            (MYSQL_DATABASE, MYSQL_DATABASE),
        )
        fks = [
            {
                "fk": f"{r[1]}.{r[2]}({r[3]})",
                "pk": f"{r[4]}.{r[5]}({r[6]})",
            }
            for r in cur.fetchall()
        ]

    schema = {
        "tables": tables,
        "columns": columns,
        "relationships": fks,
    }
    return schema


def _table_allowed(sch: str, tbl: str, include_tables: set, skip_tables: set) -> bool:
    """Return True if table should be embedded based on include/skip lists."""
    if tbl in skip_tables or f"{sch}.{tbl}" in skip_tables:
        return False
    if include_tables and not (tbl in include_tables or f"{sch}.{tbl}" in include_tables):
        return False
    return True


def _parse_table_ref(ref: str) -> Tuple[str, str]:
    """Parse 'schema.table(col)' into (schema, table)."""
    try:
        sch, rest = ref.split(".", 1)
        tbl = rest.split("(", 1)[0]
        return sch, tbl
    except Exception:
        return "", ""


def upsert_schema_into_chroma(schema: Dict[str, Any]) -> None:
    """Persist schema metadata into Chroma honoring config (descriptions, skip tables)."""
    skip_tables = set(EMBED_CONFIG.get("skip_tables", []) or [])
    include_tables = set(EMBED_CONFIG.get("include_tables", []) or [])
    descriptions = EMBED_CONFIG.get("descriptions", {}) or {}
    column_descriptions = EMBED_CONFIG.get("column_descriptions", {}) or {}

    # Tables (apply description and skip tables if configured)
    table_docs = []
    table_ids = []
    table_embs = []
    for sch, tbl in schema["tables"]:
        if not _table_allowed(sch, tbl, include_tables, skip_tables):
            continue
        desc = descriptions.get(tbl) or descriptions.get(f"{sch}.{tbl}")
        doc = f"Table: {sch}.{tbl}" + (f" - {desc}" if desc else "")
        table_docs.append(doc)
        table_ids.append(f"t::{sch}.{tbl}")
        table_embs.append(model.encode(doc).tolist())

    # Columns (skip those under skipped tables)
    col_docs = []
    col_ids = []
    col_embs = []
    for sch, tbl, col, dtype in schema["columns"]:
        if not _table_allowed(sch, tbl, include_tables, skip_tables):
            continue
        desc = column_descriptions.get(f"{sch}.{tbl}.{col}") or column_descriptions.get(f"{tbl}.{col}")
        doc = f"Column: {sch}.{tbl}.{col} :: {dtype}" + (f" - {desc}" if desc else "")
        col_docs.append(doc)
        col_ids.append(f"c::{sch}.{tbl}.{col}")
        col_embs.append(model.encode(doc).tolist())

    # Relationships (only for allowed tables)
    filtered_rels = []
    for r in schema["relationships"]:
        fk_sch, fk_tbl = _parse_table_ref(r["fk"])
        pk_sch, pk_tbl = _parse_table_ref(r["pk"])
        if not fk_sch or not pk_sch:
            continue
        if _table_allowed(fk_sch, fk_tbl, include_tables, skip_tables) and _table_allowed(pk_sch, pk_tbl, include_tables, skip_tables):
            filtered_rels.append(r)
    rel_docs = [f"FK: {r['fk']} -> {r['pk']}" for r in filtered_rels]
    rel_ids = [f"r::{idx}" for idx, _ in enumerate(rel_docs)]
    rel_embs = [model.encode(doc).tolist() for doc in rel_docs]

    # Upserts
    if table_docs:
        TABLES_COL.upsert(ids=table_ids, embeddings=table_embs, documents=table_docs)
    if col_docs:
        COLS_COL.upsert(ids=col_ids, embeddings=col_embs, documents=col_docs)
    if rel_docs:
        RELS_COL.upsert(ids=rel_ids, embeddings=rel_embs, documents=rel_docs)


def sample_distinct_values(schema: Dict[str, Any], max_per_column: int = 50) -> None:
    """For text-like columns, store up to N distinct sampled values in Chroma to reduce live hits, guided by config."""
    text_like = {"char", "nchar", "varchar", "nvarchar", "text", "ntext", "uniqueidentifier"}
    skip_tables = set(EMBED_CONFIG.get("skip_tables", []) or [])
    include_tables = set(EMBED_CONFIG.get("include_tables", []) or [])
    embed_values = EMBED_CONFIG.get("embed_values", {}) or {}
    skip_values = set((EMBED_CONFIG.get("skip_values", []) or []))

    logger.info("Starting distinct values sampling for text columns")
    sampled_count = 0
    error_count = 0
    
    with get_connection() as conn:
        cur = conn.cursor()
        for sch, tbl, col, dtype in schema["columns"]:
            if sch != MYSQL_DATABASE:
                continue
            if not _table_allowed(sch, tbl, include_tables, skip_tables):
                continue
            if (dtype or "").lower() not in text_like:
                continue
            # Only embed if configured in embed_values for this table
            allowed_cols = embed_values.get(tbl) or embed_values.get(f"{sch}.{tbl}")
            if not allowed_cols or col not in set(allowed_cols):
                continue
            # Skip specific value columns
            if f"{tbl}.{col}" in skip_values or f"{sch}.{tbl}.{col}" in skip_values:
                continue
            full = f"{sch}.{tbl}.{col}"
            try:
                cur.execute(f"SELECT DISTINCT `{col}` FROM `{sch}`.`{tbl}` WHERE `{col}` IS NOT NULL LIMIT {max_per_column}")
                values = [str(r[0]) for r in cur.fetchall() if r[0] is not None]
                if not values:
                    logger.debug(f"No distinct values found for {full}")
                    continue
                docs = [f"{full}={v}" for v in values]
                ids = [f"v::{full}::{i}" for i, _ in enumerate(values)]
                embs = [model.encode(d).tolist() for d in docs]
                VALS_COL.upsert(ids=ids, embeddings=embs, documents=docs)
                sampled_count += 1
                logger.debug(f"Sampled {len(values)} distinct values for {full}")
            except Exception as e:
                # Log per-column sampling errors instead of silently ignoring
                error_count += 1
                logger.warning(f"Failed to sample values for {full}: {e}")
                continue
    
    logger.info(f"Distinct values sampling complete: {sampled_count} columns sampled, {error_count} errors")

# =====================
# Embedding search helpers
# =====================

def _query_chroma(col, text: str, n: int):
    return col.query(query_embeddings=[model.encode(text).tolist()], n_results=n)


def _log_embed_context(kind: str, query: str, docs: List[str]):
    """Lightweight debug trace of what context is returned to the LLM."""
    try:
        sample = docs[:5] if isinstance(docs, list) else docs
        logger.debug(f"[EMBED_CTX] {kind} query='{query}' -> {sample}")
    except Exception:
        logger.debug(f"[EMBED_CTX] {kind} query='{query}' -> <unprintable>")


def search_tables(text: str, top_k: int = 25) -> List[str]:
    res = _query_chroma(TABLES_COL, text, top_k)
    docs = res.get("documents", [["No results"]])[0]
    _log_embed_context("tables", text, docs)
    return docs


def search_columns(text: str, top_k: int = 50) -> List[str]:
    res = _query_chroma(COLS_COL, text, top_k)
    docs = res.get("documents", [["No results"]])[0]
    _log_embed_context("columns", text, docs)
    return docs


def search_relationships(text: str, top_k: int = 25) -> List[str]:
    res = _query_chroma(RELS_COL, text, top_k)
    docs = res.get("documents", [["No results"]])[0]
    _log_embed_context("relationships", text, docs)
    return docs


def _tool_to_text(obj: Any, limit: int = 4000) -> str:
    """Serialize tool output to a string safe for LLM messages."""
    try:
        if isinstance(obj, dict):
            out = json.dumps(obj, default=str)
        elif isinstance(obj, list):
            out = "\n".join([str(x) for x in obj])
        else:
            out = str(obj)
        return out[:limit]
    except Exception:
        return "<unserializable tool output>"


# Tool-safe wrappers (return strings for the LLM)
def search_tables_tool(text: str, top_k: int = 25) -> str:
    return _tool_to_text(search_tables(text, top_k))


def search_columns_tool(text: str, top_k: int = 50) -> str:
    return _tool_to_text(search_columns(text, top_k))


def search_relationships_tool(text: str, top_k: int = 25) -> str:
    return _tool_to_text(search_relationships(text, top_k))


def _should_check_live(user_value: str, embedded_values: List[str], threshold: float = 0.70) -> bool:
    if not embedded_values:
        return True
    uv_emb = model.encode(user_value).reshape(1, -1)
    sims = [cosine_similarity(uv_emb, model.encode(ev).reshape(1, -1))[0][0] for ev in embedded_values]
    return max(sims) < threshold


def _auto_detect_column_from_query(query: str) -> Optional[str]:
    res = search_columns(query, top_k=1)
    if not res:
        return None
    cand = res[0]
    if cand.lower().startswith("column: "):
        return cand.split("Column: ", 1)[-1].split(" :: ")[0].strip()
    return None


def get_live_distinct_values(col_path: str, max_values: int = 100) -> List[str]:
    col_path = col_path.strip().strip("'").strip('"')
    # Respect chroma-only mode
    if OVERRIDES.get("chroma_only"):
        # Return whatever is embedded
        res = VALS_COL.query(query_embeddings=[model.encode(col_path).tolist()], n_results=max_values)
        docs = res.get("documents", [[]])[0]
        return [d.split("=", 1)[-1].strip() if "=" in d else d for d in docs]

    # Expect sch.tbl.col
    parts = col_path.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid column path (expected schema.table.column): {col_path}")
    sch, tbl, col = parts
    if sch != MYSQL_DATABASE:
        raise ValueError(f"Column path must use configured database schema: expected {MYSQL_DATABASE}, got {sch}")
    include_tables = set(EMBED_CONFIG.get("include_tables", []) or [])
    skip_tables = set(EMBED_CONFIG.get("skip_tables", []) or [])
    if not _table_allowed(sch, tbl, include_tables, skip_tables):
        raise ValueError(f"Table {sch}.{tbl} is not allowed by include/skip configuration")
    
    # Validate identifiers to prevent SQL injection
    if not all(_validate_sql_identifier(p) for p in [sch, tbl, col]):
        logger.warning(f"Invalid SQL identifier in path: {col_path}")
        raise ValueError(f"Invalid SQL identifier in column path: {col_path}")
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT DISTINCT `{col}` FROM `{sch}`.`{tbl}` WHERE `{col}` IS NOT NULL LIMIT {max_values}")
        values = [str(r[0]) for r in cur.fetchall() if r[0] is not None]
    # Upsert these values into Chroma for future lookups, unless explicitly skipped by config
    skip_values = set((EMBED_CONFIG.get("skip_values", []) or []))
    if values and (f"{tbl}.{col}" not in skip_values and f"{sch}.{tbl}.{col}" not in skip_values):
        full = f"{sch}.{tbl}.{col}"
        docs = [f"{full}={v}" for v in values]
        ids = [f"v::{full}::{i}" for i, _ in enumerate(values)]
        embs = [model.encode(d).tolist() for d in docs]
        VALS_COL.upsert(ids=ids, embeddings=embs, documents=docs)
    return values


def get_live_distinct_values_tool(col_path: str, max_values: int = 100) -> str:
    return _tool_to_text(get_live_distinct_values(col_path, max_values=max_values))


def search_values_smart(user_query: str, top_k: int = 100, similarity_threshold: float = 0.80) -> Dict[str, Any]:
    table_column = _auto_detect_column_from_query(user_query)
    if not table_column:
        return {"values": [], "needs_db_lookup": False, "error": "Could not detect column"}

    emb = VALS_COL.query(query_embeddings=[model.encode(user_query).tolist()], n_results=top_k)
    embedded_values = emb.get("documents", [[]])[0]
    _log_embed_context("values_embed", user_query, embedded_values)
    clean_values = [d.split("=", 1)[-1].strip() if "=" in d else d for d in embedded_values]

    # If chroma-only mode is enabled, do not hit DB
    if OVERRIDES.get("chroma_only"):
        return {"values": clean_values, "needs_db_lookup": False}

    user_guess = user_query.split()[-1]
    if user_guess and _should_check_live(user_guess, clean_values, similarity_threshold):
        live = get_live_distinct_values(table_column)
        return {"values": live, "needs_db_lookup": False}

    return {"values": clean_values, "needs_db_lookup": False}


def search_values_smart_tool(user_query: str, top_k: int = 100, similarity_threshold: float = 0.80) -> str:
    return _tool_to_text(search_values_smart(user_query, top_k=top_k, similarity_threshold=similarity_threshold))

# =====================
# SQL Helpers
# =====================

def _validate_sql_identifier(identifier: str) -> bool:
    """
    Validate SQL identifier to prevent injection via schema/table/column names.
    Returns True if valid, False otherwise.
    """
    if not identifier:
        return False
    # MySQL identifiers: alphanumeric, underscore, dollar, must start with letter or underscore
    if len(identifier) > 64:
        return False
    # Allow schema.table.column format
    parts = identifier.split('.')
    for part in parts:
        part = part.strip('`')  # Remove backticks if present
        if not part:
            return False
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_$]*$', part):
            return False
    return True

def clean_sql(sql: str) -> str:
    sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```", "", sql)
    return sql.strip().rstrip(";")


def validate_sql_schema(sql: str) -> Dict[str, Any]:
    """Use EXPLAIN to validate syntax and basic schema references without execution."""
    sql = clean_sql(sql)
    ok, msg = _is_select_only(sql)
    if not ok:
        logger.debug(f"SQL validation failed: {msg}")
        return {"valid": False, "error": msg}
    
    logger.debug(f"Validating SQL: {sql[:150]}...")
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"EXPLAIN {sql}")
        logger.debug("SQL validation successful")
        return {"valid": True}
    except Exception as e:
        logger.debug(f"SQL validation error: {e}")
        return {"valid": False, "error": str(e)}


def validate_sql_schema_tool(sql: str) -> str:
    return _tool_to_text(validate_sql_schema(sql))


def execute_sql(sql: str) -> Dict[str, Any]:
    """Execute a SQL query with proper validation, timeout, and error handling."""
    sql = clean_sql(sql)
    ok, msg = _is_select_only(sql)
    if not ok:
        logger.warning(f"SQL validation failed: {msg}")
        return {"error": msg}
    sql = _enforce_row_limit_on_sql(sql)
    
    logger.info(f"Executing SQL: {sql[:200]}...")
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql)
            try:
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                LAST_SQL["final_sql"] = sql
                py_rows = [list(r) for r in rows]
                logger.info(f"Query executed successfully: {len(py_rows)} rows returned")
                return {"columns": cols, "rows": py_rows}
            except pymysql.err.ProgrammingError as e:
                logger.debug(f"No results returned (ProgrammingError): {e}")
                LAST_SQL["final_sql"] = sql
                return {"columns": [], "rows": []}
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return {"error": str(e)}


def execute_sql_tool(sql: str) -> str:
    return _tool_to_text(execute_sql(sql))


def regenerate_sql_with_llm(prev_sql: str, error_msg: str) -> str:
    prev_sql = clean_sql(prev_sql)
    prompt = f"""
You are an expert SQL generator for MySQL SELECT queries.

Constraints:
- Only generate SELECT statements. Never use DML/DDL.
- Limit results to 10 rows unless explicitly requested otherwise (use LIMIT 10).
- Alias columns with user-friendly names.
- Do not assume values; use tools to validate them when necessary.

The previous SQL was:
{prev_sql}

It failed with the following error:
{error_msg}

Generate a corrected MySQL query that fixes the issue. Return only the SQL, no explanation.
"""
    llm = ChatOpenAI(api_key=OPENAI_API_KEY or None, model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE, base_url=OPENAI_BASE_URL or None)
    out = llm.invoke(prompt)
    if hasattr(out, "content"):
        out = out.content
    return clean_sql(out)


def execute_sql_with_retry(sql: str, retries: int = 2, qid: Optional[str] = None, question: Optional[str] = None) -> Dict[str, Any]:
    # Fallback to current run context if not provided
    if qid is None:
        qid = CURRENT_RUN.get("id")
    if question is None:
        question = CURRENT_RUN.get("question")
    start_dt = CURRENT_RUN.get("start") or datetime.now(timezone.utc)
    candidate = sql
    LAST_SQL["attempts"] = []
    for attempt in range(retries + 1):
        v = validate_sql_schema(candidate)
        LAST_SQL["attempts"].append({"attempt": attempt + 1, "phase": "validate", "sql": clean_sql(candidate), "ok": v.get("valid", False), "error": v.get("error")})
        if not v.get("valid"):
            if attempt < retries:
                candidate = regenerate_sql_with_llm(candidate, v.get("error", "validation error"))
                continue
            return {"error": v.get("error", "validation error")}
        # If generate-only mode, do not execute against DB
        if OVERRIDES.get("generate_only"):
            final_sql = clean_sql(candidate)
            LAST_SQL["final_sql"] = final_sql
            LAST_SQL["attempts"].append({"attempt": attempt + 1, "phase": "generate_only", "sql": final_sql, "ok": True})
            end_dt = datetime.now(timezone.utc)
            duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
            entry = {
                "id": qid or str(uuid.uuid4()),
                "ts": _format_ts(start_dt),
                "question": question,
                "attempts": LAST_SQL.get("attempts", []),
                "final_sql": final_sql,
                "ok": True,
                "error": None,
                "columns": None,
                "rows": None,
                "rows_count": None,
                "trace": list(TRACE_EVENTS),
                "duration_ms": duration_ms,
            }
            QUERY_HISTORY.append(entry)
            return {"generated_only": True, "sql": final_sql}
        res = execute_sql(candidate)
        LAST_SQL["attempts"].append({"attempt": attempt + 1, "phase": "execute", "sql": clean_sql(candidate), "ok": "error" not in res, "error": res.get("error")})
        if "error" in res and attempt < retries:
            candidate = regenerate_sql_with_llm(candidate, res["error"])
            continue
        # Log into query history for reference with query id
        end_dt = datetime.now(timezone.utc)
        duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
        entry = {
            "id": qid or str(uuid.uuid4()),
            "ts": _format_ts(start_dt),
            "question": question,
            "attempts": LAST_SQL.get("attempts", []),
            "final_sql": clean_sql(candidate),
            "ok": "error" not in res,
            "error": res.get("error"),
            "columns": res.get("columns"),
            "rows": res.get("rows"),
            "rows_count": len(res.get("rows", [])) if isinstance(res.get("rows"), list) else None,
            "trace": list(TRACE_EVENTS),
            "duration_ms": duration_ms,
        }
        QUERY_HISTORY.append(entry)
        return res
    return {"error": "Max retries reached"}


def execute_sql_with_retry_tool(sql: str, retries: int = 2) -> str:
    return _tool_to_text(execute_sql_with_retry(sql, retries=retries))

# =====================
# Session State Management (Thread-Safe)
# =====================

# Global state with locks for thread safety
_state_lock = Lock()
_global_query_history: List[Dict[str, Any]] = []
MAX_HISTORY_SIZE = int(os.getenv("MAX_QUERY_HISTORY", "1000"))

def get_query_history() -> List[Dict[str, Any]]:
    """Thread-safe getter for query history."""
    with _state_lock:
        return list(_global_query_history)

def append_to_history(entry: Dict[str, Any]) -> None:
    """Thread-safe append to query history with size limit."""
    with _state_lock:
        if len(_global_query_history) >= MAX_HISTORY_SIZE:
            _global_query_history.pop(0)
            logger.info(f"Query history reached max size ({MAX_HISTORY_SIZE}), removed oldest entry")
        _global_query_history.append(entry)
        logger.info(f"Added query to history: ID={entry.get('id')}, duration={entry.get('duration_ms')}ms")

def clear_query_history() -> None:
    """Thread-safe clear query history."""
    with _state_lock:
        _global_query_history.clear()
        logger.info("Query history cleared")

# Session state class for per-user tracking
class SessionState:
    """Per-session state to avoid global state issues."""
    def __init__(self):
        self.trace_events: List[Dict[str, Any]] = []
        self.last_sql: Dict[str, Any] = {"final_sql": None, "attempts": []}
        self.current_run: Dict[str, Any] = {"id": None, "question": None, "start": None}
    
    def clear_trace(self):
        self.trace_events.clear()
    
    def append_trace(self, event: Dict[str, Any]):
        try:
            self.trace_events.append(event)
            logger.debug(f"Trace event: {event.get('tool')} - {str(event.get('input'))[:100]}")
        except Exception as e:
            logger.warning(f"Failed to append trace event: {e}")

# Legacy global state for backward compatibility (will be phased out)
TRACE_EVENTS: List[Dict[str, Any]] = []
LAST_SQL: Dict[str, Any] = {"final_sql": None, "attempts": []}
QUERY_HISTORY: List[Dict[str, Any]] = []  # list of {id, ts, question, attempts, final_sql, ok, rows_count, trace, duration_ms}


def _format_ts(dt: datetime) -> str:
    try:
        return dt.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')
    except Exception:
        return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')
CURRENT_RUN: Dict[str, Any] = {"id": None, "question": None, "start": None}
PREFS_PATH = os.getenv("PREFS_PATH", "config/prefs.json")
OVERRIDES: Dict[str, Any] = {"default_row_limit": None, "query_timeout_seconds": None, "chroma_only": False, "stream_trace": True, "ui_density": "M", "generate_only": False, "recursion_limit": 25}

# Load persisted preferences if present
if os.path.exists(PREFS_PATH):
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                OVERRIDES.update({k: data.get(k, v) for k, v in OVERRIDES.items()})
    except Exception:
        pass


def save_prefs():
    try:
        with open(PREFS_PATH, "w", encoding="utf-8") as f:
            json.dump(OVERRIDES, f, indent=2)
        return "Preferences saved."
    except Exception as e:
        return f"Failed to save preferences: {e}"


def _append_trace(event: Dict[str, Any]):
    try:
        TRACE_EVENTS.append(event)
    except Exception:
        pass

# Safeguards via config + UI overrides

def _get_safeguards():
    sg = (EMBED_CONFIG or {}).get("safeguards", {}) or {}
    row_limit = OVERRIDES.get("default_row_limit") or sg.get("default_row_limit", 10)
    timeout = OVERRIDES.get("query_timeout_seconds") or sg.get("query_timeout_seconds", 30)
    return {
        "default_row_limit": int(row_limit),
        "query_timeout_seconds": int(timeout),
    }


def set_overrides(row_limit: Optional[int], timeout: Optional[int], chroma_only: bool, stream_trace: bool, ui_density: Optional[str] = None, generate_only: Optional[bool] = None, recursion_limit: Optional[int] = None) -> str:
    OVERRIDES["default_row_limit"] = int(row_limit) if row_limit else None
    OVERRIDES["query_timeout_seconds"] = int(timeout) if timeout else None
    OVERRIDES["chroma_only"] = bool(chroma_only)
    OVERRIDES["stream_trace"] = bool(stream_trace)
    if ui_density in {"XS", "S", "M", "L", "XL"}:
        OVERRIDES["ui_density"] = ui_density
    if generate_only is not None:
        OVERRIDES["generate_only"] = bool(generate_only)
    if recursion_limit is not None and recursion_limit in {25, 50, 75, 100}:
        OVERRIDES["recursion_limit"] = int(recursion_limit)
    return save_prefs()


def _strip_comments_and_strings(s: str) -> str:
    out = []
    i = 0
    n = len(s)
    in_str = False
    while i < n:
        ch = s[i]
        if not in_str:
            # Line comment
            if ch == '-' and i + 1 < n and s[i + 1] == '-':
                # skip until end of line
                i += 2
                while i < n and s[i] not in ('\n', '\r'):
                    i += 1
                continue
            # Block comment
            if ch == '/' and i + 1 < n and s[i + 1] == '*':
                i += 2
                while i + 1 < n and not (s[i] == '*' and s[i + 1] == '/'):
                    i += 1
                i = i + 2 if i + 1 < n else n
                continue
            if ch == "'":
                # enter string, write empty quotes to preserve word boundaries
                out.append("''")
                in_str = True
                i += 1
                continue
            out.append(ch)
            i += 1
        else:
            # inside string; handle escaped ''
            if ch == "'":
                if i + 1 < n and s[i + 1] == "'":
                    i += 2  # escaped quote
                    continue
                else:
                    in_str = False
                    i += 1
                    continue
            i += 1
    return ''.join(out)


def _is_select_only(sql: str) -> Tuple[bool, str]:
    """Ensure the SQL is a single SELECT (or WITH ... SELECT) and contains no DML/DDL keywords."""
    s = clean_sql(sql)
    s2 = _strip_comments_and_strings(s)
    low = s2.strip().lower()
    if not (low.startswith('select') or low.startswith('with')):
        return False, "Only SELECT statements are allowed (optionally starting with WITH for CTEs)."
    # reject dangerous keywords anywhere
    forbidden = r"\b(insert|update|delete|merge|alter|drop|create|exec|execute|truncate|grant|revoke|deny|backup|restore|dbcc|shutdown|use)\b"
    if re.search(forbidden, low, flags=re.IGNORECASE):
        return False, "Only read-only SELECT statements are permitted. DML/DDL is blocked."
    # prevent multiple statements
    if ';' in low:
        return False, "Multiple statements are not allowed. Submit a single SELECT query."
    return True, ""


def _enforce_row_limit_on_sql(sql: str) -> str:
    """Ensure a LIMIT clause exists for MySQL SELECT statements."""
    sql_clean = clean_sql(sql)
    ok, _ = _is_select_only(sql_clean)
    if not ok:
        return sql_clean

    # If LIMIT already present, leave unchanged
    if re.search(r"\blimit\s+\d+(\s*,\s*\d+)?", sql_clean, flags=re.IGNORECASE):
        return sql_clean

    n = _get_safeguards()["default_row_limit"]

    # Append LIMIT before trailing semicolon if present
    sql_no_semicolon = sql_clean.rstrip().rstrip(';')
    return f"{sql_no_semicolon} LIMIT {n}"

# =====================
# LLM + Tools (ReAct)
# =====================

def _make_custom_instructions() -> str:
    base = """
You are an expert SQL generator for MySQL, who helps users write correct and efficient SELECT queries based on their questions.
Currently we are dealing with a Banking database for a financial institution based in India.
Follow the ReAct format: Thought -> Action -> Action Input -> Observation -> Thought -> Final Answer.
Use provided tools for every step before final answer; do not jump straight to final answer.
When filtering values, first search embeddings in ChromaDB; if low similarity, query live DISTINCT values for that column.
Use only SELECT queries. Limit to 10 rows by default (use LIMIT 10). Provide results in a tabular form.
Prefer schema-qualified names `schema`.`table`.`column`.
"""
    if OVERRIDES.get("generate_only"):
        base += """
IMPORTANT: Generation-only mode is enabled.
- Do NOT attempt to execute SQL or refine based on execution results.
- Do NOT call any tool that executes queries.
- Your Final Answer MUST be only a single code block containing the final MySQL SELECT.
"""
    return base

custom_instructions = _make_custom_instructions()

prompt_template = f"""
{custom_instructions}

Answer the following question using the tools.

{{tools}}

Use this format:
Question: the input question
Thought: think about what to do
Action: one of [{{tool_names}}]
Action Input: input to the action
Observation: result of the action
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer to the original question

Begin!

Question: {{input}}
Thought:{{agent_scratchpad}}
"""

prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    template=prompt_template,
)

# Tools

def _build_tools():
    base_tools = [
        Tool(name="SearchTables", func=search_tables_tool, description="Find relevant MySQL tables using embeddings."),
        Tool(name="SearchColumns", func=search_columns_tool, description="Find relevant MySQL columns using embeddings."),
        Tool(name="SearchRelationships", func=search_relationships_tool, description="Find FK relationships between MySQL tables."),
        Tool(name="SearchValues", func=search_values_smart_tool, description="Smart value search; uses embeddings, falls back to live DISTINCT."),
        Tool(name="GetLiveDistinctValues", func=get_live_distinct_values_tool, description="Fetch DISTINCT values for a column directly from MySQL. Arg: 'schema.table.column'"),
        Tool(name="ValidateSQL", func=validate_sql_schema_tool, description="Validate SQL syntax using EXPLAIN."),
    ]
    if not OVERRIDES.get("generate_only"):
        base_tools.append(Tool(name="ExecuteSQL", func=execute_sql_with_retry_tool, description="Execute SQL with retries and LLM-based regeneration."))
    return base_tools

llm = ChatOpenAI(api_key=OPENAI_API_KEY or None, model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE, base_url=OPENAI_BASE_URL or None)
# Build LangGraph ReAct agent with tools based on current mode
agent = lg_create_react_agent(llm, tools=_build_tools())

# =====================
# Bootstrapping Chroma with schema
# =====================

def check_embeddings_exist() -> Dict[str, bool]:
    """
    Check if embeddings already exist in ChromaDB collections.
    Returns a dict with status for each collection type.
    """
    try:
        tables_count = TABLES_COL.count()
        cols_count = COLS_COL.count()
        rels_count = RELS_COL.count()
        vals_count = VALS_COL.count()
        
        return {
            "tables": tables_count > 0,
            "columns": cols_count > 0,
            "relationships": rels_count > 0,
            "values": vals_count > 0,
            "counts": {
                "tables": tables_count,
                "columns": cols_count,
                "relationships": rels_count,
                "values": vals_count
            }
        }
    except Exception as e:
        logger.warning(f"Error checking embeddings: {e}")
        return {
            "tables": False,
            "columns": False,
            "relationships": False,
            "values": False,
            "counts": {}
        }

def ensure_chroma_bootstrap(sample_values: bool = True, force_refresh: bool = False) -> str:
    """
    Bootstrap ChromaDB with schema embeddings.
    
    Args:
        sample_values: Whether to sample distinct values for text columns
        force_refresh: If True, re-embed even if embeddings exist. If False, skip if embeddings exist.
    
    Returns:
        Status message
    """
    # Check if embeddings already exist
    existing = check_embeddings_exist()
    
    if not force_refresh:
        if existing["tables"] and existing["columns"]:
            counts = existing["counts"]
            msg = (f"[OK] Using existing embeddings: "
                   f"{counts.get('tables', 0)} tables, "
                   f"{counts.get('columns', 0)} columns, "
                   f"{counts.get('relationships', 0)} relationships, "
                   f"{counts.get('values', 0)} values. "
                   f"Use 'Refresh Embeddings' in Settings to re-embed.")
            logger.info(msg)
            return msg
    
    logger.info(f"{'Force refreshing' if force_refresh else 'Creating'} embeddings...")
    schema = fetch_schema()
    upsert_schema_into_chroma(schema)
    
    if sample_values:
        sample_distinct_values(schema)
    
    # Get final counts
    final = check_embeddings_exist()
    counts = final["counts"]
    msg = (f"[OK] Embeddings {'refreshed' if force_refresh else 'created'}: "
           f"{counts.get('tables', 0)} tables, "
           f"{counts.get('columns', 0)} columns, "
           f"{counts.get('relationships', 0)} relationships, "
           f"{counts.get('values', 0)} values.")
    logger.info(msg)
    return msg

def reload_config_and_refresh() -> str:
    """Reload config and force re-embedding."""
    global EMBED_CONFIG
    EMBED_CONFIG = load_embedding_config()
    msg = ensure_chroma_bootstrap(sample_values=True, force_refresh=True)
    return "Config reloaded. " + msg

# =====================
# Embedding Management Functions
# =====================

def get_embedding_details() -> Dict[str, Any]:
    """Get detailed information about what's embedded in ChromaDB."""
    try:
        # Get all documents from each collection
        tables_data = TABLES_COL.get()
        cols_data = COLS_COL.get()
        rels_data = RELS_COL.get()
        vals_data = VALS_COL.get()
        
        # Extract table names
        table_names = []
        if tables_data and tables_data.get('documents'):
            for doc in tables_data['documents']:
                if doc.startswith("Table: "):
                    table_name = doc.split("Table: ")[1].split(" -")[0].strip()
                    table_names.append(table_name)
        
        # Extract column names grouped by table
        columns_by_table = {}
        if cols_data and cols_data.get('documents'):
            for doc in cols_data['documents']:
                if doc.startswith("Column: "):
                    parts = doc.split("Column: ")[1].split(" :: ")
                    if parts:
                        col_full = parts[0].strip()
                        # schema.table.column format
                        col_parts = col_full.split('.')
                        if len(col_parts) >= 2:
                            table_key = '.'.join(col_parts[:-1])
                            col_name = col_parts[-1]
                            if table_key not in columns_by_table:
                                columns_by_table[table_key] = []
                            columns_by_table[table_key].append(col_name)
        
        # Extract columns with embedded values
        value_columns = set()
        if vals_data and vals_data.get('documents'):
            for doc in vals_data['documents']:
                if '=' in doc:
                    col_path = doc.split('=')[0].strip()
                    value_columns.add(col_path)
        
        return {
            "table_names": sorted(table_names),
            "columns_by_table": columns_by_table,
            "value_columns": sorted(list(value_columns)),
            "counts": {
                "tables": len(table_names),
                "columns": len(cols_data.get('documents', [])),
                "relationships": len(rels_data.get('documents', [])),
                "values": len(vals_data.get('documents', []))
            }
        }
    except Exception as e:
        logger.error(f"Error getting embedding details: {e}")
        return {
            "table_names": [],
            "columns_by_table": {},
            "value_columns": [],
            "counts": {"tables": 0, "columns": 0, "relationships": 0, "values": 0}
        }

def format_embedding_summary() -> str:
    """Format embedding details for display in UI."""
    details = get_embedding_details()
    counts = details["counts"]
    
    lines = [
        "### Embedding Summary",
        f"- **Tables**: {counts['tables']} embedded",
        f"- **Columns**: {counts['columns']} embedded",
        f"- **Relationships**: {counts['relationships']} embedded",
        f"- **Value Samples**: {counts['values']} distinct values",
        "",
        "#### Embedded Tables:",
    ]
    
    if details["table_names"]:
        for table in details["table_names"][:20]:  # Show first 20
            lines.append(f"  - {table}")
        if len(details["table_names"]) > 20:
            lines.append(f"  ... and {len(details['table_names']) - 20} more")
    else:
        lines.append("  _(none)_")
    
    lines.append("")
    lines.append("#### Columns with Embedded Values:")
    if details["value_columns"]:
        for col in details["value_columns"][:15]:  # Show first 15
            lines.append(f"  - {col}")
        if len(details["value_columns"]) > 15:
            lines.append(f"  ... and {len(details['value_columns']) - 15} more")
    else:
        lines.append("  _(none)_")
    
    return "\n".join(lines)

def refresh_embeddings_selective(
    refresh_tables: bool,
    refresh_columns: bool,
    refresh_relationships: bool,
    refresh_values: bool
) -> str:
    """Selectively refresh specific types of embeddings."""
    if not any([refresh_tables, refresh_columns, refresh_relationships, refresh_values]):
        return "[WARNING] No embedding types selected. Nothing to refresh."
    
    logger.info(f"Selective refresh: tables={refresh_tables}, columns={refresh_columns}, "
                f"relationships={refresh_relationships}, values={refresh_values}")
    
    try:
        schema = fetch_schema()
        refreshed = []
        
        if refresh_tables:
            # Clear and re-embed tables
            skip_tables = set(EMBED_CONFIG.get("skip_tables", []) or [])
            descriptions = EMBED_CONFIG.get("descriptions", {}) or {}
            table_docs = []
            table_ids = []
            table_embs = []
            for sch, tbl in schema["tables"]:
                if tbl in skip_tables or f"{sch}.{tbl}" in skip_tables:
                    continue
                desc = descriptions.get(tbl) or descriptions.get(f"{sch}.{tbl}")
                doc = f"Table: {sch}.{tbl}" + (f" - {desc}" if desc else "")
                table_docs.append(doc)
                table_ids.append(f"t::{sch}.{tbl}")
                table_embs.append(model.encode(doc).tolist())
            
            if table_docs:
                TABLES_COL.delete(ids=TABLES_COL.get()['ids'])
                TABLES_COL.upsert(ids=table_ids, embeddings=table_embs, documents=table_docs)
                refreshed.append(f"{len(table_docs)} tables")
                logger.info(f"Refreshed {len(table_docs)} table embeddings")
        
        if refresh_columns:
            # Clear and re-embed columns
            skip_tables = set(EMBED_CONFIG.get("skip_tables", []) or [])
            col_docs = []
            col_ids = []
            col_embs = []
            for sch, tbl, col, dtype in schema["columns"]:
                if tbl in skip_tables or f"{sch}.{tbl}" in skip_tables:
                    continue
                doc = f"Column: {sch}.{tbl}.{col} :: {dtype}"
                col_docs.append(doc)
                col_ids.append(f"c::{sch}.{tbl}.{col}")
                col_embs.append(model.encode(doc).tolist())
            
            if col_docs:
                COLS_COL.delete(ids=COLS_COL.get()['ids'])
                COLS_COL.upsert(ids=col_ids, embeddings=col_embs, documents=col_docs)
                refreshed.append(f"{len(col_docs)} columns")
                logger.info(f"Refreshed {len(col_docs)} column embeddings")
        
        if refresh_relationships:
            # Clear and re-embed relationships
            rel_docs = [f"FK: {r['fk']} -> {r['pk']}" for r in schema["relationships"]]
            rel_ids = [f"r::{idx}" for idx, _ in enumerate(rel_docs)]
            rel_embs = [model.encode(doc).tolist() for doc in rel_docs]
            
            if rel_docs:
                RELS_COL.delete(ids=RELS_COL.get()['ids'])
                RELS_COL.upsert(ids=rel_ids, embeddings=rel_embs, documents=rel_docs)
                refreshed.append(f"{len(rel_docs)} relationships")
                logger.info(f"Refreshed {len(rel_docs)} relationship embeddings")
        
        if refresh_values:
            # Clear and re-sample distinct values
            VALS_COL.delete(ids=VALS_COL.get()['ids'])
            sample_distinct_values(schema, max_per_column=50)
            vals_count = VALS_COL.count()
            refreshed.append(f"{vals_count} value samples")
            logger.info(f"Refreshed {vals_count} value embeddings")
        
        msg = "[OK] Refreshed: " + ", ".join(refreshed)
        return msg
        
    except Exception as e:
        error_msg = f"[ERROR] Failed to refresh embeddings: {e}"
        logger.error(error_msg)
        return error_msg

# =====================
# Gradio UI
# =====================

def run_controller(q, chroma_only, stream_trace, rl, to):
    set_overrides(rl, to, chroma_only, stream_trace)
    # Generate a query ID and announce running
    qid = str(uuid.uuid4())
    yield f"⏳ Running... (ID: {qid})", None, None
    CURRENT_RUN["id"] = qid
    CURRENT_RUN["question"] = q
    if stream_trace:
        for triple in run_agent_stream(q, qid=qid):
            yield triple
        CURRENT_RUN["id"], CURRENT_RUN["question"] = None, None
        return
    TRACE_EVENTS.clear()
    recursion_limit = OVERRIDES.get("recursion_limit", 25)
    result = agent.invoke(
        {"messages": [SystemMessage(content=custom_instructions), HumanMessage(content=q)]},
        config={"recursion_limit": recursion_limit}
    )
    if isinstance(result, dict):
        msgs = result.get("messages") or []
        final = msgs[-1].content if msgs else str(result)
    else:
        final = str(result)
    # Clean leading 'Final Answer:' if present
    if isinstance(final, str):
        final = re.sub(r"^\s*Final Answer\s*:\s*", "", final, flags=re.IGNORECASE)
    sql_text = _format_sql_attempts()
    CURRENT_RUN["id"], CURRENT_RUN["question"], CURRENT_RUN["start"] = None, None, None
    yield final, _format_trace_events(), sql_text

def _format_result(res: Dict[str, Any]) -> str:
    if not res:
        return "No result"
    if "error" in res:
        return f"❌ Error: {res['error']}"
    cols = res.get("columns", [])
    rows = res.get("rows", [])
    if not cols:
        return "No rows"
    return f"Rows: {len(rows)} | Columns: {len(cols)}"


def _format_intermediate_steps(steps: List[Dict[str, Any]]) -> str:
    if not steps:
        return "No agent steps."
    parts = []
    for i, step in enumerate(steps, 1):
        action = step.get("action")
        observation = step.get("observation")
        parts.append(f"### Step {i}\n\n- Action: {action}\n- Observation: {observation}\n")
    return "\n".join(parts)


def _format_trace_events() -> str:
    if not TRACE_EVENTS:
        return "No trace events."
    out = []
    for i, ev in enumerate(TRACE_EVENTS, 1):
        tool = ev.get("tool")
        input_ = ev.get("input")
        output_ = ev.get("output")
        out.append(f"### Event {i}\n- Tool: {tool}\n- Input: {input_}\n- Output: {output_}\n")
    return "\n".join(out)


def _format_sql_attempts() -> str:
    if not LAST_SQL.get("attempts"):
        return "No SQL attempts."
    lines = []
    for att in LAST_SQL["attempts"]:
        lines.append(f"- Attempt {att['attempt']} [{att['phase']}]: ok={att['ok']}\n  SQL: {att['sql']}\n  Error: {att.get('error')}")
    final = LAST_SQL.get("final_sql")
    if final:
        lines.append(f"\nFinal SQL Executed:\n```sql\n{final}\n```")
    return "\n".join(lines)


def _format_history() -> str:
    if not QUERY_HISTORY:
        return "No queries run yet."
    out = ["### Query History"]
    for i, q in enumerate(QUERY_HISTORY[::-1], 1):
        out.append(f"{i}. ID={q['id']} | {q['ts']} | {q.get('duration_ms', 0)} ms | ok={q['ok']} rows={q.get('rows_count')}\nQuestion: {q.get('question')}\n\n```sql\n{q.get('final_sql') or ''}\n```\n")
    return "\n".join(out)


def _history_choices() -> List[str]:
    return [q["id"] for q in QUERY_HISTORY[::-1]]


def _history_details(qid: str) -> str:
    if not qid:
        return "Select a query ID."
    entry = next((q for q in QUERY_HISTORY if q["id"] == qid), None)
    if not entry:
        return "Query ID not found. Click Refresh History."
    lines = [
        f"### Query {entry['id']}",
        f"Timestamp: {entry['ts']}",
        f"Duration: {entry.get('duration_ms', 0)} ms",
        f"Question: {entry.get('question')}",
        f"Status: {'OK' if entry['ok'] else 'ERROR'}",
        f"Rows: {entry.get('rows_count')}",
        "\n#### Final SQL:\n```sql\n" + (entry.get('final_sql') or '') + "\n```",
        "\n#### Attempts:",
    ]
    for att in entry.get("attempts", []):
        lines.append(f"- Attempt {att['attempt']} [{att['phase']}] ok={att['ok']}\n  SQL: {att['sql']}\n  Error: {att.get('error')}")
    lines.append("\n#### Trace (truncated):\n" + ("\n".join([str(t)[:500] for t in entry.get('trace', [])]) or "(none)"))
    return "\n".join(lines)


def run_agent_stream(user_query: str, qid: Optional[str] = None):
    # Generator to stream partial trace while collecting final answer
    TRACE_EVENTS.clear()
    final_text = None
    try:
        recursion_limit = OVERRIDES.get("recursion_limit", 25)
        for ev in agent.stream(
            {"messages": [SystemMessage(content=custom_instructions), HumanMessage(content=user_query)]},
            config={"recursion_limit": recursion_limit},
            stream_mode="values"
        ):
            msgs = ev.get("messages") if isinstance(ev, dict) else None
            if msgs:
                last = msgs[-1]
                try:
                    content = last.content if hasattr(last, "content") else str(last)
                except Exception:
                    content = str(last)
                # Clean leading 'Final Answer:' if present
                if isinstance(content, str):
                    content = re.sub(r"^\s*Final Answer\s*:\s*", "", content, flags=re.IGNORECASE)
                _append_trace({"tool": "agent", "input": user_query, "output": content})
                final_text = content  # keep updating; last one after stream ends is final
                # Emit partial trace to UI while keeping results unchanged
                yield None, _format_trace_events(), _format_history()
        # After stream completes, do NOT re-invoke the agent to avoid double execution.
        sql_text = _format_sql_attempts()
        yield (final_text or "No output"), _format_trace_events(), sql_text
    except Exception as e:
        yield f"❌ Error: {str(e)}", _format_trace_events(), None

def _load_markdown(filename: str) -> str:
    """Load markdown content from external file."""
    try:
        with open(os.path.join("help_pages", filename), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"# Error loading {filename}\n\nCould not load markdown content: {str(e)}"

def _load_html(filename: str) -> str:
    """Load HTML file from help_pages/ directory."""
    path = os.path.join("help_pages", filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"<p><em>Documentation file not found: {filename}</em></p>"

# Load markdown content from external files
USER_MANUAL_MD = _load_markdown("user_manual.md")
DOCS_HTML = _load_html("docs.html")

# Generate initial CSS based on preferences
APP_CSS = get_css_for_density(OVERRIDES.get("ui_density", "M"))

# =====================
# UI Definition
# =====================

with gr.Blocks(title="ETL Insights Chat Bot") as demo:
    navbar = gr.Navbar(value=[("Docs", "/docs"), ("About", "/about")], visible=True, main_page_name="Home")
    gr.Markdown("## 💬 ETL Insights Chat Bot")
    with gr.Tabs():
        with gr.Tab("💬 Ask"):
            with gr.Group(elem_classes=["chat-container"]):
                chat = gr.Chatbot(height=450, elem_id="chatbot")
                # spinner overlay
                chat_spinner = gr.HTML(value='''<div id="chat_spinner"><span class="spinner-text">Working...</span><div class="lds-ring"><div></div><div></div><div></div><div></div></div></div>''', visible=False)
            query_box = gr.Textbox(label="Ask your question", placeholder="e.g., Show top 10 transactions over $1000 in the last month")
            with gr.Row():
                run_btn = gr.Button("Get Answers", variant="primary")
                clear_chat_btn = gr.Button("Clear Chat", variant="secondary")
            sample_examples = gr.Examples([
                ["List top 10 customers by total balance"],
                ["Show top 10 transactions over 1000 in the last month"],
                ["Find accounts opened in 2024"],
            ], inputs=[query_box])
        with gr.Tab("📊 Results"):
            results_md = gr.Markdown()
            results_table = gr.Dataframe(headers=None, value=[], interactive=False, wrap=True, elem_id="results_table", column_count=0, row_count=0)
        with gr.Tab("🔍 Inspect"):
            with gr.Accordion("Agent Trace (debug)", open=False):
                trace_md = gr.Markdown()
            with gr.Accordion("Generated SQL", open=False):
                sql_md = gr.Markdown()
            with gr.Accordion("Query History", open=False):
                hist_md = gr.Markdown()
                with gr.Row():
                    hist_select = gr.Dropdown(choices=[], label="Select Query ID")
                    btn_hist_refresh = gr.Button("Refresh History")
                    btn_hist_view = gr.Button("View Details")
                    btn_hist_rerun = gr.Button("Re-run")
                    btn_hist_clear = gr.Button("Clear History", variant="secondary")
                hist_details = gr.Markdown()
        with gr.Tab("🗂️ Schema Browser"):
            gr.Markdown("### Explore Schema")
            tbl_query = gr.Textbox(label="Search tables")
            btn_tbl = gr.Button("Search Tables", variant="primary")
            tbl_out = gr.Markdown()
            col_query = gr.Textbox(label="Search columns")
            btn_col = gr.Button("Search Columns", variant="primary")
            col_out = gr.Markdown()
        with gr.Tab("📖 User Manual"):
            # Load HTML content but replace FAQ section with native Gradio components
            try:
                with open("help_pages/user_manual.html", "r", encoding="utf-8") as f:
                    user_manual_html = f.read()
                
                # Display the full HTML with native accordion
                gr.HTML(user_manual_html)
                    
            except FileNotFoundError:
                gr.Markdown("**User manual not found.** Please ensure `help_pages/user_manual.html` exists.")
            except Exception as e:
                gr.Markdown(f"**Error loading user manual:** {e}")
        with gr.Tab("📚 Docs"):
            gr.HTML(DOCS_HTML)
        with gr.Tab("⚙️ Settings"):
            gr.Markdown("### Preferences & Maintenance")
            with gr.Row():
                with gr.Column():
                    chroma_only_chk = gr.Checkbox(False, label="Chroma-only mode (no live DB for values)")
                    stream_trace_chk = gr.Checkbox(True, label="Stream agent trace live")
                with gr.Column():
                    row_limit = gr.Slider(5, 200, value=_get_safeguards()["default_row_limit"], step=1, label="Default row limit (LIMIT N)")
                    timeout = gr.Slider(5, 120, value=_get_safeguards()["query_timeout_seconds"], step=1, label="Query timeout (seconds)")
                with gr.Column():
                    density = gr.Dropdown(["XS","S","M","L","XL"], value=OVERRIDES.get("ui_density","M"), label="Text & Spacing Density")
                    generate_only_chk = gr.Checkbox(OVERRIDES.get("generate_only", False), label="Generate SQL only (do not execute)")
            
            with gr.Row():
                recursion_limit_radio = gr.Radio(
                    choices=[25, 50, 75, 100],
                    value=OVERRIDES.get("recursion_limit", 25),
                    label="Agent Recursion Limit",
                    info="Higher values allow more reasoning steps but take longer. 25 (Default): Most queries | 50: Complex multi-table | 75: Very complex analytical | 100: Maximum (difficult questions only)"
                )
            
            with gr.Row():
                btn_apply_prefs = gr.Button("Apply Preferences", variant="secondary")
                btn_test_conn = gr.Button("Test Connection", variant="secondary")
                btn_bootstrap = gr.Button("🔄 Refresh Embeddings (Force Re-embed)", variant="secondary")
                btn_reload = gr.Button("Reload Config + Refresh", variant="secondary")
            settings_msg = gr.Markdown(visible=True)
            
            # Embedding Management Section
            with gr.Accordion("📊 Embedding Management", open=False):
                gr.Markdown("""
                **View and manage your schema embeddings.** Embeddings are created once and reused for fast startup.
                Use this section to see what's embedded and selectively refresh specific types.
                """)
                
                with gr.Row():
                    btn_view_embeddings = gr.Button("🔍 View Current Embeddings", variant="secondary")
                    btn_refresh_embeddings_view = gr.Button("🔄 Refresh View", variant="secondary")
                
                embedding_summary = gr.Markdown("Click 'View Current Embeddings' to see details...")
                
                gr.Markdown("### Selective Refresh")
                gr.Markdown("Choose which embedding types to refresh (useful after schema changes):")
                
                with gr.Row():
                    chk_refresh_tables = gr.Checkbox(label="Table Names", value=True)
                    chk_refresh_columns = gr.Checkbox(label="Column Names", value=True)
                    chk_refresh_relationships = gr.Checkbox(label="Relationships (FKs)", value=True)
                    chk_refresh_values = gr.Checkbox(label="Distinct Values", value=True)
                
                btn_selective_refresh = gr.Button("🔄 Refresh Selected Types", variant="primary")
                selective_refresh_msg = gr.Markdown("")

    def _apply_prefs(chroma_only, stream_trace, rl, to, dens, gen_only, rec_limit):
        css = get_css_for_density(dens)
        msg = set_overrides(rl, to, chroma_only, stream_trace, ui_density=dens, generate_only=gen_only, recursion_limit=rec_limit)
        # Rebuild agent and custom instructions to reflect new mode
        global agent, custom_instructions
        custom_instructions = _make_custom_instructions()
        agent = lg_create_react_agent(llm, tools=_build_tools())
        return gr.update(value=f"<style>{css}</style>"), gr.update(value=f"✅ Preferences saved.")

    # hidden HTML to inject live CSS overrides
    css_inject = gr.HTML(value=f"<style>{APP_CSS}</style>", visible=True)

    btn_apply_prefs.click(_apply_prefs, inputs=[chroma_only_chk, stream_trace_chk, row_limit, timeout, density, generate_only_chk, recursion_limit_radio], outputs=[css_inject, settings_msg])

    def _test_conn():
        try:
            with get_connection() as conn:
                c = conn.cursor()
                c.execute("SELECT 1")
                _ = c.fetchall()
            return "✅ Connection OK"
        except Exception as e:
            return f"❌ Connection failed: {e}"

    btn_test_conn.click(_test_conn, outputs=settings_msg)
    btn_bootstrap.click(lambda: ensure_chroma_bootstrap(sample_values=True, force_refresh=True), outputs=settings_msg)
    btn_reload.click(lambda: reload_config_and_refresh(), outputs=settings_msg)
    
    # Embedding Management UI event handlers
    btn_view_embeddings.click(lambda: format_embedding_summary(), outputs=embedding_summary)
    btn_refresh_embeddings_view.click(lambda: format_embedding_summary(), outputs=embedding_summary)
    btn_selective_refresh.click(
        refresh_embeddings_selective,
        inputs=[chk_refresh_tables, chk_refresh_columns, chk_refresh_relationships, chk_refresh_values],
        outputs=selective_refresh_msg
    )

    def _search_tbl(q):
        res = search_tables(q or "", top_k=20)
        return "\n".join([f"- {r}" for r in res])

    def _search_col(q):
        res = search_columns(q or "", top_k=30)
        return "\n".join([f"- {r}" for r in res])

    btn_tbl.click(_search_tbl, inputs=tbl_query, outputs=tbl_out)
    btn_col.click(_search_col, inputs=col_query, outputs=col_out)

    def _hist_refresh():
        return gr.update(choices=_history_choices()), _format_history()

    btn_hist_refresh.click(_hist_refresh, outputs=[hist_select, hist_md])

    def _hist_view(qid):
        return _history_details(qid)

    btn_hist_view.click(_hist_view, inputs=hist_select, outputs=hist_details)

    def _hist_rerun(qid):
        entry = next((q for q in QUERY_HISTORY if q["id"] == qid), None)
        if not entry:
            return "Select a valid history entry."
        return f"Paste into chat: {entry.get('question')}"

    btn_hist_rerun.click(_hist_rerun, inputs=hist_select)

    def _hist_clear():
        QUERY_HISTORY.clear()
        return gr.update(choices=[]), "History cleared.", ""

    btn_hist_clear.click(_hist_clear, outputs=[hist_select, hist_md, hist_details])

    def _clear_chat():
        # Clear only the chat UI; do not touch query history
        return []

    clear_chat_btn.click(_clear_chat, outputs=chat)

    # Disable button, show spinner, run, then enable and hide spinner
    def _disable_btn():
        return gr.update(interactive=False), gr.update(visible=True)

    def _enable_btn_done():
        return gr.update(interactive=True), gr.update(visible=False)

    def _append_chat_user(q, history):
        history = history or []
        history.append({"role": "user", "content": q})
        return history

    def _append_chat_assistant(ans, history):
        history = history or []
        history.append({"role": "assistant", "content": ans or ""})
        return history

    def _clear_query_box():
        return gr.update(value="")

    run_btn.click(_disable_btn, outputs=[run_btn, chat_spinner])\
        .then(_append_chat_user, inputs=[query_box, chat], outputs=chat)\
        .then(run_controller, inputs=[query_box, chroma_only_chk, stream_trace_chk, row_limit, timeout], outputs=[results_md, trace_md, sql_md])\
        .then(_append_chat_assistant, inputs=[results_md, chat], outputs=chat)\
        .then(lambda: gr.update(headers=[], value=[], column_count=0), outputs=[results_table])\
        .then(_clear_query_box, outputs=query_box)\
        .then(_enable_btn_done, outputs=[run_btn, chat_spinner])
    def _paginate(res: Dict[str, Any], page: int, page_size: int = 25):
        if not res or "columns" not in res:
            return gr.update(headers=[], value=[], column_count=0), 1
        cols = res.get("columns", [])
        rows = res.get("rows", []) or []
        total_pages = max(1, math.ceil(len(rows) / page_size))
        page = max(1, min(int(page or 1), total_pages))
        start = (page - 1) * page_size
        end = start + page_size
        page_rows = rows[start:end]
        return gr.update(headers=cols, value=page_rows, column_count=len(cols)), total_pages

    def run_controller(q, chroma_only, stream_trace, rl, to):
        set_overrides(rl, to, chroma_only, stream_trace)
        # Generate a query ID and announce running
        qid = str(uuid.uuid4())
        yield f"⏳ Running... (ID: {qid})", None, None
        CURRENT_RUN["id"] = qid
        CURRENT_RUN["question"] = q
        if stream_trace:
            for triple in run_agent_stream(q, qid=qid):
                yield triple
            CURRENT_RUN["id"], CURRENT_RUN["question"] = None, None
            return
        TRACE_EVENTS.clear()
        result = agent.invoke({"messages": [SystemMessage(content=custom_instructions), HumanMessage(content=q)]})
        if isinstance(result, dict):
            msgs = result.get("messages") or []
            final = msgs[-1].content if msgs else str(result)
        else:
            final = str(result)
        # Clean leading 'Final Answer:' if present
        if isinstance(final, str):
            final = re.sub(r"^\s*Final Answer\s*:\s*", "", final, flags=re.IGNORECASE)
        sql_text = _format_sql_attempts()
        CURRENT_RUN["id"], CURRENT_RUN["question"], CURRENT_RUN["start"] = None, None, None
        yield final, _format_trace_events(), sql_text

    def _append_chat_user(q, history):
        history = history or []
        history.append({"role": "user", "content": q})
        return history

    def _append_chat_assistant(ans, history):
        history = history or []
        history.append({"role": "assistant", "content": ans or ""})
        return history

    def _clear_query_box():
        return gr.update(value="")

    def _set_results_table_from_history_simple():
        last_entry = QUERY_HISTORY[-1] if QUERY_HISTORY else None
        if not last_entry or last_entry.get("columns") is None:
            return gr.update(headers=[], value=[], column_count=0)
        cols = last_entry.get("columns")
        rows = last_entry.get("rows", [])
        return gr.update(headers=cols, value=rows, column_count=len(cols))

if __name__ == "__main__":
    # Check embeddings and optionally bootstrap on startup (lazy loading - only if needed)
    try:
        startup_msg = ensure_chroma_bootstrap(sample_values=True, force_refresh=False)
        print(f">> Startup: {startup_msg}")
    except Exception as e:
        print(f">> Warning: Failed to check/bootstrap schema cache: {e}")
        print("   You can manually refresh embeddings from Settings tab")
    demo.launch(server_port=9594, css=APP_CSS)
