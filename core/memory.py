"""
Memory System
=============
三层记忆架构：
  1. ConversationBuffer — 短期记忆（滑动窗口对话历史）
  2. WorkingMemory      — 工作记忆（当前任务上下文）
  3. LongTermMemory     — 长期记忆（持久化关键词检索）
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from config.settings import get_settings


# ─── 短期记忆：对话缓冲区 ─────────────────────────────────────

@dataclass
class ConversationBuffer:
    """
    Sliding-window conversation memory.
    Keeps the last `window_size` message pairs.
    """
    window_size: int = 20
    messages: list[BaseMessage] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append(HumanMessage(content=content))
        self._trim()

    def add_ai_message(self, content: str) -> None:
        self.messages.append(AIMessage(content=content))
        self._trim()

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        self._trim()

    def get_messages(self) -> list[BaseMessage]:
        return list(self.messages)

    def get_context_string(self, exclude_last: bool = False) -> str:
        """Return conversation as a formatted string for prompt injection.

        Args:
            exclude_last: Skip the most recent message. The agent adds the
                current query to the buffer before building context, so this
                avoids echoing the current query back as history.
        """
        lines = []
        msgs = self.messages[:-1] if exclude_last else self.messages
        for msg in msgs:
            if isinstance(msg, HumanMessage):
                lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Assistant: {msg.content}")
            elif isinstance(msg, SystemMessage):
                lines.append(f"System: {msg.content}")
        return "\n".join(lines)

    def to_dict(self) -> list:  
        # Serialize messages for disk persistence.  
        out = []  
        for msg in self.messages:  
            if isinstance(msg, AIMessage):  
                role = 'ai'  
            elif isinstance(msg, SystemMessage):  
                role = 'system'  
            else:  
                role = 'human'  
            out.append({'role': role, 'content': str(msg.content)})  
        return out  
  
    def load_messages(self, data: list) -> None:  
        # Restore messages previously saved by to_dict().  
        restored = []  
        for item in data:  
            role = item.get('role', 'human')  
            content = item.get('content', '')  
            if role == 'ai':  
                restored.append(AIMessage(content=content))  
            elif role == 'system':  
                restored.append(SystemMessage(content=content))  
            else:  
                restored.append(HumanMessage(content=content))  
        self.messages = restored  
        self._trim() 
    def clear(self) -> None:
        self.messages.clear()

    def _trim(self) -> None:
        """Keep only the last `window_size` messages."""
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]


# ─── 工作记忆：当前任务上下文 ──────────────────────────────────

@dataclass
class WorkingMemory:
    """
    Scratch-pad for the current task.
    Stores key-value facts, intermediate results, and observations.
    """
    scratch_pad: dict[str, Any] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    max_observations: int = 50

    def store(self, key: str, value: Any) -> None:
        self.scratch_pad[key] = value

    def retrieve(self, key: str, default: Any = None) -> Any:
        return self.scratch_pad.get(key, default)

    def add_observation(self, observation: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.observations.append(f"[{ts}] {observation}")
        if len(self.observations) > self.max_observations:
            self.observations = self.observations[-self.max_observations:]

    def get_context(self) -> str:
        """Format working memory as context string."""
        parts = []
        if self.scratch_pad:
            parts.append("=== Working Memory ===")
            for k, v in self.scratch_pad.items():
                parts.append(f"  {k}: {v}")
        if self.observations:
            parts.append("=== Observations ===")
            for obs in self.observations[-10:]:  # Last 10
                parts.append(f"  {obs}")
        return "\n".join(parts)

    def clear(self) -> None:
        self.scratch_pad.clear()
        self.observations.clear()


# ─── 长期记忆：关键词检索持久化 ──────────────────────────────────────

class LongTermMemory:
    """
    Persistent memory backed by keyword search (Chinese via character bigrams).
    Stores and retrieves past experiences, facts, and learnings.

    Storage is SQLite (stdlib). Every mutation is a statement against the shared
    database file, never a rewrite of an in-process snapshot — that is what makes
    concurrent writers safe. Several instances do coexist: each cached Agent owns
    one, and the /api/memories routes build a fresh one per request. With
    whole-file rewrites, whichever instance wrote last silently wiped everything
    the others had added since they loaded.
    """

    MAX_ENTRIES = 500
    DB_NAME = "long_term_memory.db"
    LEGACY_JSON = "long_term_memory.json"

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS memories (
            id           TEXT PRIMARY KEY,
            content      TEXT NOT NULL,
            category     TEXT NOT NULL DEFAULT 'general',
            metadata     TEXT NOT NULL DEFAULT '{}',
            keywords     TEXT NOT NULL DEFAULT '[]',
            timestamp    REAL NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0
        )
    """

    def __init__(self, persist_dir: Optional[str] = None):
        settings = get_settings()
        self.persist_dir = Path(persist_dir or (settings.data_dir / "memory"))
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.persist_dir / self.DB_NAME
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(self._SCHEMA)
            conn.commit()
        finally:
            conn.close()
        self._migrate_legacy_json()

    @contextmanager
    def _conn(self):
        """One short-lived connection per operation.

        Connections are never shared between instances or threads, and SQLite's
        own file locking serialises writers across processes, so the CLI and the
        web server can run at the same time. Connections are cheap; correctness
        is not.
        """
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "content": row["content"],
            "category": row["category"],
            "metadata": json.loads(row["metadata"] or "{}"),
            "keywords": json.loads(row["keywords"] or "[]"),
            "timestamp": row["timestamp"],
            "access_count": row["access_count"],
        }

    def _migrate_legacy_json(self) -> None:
        """Import the pre-SQLite JSON store on first use, keeping the original.

        The old file is renamed to `.bak` rather than deleted, so a bad migration
        is always recoverable by hand.
        """
        legacy = self.persist_dir / self.LEGACY_JSON
        if not legacy.exists():
            return
        try:
            entries = json.loads(legacy.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return          # unreadable — leave it in place for manual recovery
        with self._conn() as conn:
            if conn.execute("SELECT 1 FROM memories LIMIT 1").fetchone():
                return      # database already populated; do not touch the file
            for e in entries:
                if not isinstance(e, dict) or not e.get("content"):
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO memories"
                    " (id, content, category, metadata, keywords, timestamp, access_count)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        e.get("id") or uuid.uuid4().hex[:8],   # backfill pre-id entries
                        e["content"],
                        e.get("category", "general"),
                        json.dumps(e.get("metadata") or {}, ensure_ascii=False),
                        json.dumps([k for k in (e.get("keywords") or []) if isinstance(k, str)],
                                   ensure_ascii=False),
                        float(e.get("timestamp") or time.time()),
                        int(e.get("access_count") or 0),
                    ),
                )
        backup = legacy.with_suffix(".json.bak")
        if backup.exists():
            backup = legacy.with_suffix(".json.bak.%d" % int(time.time()))
        legacy.replace(backup)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize for keyword matching, including Chinese character bigrams."""
        import re
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        cjk = re.findall(r"[一-鿿]", text)
        if cjk:
            tokens.extend("".join(cjk[i:i + 2]) for i in range(len(cjk) - 1))
        return set(tokens)

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        ta, tb = LongTermMemory._tokenize(a), LongTermMemory._tokenize(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / min(len(ta), len(tb))

    def store(self, content: str, category: str = "general", metadata: Optional[dict] = None, dedup: bool = True, keywords: Optional[list] = None) -> bool:
        """Store a memory entry.

        `keywords` are short trigger phrases the user might say when recalling
        this fact (e.g. "职业", "工作"). They improve recall when the user
        rephrases the question. Returns True if stored, False if duplicate.
        """
        with self._conn() as conn:
            if dedup:
                for row in conn.execute("SELECT id, content FROM memories"):
                    if self._similarity(content, row["content"]) >= 0.8:
                        conn.execute(
                            "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                            (row["id"],),
                        )
                        return False
            conn.execute(
                "INSERT INTO memories"
                " (id, content, category, metadata, keywords, timestamp, access_count)"
                " VALUES (?, ?, ?, ?, ?, ?, 0)",
                (
                    uuid.uuid4().hex[:8],
                    content,
                    category,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    json.dumps([k for k in (keywords or []) if isinstance(k, str)],
                               ensure_ascii=False),
                    time.time(),
                ),
            )
            # Cap memory size: drop the oldest entries beyond MAX_ENTRIES
            conn.execute(
                "DELETE FROM memories WHERE rowid NOT IN"
                " (SELECT rowid FROM memories ORDER BY rowid DESC LIMIT ?)",
                (self.MAX_ENTRIES,),
            )
        return True

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry by id. Returns True if one was removed."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cur.rowcount > 0

    def delete_matching(self, keyword: str) -> int:
        """Delete all entries matching a keyword (content or recall keywords)."""
        tokens = self._tokenize(keyword)
        with self._conn() as conn:
            victims = [
                row["id"] for row in
                conn.execute("SELECT id, content, keywords FROM memories")
                if tokens & (self._tokenize(row["content"])
                             | self._tokenize(" ".join(json.loads(row["keywords"] or "[]"))))
            ]
            conn.executemany("DELETE FROM memories WHERE id = ?", [(i,) for i in victims])
            return len(victims)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Keyword search over stored memories, matching content AND recall keywords.

        Works for Chinese via character bigrams.
        """
        query_tokens = self._tokenize(query)
        with self._conn() as conn:
            scored = []
            for row in conn.execute("SELECT * FROM memories ORDER BY rowid"):
                entry = self._row_to_entry(row)
                score = (len(query_tokens & self._tokenize(entry["content"]))
                         + len(query_tokens & self._tokenize(" ".join(entry["keywords"]))))
                if score > 0:
                    scored.append((score, entry))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [e for _, e in scored[:top_k]]
            # Bump access counts with a targeted UPDATE: reading no longer
            # rewrites the whole store, so a search can't clobber a concurrent
            # writer's entries the way it used to.
            if results:
                conn.executemany(
                    "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                    [(e["id"],) for e in results],
                )
            for r in results:
                r["access_count"] += 1
        return results

    def get_context(self, query: str, top_k: int = 3) -> str:
        """Get relevant memories as context string."""
        results = self.search(query, top_k)
        if not results:
            return ""
        lines = ["=== Relevant Past Knowledge ==="]
        for r in results:
            lines.append(f"  - [{r['category']}] {r['content']}")
        return "\n".join(lines)

    def get_all(self) -> list[dict]:
        with self._conn() as conn:
            return [self._row_to_entry(r)
                    for r in conn.execute("SELECT * FROM memories ORDER BY rowid")]

    def clear(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM memories")


# ─── 统一记忆接口 ─────────────────────────────────────────────

class Memory:
    """
    Unified memory system combining all three layers.
    """

    def __init__(self, window_size: int = 20, enable_long_term: bool = True):
        self.conversation = ConversationBuffer(window_size=window_size)
        self.working = WorkingMemory()
        self.long_term = LongTermMemory() if enable_long_term else None

    def add_user_message(self, content: str) -> None:
        self.conversation.add_user_message(content)

    def add_ai_message(self, content: str) -> None:
        self.conversation.add_ai_message(content)

    def store_fact(self, fact: str, category: str = "general", keywords: Optional[list] = None) -> bool:
        """Store a fact in long-term memory. Returns True if stored (not a duplicate)."""
        if self.long_term:
            return self.long_term.store(fact, category, keywords=keywords)
        return False

    def remember(self, fact: str) -> bool:
        """Alias of store_fact: remember a durable fact about the user."""
        return self.store_fact(fact)

    def forget(self, target: str) -> int:
        """Delete a memory by id or keyword. Returns the number removed."""
        if not self.long_term:
            return 0
        if self.long_term.delete(target):
            return 1
        return self.long_term.delete_matching(target)

    def get_full_context(self, current_query: str = "") -> str:
        """
        Assemble full memory context for the LLM prompt.
        Combines conversation history, working memory, and relevant long-term memories.
        """
        parts = []

        # Short-term memory: previous conversation (the current query is added
        # to the buffer by Agent.run() and is excluded to avoid duplication).
        convo = self.conversation.get_context_string(exclude_last=True)
        if convo:
            parts.append("=== Conversation History ===")
            parts.append(convo)

        # Working memory
        wm = self.working.get_context()
        if wm:
            parts.append(wm)

        # Long-term memory (relevant to current query)
        if self.long_term and current_query:
            ltm = self.long_term.get_context(current_query)
            if ltm:
                parts.append(ltm)

        return "\n\n".join(parts)

    def clear_all(self) -> None:
        self.conversation.clear()
        self.working.clear()

    def save_session_summary(self, summary: str) -> None:
        """Save a session summary to long-term memory."""
        if self.long_term:
            self.long_term.store(summary, category="session_summary")
