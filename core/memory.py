"""
Memory System
=============
三层记忆架构：
  1. ConversationBuffer — 短期记忆（滑动窗口对话历史）
  2. WorkingMemory      — 工作记忆（当前任务上下文）
  3. LongTermMemory     — 长期记忆（持久化关键词检索）
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory

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
    """

    MAX_ENTRIES = 500

    def __init__(self, persist_dir: Optional[str] = None):
        settings = get_settings()
        self.persist_dir = Path(persist_dir or (settings.data_dir / "memory"))
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._memory_file = self.persist_dir / "long_term_memory.json"
        self._entries: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self._memory_file.exists():
            with open(self._memory_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
            # Backfill ids for entries saved before ids existed
            for e in entries:
                e.setdefault("id", uuid.uuid4().hex[:8])
            return entries
        return []

    def _save(self) -> None:
        with open(self._memory_file, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False, indent=2)

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
        if dedup:
            for entry in self._entries:
                if self._similarity(content, entry.get("content", "")) >= 0.8:
                    entry["access_count"] = entry.get("access_count", 0) + 1
                    self._save()
                    return False
        entry = {
            "id": uuid.uuid4().hex[:8],
            "content": content,
            "category": category,
            "metadata": metadata or {},
            "keywords": [k for k in (keywords or []) if isinstance(k, str)],
            "timestamp": time.time(),
            "access_count": 0,
        }
        self._entries.append(entry)
        # Cap memory size: drop the oldest entries beyond MAX_ENTRIES
        if len(self._entries) > self.MAX_ENTRIES:
            self._entries = self._entries[-self.MAX_ENTRIES:]
        self._save()
        return True

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry by id. Returns True if one was removed."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.get("id") != memory_id]
        removed = len(self._entries) != before
        if removed:
            self._save()
        return removed

    def delete_matching(self, keyword: str) -> int:
        """Delete all entries matching a keyword (content or recall keywords)."""
        tokens = self._tokenize(keyword)
        before = len(self._entries)
        self._entries = [e for e in self._entries
                         if not (tokens & (self._tokenize(e.get("content", ""))
                                           | self._tokenize(" ".join(e.get("keywords", [])))))]
        removed = before - len(self._entries)
        if removed:
            self._save()
        return removed

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Keyword search over stored memories, matching content AND recall keywords.

        Works for Chinese via character bigrams.
        """
        query_tokens = self._tokenize(query)
        scored = []
        for entry in self._entries:
            content_tokens = self._tokenize(entry["content"])
            kw_tokens = self._tokenize(" ".join(entry.get("keywords", [])))
            score = len(query_tokens & content_tokens) + len(query_tokens & kw_tokens)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in scored[:top_k]]
        # Update access count
        for r in results:
            r["access_count"] = r.get("access_count", 0) + 1
        self._save()
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
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._save()


# ─── 统一记忆接口 ─────────────────────────────────────────────

class Memory:
    """
    Unified memory system combining all three layers.
    """

    def __init__(self, window_size: int = 20, enable_long_term: bool = True):
        settings = get_settings()
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
