"""Session branching for tree-based conversation navigation.

Inspired by Pi's session tree: messages form a DAG via parent_id
references. A leaf_id pointer tracks the current position.
Branching creates a new path from any previous message.

This module adds branching capabilities on top of the existing
HistoryDB without modifying its core schema -- it uses the
metadata column to store parent_id and branch info.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .database import HistoryDB


@dataclass(frozen=True)
class BranchPoint:
    """A point in the conversation tree where branches diverge."""

    message_id: str
    role: str
    content_preview: str
    timestamp: str
    children: tuple[str, ...] = ()  # message ids of direct children


@dataclass(frozen=True)
class SessionBranch:
    """A named branch in the session tree."""

    branch_id: str
    label: str
    leaf_id: str  # current tip of this branch
    created_at: str
    provider: str = ""
    message_count: int = 0


class BranchingSession:
    """Manages branching on top of a HistoryDB session.

    Each message gets a parent_id stored in its metadata,
    forming a DAG. The session metadata tracks:
    - leaf_id: current position in the tree
    - branches: named branches with their tips
    """

    def __init__(self, db: HistoryDB, session_id: str) -> None:
        self._db = db
        self._session_id = session_id
        self._leaf_id: Optional[str] = None
        self._branches: dict[str, SessionBranch] = {}
        self._current_branch_id: Optional[str] = None
        self._load_state()

    def _load_state(self) -> None:
        """Load branching state from session metadata."""
        session = self._db.get_session(self._session_id)
        if session is None:
            return

        meta = session.get("metadata", {})
        self._leaf_id = meta.get("leaf_id")
        self._current_branch_id = meta.get("current_branch_id")
        raw_branches = meta.get("branches", {})
        for bid, data in raw_branches.items():
            self._branches[bid] = SessionBranch(
                branch_id=bid,
                label=data.get("label", bid),
                leaf_id=data.get("leaf_id", ""),
                created_at=data.get("created_at", ""),
                provider=data.get("provider", ""),
                message_count=data.get("message_count", 0),
            )
        if self._current_branch_id not in self._branches:
            self._current_branch_id = None
        if self._leaf_id is None:
            messages = self._db.get_session_messages(self._session_id)
            if messages:
                self._leaf_id = messages[-1]["id"]

    @staticmethod
    def _message_parent_map(all_messages: list[dict]) -> dict[str, Optional[str]]:
        """Resolve parent ids, inferring a linear chain for legacy sessions."""
        parent_map: dict[str, Optional[str]] = {}
        previous_id: Optional[str] = None
        for msg in all_messages:
            parent_id = msg.get("metadata", {}).get("parent_id")
            if parent_id is None and previous_id is not None:
                parent_id = previous_id
            parent_map[msg["id"]] = parent_id
            previous_id = msg["id"]
        return parent_map

    def _save_state(self) -> None:
        """Persist branching state to session metadata."""
        branches_data = {
            bid: {
                "label": b.label,
                "leaf_id": b.leaf_id,
                "created_at": b.created_at,
                "provider": b.provider,
                "message_count": b.message_count,
            }
            for bid, b in self._branches.items()
        }
        meta = {
            "leaf_id": self._leaf_id,
            "branches": branches_data,
            "current_branch_id": self._current_branch_id,
        }
        now = datetime.now(timezone.utc).isoformat()
        self._db._conn.execute(
            "UPDATE sessions SET metadata = ?, updated_at = ? WHERE id = ?",
            (json.dumps(meta), now, self._session_id),
        )
        self._db._conn.commit()

    @property
    def leaf_id(self) -> Optional[str]:
        return self._leaf_id

    @property
    def branches(self) -> dict[str, SessionBranch]:
        return dict(self._branches)

    def add_message(
        self,
        role: str,
        content: str,
        provider: str = "",
        token_count: int = 0,
    ) -> str:
        """Add a message as a child of the current leaf. Returns message id."""
        msg = self._db.add_message(
            session_id=self._session_id,
            role=role,
            content=content,
            token_count=token_count,
            metadata={"parent_id": self._leaf_id, "provider": provider},
        )
        self._leaf_id = msg["id"]
        if self._current_branch_id and self._current_branch_id in self._branches:
            branch = self._branches[self._current_branch_id]
            self._branches[self._current_branch_id] = SessionBranch(
                branch_id=branch.branch_id,
                label=branch.label,
                leaf_id=msg["id"],
                created_at=branch.created_at,
                provider=branch.provider,
                message_count=branch.message_count + 1,
            )
        self._save_state()
        return msg["id"]

    def create_branch(
        self,
        label: str,
        from_message_id: Optional[str] = None,
        provider: str = "",
    ) -> SessionBranch:
        """Create a named branch from a specific message (or current leaf)."""
        branch_point = from_message_id or self._leaf_id
        branch_id = uuid.uuid4().hex[:8]
        now = datetime.now(timezone.utc).isoformat()

        branch = SessionBranch(
            branch_id=branch_id,
            label=label,
            leaf_id=branch_point or "",
            created_at=now,
            provider=provider,
        )
        self._branches[branch_id] = branch

        # Switch to the new branch
        self._leaf_id = branch_point
        self._current_branch_id = branch_id
        self._save_state()
        return branch

    def switch_branch(self, branch_id: str) -> bool:
        """Switch the current leaf to a branch's tip. Returns success."""
        branch = self._branches.get(branch_id)
        if branch is None:
            return False
        self._leaf_id = branch.leaf_id
        self._current_branch_id = branch_id
        self._save_state()
        return True

    def navigate_to(self, message_id: str) -> bool:
        """Move the leaf pointer to an arbitrary message. Returns success."""
        messages = self._db.get_session_messages(self._session_id)
        msg_ids = {m["id"] for m in messages}
        if message_id not in msg_ids:
            return False
        self._leaf_id = message_id
        self._current_branch_id = None
        self._save_state()
        return True

    def get_path_to_leaf(self) -> list[dict]:
        """Walk backward from leaf to root via parent_id links."""
        if self._leaf_id is None:
            return []

        all_messages = self._db.get_session_messages(self._session_id)
        by_id = {m["id"]: m for m in all_messages}
        parent_map = self._message_parent_map(all_messages)

        path = []
        current_id = self._leaf_id
        visited: set[str] = set()

        while current_id and current_id in by_id and current_id not in visited:
            visited.add(current_id)
            msg = by_id[current_id]
            path.append(msg)
            parent_id = parent_map.get(current_id)
            current_id = parent_id

        path.reverse()
        return path

    def get_tree(self) -> list[BranchPoint]:
        """Build the full branch tree for visualization."""
        all_messages = self._db.get_session_messages(self._session_id)

        # Build parent -> children index
        children_of: dict[Optional[str], list[str]] = {}
        parent_map = self._message_parent_map(all_messages)
        for msg in all_messages:
            parent_id = parent_map.get(msg["id"])
            children_of.setdefault(parent_id, []).append(msg["id"])

        # Find branch points (messages with >1 child)
        branch_points = []

        for msg in all_messages:
            kids = children_of.get(msg["id"], [])
            if len(kids) > 1:
                preview = msg["content"][:60]
                if len(msg["content"]) > 60:
                    preview += "..."
                branch_points.append(BranchPoint(
                    message_id=msg["id"],
                    role=msg["role"],
                    content_preview=preview,
                    timestamp=msg["timestamp"],
                    children=tuple(kids),
                ))

        return branch_points

    def format_tree(self) -> str:
        """Render the session tree as a human-readable string."""
        all_messages = self._db.get_session_messages(self._session_id)
        if not all_messages:
            return "(empty session)"

        # Build parent -> children index
        children_of: dict[Optional[str], list[str]] = {}
        parent_map = self._message_parent_map(all_messages)
        for msg in all_messages:
            parent_id = parent_map.get(msg["id"])
            children_of.setdefault(parent_id, []).append(msg["id"])

        by_id = {m["id"]: m for m in all_messages}

        # Get the active path for highlighting
        active_ids = {m["id"] for m in self.get_path_to_leaf()}

        lines: list[str] = []

        def _render(msg_id: str, depth: int, prefix: str) -> None:
            msg = by_id.get(msg_id)
            if msg is None:
                return

            role = msg["role"]
            content = msg["content"][:50].replace("\n", " ")
            if len(msg["content"]) > 50:
                content += "..."

            marker = "*" if msg_id == self._leaf_id else " "
            active = ">" if msg_id in active_ids else " "
            lines.append(f"{prefix}{active}{marker} [{role}] {content}")

            kids = children_of.get(msg_id, [])
            for i, kid_id in enumerate(kids):
                is_last = i == len(kids) - 1
                child_prefix = prefix + ("   " if is_last else " | ")
                connector = " \\-" if is_last else " |-"
                lines.append(f"{prefix}{connector}")
                _render(kid_id, depth + 1, child_prefix)

        # Start from root messages (those with no parent or parent=None)
        roots = children_of.get(None, [])
        for root_id in roots:
            _render(root_id, 0, "")

        if not lines:
            return "(no tree structure)"

        header = f"Session tree ({len(all_messages)} messages, {len(self._branches)} branches)"
        branch_info = ""
        if self._branches:
            branch_labels = [f"  {b.label} -> {b.leaf_id[:8]}" for b in self._branches.values()]
            branch_info = "\nBranches:\n" + "\n".join(branch_labels)

        return header + "\n" + "\n".join(lines) + branch_info
