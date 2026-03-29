"""Tests for the session branching system."""

import os
import tempfile

import pytest

from cascade.history.database import HistoryDB
from cascade.history.branching import BranchingSession, BranchPoint, SessionBranch


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = HistoryDB(db_path=path)
    yield database
    database.close()
    os.unlink(path)


@pytest.fixture
def session(db):
    """Create a test session."""
    return db.create_session(provider="claude", model="test", title="Test Session")


@pytest.fixture
def branching(db, session):
    """Create a branching session."""
    return BranchingSession(db, session["id"])


class TestBranchingSession:
    """Tests for the core branching functionality."""

    def test_empty_session(self, branching):
        assert branching.leaf_id is None
        assert branching.branches == {}

    def test_add_message_updates_leaf(self, branching):
        msg_id = branching.add_message("user", "Hello", provider="claude")
        assert branching.leaf_id == msg_id

    def test_chain_of_messages(self, branching):
        id1 = branching.add_message("user", "First")
        branching.add_message("assistant", "Response 1", provider="claude")
        branching.add_message("user", "Second")
        id4 = branching.add_message("assistant", "Response 2", provider="claude")

        assert branching.leaf_id == id4

        # Path should be the full chain
        path = branching.get_path_to_leaf()
        assert len(path) == 4
        assert path[0]["id"] == id1
        assert path[3]["id"] == id4

    def test_create_branch(self, branching):
        branching.add_message("user", "Hello")
        id2 = branching.add_message("assistant", "Hi!", provider="claude")

        branch = branching.create_branch("experiment", provider="gemini")
        assert branch.label == "experiment"
        assert branch.leaf_id == id2  # branch from current leaf
        assert branch.branch_id in branching.branches

    def test_branch_and_continue(self, branching):
        branching.add_message("user", "Hello")
        branching.add_message("assistant", "Hi!", provider="claude")

        # Create branch at id2
        branch = branching.create_branch("alt-path")

        # Continue on the branch
        branching.add_message("user", "Different question")
        id4 = branching.add_message("assistant", "Alt response", provider="gemini")

        # Path should go: id1 -> id2 -> id3 -> id4
        path = branching.get_path_to_leaf()
        assert len(path) == 4
        assert path[-1]["id"] == id4
        assert branching.branches[branch.branch_id].leaf_id == id4

    def test_switch_branch(self, branching):
        branching.add_message("user", "Hello")
        id2 = branching.add_message("assistant", "Hi!", provider="claude")

        # Save current position
        main_branch = branching.create_branch("main")

        # Add more on main
        branching.add_message("user", "Continue")
        branching.add_message("assistant", "OK", provider="claude")

        # Create alt branch from id2
        branching.create_branch("alt", from_message_id=id2)
        branching.add_message("user", "Alt question")
        branching.add_message("assistant", "Alt answer", provider="gemini")

        # Leaf should be at alt branch tip
        alt_path = branching.get_path_to_leaf()
        assert alt_path[-1]["content"] == "Alt answer"

        # Switch back to main
        assert branching.switch_branch(main_branch.branch_id) is True
        main_path = branching.get_path_to_leaf()
        assert main_path[-1]["content"] == "OK"

    def test_navigate_to_invalid(self, branching):
        assert branching.navigate_to("nonexistent") is False

    def test_navigate_to_valid(self, branching):
        id1 = branching.add_message("user", "Hello")
        branching.add_message("assistant", "Hi!")
        branching.add_message("user", "More")

        assert branching.navigate_to(id1) is True
        assert branching.leaf_id == id1

    def test_get_tree_no_branches(self, branching):
        branching.add_message("user", "Hello")
        branching.add_message("assistant", "Hi!")

        tree = branching.get_tree()
        # Linear chain has no branch points
        assert len(tree) == 0

    def test_get_tree_with_branches(self, db, session, branching):
        branching.add_message("user", "Hello")
        id2 = branching.add_message("assistant", "Hi!")

        # First path
        branching.add_message("user", "Path A")

        # Go back and create second path
        branching.navigate_to(id2)
        branching.add_message("user", "Path B")

        tree = branching.get_tree()
        assert len(tree) == 1
        assert tree[0].message_id == id2
        assert len(tree[0].children) == 2

    def test_format_tree_empty(self, branching):
        result = branching.format_tree()
        assert "empty" in result.lower()

    def test_format_tree_linear(self, branching):
        branching.add_message("user", "Hello")
        branching.add_message("assistant", "Hi!")

        result = branching.format_tree()
        assert "2 messages" in result
        assert "user" in result
        assert "assistant" in result

    def test_state_persistence(self, db, session):
        """Test that branching state survives reload."""
        bs1 = BranchingSession(db, session["id"])
        id1 = bs1.add_message("user", "Hello")
        bs1.create_branch("test-branch", provider="claude")

        # Reload
        bs2 = BranchingSession(db, session["id"])
        assert bs2.leaf_id == id1
        assert "test-branch" in [b.label for b in bs2.branches.values()]

    def test_legacy_sessions_infer_linear_path(self, db, session):
        db.add_message(session["id"], role="user", content="Hello")
        db.add_message(session["id"], role="assistant", content="Hi!")
        db.add_message(session["id"], role="user", content="Continue")

        branching = BranchingSession(db, session["id"])
        path = branching.get_path_to_leaf()

        assert [msg["content"] for msg in path] == ["Hello", "Hi!", "Continue"]
        assert branching.leaf_id == path[-1]["id"]


class TestBranchPoint:
    def test_frozen(self):
        bp = BranchPoint(
            message_id="abc",
            role="user",
            content_preview="Hello",
            timestamp="2024-01-01",
            children=("child1", "child2"),
        )
        with pytest.raises(AttributeError):
            bp.role = "assistant"


class TestSessionBranch:
    def test_frozen(self):
        sb = SessionBranch(
            branch_id="abc",
            label="test",
            leaf_id="def",
            created_at="2024-01-01",
        )
        with pytest.raises(AttributeError):
            sb.label = "changed"

    def test_defaults(self):
        sb = SessionBranch(
            branch_id="abc",
            label="test",
            leaf_id="def",
            created_at="2024-01-01",
        )
        assert sb.provider == ""
        assert sb.message_count == 0
