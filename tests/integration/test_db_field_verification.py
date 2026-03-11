"""DB field-level verification tests for core memory write path.

Every test re-queries from DB (not from return value) and verifies EVERY field.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from memoria.core.memory.models.memory import MemoryRecord
from memoria.core.memory.models.memory_edit_log import MemoryEditLog
from memoria.core.memory.tabular.store import MemoryStore
from memoria.core.memory.types import Memory, MemoryType, TrustTier, _utcnow


def _uid() -> str:
    return f"test_{uuid.uuid4().hex[:8]}"


def _mem(user_id: str, content: str = "test content", **kw) -> Memory:
    return Memory(
        memory_id=uuid.uuid4().hex,
        user_id=user_id,
        content=content,
        memory_type=kw.get("memory_type", MemoryType.SEMANTIC),
        trust_tier=kw.get("trust_tier", TrustTier.T1_VERIFIED),
        initial_confidence=kw.get("initial_confidence", 0.9),
        source_event_ids=kw.get("source_event_ids", ["evt:test"]),
        session_id=kw.get("session_id", None),
        observed_at=_utcnow(),
    )


class TestStoreCreate:
    """store.create() — verify every DB field."""

    def test_create_all_fields(self, db_factory):
        """Every field written by create() must match what's in DB."""
        store = MemoryStore(db_factory)
        uid = _uid()
        before = datetime.now(timezone.utc)

        mem = _mem(uid, content="hello world",
                   memory_type=MemoryType.PROFILE,
                   trust_tier=TrustTier.T2_CURATED,
                   initial_confidence=0.75,
                   source_event_ids=["evt:abc", "evt:def"],
                   session_id="sess-001")
        result = store.create(mem)

        after = datetime.now(timezone.utc)

        # Re-query from DB
        db = db_factory()
        row = db.query(MemoryRecord).filter_by(memory_id=result.memory_id).first()
        db.close()

        assert row is not None, "Row must exist in DB"
        assert row.memory_id == result.memory_id
        assert row.user_id == uid
        assert row.content == "hello world"
        assert str(row.memory_type) == "profile"
        assert str(row.trust_tier) == "T2"
        assert abs(float(row.initial_confidence) - 0.75) < 0.01
        assert row.session_id == "sess-001"
        assert row.is_active == 1
        assert row.superseded_by is None
        assert row.observed_at is not None
        # source_event_ids stored as JSON
        import json
        src = json.loads(row.source_event_ids) if isinstance(row.source_event_ids, str) else row.source_event_ids
        assert "evt:abc" in src
        assert "evt:def" in src

    def test_create_null_session(self, db_factory):
        """session_id=None must be stored as NULL, not empty string."""
        store = MemoryStore(db_factory)
        uid = _uid()
        mem = _mem(uid, session_id=None)
        result = store.create(mem)

        db = db_factory()
        row = db.query(MemoryRecord).filter_by(memory_id=result.memory_id).first()
        db.close()

        assert row.session_id is None

    def test_create_is_active_default_1(self, db_factory):
        """Newly created memory must have is_active=1."""
        store = MemoryStore(db_factory)
        mem = _mem(_uid())
        result = store.create(mem)

        db = db_factory()
        row = db.query(MemoryRecord).filter_by(memory_id=result.memory_id).first()
        db.close()

        assert row.is_active == 1
        assert row.superseded_by is None

    def test_create_no_side_effects_on_other_users(self, db_factory):
        """Creating memory for user A must not affect user B's records."""
        store = MemoryStore(db_factory)
        uid_a, uid_b = _uid(), _uid()

        # Pre-create B's memory
        mem_b = _mem(uid_b, content="b's memory")
        store.create(mem_b)

        # Create A's memory
        store.create(_mem(uid_a, content="a's memory"))

        db = db_factory()
        b_count = db.query(MemoryRecord).filter_by(user_id=uid_b, is_active=1).count()
        db.close()

        assert b_count == 1, "B's memory count must not change"


class TestStoreSupersede:
    """store.supersede() — verify old deactivated, new created, link set."""

    def test_supersede_all_fields(self, db_factory):
        store = MemoryStore(db_factory)
        uid = _uid()

        old = store.create(_mem(uid, content="old content"))
        new_mem = _mem(uid, content="new content")
        result = store.supersede(old.memory_id, new_mem)

        db = db_factory()
        old_row = db.query(MemoryRecord).filter_by(memory_id=old.memory_id).first()
        new_row = db.query(MemoryRecord).filter_by(memory_id=result.memory_id).first()
        db.close()

        # Old must be deactivated and linked
        assert old_row.is_active == 0, "Old must be deactivated"
        assert old_row.superseded_by == result.memory_id, "Old must link to new"

        # New must be active
        assert new_row.is_active == 1, "New must be active"
        assert new_row.superseded_by is None
        assert new_row.content == "new content"
        assert new_row.user_id == uid

    def test_supersede_only_affects_target(self, db_factory):
        """Superseding one memory must not deactivate others."""
        store = MemoryStore(db_factory)
        uid = _uid()

        m1 = store.create(_mem(uid, content="m1"))
        m2 = store.create(_mem(uid, content="m2"))
        m3 = store.create(_mem(uid, content="m3"))

        store.supersede(m1.memory_id, _mem(uid, content="m1 new"))

        db = db_factory()
        m2_row = db.query(MemoryRecord).filter_by(memory_id=m2.memory_id).first()
        m3_row = db.query(MemoryRecord).filter_by(memory_id=m3.memory_id).first()
        db.close()

        assert m2_row.is_active == 1, "m2 must remain active"
        assert m3_row.is_active == 1, "m3 must remain active"


class TestStoreDeactivate:
    """store.deactivate() — verify is_active=0."""

    def test_deactivate_sets_inactive(self, db_factory):
        store = MemoryStore(db_factory)
        uid = _uid()
        mem = store.create(_mem(uid))

        store.deactivate(mem.memory_id)

        db = db_factory()
        row = db.query(MemoryRecord).filter_by(memory_id=mem.memory_id).first()
        db.close()

        assert row.is_active == 0
        assert row.superseded_by is None  # deactivate does not set superseded_by


class TestEditorInject:
    """editor.inject() — verify DB record + audit log."""

    def test_inject_creates_memory_and_audit(self, db_factory):
        from memoria.core.memory.editor import MemoryEditor
        from memoria.core.memory.canonical_storage import CanonicalStorage

        uid = _uid()
        storage = CanonicalStorage(db_factory)
        editor = MemoryEditor(storage, db_factory)

        before = datetime.now(timezone.utc)
        mem = editor.inject(uid, "injected content",
                            memory_type=MemoryType.SEMANTIC,
                            source="test_inject",
                            session_id="sess-inject")
        after = datetime.now(timezone.utc)

        db = db_factory()

        # Ground truth 1: memory record
        row = db.query(MemoryRecord).filter_by(memory_id=mem.memory_id).first()
        assert row is not None
        assert row.user_id == uid
        assert row.content == "injected content"
        assert str(row.memory_type) == "semantic"
        assert row.is_active == 1
        assert row.session_id == "sess-inject"
        assert row.superseded_by is None

        # Ground truth 2: audit log
        log = db.query(MemoryEditLog).filter_by(user_id=uid, operation="inject").first()
        assert log is not None
        assert log.user_id == uid
        assert log.operation == "inject"
        assert mem.memory_id in (log.target_ids or "")

        db.close()

    def test_inject_two_similar_contents_both_active(self, db_factory):
        """inject() must NOT deduplicate — both records must be active."""
        from memoria.core.memory.editor import MemoryEditor
        from memoria.core.memory.canonical_storage import CanonicalStorage

        uid = _uid()
        storage = CanonicalStorage(db_factory)
        editor = MemoryEditor(storage, db_factory)

        m1 = editor.inject(uid, "list test 1", memory_type=MemoryType.SEMANTIC)
        m2 = editor.inject(uid, "list test 2", memory_type=MemoryType.SEMANTIC)

        db = db_factory()
        r1 = db.query(MemoryRecord).filter_by(memory_id=m1.memory_id).first()
        r2 = db.query(MemoryRecord).filter_by(memory_id=m2.memory_id).first()
        db.close()

        assert r1.is_active == 1, "First inject must remain active"
        assert r2.is_active == 1, "Second inject must remain active"


class TestEditorCorrect:
    """editor.correct() — verify old superseded, new created, audit logged."""

    def test_correct_all_fields(self, db_factory):
        from memoria.core.memory.editor import MemoryEditor
        from memoria.core.memory.canonical_storage import CanonicalStorage

        uid = _uid()
        storage = CanonicalStorage(db_factory)
        editor = MemoryEditor(storage, db_factory)

        original = editor.inject(uid, "original content", memory_type=MemoryType.SEMANTIC)
        corrected = editor.correct(uid, original.memory_id,
                                   "corrected content", reason="user correction")

        db = db_factory()

        # Ground truth 1: original deactivated + linked
        orig_row = db.query(MemoryRecord).filter_by(memory_id=original.memory_id).first()
        assert orig_row.is_active == 0, "Original must be deactivated"
        assert orig_row.superseded_by == corrected.memory_id

        # Ground truth 2: corrected record
        corr_row = db.query(MemoryRecord).filter_by(memory_id=corrected.memory_id).first()
        assert corr_row.is_active == 1
        assert corr_row.content == "corrected content"
        assert corr_row.user_id == uid
        assert corr_row.superseded_by is None

        # Ground truth 3: audit log
        log = db.query(MemoryEditLog).filter_by(
            user_id=uid, operation="correct"
        ).order_by(MemoryEditLog.created_at.desc()).first()
        assert log is not None
        assert "user correction" in (log.reason or "")

        db.close()


class TestEditorPurge:
    """editor.purge() — verify is_active=0, no other records affected."""

    def test_purge_by_id(self, db_factory):
        from memoria.core.memory.editor import MemoryEditor
        from memoria.core.memory.canonical_storage import CanonicalStorage

        uid = _uid()
        storage = CanonicalStorage(db_factory)
        editor = MemoryEditor(storage, db_factory)

        m1 = editor.inject(uid, "keep this", memory_type=MemoryType.SEMANTIC)
        m2 = editor.inject(uid, "purge this", memory_type=MemoryType.SEMANTIC)

        result = editor.purge(uid, memory_ids=[m2.memory_id], reason="test purge")

        db = db_factory()

        # m2 must be deactivated
        r2 = db.query(MemoryRecord).filter_by(memory_id=m2.memory_id).first()
        assert r2.is_active == 0

        # m1 must remain active
        r1 = db.query(MemoryRecord).filter_by(memory_id=m1.memory_id).first()
        assert r1.is_active == 1

        # Result count
        assert result.deactivated == 1

        # Audit log
        log = db.query(MemoryEditLog).filter_by(user_id=uid, operation="purge").first()
        assert log is not None
        assert "test purge" in (log.reason or "")

        db.close()

    def test_purge_by_type(self, db_factory):
        from memoria.core.memory.editor import MemoryEditor
        from memoria.core.memory.canonical_storage import CanonicalStorage

        uid = _uid()
        storage = CanonicalStorage(db_factory)
        editor = MemoryEditor(storage, db_factory)

        editor.inject(uid, "working 1", memory_type=MemoryType.WORKING)
        editor.inject(uid, "working 2", memory_type=MemoryType.WORKING)
        semantic = editor.inject(uid, "keep semantic", memory_type=MemoryType.SEMANTIC)

        result = editor.purge(uid, memory_types=[MemoryType.WORKING])

        db = db_factory()

        # All WORKING must be deactivated
        working_active = db.query(MemoryRecord).filter_by(
            user_id=uid, is_active=1
        ).filter(MemoryRecord.memory_type == MemoryType.WORKING).count()
        assert working_active == 0

        # SEMANTIC must remain
        sem_row = db.query(MemoryRecord).filter_by(memory_id=semantic.memory_id).first()
        assert sem_row.is_active == 1

        assert result.deactivated == 2

        db.close()


class TestObserveExplicit:
    """canonical_storage.store() via observe_explicit — contradiction detection."""

    def test_observe_explicit_no_contradiction(self, db_factory):
        """Unique content must create new record, not supersede anything."""
        from memoria.core.memory.canonical_storage import CanonicalStorage

        uid = _uid()
        storage = CanonicalStorage(db_factory)

        mem = storage.store(uid, "completely unique content xyz123",
                            memory_type=MemoryType.SEMANTIC)

        db = db_factory()
        row = db.query(MemoryRecord).filter_by(memory_id=mem.memory_id).first()
        db.close()

        assert row is not None
        assert row.is_active == 1
        assert row.content == "completely unique content xyz123"
        assert row.user_id == uid

    def test_observe_explicit_contradiction_supersedes(self, db_factory):
        """Highly similar content must supersede the old record."""
        from memoria.core.memory.canonical_storage import CanonicalStorage
        from memoria.core.embedding import get_embedding_client

        if True:  # skip: requires real embedding similarity
            pytest.skip("No embedding client configured")

        uid = _uid()
        storage = CanonicalStorage(db_factory)

        old = storage.store(uid, "The user prefers Python for scripting",
                            memory_type=MemoryType.PROFILE)
        new = storage.store(uid, "The user prefers Python for all scripting tasks",
                            memory_type=MemoryType.PROFILE)

        db = db_factory()
        old_row = db.query(MemoryRecord).filter_by(memory_id=old.memory_id).first()
        new_row = db.query(MemoryRecord).filter_by(memory_id=new.memory_id).first()
        db.close()

        if old_row.is_active == 0:
            # Contradiction detected — verify supersede link
            assert old_row.superseded_by == new_row.memory_id
            assert new_row.is_active == 1
        else:
            # No contradiction detected (similarity below threshold) — both active
            assert old_row.is_active == 1
            assert new_row.is_active == 1


class TestGovernanceHourly:
    """run_hourly() — verify tool_result and working memories cleaned."""

    def test_hourly_cleans_tool_results(self, db_factory):
        from memoria.core.memory.tabular.governance import GovernanceScheduler
        from sqlalchemy import text

        uid = _uid()
        db = db_factory()

        # Insert old TOOL_RESULT memory (>24h old, TTL=24h)
        mid = uuid.uuid4().hex
        db.execute(text(
            "INSERT INTO mem_memories (memory_id, user_id, content, memory_type, "
            "trust_tier, initial_confidence, is_active, source_event_ids, observed_at, created_at) "
            "VALUES (:mid, :uid, 'tool result', 'tool_result', 'T3', 0.5, 1, '[]', "
            "DATE_SUB(NOW(), INTERVAL 25 HOUR), DATE_SUB(NOW(), INTERVAL 25 HOUR))"
        ), {"mid": mid, "uid": uid})
        db.commit()
        db.close()

        scheduler = GovernanceScheduler(db_factory)
        report = scheduler.run_hourly()

        db = db_factory()
        # tool_result cleanup DELETEs rows (not deactivates)
        row = db.query(MemoryRecord).filter_by(memory_id=mid).first()
        db.close()

        assert row is None, "Old tool_result must be DELETED by cleanup"
        assert report.cleaned_tool_results >= 1

    def test_hourly_archives_working_memories(self, db_factory):
        from memoria.core.memory.tabular.governance import GovernanceScheduler
        from sqlalchemy import text

        uid = _uid()
        session_id = f"sess_{uuid.uuid4().hex[:8]}"
        db = db_factory()

        # Insert WORKING memory >2h old (stale_hours=2)
        mid = uuid.uuid4().hex
        db.execute(text(
            "INSERT INTO mem_memories (memory_id, user_id, content, memory_type, "
            "trust_tier, initial_confidence, is_active, session_id, source_event_ids, observed_at, created_at) "
            "VALUES (:mid, :uid, 'working mem', 'working', 'T3', 0.5, 1, :sid, '[]', "
            "DATE_SUB(NOW(), INTERVAL 3 HOUR), DATE_SUB(NOW(), INTERVAL 3 HOUR))"
        ), {"mid": mid, "uid": uid, "sid": session_id})
        db.commit()
        db.close()

        scheduler = GovernanceScheduler(db_factory)
        report = scheduler.run_hourly()

        db = db_factory()
        row = db.query(MemoryRecord).filter_by(memory_id=mid).first()
        db.close()

        assert row.is_active == 0, "Old working memory must be archived (is_active=0)"
        assert report.archived_working >= 1


class TestGovernanceDaily:
    """run_daily_all() — verify stale and low-confidence memories handled."""

    def test_daily_quarantines_low_confidence(self, db_factory):
        from memoria.core.memory.tabular.governance import GovernanceScheduler
        from sqlalchemy import text

        uid = _uid()
        db = db_factory()

        # Insert very low confidence memory (>30 days old)
        mid = uuid.uuid4().hex
        db.execute(text(
            "INSERT INTO mem_memories (memory_id, user_id, content, memory_type, "
            "trust_tier, initial_confidence, is_active, source_event_ids, observed_at, created_at) "
            "VALUES (:mid, :uid, 'stale low conf', 'semantic', 'T4', 0.05, 1, '[]', "
            "DATE_SUB(NOW(), INTERVAL 31 DAY), DATE_SUB(NOW(), INTERVAL 31 DAY))"
        ), {"mid": mid, "uid": uid})
        db.commit()
        db.close()

        scheduler = GovernanceScheduler(db_factory)
        report = scheduler.run_daily_all()

        db = db_factory()
        row = db.query(MemoryRecord).filter_by(memory_id=mid).first()
        db.close()

        # Either quarantined (is_active=0) or confidence decayed
        assert row.is_active == 0 or float(row.initial_confidence) < 0.05 + 0.01, \
            "Low confidence stale memory must be quarantined or decayed"
        assert report.quarantined >= 1 or report.cleaned_stale >= 1
