"""Tests for app registration CRUD in CleoSQLite."""

import pytest

from services.data.sql.db import CleoSQLite


@pytest.fixture
def sqlite_db(tmp_path):
    db = CleoSQLite(db_path=str(tmp_path / "test.db"))
    yield db
    db.close()


def _register_app(db, name="test_app", **kwargs):
    defaults = dict(
        description="A test app",
        app_type="on_demand",
        grpc_address="localhost:50060",
        input_schema_json='{"type": "object"}',
    )
    defaults.update(kwargs)
    return db.upsert_app(name=name, **defaults)


class TestUpsertApp:
    def test_insert_returns_created_true(self, sqlite_db):
        row_id, created = _register_app(sqlite_db)
        assert row_id >= 1
        assert created is True

    def test_update_returns_created_false(self, sqlite_db):
        _register_app(sqlite_db, name="my_app")
        row_id, created = _register_app(sqlite_db, name="my_app", description="updated")
        assert row_id >= 1
        assert created is False

    def test_update_preserves_enabled_state(self, sqlite_db):
        _register_app(sqlite_db, name="my_app")
        sqlite_db.set_app_enabled("my_app", False)

        # Re-register — enabled should stay False
        _register_app(sqlite_db, name="my_app", description="v2")
        apps = sqlite_db.list_apps()
        assert len(apps) == 1
        assert apps[0]["enabled"] == 0
        assert apps[0]["description"] == "v2"

    def test_update_refreshes_fields(self, sqlite_db):
        _register_app(sqlite_db, name="my_app", grpc_address="localhost:50060")
        _register_app(sqlite_db, name="my_app", grpc_address="localhost:50099")

        apps = sqlite_db.list_apps()
        assert apps[0]["grpc_address"] == "localhost:50099"


class TestListApps:
    def test_list_all(self, sqlite_db):
        _register_app(sqlite_db, name="app_a")
        _register_app(sqlite_db, name="app_b")
        _register_app(sqlite_db, name="app_c")

        apps = sqlite_db.list_apps()
        assert len(apps) == 3

    def test_enabled_only(self, sqlite_db):
        _register_app(sqlite_db, name="app_a")
        _register_app(sqlite_db, name="app_b")
        sqlite_db.set_app_enabled("app_b", False)

        apps = sqlite_db.list_apps(enabled_only=True)
        assert len(apps) == 1
        assert apps[0]["name"] == "app_a"

    def test_filter_by_type(self, sqlite_db):
        _register_app(sqlite_db, name="app_a", app_type="on_demand")
        _register_app(sqlite_db, name="app_b", app_type="polling")

        apps = sqlite_db.list_apps(app_type="polling")
        assert len(apps) == 1
        assert apps[0]["name"] == "app_b"

    def test_filter_by_type_and_enabled(self, sqlite_db):
        _register_app(sqlite_db, name="app_a", app_type="on_demand")
        _register_app(sqlite_db, name="app_b", app_type="on_demand")
        _register_app(sqlite_db, name="app_c", app_type="polling")
        sqlite_db.set_app_enabled("app_b", False)

        apps = sqlite_db.list_apps(enabled_only=True, app_type="on_demand")
        assert len(apps) == 1
        assert apps[0]["name"] == "app_a"


class TestSetAppEnabled:
    def test_disable_and_reenable(self, sqlite_db):
        _register_app(sqlite_db, name="my_app")

        assert sqlite_db.set_app_enabled("my_app", False) is True
        apps = sqlite_db.list_apps(enabled_only=True)
        assert len(apps) == 0

        assert sqlite_db.set_app_enabled("my_app", True) is True
        apps = sqlite_db.list_apps(enabled_only=True)
        assert len(apps) == 1

    def test_nonexistent_returns_false(self, sqlite_db):
        assert sqlite_db.set_app_enabled("no_such_app", False) is False

    def test_default_enabled(self, sqlite_db):
        _register_app(sqlite_db, name="my_app")
        apps = sqlite_db.list_apps()
        assert apps[0]["enabled"] == 1
