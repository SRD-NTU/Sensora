"""Storage regression tests; run with PYTHONPATH=src python3 -m unittest discover -s tests."""
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

from sensora import database, material_store
from sensora.supabase_store import SupabaseError, SupabaseStore, configured_store

URL = "https://test-project.supabase.co"
KEY = "sb_secret_test_fixture_only"
VALUES = dict(name="Test polymer", youngs_modulus_pa=100000, poisson_ratio=0.3,
              relative_permittivity=3, density_kg_m3=1000)


@contextmanager
def json_response(value):
    yield io.BytesIO(json.dumps(value).encode())


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        root_patch = patch("sensora.supabase_store.ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        env_patch = patch.dict(os.environ, {}, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def test_absent_configuration_is_an_error(self):
        with self.assertRaisesRegex(SupabaseError, "SUPABASE_URL and SUPABASE_SECRET_KEY"):
            database.read_database()

    def test_partial_configuration_does_not_fall_back(self):
        os.environ["SUPABASE_URL"] = URL
        with self.assertRaisesRegex(SupabaseError, "both"):
            database.read_database()

    def test_environment_overrides_project_secrets(self):
        (self.root / ".streamlit").mkdir()
        (self.root / ".streamlit/secrets.toml").write_text(
            f'SUPABASE_URL = "{URL}"\nSUPABASE_SECRET_KEY = "sb_secret_file"\n'
        )
        os.environ["SUPABASE_SECRET_KEY"] = KEY
        store = configured_store()
        self.assertEqual((store.url, store.secret_key), (URL, KEY))
        self.assertNotIn(KEY, repr(store))

    def test_invalid_configuration_is_rejected(self):
        for url, key in [("http://example.com", KEY), (URL + "/dashboard", KEY),
                         (URL, "sb_publishable_test")]:
            with self.subTest(url=url, key=key), patch.dict(os.environ, {
                "SUPABASE_URL": url, "SUPABASE_SECRET_KEY": key,
            }), self.assertRaises(SupabaseError):
                configured_store()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.store = SupabaseStore(URL, KEY)

    def test_read_continues_after_a_short_server_capped_page(self):
        pages = [
            [{"material_id": "F001", "record_json": {"material_id": "F001"}}],
            [{"material_id": "M001", "record_json": {"material_id": "M001"}}],
            [],
        ]
        with patch("sensora.supabase_store.urlopen", side_effect=lambda *a, **kw: json_response(pages.pop(0))) as send:
            self.assertEqual(self.store.read_records(), [{"material_id": "F001"}, {"material_id": "M001"}])
        requests = [call.args[0] for call in send.call_args_list]
        self.assertEqual(requests[0].get_header("Apikey"), KEY)
        self.assertIsNone(requests[0].get_header("Authorization"))
        self.assertEqual(parse_qs(urlsplit(requests[1].full_url).query)["material_id"], ["gt.F001"])

    def test_http_errors_are_actionable_and_do_not_echo_credentials(self):
        for status, code, expected in [(404, "PGRST205", "setup.sql"), (404, "PGRST202", "setup.sql"),
                                       (401, "", "denied"), (409, "23505", "already exists"),
                                       (400, "P0002", "no longer exists")]:
            error = HTTPError(URL, status, "failure", {}, io.BytesIO(json.dumps({
                "code": code, "message": KEY,
            }).encode()))
            with self.subTest(code=code), patch("sensora.supabase_store.urlopen", side_effect=error):
                with self.assertRaisesRegex(SupabaseError, expected) as caught:
                    self.store.read_records()
                self.assertNotIn(KEY, str(caught.exception))

    def test_mutations_are_not_retried_after_network_failure(self):
        with patch("sensora.supabase_store.urlopen", side_effect=URLError(KEY)) as send:
            with self.assertRaisesRegex(SupabaseError, "refresh"):
                self.store.add_record(VALUES)
        self.assertEqual(send.call_count, 1)

    def test_insert_uses_transactional_rpc(self):
        with patch("sensora.supabase_store.urlopen", return_value=json_response("M011")) as send:
            self.assertEqual(self.store.add_record(VALUES), "M011")
        request = send.call_args.args[0]
        self.assertTrue(request.full_url.endswith("rpc/sensora_add_material"))
        self.assertEqual(json.loads(request.data), {"p_record": VALUES})

    def test_delete_count_must_match_selection(self):
        with patch.object(SupabaseStore, "request", return_value=1):
            with self.assertRaisesRegex(SupabaseError, "unexpected deletion count"):
                self.store.delete_records(["M001", "M002"])


class StorageRoutingTests(unittest.TestCase):
    def test_remote_failures_are_reported(self):
        remote = SupabaseStore(URL, KEY)
        with patch("sensora.material_store.configured_store", return_value=remote), \
                patch.object(SupabaseStore, "request", side_effect=SupabaseError("Offline")):
            for action in [lambda: material_store.add_material("matrix", VALUES),
                           lambda: material_store.delete_materials(["M001"])]:
                with self.assertRaisesRegex(material_store.MaterialStoreError, "Offline"):
                    action()

    def test_remote_insert_validates_fields_before_sending(self):
        with patch("sensora.material_store.configured_store") as remote:
            for key, value in [("youngs_modulus_pa", float("nan")), ("poisson_ratio", 0.5),
                               ("name", " "), ("density_kg_m3", -1)]:
                with self.subTest(key=key), self.assertRaises(material_store.MaterialStoreError):
                    material_store.add_material("matrix", {**VALUES, key: value})
            remote.assert_not_called()

    def test_reads_and_writes_use_supabase(self):
        remote = SupabaseStore(URL, KEY)
        with patch("sensora.database.configured_store", return_value=remote), \
                patch("sensora.material_store.configured_store", return_value=remote), \
                patch.object(SupabaseStore, "read_records", return_value=[{"material_id": "M001"}]) as read, \
                patch.object(SupabaseStore, "add_record", return_value="M011") as add, \
                patch.object(SupabaseStore, "delete_records", return_value={"materials": 1}) as delete:
            self.assertEqual(database.read_database().material_candidates, [{"material_id": "M001"}])
            self.assertEqual(material_store.add_material("matrix", VALUES), "M011")
            self.assertEqual(material_store.delete_materials(["M011"]), {"materials": 1})
            read.assert_called_once_with()
            add.assert_called_once()
            delete.assert_called_once_with(["M011"])


if __name__ == "__main__":
    unittest.main()
