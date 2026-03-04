"""Tests for admin endpoint helpers: _build_users_select and _ensure_admin_columns.

NOTE: These tests use self-contained stubs that replicate the production functions.
Importing webbridge.py directly requires Flask and other production deps which are
not installed in the test environment.  When the production functions change, the
corresponding stubs here must be updated to match.
"""
import unittest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stubs — exact copy of the production helpers from webbridge.py
# ---------------------------------------------------------------------------

def _build_users_select(avail):
    def _ts(c):
        return f"to_char({c}, 'YYYY-MM-DD HH24:MI') AS {c}" if c in avail else f"NULL::text AS {c}"
    def _txt(c):
        return f"COALESCE({c}, '') AS {c}" if c in avail else f"'' AS {c}"
    def _int(c, default=0):
        return f"COALESCE({c}, {default}) AS {c}" if c in avail else f"{default} AS {c}"
    if 'userid' in avail:
        uid_expr = "userid::text AS userid"
    elif 'id' in avail:
        uid_expr = "id::text AS userid"
    else:
        uid_expr = "NULL AS userid"
    if 'role_tag' in avail:
        role_expr = "COALESCE(role_tag, '') AS role_tag"
    elif 'roletag' in avail:
        role_expr = "COALESCE(roletag, '') AS role_tag"
    else:
        role_expr = "'' AS role_tag"
    jsk_col = next((c for c in ('jskillset', 'skills', 'skillset') if c in avail), None)
    jsk_expr = f"COALESCE({jsk_col}, '') AS jskillset" if jsk_col else "'' AS jskillset"
    jd_expr = ("CASE WHEN jd IS NOT NULL AND jd != '' THEN LEFT(jd, 120) ELSE '' END AS jd"
               if 'jd' in avail else "'' AS jd")
    grt_expr = ("CASE WHEN google_refresh_token IS NOT NULL AND google_refresh_token != ''"
                "     THEN 'Set' ELSE '' END AS google_refresh_token"
                if 'google_refresh_token' in avail else "'' AS google_refresh_token")
    return f"""
        SELECT
            {uid_expr},
            username,
            {_txt('cemail')},
            {_txt('password')},
            {_txt('fullname')},
            {_txt('corporation')},
            {_ts('created_at')},
            {role_expr},
            {_int('token')},
            {jd_expr},
            {jsk_expr},
            {grt_expr},
            {_ts('google_token_expires')},
            {_int('last_result_count')},
            {_txt('last_deducted_role_tag')},
            {_ts('session')},
            {_txt('useraccess')},
            {_int('target_limit', 10)}
        FROM login ORDER BY username
    """


_ENSURE_ADMIN_DDLS = [
    "ALTER TABLE login ADD COLUMN IF NOT EXISTS target_limit INTEGER DEFAULT 10",
    "ALTER TABLE login ADD COLUMN IF NOT EXISTS last_result_count INTEGER",
    "ALTER TABLE login ADD COLUMN IF NOT EXISTS last_deducted_role_tag TEXT",
    "ALTER TABLE login ADD COLUMN IF NOT EXISTS session TIMESTAMPTZ",
    "ALTER TABLE login ADD COLUMN IF NOT EXISTS google_refresh_token TEXT",
    "ALTER TABLE login ADD COLUMN IF NOT EXISTS google_token_expires TIMESTAMP",
    "ALTER TABLE login ADD COLUMN IF NOT EXISTS corporation TEXT",
    "ALTER TABLE login ADD COLUMN IF NOT EXISTS useraccess TEXT",
]


def _ensure_admin_columns(cur):
    for i, ddl in enumerate(_ENSURE_ADMIN_DDLS):
        sp = f"_adm_col_{i}"
        try:
            cur.execute(f"SAVEPOINT {sp}")
            cur.execute(ddl)
            cur.execute(f"RELEASE SAVEPOINT {sp}")
        except Exception:
            try:
                cur.execute(f"ROLLBACK TO SAVEPOINT {sp}")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildUsersSelect(unittest.TestCase):

    def test_full_schema_contains_all_aliases(self):
        """A fully-populated schema produces a SELECT with all expected aliases."""
        avail = {
            'userid', 'username', 'cemail', 'password', 'fullname', 'corporation',
            'created_at', 'role_tag', 'token', 'jd', 'jskillset',
            'google_refresh_token', 'google_token_expires',
            'last_result_count', 'last_deducted_role_tag', 'session',
            'useraccess', 'target_limit',
        }
        sql = _build_users_select(avail)
        for alias in ('userid', 'cemail', 'password', 'fullname', 'corporation',
                      'created_at', 'role_tag', 'token', 'jd', 'jskillset',
                      'google_refresh_token', 'google_token_expires',
                      'last_result_count', 'last_deducted_role_tag', 'session',
                      'useraccess', 'target_limit'):
            self.assertIn(f'AS {alias}', sql, f"Expected alias '{alias}' in SELECT")
        self.assertIn('FROM login', sql)

    def test_minimal_schema_uses_safe_fallbacks(self):
        """A login table with only username/password produces safe literal fallbacks."""
        sql = _build_users_select({'username', 'password'})
        self.assertIn('NULL AS userid', sql)
        self.assertIn("'' AS role_tag", sql)
        self.assertIn('0 AS token', sql)
        self.assertIn('10 AS target_limit', sql)
        self.assertIn('FROM login', sql)

    def test_roletag_fallback(self):
        """'roletag' column maps correctly to the role_tag alias."""
        sql = _build_users_select({'username', 'roletag'})
        self.assertIn('roletag', sql)
        self.assertIn('AS role_tag', sql)
        self.assertNotIn("'' AS role_tag", sql)

    def test_skills_fallback_for_jskillset(self):
        """'skills' column is accepted as fallback for jskillset."""
        sql = _build_users_select({'username', 'skills'})
        self.assertIn('skills', sql)
        self.assertIn('AS jskillset', sql)

    def test_id_fallback_for_userid(self):
        """'id' column is used when 'userid' is absent."""
        sql = _build_users_select({'id', 'username'})
        self.assertIn('id::text AS userid', sql)

    def test_google_refresh_token_always_masked(self):
        """google_refresh_token must never expose the raw value."""
        sql = _build_users_select({'username', 'google_refresh_token'})
        self.assertIn("'Set'", sql)
        self.assertNotIn('COALESCE(google_refresh_token', sql)

    def test_no_alias_duplicates(self):
        """Each output alias appears exactly once."""
        avail = {
            'userid', 'username', 'cemail', 'password', 'fullname', 'corporation',
            'created_at', 'role_tag', 'token', 'jd', 'jskillset',
            'google_refresh_token', 'google_token_expires',
            'last_result_count', 'last_deducted_role_tag', 'session',
            'useraccess', 'target_limit',
        }
        sql = _build_users_select(avail)
        for alias in ('userid', 'cemail', 'role_tag', 'jskillset', 'target_limit'):
            count = sql.count(f'AS {alias}')
            self.assertEqual(count, 1, f"Alias '{alias}' appears {count} times, expected 1")

    def test_missing_created_at_does_not_raise(self):
        """Query builds fine even when created_at is absent (common in older schemas)."""
        sql = _build_users_select({'username', 'password', 'userid'})
        self.assertIn('NULL::text AS created_at', sql)

    def test_missing_google_token_expires_does_not_raise(self):
        """Query builds fine even when google_token_expires is absent."""
        sql = _build_users_select({'username', 'password', 'userid'})
        self.assertIn('NULL::text AS google_token_expires', sql)


class TestEnsureAdminColumns(unittest.TestCase):

    def test_all_ddls_succeed_releases_savepoints(self):
        """When all DDLs succeed every SAVEPOINT is RELEASEd, none rolled back."""
        cur = MagicMock()
        _ensure_admin_columns(cur)
        calls = [str(c) for c in cur.execute.call_args_list]
        releases  = [c for c in calls if 'RELEASE SAVEPOINT' in c]
        rollbacks = [c for c in calls if 'ROLLBACK TO SAVEPOINT' in c]
        self.assertEqual(len(rollbacks), 0)
        self.assertEqual(len(releases), len(_ENSURE_ADMIN_DDLS))

    def test_failed_ddl_rolls_back_and_continues(self):
        """A failing DDL rolls back its savepoint; remaining DDLs still run."""
        executed = []

        def side_effect(sql, *a, **kw):
            s = str(sql)
            executed.append(s)
            # Fail the very first ADD COLUMN
            if 'ADD COLUMN' in s and len([x for x in executed if 'ADD COLUMN' in x]) == 1:
                raise Exception("column already exists")

        cur = MagicMock()
        cur.execute.side_effect = side_effect
        _ensure_admin_columns(cur)   # must not propagate

        calls = ' '.join(executed)
        # A rollback must have occurred for the failed DDL
        self.assertIn('ROLLBACK TO SAVEPOINT', calls)
        # Remaining DDLs must still have been attempted
        remaining = [x for x in executed if 'ADD COLUMN' in x]
        self.assertGreater(len(remaining), 1)

    def test_covers_all_optional_columns(self):
        """Every column used by the admin SELECT is in the DDL list."""
        ddl_text = ' '.join(_ENSURE_ADMIN_DDLS)
        for col in ('target_limit', 'last_result_count', 'last_deducted_role_tag',
                    'session', 'google_refresh_token', 'google_token_expires',
                    'corporation', 'useraccess'):
            self.assertIn(col, ddl_text, f"'{col}' missing from _ENSURE_ADMIN_DDLS")


if __name__ == '__main__':
    unittest.main()

