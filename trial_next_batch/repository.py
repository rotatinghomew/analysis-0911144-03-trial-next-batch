import hashlib
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

from .domain import validate


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class Conflict(ValueError):
    pass


class Repository:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS revisions(
                    revision INTEGER PRIMARY KEY AUTOINCREMENT,
                    digest TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS receipts(
                    request_key TEXT PRIMARY KEY,
                    digest TEXT NOT NULL,
                    revision INTEGER NOT NULL REFERENCES revisions(revision)
                );
                CREATE TABLE IF NOT EXISTS plans(
                    plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    revision INTEGER NOT NULL REFERENCES revisions(revision),
                    data_digest TEXT NOT NULL,
                    request_digest TEXT NOT NULL,
                    request TEXT NOT NULL,
                    result TEXT NOT NULL,
                    engine_version TEXT NOT NULL
                );
            ''')

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute('PRAGMA foreign_keys=ON')
        return connection

    def import_data(self, data, request_key):
        if not isinstance(request_key, str) or not request_key or len(request_key) > 200:
            raise ValueError('request_key 必须是长度不超过 200 的非空字符串')
        if not isinstance(data, dict):
            raise ValueError('资料必须是对象')
        try:
            validate(data)
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError('资料结构不完整或字段类型错误') from error
        payload = canonical(data)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            receipt = connection.execute('SELECT digest, revision FROM receipts WHERE request_key=?', (request_key,)).fetchone()
            if receipt:
                if receipt[0] != digest:
                    raise Conflict('相同请求号不能导入不同资料')
                return {'revision': receipt[1], 'digest': digest, 'replayed': True}
            connection.execute('INSERT OR IGNORE INTO revisions(digest,payload) VALUES (?,?)', (digest, payload))
            revision = connection.execute('SELECT revision FROM revisions WHERE digest=?', (digest,)).fetchone()[0]
            connection.execute('INSERT INTO receipts VALUES(?,?,?)', (request_key, digest, revision))
        return {'revision': revision, 'digest': digest, 'replayed': False}

    def get(self, revision=None):
        with self.connect() as connection:
            if revision is None:
                row = connection.execute('SELECT revision,digest,payload FROM revisions ORDER BY revision DESC LIMIT 1').fetchone()
            else:
                row = connection.execute('SELECT revision,digest,payload FROM revisions WHERE revision=?', (revision,)).fetchone()
        if row is None:
            raise LookupError('资料版本不存在')
        return {'revision': row[0], 'digest': row[1], 'data': json.loads(row[2])}

    def history(self):
        with self.connect() as connection:
            return [{'revision': row[0], 'digest': row[1]} for row in connection.execute('SELECT revision,digest FROM revisions ORDER BY revision')]

    def save_plan(self, revision, request, result):
        """Store a plan snapshot.

        The row keeps the revision and digest it was computed from, so a later
        import cannot silently rewrite what the planner saw.  The whole result
        is stored rather than re-derived on read.
        """
        if not isinstance(result, dict) or 'status' not in result:
            raise ValueError('规划结果不完整')
        snapshot = self.get(revision)
        request_payload = canonical(request)
        result_payload = canonical(result)
        created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        engine = (result.get('engine') or {}).get('version', 'unknown')
        with self.connect() as connection:
            cursor = connection.execute(
                'INSERT INTO plans(created_at,revision,data_digest,request_digest,request,result,engine_version)'
                ' VALUES (?,?,?,?,?,?,?)',
                (
                    created,
                    snapshot['revision'],
                    snapshot['digest'],
                    hashlib.sha256(request_payload.encode()).hexdigest(),
                    request_payload,
                    result_payload,
                    engine,
                ),
            )
            plan_id = cursor.lastrowid
        return {
            'plan_id': plan_id,
            'created_at': created,
            'revision': snapshot['revision'],
            'data_digest': snapshot['digest'],
            'request_digest': hashlib.sha256(request_payload.encode()).hexdigest(),
            'engine_version': engine,
        }

    def get_plan(self, plan_id):
        with self.connect() as connection:
            row = connection.execute(
                'SELECT plan_id,created_at,revision,data_digest,request_digest,request,result,engine_version'
                ' FROM plans WHERE plan_id=?',
                (plan_id,),
            ).fetchone()
        if row is None:
            raise LookupError('规划记录不存在')
        return {
            'plan_id': row[0],
            'created_at': row[1],
            'revision': row[2],
            'data_digest': row[3],
            'request_digest': row[4],
            'request': json.loads(row[5]),
            'result': json.loads(row[6]),
            'engine_version': row[7],
        }

    def list_plans(self, revision=None):
        query = 'SELECT plan_id,created_at,revision,data_digest,engine_version FROM plans'
        parameters = ()
        if revision is not None:
            query += ' WHERE revision=?'
            parameters = (revision,)
        query += ' ORDER BY plan_id'
        with self.connect() as connection:
            return [
                {
                    'plan_id': row[0],
                    'created_at': row[1],
                    'revision': row[2],
                    'data_digest': row[3],
                    'engine_version': row[4],
                }
                for row in connection.execute(query, parameters)
            ]
