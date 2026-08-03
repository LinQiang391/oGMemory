"""Thread-safe shared connection pool for all SQL stores.

All SQL-backed stores (SQLContextFS, SQLOutboxStore, SQLRelationStore,
SQLSessionArchiveStore) share one pool instance so that:

1. Connection count stays bounded (one pool, not N stores × pool_size).
2. Cross-store operations (e.g. write_node + outbox INSERT) can use the
   same connection for a single atomic transaction.

The pool enforces a hard cap on total connections (idle + active) via a
semaphore.  When ``pool_size`` connections are already checked out, the
next ``get_connection()`` call blocks until one is returned.
"""

import logging
import threading

try:
    import psycopg2

    _HAS_PSYCOPG2 = True
except ImportError:
    _HAS_PSYCOPG2 = False
    psycopg2 = None

logger = logging.getLogger(__name__)


class SharedConnectionPool:
    """Process-level, thread-safe psycopg2 connection pool.

    ``pool_size`` caps the total number of live connections (idle + active).
    ``get_connection()`` blocks when the cap is reached until a connection
    is returned.

    Idle connections are validated by default before reuse. Callers whose
    operations are independently bounded and retried may skip the preflight.

    Usage::

        pool = SharedConnectionPool(dsn="...", pool_size=10)
        conn = pool.get_connection()
        try:
            ...
            conn.commit()
        except:
            conn.rollback()
            raise
        finally:
            pool.return_connection(conn)
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        connect_timeout_seconds: int = 10,
    ):
        if not _HAS_PSYCOPG2:
            raise ImportError(
                "psycopg2 is required for SharedConnectionPool. "
                "Install with: pip install psycopg2-binary"
            )
        if connect_timeout_seconds < 1:
            raise ValueError("connect_timeout_seconds must be positive")
        self._connection_string = connection_string
        self._connect_timeout_seconds = connect_timeout_seconds
        self._pool_size = pool_size
        self._pool: list = []
        self._lock = threading.Lock()
        # Semaphore limits total concurrent connections to pool_size
        self._semaphore = threading.Semaphore(pool_size)

    @property
    def connection_string(self) -> str:
        return self._connection_string

    def _is_connection_alive(self, conn) -> bool:
        """Check that an idle connection is still usable."""
        if conn.closed != 0:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            conn.rollback()
            return True
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            return False

    def get_connection(
        self,
        timeout: float | None = 30.0,
        *,
        validate: bool = True,
    ):
        """Borrow a connection (blocks if pool is exhausted).

        Args:
            timeout: Seconds to wait for a free slot. ``None`` waits forever.
                     Raises ``RuntimeError`` on timeout.

        Returns:
            A live psycopg2 connection.
        """
        acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            raise RuntimeError(
                f"SharedConnectionPool: timed out waiting for a connection "
                f"slot (pool_size={self._pool_size})"
            )

        while True:
            with self._lock:
                if not self._pool:
                    break
                conn = self._pool.pop()
            if conn.closed == 0 and (
                not validate or self._is_connection_alive(conn)
            ):
                return conn
            # The acquired slot remains reserved while unusable idle
            # connections are discarded and replaced.

        try:
            return psycopg2.connect(
                self._connection_string,
                connect_timeout=self._connect_timeout_seconds,
            )
        except Exception:
            # Connection creation failed — release the slot
            self._semaphore.release()
            raise

    def return_connection(self, conn):
        """Return a connection to the pool for reuse."""
        with self._lock:
            if len(self._pool) < self._pool_size and conn.closed == 0:
                self._pool.append(conn)
                self._semaphore.release()
                return
        # Pool full or connection closed — discard
        if conn.closed == 0:
            conn.close()
        self._semaphore.release()

    def close_all(self):
        """Close every idle connection in the pool (for shutdown / tests)."""
        with self._lock:
            while self._pool:
                conn = self._pool.pop()
                if conn.closed == 0:
                    conn.close()


class PoolAdapterMixin:
    """Mixin that adds optional shared-pool support to a SQL store.

    Subclasses still define their own ``_get_connection`` / ``_return_connection``
    for the standalone (no-pool) case; this mixin wires those methods to the
    shared pool when one is provided.

    Usage in a store::

        class SQLSomeStore(PoolAdapterMixin):
            def __init__(self, connection_string=None, pool_size=5, pool=None):
                self._init_pool(pool, connection_string, pool_size)
                self._ensure_table()

            # Optional: override standalone pool methods
            def _get_connection(self): ...
            def _return_connection(self, conn): ...
    """

    def _init_pool(self, pool, connection_string, pool_size):
        """Initialise pool wiring. Call from ``__init__``."""
        self._shared_pool = pool
        if pool is not None:
            self._connection_string = pool.connection_string
            self._get_connection = pool.get_connection
            self._return_connection = pool.return_connection
        else:
            if not connection_string:
                raise ValueError(
                    "connection_string is required when pool is not provided"
                )
            self._connection_string = connection_string
            self._pool_size = pool_size
            self._pool: list = []

    @staticmethod
    def bind_tenant(conn, account_id: str) -> None:
        """Bind tenant identity to the current transaction for RLS.

        Must be called at the start of every transaction that touches
        RLS-protected tables.  Uses ``SET LOCAL`` so the binding
        auto-resets when the transaction ends and the connection returns
        to the pool — no risk of leaking identity to the next request.
        """
        with conn.cursor() as cur:
            cur.execute("SET LOCAL app.account_id = %s", (account_id,))
