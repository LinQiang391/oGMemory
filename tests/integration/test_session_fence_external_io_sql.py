"""Final-review regressions: real SQL archives and artifact I/O, no models."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import psycopg2
import pytest

from core.models import Role
from features.tool_result_externalization.externalizer import (
    BoundedDaemonIORunner,
    ExternalizationTimeoutError,
)
from features.tool_result_externalization.store_sql import SQLToolResultArtifactStore
from fs.sql_adapter.session_mutation import SessionMutationLostError, active_session_guard
from providers.unified_config import OgMemConfig
from server.memory_service import MemoryService
from session.session_manager import SessionManager
from session.session_state import SessionState
from tests.integration.test_session_mutation_sql import ctx as ctx
from tests.integration.test_session_mutation_sql import database as database
from tests.unit.features.test_tool_result_externalization import _foundation_metadata
















def _seed_archives(fs, archive, ctx):
    with fs.session_mutation_lock(SessionState.state_uri(ctx.account_id, ctx.session_id), ctx):
        for index in range(2):
            assert archive.write_archive(session_id=ctx.session_id, ctx=ctx, archive_id=f"seed-{index}",
                overview="Original overview", abstract="Original abstract",
                messages=[{"role": "user", "content": f"Original source {index}"}]).success
