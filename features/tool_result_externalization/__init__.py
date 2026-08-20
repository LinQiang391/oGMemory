"""Large tool-result externalization.

The package deliberately has no dependency on the HTTP or MCP transports.
"""

from .externalizer import ToolResultExternalizer
from .models import ArtifactMetadata, NormalizedMessage, ToolResultPart
from .normalizer import ToolResultNormalizer
from .turn_ledger import ToolResultTurnLedger, TurnLedgerRegistration, TurnLedgerRestore
from .policy import ToolResultPolicy
from .store_base import (
    ArtifactNotVisibleError,
    ArtifactStoreUnavailableError,
    ToolResultArtifactStore,
)
from .store_local import LocalToolResultArtifactStore
from .synopsis import ToolResultSynopsisBuilder

__all__ = [
    "ArtifactMetadata",
    "ArtifactNotVisibleError",
    "ArtifactStoreUnavailableError",
    "LocalToolResultArtifactStore",
    "NormalizedMessage",
    "ToolResultArtifactStore",
    "ToolResultExternalizer",
    "ToolResultNormalizer",
    "ToolResultPart",
    "ToolResultPolicy",
    "ToolResultSynopsisBuilder",
    "ToolResultTurnLedger",
    "TurnLedgerRegistration",
    "TurnLedgerRestore",
]
