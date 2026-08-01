"""Model persistence and the HTTP service that exposes it."""

from .artifacts import ArtifactRegistry, ModelArtifact, build_artifact

__all__ = ["ArtifactRegistry", "ModelArtifact", "build_artifact", "create_app"]


def create_app():
    """Lazily build the FastAPI app so importing this package stays cheap."""
    from .app import create_app as _create_app

    return _create_app()
