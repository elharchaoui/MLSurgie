"""Single source for the package version (pyproject.toml is authoritative when installed)."""
from importlib import metadata

try:
    VERSION = metadata.version("mlsurgie")
except metadata.PackageNotFoundError:  # running from a source tree without installation
    VERSION = "0.2.0"
