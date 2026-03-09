#!/usr/bin/env python3
"""
Config Version Management

Generates version strings for feature dimension configs to enable
backward-compatible storage in InfluxDB.

When config/feature_dimensions.yaml changes, the version hash changes,
allowing old and new data to coexist with proper tagging.
"""

import hashlib
import yaml
from pathlib import Path
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConfigVersionManager:
    """
    Manages config versioning for feature vector storage.

    Uses MD5 hash of feature_dimensions.yaml content to detect changes.
    When dimensions change, automatically creates new version tag.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config version manager.

        Args:
            config_path: Path to feature_dimensions.yaml (auto-detect if None)
        """
        if config_path is None:
            # Auto-detect config path
            config_path = Path(__file__).parent.parent.parent / "config" / "feature_dimensions.yaml"

        self.config_path = config_path
        self._cached_version = None
        self._cached_hash = None
        self._cached_timestamp = None

    def get_version(self) -> str:
        """
        Get current config version string.

        Returns:
            Version string like "v1.0" or "va3f2c1b" (hash-based)

        Format:
            - Explicit version from YAML (if present): "v1.0"
            - Auto-generated from hash: "v{first_8_chars_of_md5}"
        """
        # Check if we have cached version
        if self._cached_version and self._is_cache_valid():
            return self._cached_version

        # Read config and generate version
        with open(self.config_path) as f:
            config_content = f.read()

        # Check for explicit version in YAML
        config = yaml.safe_load(config_content)
        if 'version' in config:
            version = config['version']
            logger.info(f"Using explicit config version: {version}")
        else:
            # Generate hash-based version
            config_hash = hashlib.md5(config_content.encode()).hexdigest()[:8]
            version = f"v{config_hash}"
            logger.info(f"Generated hash-based version: {version}")

        # Cache the result
        self._cached_version = version
        self._cached_hash = hashlib.md5(config_content.encode()).hexdigest()
        self._cached_timestamp = datetime.now()

        return version

    def get_full_hash(self) -> str:
        """
        Get full MD5 hash of config file.

        Returns:
            32-character MD5 hash
        """
        with open(self.config_path) as f:
            content = f.read()
        return hashlib.md5(content.encode()).hexdigest()

    def get_config_snapshot(self) -> dict:
        """
        Get complete config snapshot for storage.

        Returns:
            Dictionary with:
            - version: Version string
            - hash: Full MD5 hash
            - timestamp: When snapshot was created
            - yaml_content: Full YAML content
            - dimensions: Key dimension values
        """
        from config.dimensions import DIMS

        with open(self.config_path) as f:
            yaml_content = f.read()

        return {
            'version': self.get_version(),
            'hash': self.get_full_hash(),
            'timestamp': datetime.now().isoformat(),
            'yaml_content': yaml_content,
            'dimensions': {
                'fincoll_total': DIMS.fincoll_total,
                'senvec_total': DIMS.senvec_total,
                'model_input': DIMS.model_input,
                'model_output': DIMS.model_output,
            }
        }

    def _is_cache_valid(self, ttl_seconds: int = 300) -> bool:
        """
        Check if cached version is still valid.

        Args:
            ttl_seconds: Cache TTL in seconds (default 5 minutes)

        Returns:
            True if cache is valid
        """
        if not self._cached_timestamp:
            return False

        age = (datetime.now() - self._cached_timestamp).total_seconds()
        return age < ttl_seconds

    def invalidate_cache(self):
        """Invalidate version cache (force re-read on next get_version())."""
        self._cached_version = None
        self._cached_hash = None
        self._cached_timestamp = None


# Global singleton instance
_version_manager: Optional[ConfigVersionManager] = None


def get_config_version() -> str:
    """
    Get current config version string (singleton pattern).

    Returns:
        Version string like "v1.0" or "va3f2c1b"
    """
    global _version_manager
    if _version_manager is None:
        _version_manager = ConfigVersionManager()
    return _version_manager.get_version()


def get_config_snapshot() -> dict:
    """
    Get complete config snapshot (singleton pattern).

    Returns:
        Config snapshot dictionary
    """
    global _version_manager
    if _version_manager is None:
        _version_manager = ConfigVersionManager()
    return _version_manager.get_config_snapshot()


if __name__ == "__main__":
    # Test config version detection
    manager = ConfigVersionManager()

    print("Config Version Manager Test")
    print("=" * 60)
    print(f"Config path: {manager.config_path}")
    print(f"Version: {manager.get_version()}")
    print(f"Full hash: {manager.get_full_hash()}")
    print()

    snapshot = manager.get_config_snapshot()
    print("Snapshot:")
    print(f"  Version: {snapshot['version']}")
    print(f"  Hash: {snapshot['hash']}")
    print(f"  Timestamp: {snapshot['timestamp']}")
    print(f"  Dimensions:")
    for key, value in snapshot['dimensions'].items():
        print(f"    {key}: {value}")
    print()
    print("✅ Config version manager working correctly")
