"""
Unit tests for ShardedKV distributed key-value store.

Tests cover:
- Shard initialization and basic operations
- ShardedKV initialization and configuration
- Single key get/insert operations
- Batch get/insert operations
- Data hydration from Ray datasets
- Input validation and error handling
- Consistent hashing behavior
"""

import pytest
import ray
from unittest.mock import Mock, patch, MagicMock
import pyarrow as pa

from kv import Shard, ShardedKV, ShardedKVError, ShardUnavailableError, ValidationError


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for testing."""
    if not ray.is_initialized():
        try:
            # Try to initialize a new local cluster
            ray.init(num_cpus=4, ignore_reinit_error=True)
        except ValueError:
            # If already connected to a cluster, just use it
            ray.init(ignore_reinit_error=True)
    yield
    # Don't shutdown Ray - it may be used by other tests or the environment


class TestShard:
    """Test cases for individual Shard actors."""

    def test_shard_initialization(self, ray_context):
        """Test that a shard initializes correctly."""
        shard = Shard.remote(index=0)
        info = ray.get(shard.info.remote())
        assert info['index'] == 0
        assert info['len'] == 0

    def test_shard_insert_and_get(self, ray_context):
        """Test basic insert and get operations on a shard."""
        shard = Shard.remote(index=0)

        # Insert a value
        ray.get(shard.insert.remote("key1", "value1"))

        # Retrieve the value
        result = ray.get(shard.get.remote("key1"))
        assert result == "value1"

    def test_shard_get_default(self, ray_context):
        """Test that get returns default value for missing keys."""
        shard = Shard.remote(index=0)
        result = ray.get(shard.get.remote("nonexistent", default="default_value"))
        assert result == "default_value"

    def test_shard_batch_operations(self, ray_context):
        """Test batch insert and get operations."""
        shard = Shard.remote(index=0)

        # Batch insert
        items = [("key1", "value1"), ("key2", "value2"), ("key3", "value3")]
        ray.get(shard.insert_batch.remote(items))

        # Batch get
        keys = ["key1", "key2", "key3", "nonexistent"]
        results = ray.get(shard.get_batch.remote(keys, default=None))
        assert results == ["value1", "value2", "value3", None]

    def test_shard_info(self, ray_context):
        """Test that shard info returns correct metadata."""
        shard = Shard.remote(index=5)
        ray.get(shard.insert.remote("key1", "value1"))
        ray.get(shard.insert.remote("key2", "value2"))

        info = ray.get(shard.info.remote())
        assert info['index'] == 5
        assert info['len'] == 2


class TestShardedKVInitialization:
    """Test cases for ShardedKV initialization."""

    def test_initialization(self, ray_context):
        """Test that ShardedKV initializes with correct parameters."""
        total_memory = 4 * 1024 ** 3  # 4 GB
        num_shards = 4

        kv = ShardedKV(num_shards=num_shards, total_memory=total_memory)

        assert kv.num_shards == num_shards
        assert kv.memory_per_shard == total_memory // num_shards
        assert len(kv.shards) == num_shards

    def test_initialization_custom_params(self, ray_context):
        """Test initialization with custom CPU."""
        kv = ShardedKV(
            num_shards=2,
            total_memory=1024 ** 3,
            cpu_per_shard=0.5,
        )

        assert kv.cpu_per_shard == 0.5

    def test_len_empty(self, ray_context):
        """Test that len returns 0 for empty store."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)
        assert len(kv) == 0

    def test_iteration(self, ray_context):
        """Test that iteration over shards works."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)
        shard_count = sum(1 for _ in kv)
        assert shard_count == 4


class TestShardedKVValidation:
    """Test cases for input validation."""

    def test_validate_key_invalid_type(self, ray_context):
        """Test that non-string keys are rejected."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        with pytest.raises(ValidationError, match="Key must be a string"):
            kv._validate_key(123)

    def test_validate_key_empty(self, ray_context):
        """Test that empty keys are rejected."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        with pytest.raises(ValidationError, match="Key cannot be empty"):
            kv._validate_key("")

    def test_validate_key_too_large(self, ray_context):
        """Test that oversized keys are rejected."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)
        large_key = "x" * 1025

        with pytest.raises(ValidationError, match="Key too large"):
            kv._validate_key(large_key)

    def test_validate_value_none(self, ray_context):
        """Test that None values are rejected."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        with pytest.raises(ValidationError, match="Value cannot be None"):
            kv._validate_value(None)

    def test_validate_keys_not_list(self, ray_context):
        """Test that non-list keys parameter is rejected."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        with pytest.raises(ValidationError, match="Keys must be a list"):
            kv._validate_keys("not_a_list")

    def test_validate_keys_empty_list(self, ray_context):
        """Test that empty keys list is rejected."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        with pytest.raises(ValidationError, match="Keys list cannot be empty"):
            kv._validate_keys([])

    def test_validate_items_invalid_tuple(self, ray_context):
        """Test that invalid item tuples are rejected."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        with pytest.raises(ValidationError, match="must be a \\(key, value\\) tuple"):
            kv._validate_items([("key1",)])  # Only one element

    def test_validate_items_not_list(self, ray_context):
        """Test that non-list items parameter is rejected."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        with pytest.raises(ValidationError, match="Items must be a list"):
            kv._validate_items("not_a_list")


class TestShardedKVOperations:
    """Test cases for get/insert operations."""

    def test_insert_and_get(self, ray_context):
        """Test basic insert and get operations."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        # Insert
        ray.get(kv.insert("user:1", {"name": "Alice"}))

        # Get
        result = ray.get(kv.get("user:1"))
        assert result == {"name": "Alice"}

    def test_get_with_default(self, ray_context):
        """Test get with default value for missing keys."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        result = ray.get(kv.get("nonexistent", default="not_found"))
        assert result == "not_found"

    def test_getitem(self, ray_context):
        """Test dictionary-style access."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)
        ray.get(kv.insert("key1", "value1"))

        result = ray.get(kv["key1"])
        assert result == "value1"

    def test_insert_validation(self, ray_context):
        """Test that insert validates inputs."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        with pytest.raises(ValidationError):
            kv.insert("", "value")  # Empty key

        with pytest.raises(ValidationError):
            kv.insert("key", None)  # None value

    def test_batch_insert_and_get(self, ray_context):
        """Test batch operations."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        # Batch insert
        items = [
            ("user:1", {"name": "Alice"}),
            ("user:2", {"name": "Bob"}),
            ("user:3", {"name": "Carol"}),
        ]
        kv.insert_batch(items)

        # Batch get
        keys = ["user:1", "user:2", "user:3", "user:999"]
        results = kv.get_batch(keys, default=None)

        assert results[0] == {"name": "Alice"}
        assert results[1] == {"name": "Bob"}
        assert results[2] == {"name": "Carol"}
        assert results[3] is None

    def test_len_after_inserts(self, ray_context):
        """Test that len returns correct count after inserts."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        items = [(f"key{i}", f"value{i}") for i in range(10)]
        kv.insert_batch(items)

        assert len(kv) == 10

    def test_info(self, ray_context):
        """Test that info returns shard statistics."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        items = [(f"key{i}", f"value{i}") for i in range(10)]
        kv.insert_batch(items)

        info = kv.info()
        assert len(info) == 4
        total_items = sum(shard['len'] for shard in info)
        assert total_items == 10


class TestShardedKVHashing:
    """Test cases for consistent hashing."""

    def test_shard_index_consistency(self, ray_context):
        """Test that the same key always maps to the same shard."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        key = "test_key"
        index1 = kv._shard_index_from_key(key)
        index2 = kv._shard_index_from_key(key)

        assert index1 == index2

    def test_shard_index_range(self, ray_context):
        """Test that shard indices are within valid range."""
        kv = ShardedKV(num_shards=8, total_memory=1024 ** 3)

        for i in range(100):
            key = f"key_{i}"
            index = kv._shard_index_from_key(key)
            assert 0 <= index < 8

    def test_shard_distribution(self, ray_context):
        """Test that keys are distributed across shards."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        # Insert many items
        items = [(f"key{i}", f"value{i}") for i in range(100)]
        kv.insert_batch(items)

        # Check distribution
        info = kv.info()
        for shard_info in info:
            # Each shard should have at least some items (not perfect distribution)
            assert shard_info['len'] > 0


class TestShardedKVHydrate:
    """Test cases for data hydration from Ray datasets."""

    def test_hydrate_basic(self, ray_context):
        """Test basic hydration from a dataset."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        # Create sample dataset
        data = [
            {"id": "1", "name": "Alice", "age": 30},
            {"id": "2", "name": "Bob", "age": 25},
            {"id": "3", "name": "Carol", "age": 35},
        ]
        ds = ray.data.from_items(data)

        # Hydrate
        num_rows = kv.hydrate(ds, key_column="id")

        assert num_rows == 3
        assert len(kv) == 3

    def test_hydrate_custom_key_column(self, ray_context):
        """Test hydration with custom key column."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        data = [
            {"user_id": "user:1", "name": "Alice"},
            {"user_id": "user:2", "name": "Bob"},
        ]
        ds = ray.data.from_items(data)

        num_rows = kv.hydrate(ds, key_column="user_id")

        assert num_rows == 2
        result = ray.get(kv.get("user:1"))
        assert result["name"] == "Alice"

    def test_hydrate_validation_none_dataset(self, ray_context):
        """Test that hydrate rejects None dataset."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        with pytest.raises(ValidationError, match="Dataset cannot be None"):
            kv.hydrate(None, key_column="id")

    def test_hydrate_validation_invalid_key_column(self, ray_context):
        """Test that hydrate validates key_column parameter."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)
        ds = ray.data.from_items([{"id": "1", "name": "Alice"}])

        with pytest.raises(ValidationError, match="key_column must be a non-empty string"):
            kv.hydrate(ds, key_column="")

    def test_hydrate_validation_missing_key_column(self, ray_context):
        """Test that hydrate detects missing key column."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)
        data = [{"id": "1", "name": "Alice"}]
        ds = ray.data.from_items(data)

        with pytest.raises(ValidationError, match="key_column 'user_id' not found"):
            kv.hydrate(ds, key_column="user_id")

    def test_hydrate_with_null_keys(self, ray_context):
        """Test that hydrate skips rows with null keys."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        # Create dataset with some null keys
        data = [
            {"id": "1", "name": "Alice"},
            {"id": None, "name": "Bob"},  # Null key
            {"id": "3", "name": "Carol"},
        ]
        ds = ray.data.from_items(data)

        num_rows = kv.hydrate(ds, key_column="id")

        # Should process all rows but skip the null key
        assert num_rows == 3
        # Only 2 items should be in the store (null key skipped)
        assert len(kv) == 2


class TestShardedKVErrorHandling:
    """Test cases for error handling."""

    def test_shard_unavailable_error(self, ray_context):
        """Test handling of unavailable shards."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        # Mock a shard to raise RayActorError
        with patch.object(kv.shards[0], 'get') as mock_get:
            mock_get.remote.side_effect = ray.exceptions.RayActorError()

            # Determine which key goes to shard 0
            test_key = None
            for i in range(1000):
                key = f"test_key_{i}"
                if kv._shard_index_from_key(key) == 0:
                    test_key = key
                    break

            if test_key:
                with pytest.raises(ShardUnavailableError):
                    ray.get(kv.get(test_key))

    def test_insert_batch_empty_list(self, ray_context):
        """Test that insert_batch rejects empty list."""
        kv = ShardedKV(num_shards=4, total_memory=1024 ** 3)

        with pytest.raises(ValidationError, match="Items list cannot be empty"):
            kv.insert_batch([])


class TestShardedKVIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, ray_context):
        """Test a complete insert-query workflow."""
        kv = ShardedKV(num_shards=8, total_memory=8 * 1024 ** 3)

        # Insert data
        users = [
            (f"user:{i}", {"name": f"User{i}", "age": 20 + i})
            for i in range(50)
        ]
        kv.insert_batch(users)

        # Verify length
        assert len(kv) == 50

        # Query individual items
        user_10 = ray.get(kv.get("user:10"))
        assert user_10["name"] == "User10"

        # Query batch
        keys = [f"user:{i}" for i in range(0, 10)]
        results = kv.get_batch(keys)
        assert len(results) == 10
        assert all(r is not None for r in results)

        # Check shard distribution
        info = kv.info()
        assert len(info) == 8
        total = sum(s['len'] for s in info)
        assert total == 50
