"""
Sharded key-value store using Ray actors.

This module implements a distributed in-memory key-value store using Ray actors.
Data is sharded across multiple Ray actors for horizontal scalability.

Components:
- Shard: Ray actor storing a subset of key-value pairs
- ShardedKV: Management layer providing a unified interface to distributed shards

"""

from typing import Any, Dict, List, Iterator
import logging

import hashlib
import ray

# Module-level logger
logger = logging.getLogger(__name__)


# Custom exceptions
class ShardedKVError(Exception):
    """Base exception for ShardedKV errors."""
    pass


class ShardUnavailableError(ShardedKVError):
    """Raised when a shard is unavailable."""
    pass


class ValidationError(ShardedKVError):
    """Raised when input validation fails."""
    pass


@ray.remote
class Shard():
    """
    A Ray actor representing a single shard of the distributed lookup table.

    Each shard stores a subset of the total data in memory, allowing horizontal
    scaling across multiple nodes. Shards are stateful actors that persist data
    in memory for fast lookups.

    TODO: Data Reliability Improvements Needed for Production
    - Add persistence/snapshotting to disk for recovery after restarts
    - Implement replication across multiple nodes (primary + replicas)
    - Add health check mechanism to detect shard failures
    - Implement automatic failover to replica shards
    - Add periodic checkpointing for faster recovery
    - Consider write-ahead logging (WAL) for durability guarantees
    - Add shard backup/restore functionality
    - Implement data consistency verification (checksums, validation)

    Attributes:
        d (dict): In-memory dictionary storing key-value pairs for this shard
        index (int): The shard's index number for identification
    """

    def __init__(self, index: int):
        """
        Initialize a lookup shard.

        Args:
            index (int): Unique identifier for this shard
        """
        self.d: Dict[str, Any] = {}
        self.index = index

    def get(self, key, default=None):
        """
        Retrieve a value from this shard.

        Args:
            key: The key to look up
            default: Value to return if key is not found (default: None)

        Returns:
            The value associated with the key, or default if not found
        """
        return self.d.get(key, default)

    def get_batch(self, keys, default=None):
        """
        Retrieve multiple values from this shard in a single operation.

        This is more efficient than multiple individual get() calls as it
        reduces the number of remote method invocations.

        Args:
            keys: List of keys to look up
            default: Value to return for keys that are not found (default: None)

        Returns:
            list: List of values corresponding to the input keys, in the same order.
                 Missing keys will have the default value.
        """
        return [self.d.get(key, default) for key in keys]

    def insert(self, key, value):
        """
        Insert a single key-value pair into this shard.

        Args:
            key: The key to insert
            value: The value to associate with the key
        """
        self.d[key] = value

    def insert_batch(self, items: list[tuple[Any, Any]]):
        """
        Insert multiple key-value pairs in a single operation.

        This is more efficient than multiple individual insert() calls as it
        reduces the number of remote method invocations.

        Args:
            items: List of (key, value) tuples to insert
        """
        for key, value in items:
            self.d[key] = value

    def info(self):
        """
        Get metadata about this shard.

        Returns:
            dict: Dictionary containing shard index and number of items stored
        """
        return {
            "index": self.index,
            "len": len(self.d),
        }


class ShardedKV:
    """
    Distributed in-memory key-value store with consistent hashing across Ray actors.

    ShardedKV manages a collection of Shard actors and provides a unified interface
    for storing and retrieving data. Keys are distributed across shards using consistent
    hashing (MD5) for even load distribution.

    The class supports both direct usage and integration with Ray Serve via LookupService
    for production deployments with multiple replicas and load balancing.

    Key Operations:
        - get(key) / sharded_kv[key]: Retrieve a single value
        - get_batch(keys): Retrieve multiple values efficiently
        - insert(key, value): Insert a single key-value pair
        - insert_batch(items): Insert multiple key-value pairs efficiently
        - hydrate(dataset): Bulk load from Ray Dataset
        - len(sharded_kv): Total items across all shards
        - for shard in sharded_kv: Iterate over shard actors

    Attributes:
        shards (List): List of Shard actor handles
        num_shards (int): Total number of shards
        memory_per_shard (int): Memory allocated per shard in bytes
        cpu_per_shard (float): CPU cores allocated per shard
        shard_name_prefix (str): Prefix for naming shard actors

    Example:
        >>> # Create a sharded store with 16 shards and 32GB total memory
        >>> kv = ShardedKV(num_shards=16, total_memory=32*1024**3)
        >>>
        >>> # Insert data
        >>> kv.insert_batch([("user:1", {"name": "Alice"}), ("user:2", {"name": "Bob"})])
        >>>
        >>> # Query data
        >>> user = kv["user:1"]  # Dictionary-style access
        >>> users = kv.get_batch(["user:1", "user:2"])  # Batch query
        >>>
        >>> # Hydrate from Ray Dataset
        >>> ds = ray.data.read_parquet("s3://bucket/data.parquet")
        >>> kv.hydrate(ds, key_column="user_id")
        >>>
        >>> # Check size
        >>> print(f"Total items: {len(kv)}")
    """

    def __init__(
        self,
        num_shards: int,
        total_memory: int,
        cpu_per_shard: float = 0.1,
    ):
        """
        Initialize a sharded key-value store.

        Creates the specified number of Shard actors with evenly distributed memory
        allocation.

        Args:
            num_shards: Number of shards to create (determines parallelism)
            total_memory: Total memory in bytes to allocate across all shards
            cpu_per_shard: CPU cores per shard (default: 0.1 for lightweight actors)

        Example:
            >>> # 16 shards, 32GB total, 0.1 CPU per shard
            >>> kv = ShardedKV(num_shards=16, total_memory=32*1024**3)
        """
        self.num_shards = num_shards
        self.memory_per_shard = total_memory // num_shards
        self.cpu_per_shard = cpu_per_shard

        # Create shard actors with resource specifications (anonymous actors)
        self.shards = [
            Shard.options(
                num_cpus=cpu_per_shard,
                memory=self.memory_per_shard,
            ).remote(index=i)
            for i in range(num_shards)
        ]

    # ==================== Magic Methods ====================

    def __getitem__(self, key: str) -> ray.ObjectRef:
        """
        Look up a value by key using dictionary-style syntax.

        Returns an ObjectRef like get() for consistency. Use ray.get() to resolve
        the value, or use a regular get() call with a default parameter.

        Args:
            key: The key to look up

        Returns:
            ray.ObjectRef: Reference to the value (raises KeyError if not found)

        Raises:
            ValidationError: If key is invalid
            ShardUnavailableError: If the shard is unavailable

        Example:
            >>> sharded_kv = ShardedKV(num_shards=4, total_memory=1024**3)
            >>> obj_ref = sharded_kv["my_key"]
            >>> value = ray.get(obj_ref)  # Resolve the ObjectRef
        """
        # Validate the key
        self._validate_key(key)

        # get() will raise if key not found (no default)
        return self.get(key)

    def __len__(self) -> int:
        """
        Return the total number of items stored across all shards.

        Returns:
            int: Total count of key-value pairs in the store

        Example:
            >>> sharded_kv = ShardedKV(num_shards=4, total_memory=1024**3)
            >>> len(sharded_kv)
            1000
        """
        info = self.info()
        return sum(shard_info['len'] for shard_info in info)

    def __iter__(self) -> Iterator:
        """
        Iterate over the shard actors.

        This allows advanced users to access shards directly when needed.
        Regular users should use get/get_batch methods instead.

        Returns:
            iterator: Iterator over Shard actor handles

        Example:
            >>> sharded_kv = ShardedKV(num_shards=4, total_memory=1024**3)
            >>> for shard in sharded_kv:
            ...     info = ray.get(shard.info.remote())
        """
        return iter(self.shards)

    # ==================== Validation ====================

    def _validate_key(self, key: str) -> None:
        """
        Validate a key for get/insert operations.

        Args:
            key: The key to validate

        Raises:
            ValidationError: If the key is invalid
        """
        if not isinstance(key, str):
            raise ValidationError(f"Key must be a string, got {type(key).__name__}")
        if not key:
            raise ValidationError("Key cannot be empty")
        if len(key) > 1024:
            raise ValidationError(f"Key too large: {len(key)} bytes (max 1024)")

    def _validate_keys(self, keys: List[str]) -> None:
        """
        Validate a list of keys for batch operations.

        Args:
            keys: The keys to validate

        Raises:
            ValidationError: If any key is invalid
        """
        if not isinstance(keys, list):
            raise ValidationError(f"Keys must be a list, got {type(keys).__name__}")
        if not keys:
            raise ValidationError("Keys list cannot be empty")
        for key in keys:
            self._validate_key(key)

    def _validate_value(self, value: Any, max_size_bytes: int = 10 * 1024 * 1024) -> None:
        """
        Validate a value for insert operations.

        Args:
            value: The value to validate
            max_size_bytes: Maximum allowed size in bytes (default: 10MB)

        Raises:
            ValidationError: If the value is invalid
        """
        if value is None:
            raise ValidationError("Value cannot be None")

        # Estimate size (rough approximation)
        try:
            import sys
            value_size = sys.getsizeof(value)
            if value_size > max_size_bytes:
                raise ValidationError(
                    f"Value too large: {value_size} bytes (max {max_size_bytes} bytes)"
                )
        except Exception as e:
            # If we can't determine size, log warning but allow it
            logger.warning(f"Could not validate value size: {e}")

    def _validate_items(self, items: list) -> None:
        """
        Validate a list of (key, value) tuples for batch operations.

        Args:
            items: List of (key, value) tuples to validate

        Raises:
            ValidationError: If items list or any item is invalid
        """
        if not isinstance(items, list):
            raise ValidationError(f"Items must be a list, got {type(items).__name__}")
        if not items:
            raise ValidationError("Items list cannot be empty")

        for i, item in enumerate(items):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValidationError(
                    f"Item {i} must be a (key, value) tuple, got {type(item).__name__}"
                )
            key, value = item
            self._validate_key(key)
            self._validate_value(value)

    # ==================== Core Operations ====================

    def get(self, key: str, default: Any = None, timeout: float = 10.0) -> ray.ObjectRef:
        """
        Retrieve a value by key from the appropriate shard.

        Uses consistent hashing to route the request to the correct shard.
        Returns a Ray ObjectRef which must be resolved with ray.get() to obtain
        the actual value.

        Args:
            key: The key to look up
            default: Value to return if key is not found (default: None)
            timeout: Timeout in seconds (not used but kept for API consistency)

        Returns:
            ray.ObjectRef: Reference to the value (use ray.get() to resolve)

        Raises:
            ValidationError: If the key is invalid
            ShardUnavailableError: If the shard is unavailable

        Example:
            >>> kv = ShardedKV(num_shards=4, total_memory=1024**3)
            >>> obj_ref = kv.get("user:1")
            >>> value = ray.get(obj_ref)
        """
        # Validate input
        self._validate_key(key)

        try:
            shard_index = self._shard_index_from_key(key)
            shard = self.shards[shard_index]
            return shard.get.remote(key, default)
        except ray.exceptions.RayActorError as e:
            logger.error(f"Shard {shard_index} unavailable for key '{key}': {e}")
            raise ShardUnavailableError(f"Shard {shard_index} is unavailable") from e
        except Exception as e:
            logger.error(f"Unexpected error getting key '{key}': {e}")
            raise

    def insert(self, key, value):
        """
        Insert a single key-value pair into the appropriate shard.

        Uses consistent hashing to route the insert to the correct shard.
        Returns a Ray ObjectRef which completes when the insert finishes.
        For inserting multiple items efficiently, use insert_batch() instead.

        Args:
            key: The key to insert
            value: The value to associate with the key

        Returns:
            ray.ObjectRef: Reference that completes when insert finishes

        Raises:
            ValidationError: If key or value is invalid

        Example:
            >>> kv = ShardedKV(num_shards=4, total_memory=1024**3)
            >>>
            >>> # Insert single item
            >>> kv.insert("user:1", {"name": "Alice", "age": 30})
            >>> ray.get(kv.insert("user:2", {"name": "Bob"}))  # Wait for completion
            >>>
            >>> # For multiple inserts, use batch operation instead
            >>> kv.insert_batch([
            ...     ("user:3", {"name": "Carol"}),
            ...     ("user:4", {"name": "Dave"})
            ... ])
        """
        # Validate inputs
        self._validate_key(key)
        self._validate_value(value)

        shard_index = self._shard_index_from_key(key)
        shard = self.shards[shard_index]
        return shard.insert.remote(key, value)

    def get_batch(self, keys, default=None):
        """
        Retrieve multiple values efficiently in a single batch operation.

        Groups keys by their target shard and makes parallel requests to all shards,
        then reconstructs results in the original order. This is significantly more
        efficient than calling get() for each key individually, as it minimizes
        network round trips.

        Args:
            keys: List of keys to retrieve
            default: Value to return for missing keys (default: None)

        Returns:
            list: Values in the same order as input keys. Missing keys have default value.

        Example:
            >>> kv = ShardedKV(num_shards=4, total_memory=1024**3)
            >>> kv.insert_batch([
            ...     ("user:1", {"name": "Alice"}),
            ...     ("user:2", {"name": "Bob"}),
            ...     ("user:3", {"name": "Carol"})
            ... ])
            >>>
            >>> # Batch get - much faster than 3 individual gets
            >>> users = kv.get_batch(["user:1", "user:2", "user:99"])
            >>> # Returns: [{"name": "Alice"}, {"name": "Bob"}, None]
        """
        # Create items with (index, key) pairs
        indexed_keys = [(i, key) for i, key in enumerate(keys)]

        # Group keys by shard
        shard_batches = self._create_shard_batches(
            items=indexed_keys,
            key_fn=lambda item: item[1]  # Extract key from (index, key)
        )

        # Make batched requests to each shard
        futures = []
        shard_indices = []

        for shard_index, batch_items in enumerate(shard_batches):
            if batch_items:  # Only call if there are items for this shard
                shard = self.shards[shard_index]
                # Extract just the keys for the batch request
                batch_keys = [key for _, key in batch_items]
                future = shard.get_batch.remote(batch_keys, default)
                futures.append(future)
                shard_indices.append(shard_index)

        # Wait for all shard responses
        shard_results = ray.get(futures)

        # Reconstruct results in original order
        results = [None] * len(keys)

        for shard_idx, shard_result in zip(shard_indices, shard_results):
            batch_items = shard_batches[shard_idx]
            for (original_index, _), value in zip(batch_items, shard_result):
                results[original_index] = value

        return results

    def insert_batch(self, items: list[tuple[Any, Any]]):
        """
        Insert multiple key-value pairs efficiently in a single batch operation.

        Groups items by their target shard and makes parallel insert requests to all
        shards. This is significantly more efficient than calling individual insert
        operations, as it minimizes network round trips and allows shards to process
        batches in parallel.

        Args:
            items: List of (key, value) tuples to insert

        Raises:
            ValidationError: If items list or any item is invalid

        Example:
            >>> kv = ShardedKV(num_shards=4, total_memory=1024**3)
            >>>
            >>> # Insert multiple users at once
            >>> kv.insert_batch([
            ...     ("user:1", {"name": "Alice", "age": 30}),
            ...     ("user:2", {"name": "Bob", "age": 25}),
            ...     ("user:3", {"name": "Carol", "age": 35})
            ... ])
            >>>
            >>> # Verify insertion
            >>> print(len(kv))  # 3
        """
        # Validate all items before processing
        self._validate_items(items)

        # Group items by shard
        shard_batches = self._create_shard_batches(
            items=items,
            key_fn=lambda item: item[0]  # Extract key from (key, value)
        )

        # Make batched insert requests to each shard
        futures = []

        for shard_index, batch_items in enumerate(shard_batches):
            if batch_items:  # Only insert if there are items for this shard
                shard = self.shards[shard_index]
                future = shard.insert_batch.remote(batch_items)
                futures.append(future)

        # Wait for all insertions to complete
        ray.get(futures)

    def hydrate(self, dataset, key_column: str = "id"):
        """
        Bulk load data from a Ray Dataset into the sharded store.

        This method efficiently distributes data from any Ray Data source (BigQuery,
        S3, Parquet, CSV, etc.) across shards using parallel processing. Each row
        becomes a key-value pair where the key comes from the specified column and
        the value is the entire row as a dictionary.

        Implementation uses Ray Data's map_batches() for efficient parallel processing,
        converting PyArrow batches to (key, row_dict) tuples and distributing them
        via insert_batch().

        Args:
            dataset: Ray Dataset containing the data to load
            key_column: Column name to use as keys (default: "id")

        Returns:
            int: Total number of rows successfully inserted

        Raises:
            ValidationError: If key_column is invalid or dataset is None
            ShardedKVError: If hydration fails due to shard errors

        Example:
            >>> kv = ShardedKV(num_shards=16, total_memory=32*1024**3)
            >>>
            >>> # Load from Parquet
            >>> ds = ray.data.read_parquet("s3://bucket/users.parquet")
            >>> count = kv.hydrate(ds, key_column="user_id")
            >>> print(f"Loaded {count} users")
            >>>
            >>> # Load from BigQuery
            >>> ds = ray.data.read_bigquery(
            ...     project_id="my-project",
            ...     dataset="analytics",
            ...     query="SELECT * FROM users WHERE active = true"
            ... )
            >>> kv.hydrate(ds, key_column="id")
        """
        # Validate inputs
        if dataset is None:
            raise ValidationError("Dataset cannot be None")
        if not isinstance(key_column, str) or not key_column:
            raise ValidationError(f"key_column must be a non-empty string, got {type(key_column).__name__}")

        # Verify key column exists in dataset schema
        try:
            schema = dataset.schema()
            if schema is not None:
                column_names = schema.names
                if key_column not in column_names:
                    raise ValidationError(
                        f"key_column '{key_column}' not found in dataset. "
                        f"Available columns: {', '.join(column_names)}"
                    )
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # For other errors (e.g., schema not available), just log and continue
            logger.warning(f"Could not validate schema: {e}")

        def process_batch(batch):
            try:
                # Validate batch has the key column
                if key_column not in batch.column_names:
                    raise ValidationError(f"Batch missing key column '{key_column}'")

                # Convert PyArrow batch to (key, row_dict) tuples
                keys = batch[key_column].to_numpy()

                # Check for null/empty keys
                items = []
                for i in range(len(batch)):
                    key = keys[i]

                    # Skip null keys
                    if key is None or (isinstance(key, str) and not key):
                        logger.warning(f"Skipping row {i} with null/empty key")
                        continue

                    # Convert key to string for consistent hashing
                    key_str = str(key)

                    # Build row dictionary
                    try:
                        row_dict = {col: batch[col][i].as_py() for col in batch.column_names}
                        items.append((key_str, row_dict))
                    except Exception as e:
                        logger.error(f"Failed to convert row {i} to dict: {e}")
                        continue

                # Insert batch if we have valid items
                if items:
                    self.insert_batch(items)
                else:
                    logger.warning("Batch had no valid items to insert")

                return batch

            except ValidationError:
                # Re-raise validation errors
                raise
            except Exception as e:
                # Log and re-raise unexpected errors
                logger.error(f"Error processing batch during hydration: {e}")
                raise ShardedKVError(f"Batch processing failed: {e}") from e

        try:
            # Process dataset with error handling
            num_rows = dataset.map_batches(
                fn=process_batch,
                batch_format="pyarrow",
            ).count()

            logger.info(f"Successfully hydrated {num_rows} rows")
            return num_rows

        except ray.exceptions.RayTaskError as e:
            logger.error(f"Ray task error during hydration: {e}")
            raise ShardedKVError(f"Hydration failed due to Ray task error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during hydration: {e}")
            raise ShardedKVError(f"Hydration failed: {e}") from e

    # ==================== Internal Utilities ====================

    def _shard_index_from_key(self, key: str) -> int:
        """
        Determine which shard should store a given key using consistent hashing.

        Uses MD5 hashing to ensure keys are evenly distributed across shards.
        The same key will always map to the same shard index, enabling reliable
        routing for get and insert operations.

        Args:
            key: The key to hash (strings are UTF-8 encoded)

        Returns:
            int: Shard index from 0 to num_shards-1

        Example:
            >>> kv = ShardedKV(num_shards=4, total_memory=1024**3)
            >>> shard_idx = kv.shard_index_from_key("user:123")
            >>> print(f"Key 'user:123' belongs to shard {shard_idx}")
        """
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.num_shards

    def _create_shard_batches(self, items, key_fn):
        """
        Group items by their target shard for efficient batch operations.

        This internal utility method distributes items across shards using consistent
        hashing. It's the core building block for all batch operations (get_batch,
        insert_batch, and hydrate), ensuring items are properly grouped before making
        remote calls to minimize network overhead.

        Args:
            items: Iterable of items to distribute
            key_fn: Function extracting the key from each item for hashing

        Returns:
            list[list]: List of num_shards lists, where result[i] contains all items
                       destined for shard i. Empty lists for shards with no items.

        Example:
            >>> kv = ShardedKV(num_shards=4, total_memory=1024**3)
            >>> items = [("user:1", "Alice"), ("user:2", "Bob"), ("user:3", "Carol")]
            >>> # Group by key (first element of tuple)
            >>> batches = kv.create_shard_batches(items, key_fn=lambda x: x[0])
            >>> # batches[0] might contain [("user:1", "Alice")]
            >>> # batches[2] might contain [("user:2", "Bob"), ("user:3", "Carol")]
        """
        shard_batches = [[] for _ in range(self.num_shards)]

        for item in items:
            key = key_fn(item)
            shard_index = self._shard_index_from_key(str(key))
            shard_batches[shard_index].append(item)

        return shard_batches


    def info(self):
        """
        Get metadata from all shards in parallel.

        Retrieves statistics from each shard including shard index and number of
        items stored. Useful for debugging, monitoring, and verifying data distribution.

        Returns:
            list[dict]: List of dictionaries with keys 'index' and 'len' for each shard

        Example:
            >>> kv = ShardedKV(num_shards=4, total_memory=1024**3)
            >>> kv.insert_batch([("user:1", "Alice"), ("user:2", "Bob")])
            >>> info = kv.info()
            >>> # [{'index': 0, 'len': 1}, {'index': 1, 'len': 0},
            >>> #  {'index': 2, 'len': 1}, {'index': 3, 'len': 0}]
            >>>
            >>> # Check distribution
            >>> for shard_info in info:
            ...     print(f"Shard {shard_info['index']}: {shard_info['len']} items")
        """
        futures = [shard.info.remote() for shard in self.shards]
        return ray.get(futures)
