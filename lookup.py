"""
Distributed lookup service using Ray Serve.

This module provides a Ray Serve deployment wrapper for ShardedKV, enabling
production-ready serving with multiple replicas, load balancing, and HTTP/gRPC
access to the distributed key-value store.

Architecture:
- LookupService: Thin Ray Serve deployment that delegates all operations to ShardedKV
- Multiple replicas can be deployed for high availability
- Each replica shares access to the same set of Shard actors

Usage:
    >>> from ray import serve
    >>> from kv import ShardedKV
    >>> from lookup import LookupService
    >>>
    >>> # Create sharded store
    >>> kv = ShardedKV(num_shards=16, total_memory=32*1024**3)
    >>> kv.hydrate(dataset, key_column="id")
    >>>
    >>> # Deploy service with 4 replicas
    >>> serve.run(
    ...     LookupService.bind(kv),
    ...     name="lookup",
    ...     route_prefix="/lookup",
    ...     num_replicas=4
    ... )
"""

from ray import serve

from kv import ShardedKV


@serve.deployment
class LookupService():
    """
    Ray Serve deployment providing HTTP/gRPC access to a ShardedKV store.

    LookupService is a thin wrapper that delegates all operations to ShardedKV,
    making it accessible via Ray Serve's HTTP and gRPC interfaces. Multiple replicas
    can be deployed for high availability and load balancing, with all replicas
    sharing access to the same underlying Shard actors.

    The service exposes three main operations:
    - get(key, default): Single key lookup
    - get_batch(keys, default): Batch key lookup
    - info(): Shard statistics

    All Ray Serve deployments automatically handle ObjectRef resolution, so clients
    receive actual values rather than ObjectRefs.

    Attributes:
        sharded_kv (ShardedKV): The underlying sharded key-value store

    Example:
        >>> from ray import serve
        >>> from kv import ShardedKV
        >>> from lookup import LookupService
        >>>
        >>> # Initialize store and load data
        >>> kv = ShardedKV(num_shards=16, total_memory=32*1024**3)
        >>> kv.insert_batch([("user:1", {"name": "Alice"})])
        >>>
        >>> # Deploy with 4 replicas for load balancing
        >>> handle = serve.run(
        ...     LookupService.bind(kv),
        ...     name="lookup-service",
        ...     num_replicas=4
        ... )
        >>>
        >>> # Query via handle
        >>> user = await handle.get.remote("user:1")
        >>> users = await handle.get_batch.remote(["user:1", "user:2"])
    """

    def __init__(self, sharded_kv: ShardedKV):
        """
        Initialize the lookup service.

        Args:
            sharded_kv: ShardedKV instance to serve
        """
        self.sharded_kv = sharded_kv

    async def get(self, key, default=None):
        """
        Retrieve a single value by key.

        Delegates to ShardedKV.get() which routes the request to the appropriate
        shard using consistent hashing. Ray Serve automatically resolves the
        ObjectRef before returning the value to the client.

        Args:
            key: The key to look up
            default: Value to return if key not found (default: None)

        Returns:
            The value associated with the key, or default if not found

        Example:
            >>> # Via deployment handle
            >>> user = await handle.get.remote("user:123")
            >>>
            >>> # Via HTTP
            >>> # POST to /lookup with JSON: {"key": "user:123"}
        """
        return self.sharded_kv.get(key, default)

    async def get_batch(self, keys, default=None):
        """
        Retrieve multiple values efficiently in a batch operation.

        Delegates to ShardedKV.get_batch() which groups keys by target shard and
        makes parallel requests, significantly reducing latency compared to individual
        get() calls. Results are returned in the same order as input keys.

        Args:
            keys: List of keys to retrieve
            default: Value to return for missing keys (default: None)

        Returns:
            list: Values in the same order as input keys, with default for missing keys

        Example:
            >>> # Via deployment handle
            >>> users = await handle.get_batch.remote(["user:1", "user:2", "user:3"])
            >>>
            >>> # Via HTTP
            >>> # POST to /lookup with JSON: {"keys": ["user:1", "user:2"]}
        """
        return self.sharded_kv.get_batch(keys, default)

    async def info(self):
        """
        Retrieve metadata and statistics from all shards.

        Delegates to ShardedKV.info() to fetch information from all shards in parallel.
        Useful for monitoring shard distribution, debugging, and capacity planning.

        Returns:
            list[dict]: List of shard info dictionaries with 'index' and 'len' keys

        Example:
            >>> # Via deployment handle
            >>> stats = await handle.info.remote()
            >>> # [{'index': 0, 'len': 1523}, {'index': 1, 'len': 1491}, ...]
            >>>
            >>> total_items = sum(s['len'] for s in stats)
            >>> print(f"Total items across all shards: {total_items}")
        """
        return self.sharded_kv.info()