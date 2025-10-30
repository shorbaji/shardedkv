"""
Example script demonstrating a distributed product lookup service.

This example shows how to build a scalable, distributed in-memory lookup service
using Ray and Ray Serve. It walks through a complete workflow:

1. Creating a ShardedKV store with distributed shards across the cluster
2. Reading product catalog data from BigQuery
3. Hydrating the ShardedKV store with product data
4. Deploying a LookupService with Ray Serve for high-availability access
5. Enriching downstream datasets with product information via batched lookups

Architecture:
- ShardedKV: Distributed storage layer with consistent hashing
- LookupService: Ray Serve deployment for HTTP/remote access

This is a demonstration/reference implementation showing best practices for
building distributed lookup services with Ray.
"""

import logging
from typing import Any, Dict

import ray
from ray import serve
from ray.serve.config import AutoscalingConfig

from kv import ShardedKV
from lookup import LookupService

# Module-level logger
logger = logging.getLogger(__name__)



def create_sharded_kv(config: Dict[str, Any]) -> ShardedKV:
    """
    Create and initialize a ShardedKV store.

    Args:
        config: Configuration dictionary with shard settings

    Returns:
        Initialized ShardedKV instance with distributed shards
    """
    logger.info("creating sharded key-value store ...")

    sharded_kv = ShardedKV(
        num_shards=config["num_shards"],
        total_memory=config["total_memory"],
        cpu_per_shard=config["shard_num_cpus"],
    )

    logger.info(f"created ShardedKV with {config['num_shards']} shards, "
                f"{sharded_kv.memory_per_shard} bytes memory each")

    return sharded_kv


def load_data(config: Dict[str, Any]):
    """
    Load product catalog data from BigQuery.

    Args:
        config: Configuration dictionary with BigQuery settings

    Returns:
        Ray Dataset containing product catalog data
    """
    logger.info(f"reading from BigQuery: {config['bigquery_project_id']}.{config['bigquery_dataset']}")

    # Read product catalog data from BigQuery
    ds = ray.data.read_bigquery(
        project_id=config["bigquery_project_id"],
        dataset=config["bigquery_dataset"],
    )

    return ds


def hydrate_sharded_kv(sharded_kv: ShardedKV, dataset) -> int:
    """
    Hydrate the ShardedKV store with data from a Ray dataset.

    Args:
        sharded_kv: The ShardedKV instance to populate
        dataset: Ray Dataset containing the data to load

    Returns:
        Number of rows inserted into the shards
    """
    logger.info("hydrating sharded key-value store")

    # Use ShardedKV's hydrate method to populate shards
    num_rows = sharded_kv.hydrate(dataset, key_column="product_id")

    logger.info(f"inserted {num_rows} rows into shards")

    return num_rows


def deploy_lookup_service(sharded_kv: ShardedKV, config: Dict[str, Any]):
    """
    Deploy the LookupService with Ray Serve and autoscaling.

    Args:
        sharded_kv: The hydrated ShardedKV instance
        config: Configuration dictionary with service settings

    Returns:
        Ray Serve deployment handle for the LookupService
    """
    logger.info("creating lookup service with autoscaling")

    # Create Ray Serve deployment with autoscaling for dynamic load handling
    lookup_app = LookupService.options(
        autoscaling_config=AutoscalingConfig(
            min_replicas=config["lookup_service_min_replicas"],
            max_replicas=config["lookup_service_max_replicas"],
            target_num_ongoing_requests_per_replica=config["lookup_service_target_requests_per_replica"],
        ),
        ray_actor_options={
            "num_cpus": config["lookup_service_num_cpus"]
        }
    ).bind(sharded_kv)

    # Deploy the service
    lookup_service = serve.run(lookup_app)

    logger.info(
        f"created lookup service with autoscaling "
        f"(min: {config['lookup_service_min_replicas']}, "
        f"max: {config['lookup_service_max_replicas']} replicas)"
    )

    return lookup_service


def enrich_with_product_data(batch: Dict[str, Any], lookup_service) -> Dict[str, Any]:
    """
    Enrich a batch of data with product information from the lookup service.

    This function is used in downstream data processing to add product details
    to transaction records, user interactions, or other datasets that reference
    products by ID. It uses the batched get_batch() method for efficient lookups.

    Args:
        batch: PyArrow RecordBatch containing rows with a 'product_id' column
        lookup_service: Ray Serve handle to the LookupService deployment

    Returns:
        Dict[str, Any]: The input batch enriched with a 'product_data' column

    Example:
        >>> enriched_ds = ds.map_batches(
        ...     fn=enrich_with_product_data,
        ...     fn_kwargs={"lookup_service": lookup_service},
        ...     batch_format="pyarrow",
        ... )
    """
    # Access product_id column as a list
    product_ids = batch['product_id'].to_pylist()

    # Use batched lookup for efficiency - single remote call instead of N calls
    product_data_results = lookup_service.get_batch.remote(product_ids).result()

    # Add the product_data as a new column to the batch
    # Convert PyArrow batch to dict for modification
    batch_dict = {col: batch[col].to_pylist() for col in batch.column_names}
    batch_dict['product_data'] = product_data_results

    return batch_dict


def example_usage(lookup_service):
    """
    Example usage of the lookup service to enrich a dataset.

    This function demonstrates how to:
    1. Create a sample interactions dataset with known product IDs
    2. Enrich the dataset with product information via batched lookups
    3. Display the enriched results

    Args:
        lookup_service: Ray Serve deployment handle
    """
    # Create a sample dataset with product IDs for enrichment
    # Using actual product ID format from the catalog (PROD-XXXXXXXXXX)
    # Starting from 2 since product ID 1 may not exist
    sample_data = {
        "product_id": [
            "PROD-0000000002",
            "PROD-0000000003",
            "PROD-0000000004",
            "PROD-0000000005",
            "PROD-0000000006",
        ],
        "quantity": [2, 1, 5, 3, 1],
        "customer_id": ["CUST_A", "CUST_B", "CUST_C", "CUST_A", "CUST_D"],
        "timestamp": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
    }

    logger.info(f"created sample interactions with {len(sample_data['product_id'])} product IDs")

    # Create Ray dataset from the sample data
    interactions_ds = ray.data.from_items([
        {
            "product_id": sample_data["product_id"][i],
            "quantity": sample_data["quantity"][i],
            "customer_id": sample_data["customer_id"][i],
            "timestamp": sample_data["timestamp"][i],
        }
        for i in range(len(sample_data["product_id"]))
    ])

    logger.info(f"created interactions dataset with {interactions_ds.count()} rows")

    # Enrich the interactions dataset with product information from lookup service
    enriched_ds = interactions_ds.map_batches(
        fn=enrich_with_product_data,
        fn_kwargs={"lookup_service": lookup_service},
        batch_format="pyarrow",
    )

    logger.info("enriching interactions with product data")

    # Display enriched results
    enriched_results = enriched_ds.take_all()
    logger.info(f"enriched {len(enriched_results)} rows")

    print("\n=== Enriched Dataset Sample ===")
    for i, row in enumerate(enriched_results[:3]):  # Show first 3 rows
        print(f"\nRow {i+1}:")
        for key, value in row.items():
            print(f"  {key}: {value}")


def main() -> None:
    """
    Example workflow demonstrating the distributed lookup service.

    This example orchestrates the complete lifecycle of a distributed lookup service:
    1. Configures resource allocation (memory, CPUs, replicas)
    2. Creates a ShardedKV store with 16 shards across 32GB memory
    3. Loads product catalog data from BigQuery (limited to 1000 rows for this demo)
    4. Hydrates the ShardedKV store by distributing data across shards
    5. Deploys a LookupService with 4 replicas for load balancing
    6. Demonstrates lookups by fetching sample data from the service
    7. Shows enrichment workflow by joining product data with interaction records

    Note: The ShardedKV is hydrated before the LookupService deployment to ensure
    all data is loaded before the service begins accepting requests. This is a
    recommended pattern for production use.
    """
    # Configuration for the distributed lookup service
    config: Dict[str, Any] = {
        # BigQuery settings
        "bigquery_project_id": "anyscale-gcp-vms",
        "bigquery_dataset": "tmp.catalog_published",

        # ShardedKV settings
        "total_memory": 32 * 1024 ** 3,  # 32 GB total memory across all shards
        "num_shards": 16,  # Number of shards to distribute data across
        "shard_num_cpus": 0.1,  # CPU allocation per shard

        # LookupService settings with autoscaling
        "lookup_service_min_replicas": 2,  # Minimum number of service replicas
        "lookup_service_max_replicas": 10,  # Maximum number of service replicas
        "lookup_service_target_requests_per_replica": 10,  # Autoscaling trigger threshold
        "lookup_service_num_cpus": 0.1,  # CPU allocation per service replica
    }

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("driver main")

    # Step 1: Create ShardedKV store
    sharded_kv: ShardedKV = create_sharded_kv(config)

    # Step 2: Load data from BigQuery
    dataset: ray.data.Dataset = load_data(config)

    # Step 3: Hydrate the ShardedKV store
    hydrate_sharded_kv(sharded_kv, dataset)

    # Step 4: Deploy LookupService
    lookup_service: serve.handle.DeploymentHandle = deploy_lookup_service(sharded_kv, config)

    # Step 5: Example usage - enrich dataset with lookups
    example_usage(lookup_service)


if __name__ == "__main__":
    main()
