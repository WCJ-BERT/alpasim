# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
Server script for MTGS Sensorsim Service.

This script starts a gRPC server that exposes the MTGS renderer
as a SensorsimService. It uses UserSimulatorConfig for data loading,
ensuring consistency with the runtime.

Usage:
    python -m render.mtgs_sensorsim_server \
        --user-config user_config.yaml \
        --host 0.0.0.0 \
        --port 8080
"""

import argparse
import logging
from concurrent import futures
from pathlib import Path
from typing import Callable

import grpc

from alpasim_grpc.v0.sensorsim_pb2_grpc import add_SensorsimServiceServicer_to_server
from alpasim_runtime.config import UserSimulatorConfig, typed_parse_config
from render.mtgs_sensorsim_service import MTGSSensorsimService

# Optional trajdata support
try:
    from trajdata.dataset import UnifiedDataset
    from alpasim_utils.trajdata_data_source import TrajdataDataSource

    TRAJDATA_AVAILABLE = True
except ImportError:
    TRAJDATA_AVAILABLE = False
    UnifiedDataset = None
    TrajdataDataSource = None

logger = logging.getLogger(__name__)

# MTGS-specific: Build asset path with dataset name mapping and 'assets' subdirectory
    # Map desired_data names to asset folder names (e.g., nuplan_test -> navtest)
DATASET_NAME_MAPPING = {
        "nuplan_test": "navtest",
        "nuplan_mini": "navtest",
        # Add more mappings as needed
    }

def parse_args(arg_list: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MTGS Sensorsim Service Server - uses UserSimulatorConfig",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--user-config",
        type=str,
        required=True,
        help="Path to user config YAML file (same format as runtime)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=2,
        help="Number of renderer instances to keep in LRU cache",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for rendering",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args(arg_list)
    return args


def create_get_scene_function(
    user_config: UserSimulatorConfig,
) -> tuple[Callable[[str], TrajdataDataSource], Callable[[], list[str]]]:
    """
    Create a get_scene function that loads scenes on-demand from UnifiedDataset.

    This mimics the Dispatcher.get_scene() behavior for standalone server use.

    Args:
        user_config: User simulator configuration with data_source section

    Returns:
        Tuple of (get_scene function, get_available_scene_ids function)
    """
    if not TRAJDATA_AVAILABLE:
        raise ImportError(
            "trajdata is required for MTGS sensorsim server. Please install trajdata."
        )

    data_source_config = user_config.data_source
    if data_source_config is None:
        raise ValueError(
            "data_source is required in user config. "
            "Please add data_source section to your YAML config."
        )

    # Get asset base path from config
    asset_base_path_config = data_source_config.asset_base_path

    # Build MTGS-specific asset path: {base_path}/{mapped_dataset_name}/assets
    mtgs_asset_base_path = None
    if asset_base_path_config and data_source_config.desired_data:
        # Use first desired_data as the dataset name
        desired_data_name = data_source_config.desired_data[0]
        # Apply mapping if exists
        mapped_name = DATASET_NAME_MAPPING.get(desired_data_name, desired_data_name)
        # Construct full path: base_path/dataset_name/assets
        mtgs_asset_base_path = str(Path(asset_base_path_config) / mapped_name / "assets")
        logger.info(f"MTGS asset path: {mtgs_asset_base_path}")

    logger.info("Creating UnifiedDataset from config")
    dataset = UnifiedDataset(
        desired_data=data_source_config.desired_data,
        data_dirs=data_source_config.data_dirs,
        cache_location=data_source_config.cache_location,
        incl_vector_map=data_source_config.incl_vector_map,
        rebuild_cache=data_source_config.rebuild_cache,
        rebuild_maps=data_source_config.rebuild_maps,
        desired_dt=data_source_config.desired_dt,
        num_workers=data_source_config.num_workers,
    )
    logger.info(
        f"Created UnifiedDataset with {dataset.num_scenes()} scenes, "
        f"desired_data={data_source_config.desired_data}"
    )

    # Build scene_id to index mapping (same as Dispatcher)
    scene_id_to_idx = {}
    num_scenes = dataset.num_scenes()
    for idx in range(num_scenes):
        try:
            scene = dataset.get_scene(idx)
            scene_id_to_idx[scene.name] = idx
        except Exception as e:
            logger.warning(f"Failed to get scene at index {idx}: {e}")
            continue
    logger.info(f"Built scene_id mapping for {len(scene_id_to_idx)} scenes")

    # Scene cache for loaded scenes
    scene_cache = {}

    def get_scene(scene_id: str) -> TrajdataDataSource:
        """Load scene on-demand from UnifiedDataset."""
        # Check cache first
        if scene_id in scene_cache:
            return scene_cache[scene_id]

        # Look up scene index
        scene_idx = scene_id_to_idx.get(scene_id)
        if scene_idx is None:
            raise KeyError(f"Scene {scene_id} not found in dataset")

        # Load scene from dataset
        scene = dataset.get_scene(scene_idx)
        if scene is None:
            raise KeyError(f"Scene at index {scene_idx} not found")

        # Create TrajdataDataSource with MTGS-specific asset path
        data_source = TrajdataDataSource.from_trajdata_scene(
            scene=scene,
            dataset=dataset,
            smooth_trajectories=user_config.smooth_trajectories,
            asset_base_path=mtgs_asset_base_path,  # Use MTGS-specific path with dataset/assets
        )

        # Cache for future use
        scene_cache[scene_id] = data_source
        logger.info(
            f"Loaded scene {scene_id} on-demand, asset_path={data_source.asset_path}"
        )
        return data_source

    def get_available_scene_ids() -> list[str]:
        """Return list of all available scene IDs."""
        return list(scene_id_to_idx.keys())

    return get_scene, get_available_scene_ids


def main(arg_list: list[str] | None = None) -> None:
    """Main entry point for the server."""
    args = parse_args(arg_list)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 80)
    logger.info("MTGS Sensorsim Service Server")
    logger.info("=" * 80)
    logger.info(f"User config: {args.user_config}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Cache size: {args.cache_size}")

    # Load user config
    try:
        user_config = typed_parse_config(args.user_config, UserSimulatorConfig)
    except Exception as e:
        logger.error(f"Failed to load user config: {e}")
        import traceback

        traceback.print_exc()
        return

    if user_config.data_source is None:
        logger.error(
            "data_source is required in user config. "
            "Please add a 'data_source' section with desired_data, data_dirs, "
            "cache_location, and asset_base_path."
        )
        return

    logger.info(f"Data source: {user_config.data_source.desired_data}")
    logger.info(f"Cache location: {user_config.data_source.cache_location}")
    logger.info(f"Asset base path: {user_config.data_source.asset_base_path or '(not configured)'}")

    # Create get_scene function
    try:
        get_scene, get_available_scene_ids = create_get_scene_function(user_config)
    except Exception as e:
        logger.error(f"Failed to create get_scene function: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    address = f"{args.host}:{args.port}"
    server.add_insecure_port(address)

    # Create and register service
    service = MTGSSensorsimService(
        server=server,
        get_scene=get_scene,
        get_available_scene_ids=get_available_scene_ids,
        cache_size=args.cache_size,
        device=args.device,
    )
    add_SensorsimServiceServicer_to_server(service, server)

    logger.info("=" * 80)
    logger.info(f"✓ Server ready on {address}")
    logger.info("=" * 80)

    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        server.stop(0)


if __name__ == "__main__":
    main()
