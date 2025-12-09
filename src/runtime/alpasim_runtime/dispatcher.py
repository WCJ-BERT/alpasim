# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

from __future__ import annotations

"""
Implements a Dispatcher type which manages a pool of available egodriver endpoints
and sensor simulation endpoints and assigns them to simulation rollouts as they
come available.
"""

import asyncio
import functools
import logging
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

import alpasim_runtime
from alpasim_grpc.v0.common_pb2 import VersionId
from alpasim_grpc.v0.logging_pb2 import RolloutMetadata
from alpasim_runtime.camera_catalog import CameraCatalog
from alpasim_runtime.config import DataSourceConfig, ScenarioConfig, UserSimulatorConfig
from alpasim_runtime.loop import UnboundRollout
from alpasim_runtime.services.controller_service import ControllerService
from alpasim_runtime.services.driver_service import DriverService
from alpasim_runtime.services.physics_service import PhysicsService
from alpasim_runtime.services.sensorsim_service import SensorsimService
from alpasim_runtime.services.service_pool import ServicePool
from alpasim_runtime.services.traffic_service import TrafficService
from alpasim_runtime.worker.ipc import JobResult, RolloutJob, ServiceAllocations
from alpasim_utils.scene_data_source import SceneDataSource

from eval.schema import EvalConfig

logger = logging.getLogger(__name__)

try:
    from trajdata.dataset import UnifiedDataset
    from alpasim_utils.trajdata_data_source import TrajdataDataSource

    TRAJDATA_AVAILABLE = True

except ImportError:
    TRAJDATA_AVAILABLE = False
    UnifiedDataset = None

# Optional USDZ artifact support (for independent services like mtgs_sensorsim)
try:
    from alpasim_utils.artifact import Artifact
    USDZ_AVAILABLE = True
except ImportError:
    USDZ_AVAILABLE = False
    Artifact = None


@dataclass
class Dispatcher:
    """
    Keeps track of contention of each microservice and assigns tasks as they come available.
    Preserves named pool attributes for type safety.

    The Dispatcher uses config-based data loading via DataSourceConfig from UserSimulatorConfig.
    """

    driver_pool: ServicePool[DriverService]
    sensorsim_pool: ServicePool[SensorsimService]
    physics_pool: ServicePool[PhysicsService]
    trafficsim_pool: ServicePool[TrafficService]
    controller_pool: ServicePool[ControllerService]

    camera_catalog: CameraCatalog

    user_config: UserSimulatorConfig
    version_ids: RolloutMetadata.VersionIds
    rollouts_dir: str
    eval_config: EvalConfig  # Eval config for in-runtime evaluation

    # Scene data management - on-demand loading via get_scene()
    _scene_cache: dict[str, SceneDataSource] = field(default_factory=dict)
    _data_source_config: Optional[DataSourceConfig] = None
    _dataset: Optional[Any] = None  # UnifiedDataset when available
    _scene_id_to_idx: dict[str, int] = field(default_factory=dict)  # Map scene_id to dataset index
    _smooth_trajectories: bool = True

    def get_scene(self, scene_id: str) -> SceneDataSource:
        """
        Get a scene data source by ID, loading on-demand if necessary.

        This method provides lazy loading of scene data, reducing memory usage
        and startup time compared to pre-loading all scenes.

        Args:
            scene_id: Unique identifier for the scene

        Returns:
            SceneDataSource implementation (TrajdataDataSource)

        Raises:
            KeyError: If scene_id is not found in available scenes
        """
        # Check cache first
        if scene_id in self._scene_cache:
            return self._scene_cache[scene_id]

        # On-demand loading from UnifiedDataset
        if self._dataset is not None and TRAJDATA_AVAILABLE:
            try:
                # Look up scene index by scene_id
                scene_idx = self._scene_id_to_idx.get(scene_id)
                if scene_idx is None:
                    raise KeyError(f"Scene {scene_id} not found in dataset")

                # Get scene from dataset using index
                scene = self._dataset.get_scene(scene_idx)
                if scene is None:
                    raise KeyError(f"Scene at index {scene_idx} not found in dataset")

                # Get asset_base_path from config
                asset_base_path = None
                if self._data_source_config is not None:
                    asset_base_path = self._data_source_config.asset_base_path

                # Create TrajdataDataSource
                data_source = TrajdataDataSource.from_trajdata_scene(
                    scene=scene,
                    dataset=self._dataset,
                    smooth_trajectories=self._smooth_trajectories,
                    asset_base_path=asset_base_path,
                )

                # Cache for future use
                self._scene_cache[scene_id] = data_source
                logger.debug(f"Loaded scene {scene_id} on-demand")
                return data_source

            except Exception as e:
                logger.error(f"Failed to load scene {scene_id}: {e}")
                raise KeyError(f"Scene {scene_id} not found or failed to load: {e}")

        raise KeyError(
            f"Scene {scene_id} not found. Available scenes: "
            f"{list(self._scene_cache.keys())}"
        )

    # Scene data management - on-demand loading via get_scene()
    _scene_cache: dict[str, SceneDataSource] = field(default_factory=dict)
    _data_source_config: Optional[DataSourceConfig] = None
    _dataset: Optional[Any] = None  # UnifiedDataset when available
    _scene_id_to_idx: dict[str, int] = field(default_factory=dict)  # Map scene_id to dataset index
    _smooth_trajectories: bool = True

    def get_scene(self, scene_id: str) -> SceneDataSource:
        """
        Get a scene data source by ID, loading on-demand if necessary.

        This method provides lazy loading of scene data, reducing memory usage
        and startup time compared to pre-loading all scenes.

        Args:
            scene_id: Unique identifier for the scene

        Returns:
            SceneDataSource implementation (TrajdataDataSource)

        Raises:
            KeyError: If scene_id is not found in available scenes
        """
        # Check cache first
        if scene_id in self._scene_cache:
            return self._scene_cache[scene_id]

        # On-demand loading from UnifiedDataset
        if self._dataset is not None and TRAJDATA_AVAILABLE:
            try:
                # Look up scene index by scene_id
                scene_idx = self._scene_id_to_idx.get(scene_id)
                if scene_idx is None:
                    raise KeyError(f"Scene {scene_id} not found in dataset")

                # Get scene from dataset using index
                scene = self._dataset.get_scene(scene_idx)
                if scene is None:
                    raise KeyError(f"Scene at index {scene_idx} not found in dataset")

                # Get asset_base_path from config
                asset_base_path = None
                if self._data_source_config is not None:
                    asset_base_path = self._data_source_config.asset_base_path

                # Create TrajdataDataSource
                data_source = TrajdataDataSource.from_trajdata_scene(
                    scene=scene,
                    dataset=self._dataset,
                    smooth_trajectories=self._smooth_trajectories,
                    asset_base_path=asset_base_path,
                )

                # Cache for future use
                self._scene_cache[scene_id] = data_source
                logger.debug(f"Loaded scene {scene_id} on-demand")
                return data_source

            except Exception as e:
                logger.error(f"Failed to load scene {scene_id}: {e}")
                raise KeyError(f"Scene {scene_id} not found or failed to load: {e}")

        raise KeyError(
            f"Scene {scene_id} not found. Available scenes: "
            f"{list(self._scene_cache.keys())}"
        )

    async def find_scenario_incompatibilities(
        self, scenario: ScenarioConfig
    ) -> list[str]:
        results = await asyncio.gather(
            self.driver_pool.find_scenario_incompatibilities(scenario),
            self.sensorsim_pool.find_scenario_incompatibilities(scenario),
            self.physics_pool.find_scenario_incompatibilities(scenario),
            self.trafficsim_pool.find_scenario_incompatibilities(scenario),
            self.controller_pool.find_scenario_incompatibilities(scenario),
        )

        return [item for sublist in results for item in sublist]

    @staticmethod
    async def create(
        user_config: UserSimulatorConfig,
        allocations: ServiceAllocations,
        rollouts_dir: str = "",
        eval_config: EvalConfig,
    ) -> Dispatcher:
        """
        Initialize dispatcher: create UnifiedDataset from config, build service pools.

        Args:
            user_config: User simulator configuration (must include data_source)
            allocations: Service allocations
            asl_dir: Directory for ASL output files
        """
        camera_catalog = CameraCatalog(user_config.extra_cameras)

        # Initialize data source fields
        scene_cache: dict[str, SceneDataSource] = {}
        dataset = None
        data_source_config = user_config.data_source
        smooth_trajectories = user_config.smooth_trajectories

        # Config-based data loading (required)
        if data_source_config is None:
            raise ValueError(
                "data_source is required in user config. "
                "Please set data_source in your YAML config file."
            )

        logger.info("Creating UnifiedDataset from config")
        if not TRAJDATA_AVAILABLE:
            raise ImportError(
                "trajdata is required for data source loading. "
                "Please install trajdata."
            )

        # Create UnifiedDataset from config
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

        # Build scene_id to index mapping
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

        endpoints = user_config.endpoints
        timeout = endpoints.startup_timeout_s

        logger.info("Acquiring physics connections: %s", allocations.physics)
        physics = await ServicePool.create_from_allocation(
            PhysicsService,
            allocations.physics,
            skip=endpoints.physics.skip,
            connection_timeout_s=timeout,
        )

        logger.info("Acquiring controller connections: %s", allocations.controller)
        controller = await ServicePool.create_from_allocation(
            ControllerService,
            allocations.controller,
            skip=endpoints.controller.skip,
            connection_timeout_s=timeout,
        )

        logger.info("Acquiring traffic connections: %s", allocations.trafficsim)
        traffic = await ServicePool.create_from_allocation(
            TrafficService,
            allocations.trafficsim,
            skip=endpoints.trafficsim.skip,
            connection_timeout_s=timeout,
        )

        logger.info("Acquiring sensorsim connections: %s", allocations.sensorsim)
        sensorsim = await ServicePool.create_from_allocation(
            SensorsimService,
            allocations.sensorsim,
            skip=endpoints.sensorsim.skip,
            connection_timeout_s=timeout,
            camera_catalog=camera_catalog,
        )

        logger.info("Acquiring driver connections: %s", allocations.driver)
        driver = await ServicePool.create_from_allocation(
            DriverService,
            allocations.driver,
            skip=endpoints.driver.skip,
            connection_timeout_s=timeout,
        )

        # Gather version info from each service pool
        version_ids = await _gather_versions_from_pools(
            driver=driver,
            sensorsim=sensorsim,
            physics=physics,
            trafficsim=traffic,
            controller=controller,
        )

        logger.info("Dispatcher ready.")
        return Dispatcher(
            driver_pool=driver,
            sensorsim_pool=sensorsim,
            physics_pool=physics,
            trafficsim_pool=traffic,
            controller_pool=controller,
            camera_catalog=camera_catalog,
            user_config=user_config,
            version_ids=version_ids,
            rollouts_dir=rollouts_dir,
            _scene_cache=scene_cache,
            _data_source_config=data_source_config,
            _dataset=dataset,
            _scene_id_to_idx=scene_id_to_idx,
            _smooth_trajectories=smooth_trajectories,
            eval_config=eval_config,
        )

    def get_pool_capacity(self) -> int:
        """Return how many concurrent rollouts this dispatcher can handle."""
        return min(
            self.driver_pool.get_number_of_services(),
            self.sensorsim_pool.get_number_of_services(),
            self.physics_pool.get_number_of_services(),
            self.trafficsim_pool.get_number_of_services(),
            self.controller_pool.get_number_of_services(),
        )

    @asynccontextmanager
    async def acquire_all_services(
        self,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Atomically acquire all services, releasing on failure.

        This context manager ensures that if any service acquisition fails,
        all previously acquired services are returned to their pools before
        the exception propagates. On successful acquisition, all services
        are released when the context exits (whether normally or via exception).

        Yields:
            dict with keys: 'driver', 'sensorsim', 'physics', 'trafficsim', 'controller'
        """
        services: dict[str, Any] = {}
        pools: list[tuple[str, ServicePool]] = [
            ("driver", self.driver_pool),
            ("sensorsim", self.sensorsim_pool),
            ("physics", self.physics_pool),
            ("trafficsim", self.trafficsim_pool),
            ("controller", self.controller_pool),
        ]

        try:
            for name, pool in pools:
                services[name] = await pool.get()
            yield services
        finally:
            # Release all acquired services back to their pools
            for name, pool in pools:
                service = services.get(name)
                if service is not None:
                    await pool.put_back(service)

    async def run_job(self, job: RolloutJob) -> JobResult:
        """Execute a single rollout job."""
        rollout: UnboundRollout | None = None

        try:
            # Offload CPU-bound rollout preparation to thread to keep event loop responsive.
            loop = asyncio.get_running_loop()
            rollout = await loop.run_in_executor(
                None,
                functools.partial(
                    UnboundRollout.create,
                    config=self.user_config,
                    scenario=job.scenario,
                    version_ids=self.version_ids,
                    random_seed=job.seed,
                    get_scene=self.get_scene,
                    rollouts_dir=self.rollouts_dir,
                ),
            )

            # Acquire all services atomically with automatic cleanup
            async with self.acquire_all_services() as services:
                eval_result = await rollout.bind(
                    services["driver"],
                    services["sensorsim"],
                    services["physics"],
                    services["trafficsim"],
                    services["controller"],
                    self.camera_catalog,
                    eval_config=self.eval_config,
                ).run()

            return JobResult(
                job_id=job.job_id,
                success=True,
                error=None,
                error_traceback=None,
                rollout_uuid=rollout.rollout_uuid,
                eval_result=eval_result,
            )

        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.warning(
                "Rollout FAILED: job=%s scene=%s uuid=%s error=%s\n%s",
                job.job_id,
                rollout.scene_id if rollout else "N/A",
                rollout.rollout_uuid if rollout else "N/A",
                exc,
                tb,
            )
            return JobResult(
                job_id=job.job_id,
                success=False,
                error=str(exc),
                error_traceback=tb,
                rollout_uuid=rollout.rollout_uuid if rollout else None,
            )


async def _gather_versions_from_pools(
    driver: ServicePool[DriverService],
    sensorsim: ServicePool[SensorsimService],
    physics: ServicePool[PhysicsService],
    trafficsim: ServicePool[TrafficService],
    controller: ServicePool[ControllerService],
) -> RolloutMetadata.VersionIds:
    """
    Gather version info from each service pool.
    """
    versions: dict[str, VersionId] = {}

    # Query version from first service in each pool
    versions["driver"] = await driver.services[0].get_version()
    versions["sensorsim"] = await sensorsim.services[0].get_version()
    versions["physics"] = await physics.services[0].get_version()
    versions["trafficsim"] = await trafficsim.services[0].get_version()
    versions["controller"] = await controller.services[0].get_version()

    return RolloutMetadata.VersionIds(
        runtime_version=alpasim_runtime.VERSION_MESSAGE,
        egodriver_version=versions.get("driver"),
        sensorsim_version=versions.get("sensorsim"),
        physics_version=versions.get("physics"),
        traffic_version=versions.get("trafficsim"),
    )
