# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

"""
MTGS Sensorsim Service implementation.

This service wraps the MTGS renderer and exposes it via gRPC
as a SensorsimService, allowing it to replace the default sensorsim in alpasim.
"""

from __future__ import annotations

import functools
import io
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import grpc
import numpy as np
import torch

from alpasim_grpc.v0.common_pb2 import AvailableScenesReturn, Empty, VersionId
from alpasim_grpc.v0.sensorsim_pb2 import (
    AggregatedRenderRequest,
    AggregatedRenderReturn,
    AvailableCamerasRequest,
    AvailableCamerasReturn,
    AvailableEgoMasksReturn,
    AvailableTrajectoriesRequest,
    AvailableTrajectoriesReturn,
    RGBRenderRequest,
    RGBRenderReturn,
)
from alpasim_grpc.v0.sensorsim_pb2_grpc import SensorsimServiceServicer
from alpasim_utils.mtgs_artifact_adapter import (
    get_available_cameras_from_data_source,
    rgb_render_request_to_render_state,
)

logger = logging.getLogger(__name__)

# Import MTGS renderer from local render module
try:
    from render.src.mtgs import MTGS
    from render.base_renderer import RenderState
    MTGS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MTGS renderer not available: {e}")
    MTGS_AVAILABLE = False
    MTGS = None
    RenderState = None


VERSION_MESSAGE = VersionId(
    version_id="mtgs-sensorsim-1.0.0",
    git_hash="unknown",
)


class MTGSSensorsimService(SensorsimServiceServicer):
    """
    gRPC service that wraps MTGS renderer.
    
    This service implements the SensorsimServiceServicer interface, allowing
    MTGS to be used as a drop-in replacement for the default sensorsim.
    """

    def __init__(
        self,
        server: grpc.Server,
        get_scene: callable,
        get_available_scene_ids: callable,
        cache_size: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        Initialize the MTGS sensorsim service.

        Args:
            server: gRPC server instance
            get_scene: Function that takes scene_id and returns TrajdataDataSource
            get_available_scene_ids: Function that returns list of all available scene IDs
            cache_size: Number of renderer instances to cache
            device: Device to use for rendering (cuda or cpu)
        """
        if not MTGS_AVAILABLE:
            raise ImportError("MTGS renderer is required for MTGSSensorsimService")

        self.server = server
        self.get_scene = get_scene
        self.get_available_scene_ids = get_available_scene_ids
        self.device = device
        self.cache_size = cache_size

        logger.info(f"MTGS Sensorsim Service initialized")
        logger.info(f"Device: {device}")
        logger.info(f"Cache size: {cache_size}")

        # Cache renderer instances per scene
        self._renderer_cache: Dict[str, MTGS] = {}
        # Track loaded scene IDs for debugging/monitoring
        self._loaded_scene_ids: set[str] = set()

        def get_renderer(scene_id: str) -> MTGS:
            """Get or create renderer for a scene."""
            if scene_id in self._renderer_cache:
                logger.debug(f"Renderer cache hit for scene {scene_id}")
                return self._renderer_cache[scene_id]

            logger.info(f"Renderer cache miss, loading for scene {scene_id}")

            # Load scene data source using get_scene function
            try:
                data_source = self.get_scene(scene_id)
            except Exception as e:
                logger.error(f"Failed to load scene {scene_id}: {e}")
                raise KeyError(f"Scene {scene_id} not available: {e}")

            # Get asset_path from data source
            asset_path = data_source.asset_path
            if asset_path is None:
                raise ValueError(
                    f"Scene {scene_id} has no asset_path. "
                    "Make sure asset_base_path is configured in user config."
                )

            logger.info(f"Asset path for scene {scene_id}: {asset_path}")

            # Extract asset folder path (parent directory)
            # E.g., "/data/mtgs/assets/2021.05.12.22.00.38/" -> "/data/mtgs/assets/"
            asset_folder_path = Path(asset_path).parent

            # Extract clean asset_id from asset_path
            # E.g., "/data/mtgs/assets/2021.05.12.22.00.38/" -> "2021.05.12.22.00.38"
            asset_id = Path(asset_path).name
            if not asset_id:
                # If asset_path ends with /, name will be empty
                asset_id = Path(asset_path).parent.name

            # Clean asset_id (remove suffix like "-001" or "-002")
            if asset_id.endswith(("-001", "-002", "-003")):
                asset_id = asset_id[:-4]

            logger.info(f"Creating MTGS renderer for scene {scene_id}, asset_id={asset_id}")

            # Create renderer with asset_folder_path
            renderer = MTGS(
                device=self.device,
                asset_folder_path=asset_folder_path
            )

            # Reset renderer with asset_id
            renderer.reset(current_scene_id=scene_id, asset_id=asset_id)
            
            # Calculate ego2globals - priority: video_scene_dict > rig trajectory
            ego2globals = None
            
            # First, try to get from video_scene_dict.pkl (preferred source)
            try:
                if hasattr(renderer.asset_manager, 'video_scene_dict') and renderer.asset_manager.video_scene_dict:
                    video_dict_raw = renderer.asset_manager.video_scene_dict
                    
                    # Handle nested structure: video_scene_dict may be {scene_id: {frame_infos, ego2global, ...}}
                    # or directly {frame_infos, ego2global, ...}
                    video_dict = video_dict_raw
                    if isinstance(video_dict_raw, dict) and 'ego2global' not in video_dict_raw and 'frame_infos' not in video_dict_raw:
                        # Nested structure - get the first (and usually only) scene data
                        first_key = next(iter(video_dict_raw.keys()), None)
                        if first_key and isinstance(video_dict_raw[first_key], dict):
                            video_dict = video_dict_raw[first_key]
                            logger.info(f"Accessing nested video_scene_dict with key: {first_key}")
                    
                    # Check for frame_infos FIRST (more common case with 301 frames)
                    if 'frame_infos' in video_dict and len(video_dict['frame_infos']) > 0:
                        # Extract ego2global from frame_infos (each frame has its own ego2global)
                        frame_infos = video_dict['frame_infos']
                        logger.info(f"Found {len(frame_infos)} frames in frame_infos, extracting ego2globals...")
                        ego2globals_list = []
                        for fi in frame_infos:
                            if 'ego2global' in fi:
                                ego2globals_list.append(np.array(fi['ego2global']))
                            elif 'ego2global_translation' in fi and 'ego2global_rotation' in fi:
                                transform = np.eye(4)
                                transform[:3, 3] = np.array(fi['ego2global_translation'])
                                rot = np.array(fi['ego2global_rotation'])
                                if rot.shape == (3, 3):
                                    transform[:3, :3] = rot
                                elif rot.shape == (4,):  # quaternion
                                    from pyquaternion import Quaternion
                                    q = Quaternion(rot[0], rot[1], rot[2], rot[3])
                                    transform[:3, :3] = q.rotation_matrix
                                ego2globals_list.append(transform)
                        if ego2globals_list:
                            ego2globals = np.stack(ego2globals_list)
                            logger.info(f"✓ Extracted ego2globals from frame_infos: shape {ego2globals.shape}, first translation: {ego2globals[0, :3, 3]}")
                    elif 'ego2global' in video_dict:
                        ego2globals = np.array(video_dict['ego2global'])
                        if ego2globals.ndim == 2:
                            # Single matrix, add batch dimension
                            ego2globals = ego2globals[np.newaxis, ...]
                        logger.info(f"Loaded ego2globals from video_scene_dict: shape {ego2globals.shape}")
                    elif 'ego2global_translation' in video_dict and 'ego2global_rotation' in video_dict:
                        # Reconstruct from translation and rotation
                        translations = np.array(video_dict['ego2global_translation'])
                        rotations = np.array(video_dict['ego2global_rotation'])
                        if translations.ndim == 1:
                            translations = translations[np.newaxis, ...]
                        if rotations.ndim == 2:
                            rotations = rotations[np.newaxis, ...]
                        
                        ego2globals_list = []
                        for i in range(len(translations)):
                            transform = np.eye(4)
                            transform[:3, :3] = rotations[i] if rotations.shape[-1] == 3 else rotations[i].reshape(3, 3)
                            transform[:3, 3] = translations[i]
                            ego2globals_list.append(transform)
                        ego2globals = np.stack(ego2globals_list)
                        logger.info(f"Reconstructed ego2globals from video_scene_dict: shape {ego2globals.shape}")
            except Exception as e:
                logger.warning(f"Could not load ego2globals from video_scene_dict: {e}")
            
            # Fallback: compute from rig trajectory if video_scene_dict not available
            if ego2globals is None:
                try:
                    rig = data_source.rig
                    if rig.trajectory.poses is not None and len(rig.trajectory.poses) > 0:
                        # Convert rig trajectory poses to ego2global transformation matrices
                        poses = rig.trajectory.poses
                        ego2globals_list = []
                        for pose in poses:
                            # Create 4x4 transformation matrix from pose
                            # Handle different pose formats (gRPC objects vs numpy arrays)
                            if hasattr(pose.vec3, '__getitem__'):  # numpy array or similar
                                trans = np.array([pose.vec3[0], pose.vec3[1], pose.vec3[2]])
                            else:  # gRPC object with .x, .y, .z attributes
                                trans = np.array([pose.vec3.x, pose.vec3.y, pose.vec3.z])

                            if hasattr(pose.quat, 'w'):  # gRPC object with .w, .x, .y, .z attributes
                                quat = np.array([pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z])
                            elif hasattr(pose.quat, '__getitem__'):  # numpy array or similar
                                quat = np.array(pose.quat)
                            else:
                                raise ValueError(f"Unsupported quaternion format: {type(pose.quat)}")
                            
                            # Convert quaternion to rotation matrix using proper function
                            from .src.utils.gaussian_utils import quat_to_rotmat
                            quat_tensor = torch.tensor(quat, dtype=torch.float64).unsqueeze(0)  # [1, 4]
                            rot_mat = quat_to_rotmat(quat_tensor).squeeze(0).numpy()  # [3, 3]
                            
                            # Create 4x4 transformation matrix
                            transform = np.eye(4)
                            transform[:3, :3] = rot_mat
                            transform[:3, 3] = trans
                            ego2globals_list.append(transform)
                        
                        ego2globals = np.stack(ego2globals_list)
                        logger.info(f"Computed ego2globals from rig trajectory: shape {ego2globals.shape}")
                except Exception as e:
                    logger.warning(f"Could not compute ego2globals from rig trajectory: {e}")
            
            # Set world_to_nre transformation from Rig
            # This enables Simulator Local → Nuplan Global coordinate conversion in render()
            if hasattr(data_source, 'rig') and data_source.rig is not None:
                if hasattr(data_source.rig, 'world_to_nre') and data_source.rig.world_to_nre is not None:
                    renderer.set_world_to_nre(data_source.rig.world_to_nre)
                    logger.info(f"Set world_to_nre from rig: {data_source.rig.world_to_nre[:3, 3]}")
                else:
                    logger.warning("data_source.rig exists but world_to_nre is None")
            else:
                logger.warning("data_source.rig not available, world_to_nre not set")
            
            # Calibrate agent states (will use ego2globals if provided)
            renderer.mtgs_agent2states = renderer.calibrate_agent_state(ego2globals=ego2globals)
            
            # NOTE: No coordinate offset conversion needed!
            # - Runtime sends ego_pose already in global coordinates
            # - MTGS.render() will subtract recon2global_translation internally
            # - get_agent_pose() does nearest-neighbor search in reconstruction coordinate system
            # 
            # Coordinate flow:
            # 1. Runtime: ego_pose in global frame [x_global, y_global, ...]
            # 2. Service: passes through unchanged
            # 3. MTGS.render(): subtracts recon2global_translation → reconstruction frame
            # 4. get_agent_pose(): matches in reconstruction frame against mtgs_agent2states
            # 5. mtgs_agent2states contains: video_scene_dict.ego2global (global) - recon2global
            #
            # Therefore: NO offset conversion is needed in the service layer
            renderer.local_to_global_offset = None  # Not used
            logger.info("Coordinate system: Runtime sends global, MTGS handles conversion internally")
            
            renderer._scene_id = scene_id
            renderer._asset_id = asset_id
            renderer._initialized = True
            
            # Cache the renderer (simple LRU: keep only cache_size most recent)
            if len(self._renderer_cache) >= cache_size:
                # Remove oldest entry (first key)
                oldest_key = next(iter(self._renderer_cache))
                del self._renderer_cache[oldest_key]
            
            self._renderer_cache[scene_id] = renderer
            # Track this scene as loaded
            self._loaded_scene_ids.add(scene_id)
            return renderer
        
        self.get_renderer = get_renderer

    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        """Return version information."""
        return VERSION_MESSAGE

    def get_available_scenes(
        self, _request: Empty, _context: grpc.ServicerContext
    ) -> AvailableScenesReturn:
        """Return list of available scene IDs.

        Returns all scenes available in the dataset via get_available_scene_ids function.
        """
        try:
            scene_ids = self.get_available_scene_ids()
            logger.info(f"Available scenes: {len(scene_ids)} total")
            return AvailableScenesReturn(scene_ids=scene_ids)
        except Exception as e:
            logger.error(f"Failed to get available scenes: {e}")
            # Return empty list on error
            return AvailableScenesReturn(scene_ids=[])

    def get_available_cameras(
        self,
        request: AvailableCamerasRequest,
        context: grpc.ServicerContext,
    ) -> AvailableCamerasReturn:
        """Return available cameras for a scene."""
        scene_id = request.scene_id

        try:
            renderer = self.get_renderer(scene_id)
            cameras = get_available_cameras_from_data_source(asset_manager=renderer.asset_manager)
            return AvailableCamerasReturn(available_cameras=cameras)
        except Exception as e:
            logger.error(f"Failed to get cameras for scene {scene_id}: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Scene {scene_id} not found: {e}")
            return AvailableCamerasReturn()

    def get_available_trajectories(
        self,
        request: AvailableTrajectoriesRequest,
        context: grpc.ServicerContext,
    ) -> AvailableTrajectoriesReturn:
        """Return available trajectories for a scene."""
        scene_id = request.scene_id

        try:
            data_source = self.get_scene(scene_id)
            rig = data_source.rig

            return AvailableTrajectoriesReturn(
                available_trajectories=[
                    AvailableTrajectoriesReturn.AvailableTrajectory(
                        trajectory=rig.trajectory.to_grpc()
                    )
                ]
            )
        except Exception as e:
            logger.error(f"Failed to get trajectories for scene {scene_id}: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Scene {scene_id} not found: {e}")
            return AvailableTrajectoriesReturn()

    def get_available_ego_masks(
        self, _request: Empty, _context: grpc.ServicerContext
    ) -> AvailableEgoMasksReturn:
        """Return available ego masks (empty for now)."""
        # MTGS doesn't support ego masks yet
        return AvailableEgoMasksReturn()

    def render_rgb(
        self, request: RGBRenderRequest, context: grpc.ServicerContext
    ) -> RGBRenderReturn:
        """Render an RGB image using MTGS."""
        try:
            scene_id = request.scene_id
            camera_name = request.camera_intrinsics.logical_id

            # Get or create renderer for this scene
            renderer = self.get_renderer(scene_id)

            # Convert RGBRenderRequest to RenderState
            # Pass asset_manager to allow access to video_scene_dict for ego2global info
            render_state_dict = rgb_render_request_to_render_state(
                request,
                asset_manager=renderer.asset_manager
            )

            # NOTE: No coordinate conversion needed - Runtime sends global coordinates
            # MTGS.render() will handle the conversion internally by subtracting recon2global_translation

            # Create RenderState object
            # Note: RenderState is a dict subclass, so we can use dict assignment
            render_state = RenderState()
            render_state[RenderState.TIMESTAMP] = render_state_dict["timestamp"]
            render_state[RenderState.AGENT_STATE] = render_state_dict["agent_state"]
            render_state[RenderState.CAMERAS] = render_state_dict["cameras"]
            render_state[RenderState.LIDAR] = render_state_dict.get("lidar", {})

            # Set recon2global_translation if not set (needed for coordinate transformation)
            if not hasattr(renderer, 'recon2global_translation'):
                # Try to get from asset manager
                if hasattr(renderer.asset_manager, 'background_asset_dict') and renderer.asset_manager.background_asset_dict:
                    bg_asset = renderer.asset_manager.background_asset_dict
                    if 'background' in bg_asset and 'config' in bg_asset['background']:
                        recon2global = bg_asset['background']['config'].get('recon2world_translation', [0.0, 0.0, 0.0])
                        renderer.recon2global_translation = torch.tensor(recon2global[:2], device=self.device)
                    else:
                        renderer.recon2global_translation = torch.zeros(2, device=self.device)
                else:
                    renderer.recon2global_translation = torch.zeros(2, device=self.device)

            # Ensure renderer is initialized for this scene
            # The renderer should already be initialized in get_renderer
            # This check is just for safety
            if not hasattr(renderer, '_initialized') or not renderer._initialized:
                logger.error(f"Renderer not properly initialized for {scene_id}")
                raise ValueError(f"Renderer not initialized for scene {scene_id}")
            
            # Render
            result = renderer.render(render_state)
            
            # Extract image from result
            # result should be a dict with 'cameras' key containing camera images
            if "cameras" not in result:
                raise ValueError("Renderer did not return cameras in result")
            
            cameras_dict = result["cameras"]
            if not cameras_dict:
                raise ValueError("No cameras in render result")
            
            # Get the first camera image (or the requested camera)
            camera_name = request.camera_intrinsics.logical_id
            if camera_name in cameras_dict and "image" in cameras_dict[camera_name]:
                image = cameras_dict[camera_name]["image"]
            else:
                # Fallback to first available camera
                first_camera = next(iter(cameras_dict.values()))
                image = first_camera.get("image")
            
            if image is None:
                raise ValueError("No image found in render result")
            
            # Convert image to bytes
            # image should be a numpy array (BGR format from OpenCV)
            if isinstance(image, np.ndarray):
                # Encode as JPEG or PNG based on request
                if request.image_format == 1:  # PNG
                    success, image_bytes = cv2.imencode(".png", image)
                elif request.image_format == 2:  # JPEG
                    success, image_bytes = cv2.imencode(
                        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(request.image_quality)]
                    )
                else:
                    # Default to JPEG
                    success, image_bytes = cv2.imencode(".jpg", image)
                
                if not success:
                    raise ValueError("Failed to encode image")
                
                # Save rendered image to disk for debugging/visualization
                self._save_rendered_image(image, scene_id, camera_name, request.frame_start_us)
                
                return RGBRenderReturn(image_bytes=image_bytes.tobytes())
            else:
                raise ValueError(f"Unexpected image type: {type(image)}")
        
        except Exception as e:
            logger.exception(f"Error rendering RGB image: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            raise

    def render_aggregated(
        self,
        request: AggregatedRenderRequest,
        context: grpc.ServicerContext,
    ) -> AggregatedRenderReturn:
        """Render multiple RGB images efficiently by batching."""
        if not request.rgb_requests:
            return AggregatedRenderReturn()
        
        # Check if all requests can be batched (same scene, same timestamp)
        first_request = request.rgb_requests[0]
        scene_id = first_request.scene_id
        frame_start_us = first_request.frame_start_us
        
        can_batch = all(
            req.scene_id == scene_id and req.frame_start_us == frame_start_us
            for req in request.rgb_requests
        )
        
        if not can_batch:
            # Fallback: render each request individually
            logger.warning(
                f"Cannot batch render_aggregated: requests have different scenes or timestamps. "
                f"Falling back to sequential rendering."
            )
            rgb_returns = []
            for rgb_request in request.rgb_requests:
                rgb_return = self.render_rgb(rgb_request, context)
                rgb_returns.append(rgb_return)
            return AggregatedRenderReturn(rgb_returns=rgb_returns)
        
        # Optimized path: batch render all cameras in one call
        try:
            return self._render_aggregated_batch(request, context)
        except Exception as e:
            logger.exception(f"Batch rendering failed: {e}. Falling back to sequential rendering.")
            # Fallback to sequential rendering
            rgb_returns = []
            for rgb_request in request.rgb_requests:
                try:
                    rgb_return = self.render_rgb(rgb_request, context)
                    rgb_returns.append(rgb_return)
                except Exception as req_error:
                    logger.error(f"Failed to render camera {rgb_request.camera_intrinsics.logical_id}: {req_error}")
                    rgb_returns.append(RGBRenderReturn())  # Empty response for failed camera
            return AggregatedRenderReturn(rgb_returns=rgb_returns)

    def _render_aggregated_batch(
        self,
        request: AggregatedRenderRequest,
        context: grpc.ServicerContext,
    ) -> AggregatedRenderReturn:
        """
        Optimized batch rendering: render once, extract all cameras.
        
        This method assumes all requests have the same scene_id and frame_start_us.
        """
        first_request = request.rgb_requests[0]
        scene_id = first_request.scene_id

        # Get renderer (this internally loads scene on-demand via get_scene)
        try:
            renderer = self.get_renderer(scene_id)
        except Exception as e:
            logger.error(f"Failed to get renderer for scene {scene_id}: {e}")
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Failed to load scene {scene_id}: {e}")
            return AggregatedRenderReturn()
        
        # Convert first request to RenderState (all requests share same ego state and timestamp)
        render_state_dict = rgb_render_request_to_render_state(
            first_request, 
            asset_manager=renderer.asset_manager
        )
        render_state = RenderState(**render_state_dict)
        
        # Single render call for all cameras
        logger.info(f"Batch rendering {len(request.rgb_requests)} cameras for scene {scene_id}")
        result = renderer.render(render_state)
        
        if "cameras" not in result:
            raise ValueError("Renderer did not return cameras in result")
        
        cameras_dict = result["cameras"]
        if not cameras_dict:
            raise ValueError("No cameras in render result")
        
        # Extract and encode images for all requested cameras
        rgb_returns = []
        for rgb_request in request.rgb_requests:
            camera_name = rgb_request.camera_intrinsics.logical_id
            
            # Get image for this camera
            if camera_name in cameras_dict and "image" in cameras_dict[camera_name]:
                image = cameras_dict[camera_name]["image"]
            else:
                logger.warning(f"Camera {camera_name} not found in render result, using first available")
                first_camera = next(iter(cameras_dict.values()))
                image = first_camera.get("image")
            
            if image is None:
                logger.error(f"No image found for camera {camera_name}")
                rgb_returns.append(RGBRenderReturn())
                continue
            
            # Encode image
            if not isinstance(image, np.ndarray):
                logger.error(f"Unexpected image type for camera {camera_name}: {type(image)}")
                rgb_returns.append(RGBRenderReturn())
                continue
            
            # Encode based on format
            if rgb_request.image_format == 1:  # PNG
                success, image_bytes = cv2.imencode(".png", image)
            elif rgb_request.image_format == 2:  # JPEG
                success, image_bytes = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(rgb_request.image_quality)]
                )
            else:
                # Default to JPEG
                success, image_bytes = cv2.imencode(".jpg", image)
            
            if not success:
                logger.error(f"Failed to encode image for camera {camera_name}")
                rgb_returns.append(RGBRenderReturn())
                continue
            
            # Save rendered image to disk for debugging/visualization
            self._save_rendered_image(image, scene_id, camera_name, rgb_request.frame_start_us)
            
            rgb_returns.append(RGBRenderReturn(image_bytes=image_bytes.tobytes()))
        
        logger.info(f"Batch rendering completed: {len(rgb_returns)}/{len(request.rgb_requests)} cameras successful")
        
        # Save panorama view combining all cameras
        self._save_panorama_view(cameras_dict, scene_id, first_request.frame_start_us)
        
        return AggregatedRenderReturn(rgb_returns=rgb_returns)

    def render_lidar(
        self, _request, context: grpc.ServicerContext
    ):
        """Render LiDAR data (not implemented yet)."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("LiDAR rendering not implemented for MTGS")
        raise NotImplementedError("LiDAR rendering not implemented")

    def shut_down(self, _request: Empty, context: grpc.ServicerContext) -> Empty:
        """Shut down the service."""
        logger.info("shut_down")
        context.add_callback(self._shut_down)
        return Empty()

    def _save_rendered_image(
        self, 
        image: np.ndarray, 
        scene_id: str, 
        camera_id: str, 
        timestamp_us: int
    ) -> None:
        """Save rendered image to disk for visualization.
        
        Images are saved to: {workspace}/rendered_images/{scene_id}/{camera_id}_{timestamp}.jpg
        """
        import os
        
        # Create output directory
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "rendered_images",
            scene_id
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Save image
        filename = f"{camera_id}_{timestamp_us}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        success = cv2.imwrite(filepath, image)
        if success:
            logger.debug(f"Saved rendered image: {filepath}")
        else:
            logger.warning(f"Failed to save rendered image: {filepath}")
    
    def _save_panorama_view(
        self,
        cameras_dict: dict,
        scene_id: str,
        timestamp_us: int,
        target_size: tuple = (640, 360) 
    ) -> None:
        """
        Save a 3x3 panorama view combining all 8 cameras with center black.
        
        Layout:
            CAM_L0  CAM_F0  CAM_R0
            CAM_L1  BLACK   CAM_R1
            CAM_L2  CAM_B0  CAM_R2
        
        Args:
            cameras_dict: Dictionary of camera images from renderer
            scene_id: Scene ID for output directory
            timestamp_us: Timestamp for filename
            target_size: Size to resize each camera view (width, height)
        """
        import os
        
        # Camera layout mapping (3x3 grid)
        layout = [
            ['CAM_L0', 'CAM_F0', 'CAM_R0'],
            ['CAM_L1', 'BLACK',  'CAM_R1'],
            ['CAM_L2', 'CAM_B0', 'CAM_R2']
        ]
        
        # Create black placeholder
        black_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Build panorama row by row
        rows = []
        for row_cameras in layout:
            row_images = []
            for camera_id in row_cameras:
                if camera_id == 'BLACK':
                    row_images.append(black_image)
                elif camera_id in cameras_dict and 'image' in cameras_dict[camera_id]:
                    # Resize camera image to target size
                    img = cameras_dict[camera_id]['image']
                    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                    row_images.append(resized)
                else:
                    # Missing camera - use black placeholder
                    logger.warning(f"Camera {camera_id} not found for panorama view")
                    row_images.append(black_image)
            
            # Concatenate images horizontally
            row_image = np.hstack(row_images)
            rows.append(row_image)
        
        # Concatenate all rows vertically
        panorama = np.vstack(rows)
        
        # Save panorama
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "rendered_images",
            scene_id
        )
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"panorama_{timestamp_us}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        success = cv2.imwrite(filepath, panorama)
        if success:
            logger.info(f"Saved panorama view ({panorama.shape[1]}x{panorama.shape[0]}): {filepath}")
        else:
            logger.warning(f"Failed to save panorama view: {filepath}")

    def _shut_down(self) -> None:
        """Internal shutdown callback."""
        self.server.stop(0)
