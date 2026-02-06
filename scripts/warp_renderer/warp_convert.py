from isaaclab.utils.math import convert_camera_frame_orientation_convention
from isaaclab.scene import InteractiveScene
import isaaclab.sim as isaaclab_sim

from dataclasses import dataclass, field
from warp_raytrace import RenderContext, RenderShapeType
from pxr import Usd, UsdGeom

import numpy as np
import warp as wp
import usdrt
import re
import os


@wp.kernel(enable_backward=False)
def compute_triangle_count(num_face_counts: wp.int32, face_counts: wp.array(dtype=wp.int32), out_face_offsets: wp.array(dtype=wp.int32)):
    offset = wp.int32(0)
    for i in range(num_face_counts):
        out_face_offsets[i] = offset
        offset += face_counts[i]
    out_face_offsets[num_face_counts] = offset - num_face_counts * 2


@wp.kernel(enable_backward=False)
def triangulate_faces(num_face_counts: wp.int32, face_counts: wp.array(dtype=wp.int32), face_indices: wp.array(dtype=wp.int32), face_offsets: wp.array(dtype=wp.int32), out_triangles: wp.array(dtype=wp.int32)):
    offset = face_offsets[wp.tid()]
    num_triangles = face_counts[wp.tid()] - 2
    tri = wp.atomic_add(face_offsets, num_face_counts + 1, num_triangles * 3)

    for i in range(num_triangles):
        out_triangles[tri + i * 3 + 0] = face_indices[offset]
        out_triangles[tri + i * 3 + 1] = face_indices[offset + i + 1]
        out_triangles[tri + i * 3 + 2] = face_indices[offset + i + 2]


@wp.kernel(enable_backward=False)
def update_transforms(fabric_matrices: wp.fabricarray(dtype=wp.mat44d), mapping: wp.array(dtype=wp.int32), transforms: wp.array(dtype=wp.transformf)):
    tid = wp.tid()
    if mapping[tid] > -1:
        m = fabric_matrices[mapping[tid]]

        orientation = wp.mat33f(wp.float32(m[0, 0]), wp.float32(m[1, 0]), wp.float32(m[2, 0]),
                                wp.float32(m[0, 1]), wp.float32(m[1, 1]), wp.float32(m[2, 1]),
                                wp.float32(m[0, 2]), wp.float32(m[1, 2]), wp.float32(m[2, 2]))

        position = wp.vec3f(wp.float32(m[3, 0]), wp.float32(m[3, 1]), wp.float32(m[3, 2]))

        transforms[tid] = wp.transformf(position, wp.normalize(wp.quat_from_matrix(orientation)))


class WarpRenderer:
    # Class variable to control verbose logging (set to False for production)
    VERBOSE_LOGGING = True
    
    @dataclass
    class PrimData:
        is_shared: bool = False
        shape_type: RenderShapeType = RenderShapeType.NONE
        master_prim: Usd.Prim | None = None
        prims: list[tuple[int, Usd.Prim]] = field(default_factory=lambda: [])

    @dataclass
    class CameraData:
        is_shared: bool = False
        master_prim: Usd.Prim | None = None
        prims: list[tuple[int, Usd.Prim]] = field(default_factory=lambda: [])

    def _verbose_print(self, msg: str):
        """Print verbose logging message if VERBOSE_LOGGING is enabled."""
        if WarpRenderer.VERBOSE_LOGGING:
            print(f"[VERBOSE] {msg}", flush=True)
    
    def __init__(self, scene: InteractiveScene, width: int, height: int):
        self._verbose_print(f"WarpRenderer.__init__ starting (width={width}, height={height})...")
        self.num_worlds = 0
        self.num_cameras = 1
        self.camera_data: dict[str, WarpRenderer.CameraData] = {}
        self.prim_data: dict[str, WarpRenderer.PrimData] = {}
        print(f"[VERBOSE] Collecting prims from stage...", flush=True)
        self.__collect_prims(isaaclab_sim.get_current_stage())
        print(f"[VERBOSE] Collected prims: {len(self.prim_data)} prim groups, {self.num_worlds} worlds", flush=True)

        shape_transforms = []
        shape_sizes = []
        shape_types = []
        shape_mesh_indices = []
        shape_world_indices = []

        self.__warp_meshes = []
        self.__fabric_stage_id = None
        self.__fabric_selection: usdrt.RtPrimSelection = None
        self.__fabric_selection_mapping: list[int] = []
        self.__fabric_selection_mapping_array: wp.array = None  # Cache for wp.array conversion
        self.__camera_transforms_cache: wp.array = None  # Cache for static camera transforms
        self.__camera_transforms_valid: bool = False  # Track if cache is valid

        for prim_path, prim_data in self.prim_data.items():
            if prim_data.is_shared:
                mesh_index = -1
                if prim_data.shape_type == RenderShapeType.MESH:
                    mesh_index = len(self.__warp_meshes)
                    self.__warp_meshes.append(self.__build_mesh(prim_data.master_prim))

                for world_id, prim in prim_data.prims:
                    shape_types.append(prim_data.shape_type)
                    shape_transforms.append(self.__resolve_transform(prim))
                    shape_sizes.append(self.__resolve_shape_size(prim_data.shape_type, prim))
                    shape_mesh_indices.append(mesh_index)
                    shape_world_indices.append(world_id)

        print(f"[VERBOSE] Creating RenderContext ({width}x{height}, {self.num_worlds} worlds, {self.num_cameras} cameras)...", flush=True)
        self.render_context = RenderContext(width, height, self.num_worlds, self.num_cameras)
        print(f"[VERBOSE] RenderContext created, setting up shapes ({len(shape_transforms)} total)...", flush=True)
        self.render_context.num_shapes_total = len(shape_transforms)
        self.render_context.num_shapes_enabled = self.render_context.num_shapes_total
        self.render_context.mesh_ids = wp.array([mesh.id for mesh in self.__warp_meshes], dtype=wp.uint64)
        self.render_context.shape_mesh_indices = wp.array(shape_mesh_indices, dtype=wp.int32)
        self.render_context.shape_types = wp.array(shape_types, dtype=wp.int32)
        self.render_context.mesh_bounds = wp.empty((self.render_context.num_shapes_total, 2), dtype=wp.vec3f, ndim=2)
        self.render_context.shape_enabled = wp.array(np.arange(self.render_context.num_shapes_total), dtype=wp.uint32)
        self.render_context.shape_sizes = wp.array(shape_sizes, dtype=wp.vec3f)
        self.render_context.shape_transforms = wp.array(shape_transforms, dtype=wp.transformf)
        self.render_context.shape_materials = wp.array(np.full(self.render_context.num_shapes_total, fill_value=-1, dtype=np.int32), dtype=wp.int32)
        self.render_context.shape_world_index = wp.array(shape_world_indices, dtype=wp.int32)
        print(f"[VERBOSE] Computing mesh bounds...", flush=True)
        self.render_context.utils.compute_mesh_bounds()
        print(f"[VERBOSE] Mesh bounds computed", flush=True)

        print(f"[VERBOSE] Creating default light and assigning colors...", flush=True)
        self.render_context.utils.create_default_light(False)
        self.render_context.utils.assign_random_colors_per_shape()

        print(f"[VERBOSE] Setting up camera rays...", flush=True)
        self.camera_fovs = wp.array([20.0] * self.render_context.num_cameras, dtype=wp.float32)
        self.camera_rays = self.render_context.utils.compute_pinhole_camera_rays(width, height, self.camera_fovs)
        self.color_image = self.render_context.create_color_image_output()
        print(f"[VERBOSE] WarpRenderer.__init__ complete!", flush=True)

    def __update_fabric_selection(self):
        print(f"[VERBOSE] WarpRenderer.__update_fabric_selection() - getting stage ID...", flush=True)
        stage_id = isaaclab_sim.get_current_stage_id()
        print(f"[VERBOSE] Stage ID: {stage_id}, previous fabric_stage_id: {self.__fabric_stage_id}", flush=True)
        if self.__fabric_stage_id != stage_id:
            print(f"[VERBOSE] Stage ID changed, resetting fabric selection...", flush=True)
            self.__fabric_stage_id = stage_id
            self.__fabric_selection = None

        print(f"[VERBOSE] Attaching to USD stage...", flush=True)
        stage = usdrt.Usd.Stage.Attach(stage_id)
        print(f"[VERBOSE] Stage attached successfully", flush=True)

        update_mapping = False
        if self.__fabric_selection is None:
            print(f"[VERBOSE] Fabric selection is None, selecting prims...", flush=True)
            self.__fabric_selection = stage.SelectPrims(require_attrs=[(usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read)], want_paths=True, device=wp.get_device().alias)
            print(f"[VERBOSE] Prims selected successfully", flush=True)
            update_mapping = True
        else:
            print(f"[VERBOSE] PrepareForReuse() on existing fabric selection...", flush=True)
            update_mapping = self.__fabric_selection.PrepareForReuse()
            print(f"[VERBOSE] PrepareForReuse() completed, update_mapping={update_mapping}", flush=True)

        if update_mapping:
            print(f"[VERBOSE] update_mapping is True, building fabric_selection_mapping...", flush=True)
            self.__fabric_selection_mapping = []
            print(f"[VERBOSE] Getting paths from fabric selection...", flush=True)
            selection_paths = self.__fabric_selection.GetPaths()
            print(f"[VERBOSE] Got {len(selection_paths)} selection paths", flush=True)
            
            # Convert to dict for O(1) lookup instead of O(n)
            print(f"[VERBOSE] Converting selection_paths to dict for fast lookup...", flush=True)
            selection_paths_dict = {path: idx for idx, path in enumerate(selection_paths)}
            print(f"[VERBOSE] Dict conversion complete, iterating through prim_data ({len(self.prim_data)} items)...", flush=True)
            
            for idx, (prim_path, prim_data) in enumerate(self.prim_data.items()):
                if idx % 10 == 0:
                    print(f"[VERBOSE] Processing prim {idx}/{len(self.prim_data)}: {prim_path}", flush=True)
                if prim_data.is_shared:
                    for world_id, prim in prim_data.prims:
                        path_key = prim_path % world_id
                        path_idx = selection_paths_dict.get(path_key, -1)
                        self.__fabric_selection_mapping.append(path_idx)
            print(f"[VERBOSE] Finished building fabric_selection_mapping with {len(self.__fabric_selection_mapping)} items", flush=True)
            
            # Cache the wp.array conversion (optimization)
            self.__fabric_selection_mapping_array = wp.array(self.__fabric_selection_mapping, dtype=wp.int32)
            print(f"[VERBOSE] Cached fabric_selection_mapping as wp.array", flush=True)
        else:
            print(f"[VERBOSE] update_mapping is False, skipping mapping build", flush=True)
        print(f"[VERBOSE] __update_fabric_selection() completed!", flush=True)

    def update(self):
        print(f"[VERBOSE] WarpRenderer.update() - calling __update_fabric_selection()...", flush=True)
        self.__update_fabric_selection()
        print(f"[VERBOSE] WarpRenderer.update() - __update_fabric_selection() completed", flush=True)
        
        print(f"[VERBOSE] WarpRenderer.update() - creating fabric_matrices array...", flush=True)
        fabric_matrices = wp.fabricarray(self.__fabric_selection.__fabric_arrays_interface__, "omni:fabric:worldMatrix", dtype=wp.mat44d)
        print(f"[VERBOSE] WarpRenderer.update() - fabric_matrices created, launching update_transforms kernel...", flush=True)
        
        # Use cached wp.array instead of creating it every frame (optimization)
        wp.launch(update_transforms, len(self.__fabric_selection_mapping), [fabric_matrices, self.__fabric_selection_mapping_array, self.render_context.shape_transforms])
        print(f"[VERBOSE] WarpRenderer.update() - update_transforms kernel launched and completed!", flush=True)

    def render(self):
        print(f"[VERBOSE] WarpRenderer.render() - getting camera transforms...", flush=True)
        # Use cached camera transforms if available (optimization for static cameras)
        if not self.__camera_transforms_valid:
            self.__camera_transforms_cache = self.__get_camera_transforms()
            self.__camera_transforms_valid = True
            print(f"[VERBOSE] Camera transforms computed and cached", flush=True)
        else:
            print(f"[VERBOSE] Using cached camera transforms", flush=True)
        
        print(f"[VERBOSE] WarpRenderer.render() - calling render_context.render()...", flush=True)
        self.render_context.render(self.__camera_transforms_cache, self.camera_rays, self.color_image)
        print(f"[VERBOSE] WarpRenderer.render() - render_context.render() completed!", flush=True)

    def save_image(self, filename: str):
        # Reshape color_image from (num_worlds, num_cameras, height*width) 
        # to (num_worlds, num_cameras, height, width)
        num_worlds = self.color_image.shape[0]
        num_cameras = self.color_image.shape[1]
        height = self.render_context.height
        width = self.render_context.width
        
        # Create a reshaped view of the color image
        color_image_reshaped = wp.array(
            ptr=self.color_image.ptr,
            dtype=self.color_image.dtype,
            shape=(num_worlds, num_cameras, height, width),
            device=self.color_image.device,
            copy=False
        )
        
        color_data = self.render_context.utils.flatten_color_image_to_rgba(color_image_reshaped)
        
        from PIL import Image
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        Image.fromarray(color_data.numpy()).save(filename)

    def __collect_prims(self, stage: Usd.Stage, env_regex: str = r"(?P<root>/World/envs/env_)(?P<id>\d+)(?P<path>/.*)"):
        env_pattern = re.compile(env_regex)

        num_worlds = -1
        stage_prims: list[Usd.Prim] = [stage.GetPseudoRoot()]
        while stage_prims:
            prim = stage_prims.pop(0)
            prim_path = prim.GetPath().pathString

            world_id = -1
            if match := env_pattern.match(prim_path):
                world_id = int(match.group("id"))
                prim_path = match.group("root") + "%d" + match.group("path")

            if world_id > -1:
                if world_id > num_worlds:
                    num_worlds = world_id

            imageable = UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue

            shape_type = RenderShapeType.NONE
            
            if prim.IsA(UsdGeom.Mesh):
                shape_type = RenderShapeType.MESH
            elif prim.IsA(UsdGeom.Sphere):
                shape_type = RenderShapeType.SPHERE
            elif prim.IsA(UsdGeom.Capsule):
                shape_type = RenderShapeType.CAPSULE
            elif prim.IsA(UsdGeom.Cube):
                shape_type = RenderShapeType.BOX
            elif prim.IsA(UsdGeom.Cylinder):
                shape_type = RenderShapeType.CYLINDER
            elif prim.IsA(UsdGeom.Cone):
                shape_type = RenderShapeType.CONE
            elif prim.IsA(UsdGeom.Plane):
                shape_type = RenderShapeType.PLANE
            elif prim.IsA(UsdGeom.Camera):
                if not prim_path in self.camera_data:
                    self.camera_data[prim_path] = WarpRenderer.CameraData(world_id > -1, prim)
                self.camera_data[prim_path].prims.append((world_id, prim))

            if shape_type != RenderShapeType.NONE:
                if not prim_path in self.prim_data:
                    self.prim_data[prim_path] = WarpRenderer.PrimData(world_id > -1, shape_type, prim)
                self.prim_data[prim_path].prims.append((world_id, prim))

            if child_prims := prim.GetFilteredChildren(Usd.TraverseInstanceProxies()):
                stage_prims.extend(child_prims)

        self.num_cameras = 0
        for prim_path, camera_data in self.camera_data.items():
            if camera_data.is_shared:
                self.num_cameras += 1
        self.num_worlds = num_worlds + 1

    def __build_mesh(self, prim: Usd.Prim):
        mesh = UsdGeom.Mesh(prim)
        points = wp.array(mesh.GetPointsAttr().Get(), dtype=wp.vec3f)
        face_vertex_counts = wp.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=wp.int32)
        face_vertex_indices = wp.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=wp.int32)

        num_face_counts = face_vertex_counts.size
        wp_face_offsets = wp.zeros(num_face_counts + 2, dtype=wp.int32)

        wp.launch(kernel=compute_triangle_count, dim=1, inputs=[num_face_counts, face_vertex_counts, wp_face_offsets])

        num_triangles = wp_face_offsets.numpy()[num_face_counts].item()
        wp_triangle_indices = wp.empty(num_triangles * 3, dtype=wp.int32)

        wp.launch(kernel=triangulate_faces, dim=num_face_counts, inputs=[num_face_counts, face_vertex_counts, face_vertex_indices, wp_face_offsets, wp_triangle_indices])

        return wp.Mesh(points=points, velocities=None, indices=wp_triangle_indices)

    def __resolve_transform(self, prim: Usd.Prim) -> wp.transformf:
        position, orientation = isaaclab_sim.resolve_prim_pose(prim)
        return wp.transformf(position, wp.quatf(orientation[1], orientation[2], orientation[3], orientation[0]))

    def __resolve_scale(self, prim: Usd.Prim) -> wp.vec3f:
        scale = isaaclab_sim.resolve_prim_scale(prim)
        return wp.vec3f(scale)

    def __resolve_camera_transform(self, prim: Usd.Prim) -> wp.transformf:
        position, orientation = isaaclab_sim.resolve_prim_pose(prim)
        return wp.transformf(position, wp.quatf(orientation[1], -orientation[2], -orientation[3], orientation[0]))
        # position, orientation = isaaclab_sim.resolve_prim_pose(prim)
        # t = torch.tensor(orientation, dtype=torch.float32, device="cpu").unsqueeze(0)
        # t = convert_camera_frame_orientation_convention(t)
        # orientation = t.squeeze(0).cpu().numpy()
        # return wp.transformf(position, wp.quatf(orientation))

    def __resolve_shape_size(self, shape_type: RenderShapeType, prim: Usd.Prim) -> wp.vec3f:
        if shape_type == RenderShapeType.SPHERE:
            return self.__resolve_shape_size_sphere(prim)
        if shape_type == RenderShapeType.CAPSULE:
            return self.__resolve_shape_size_capsule(prim)
        if shape_type == RenderShapeType.BOX:
            return self.__resolve_shape_size_box(prim)
        if shape_type == RenderShapeType.CYLINDER:
            return self.__resolve_shape_size_cylinder(prim)
        if shape_type == RenderShapeType.CONE:
            return self.__resolve_shape_size_cone(prim)
        if shape_type == RenderShapeType.PLANE:
            return self.__resolve_shape_size_plane(prim)
        return self.__resolve_scale(prim)

    def __resolve_shape_size_sphere(self, prim: Usd.Prim) -> wp.vec3f:
        sphere = UsdGeom.Sphere(prim)
        radius = sphere.GetRadiusAttr().Get()
        if radius is None:
            radius = 0.5  # Default sphere radius
        return wp.vec3f(radius, 0.0, 0.0)

    def __resolve_shape_size_capsule(self, prim: Usd.Prim) -> wp.vec3f:
        capsule = UsdGeom.Capsule(prim)
        radius = capsule.GetRadiusAttr().Get()
        height = capsule.GetHeightAttr().Get()
        if radius is None:
            radius = 0.5  # Default capsule radius
        if height is None:
            height = 1.0  # Default capsule height
        return wp.vec3f(radius, height, 0.0)

    def __resolve_shape_size_box(self, prim: Usd.Prim) -> wp.vec3f:
        cube = UsdGeom.Cube(prim)
        size = cube.GetSizeAttr().Get()
        if size is None:
            size = 1.0  # Default cube size
        scale = self.__resolve_scale(prim)
        return wp.vec3f(size * scale[0], size * scale[1], size * scale[2])

    def __resolve_shape_size_cylinder(self, prim: Usd.Prim) -> wp.vec3f:
        cylinder = UsdGeom.Cylinder(prim)
        radius = cylinder.GetRadiusAttr().Get()
        height = cylinder.GetHeightAttr().Get()
        if radius is None:
            radius = 0.5  # Default cylinder radius
        if height is None:
            height = 1.0  # Default cylinder height
        return wp.vec3f(radius, height, 0.0)

    def __resolve_shape_size_cone(self, prim: Usd.Prim) -> wp.vec3f:
        cone = UsdGeom.Cone(prim)
        radius = cone.GetRadiusAttr().Get()
        height = cone.GetHeightAttr().Get()
        if radius is None:
            radius = 0.5  # Default cone radius
        if height is None:
            height = 1.0  # Default cone height
        return wp.vec3f(radius, height, 0.0)

    def __resolve_shape_size_plane(self, prim: Usd.Prim) -> wp.vec3f:
        return wp.vec3f(0.0)

    def __get_camera_transforms(self) -> wp.array(dtype=wp.transformf):
        camera_transforms = [[] for i in range(self.num_worlds)]
        for prim_path, camera_data in self.camera_data.items():
            if camera_data.is_shared:
                for world_id, camera_prim in camera_data.prims:
                    camera_transforms[world_id].append(self.__resolve_camera_transform(camera_prim))

        return wp.array(camera_transforms, dtype=wp.transformf)
