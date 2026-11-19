# SPDX-FileCopyrightText: Copyright (c) 2026 Skild AI and the Steel contributors
# SPDX-License-Identifier: Apache-2.0

"""Hard-edged (flat) shading for meshes the viewer would otherwise smooth.

Newton's OpenGL viewer computes per-vertex normals averaged over the triangles that share a vertex, which rounds
faceted shapes (stones, cubes, a squeezed soft block) into blobs. Both helpers draw the same triangles with
unshared vertices, so every face gets its own normal. Neither touches collision geometry, mass or the simulation.
"""

from __future__ import annotations

import newton
import numpy as np
import warp as wp


def flat_shade_meshes(builder: newton.ModelBuilder) -> None:
    """Hide every mesh shape and add a visual-only copy with unshared vertices and face normals to the same body."""
    visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    for s in range(builder.shape_count):
        mesh = builder.shape_source[s]
        if builder.shape_type[s] != newton.GeoType.MESH or mesh is None:
            continue
        tris = mesh.vertices[mesh.indices.reshape(-1, 3)]  # (T, 3, 3)
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        flat = newton.Mesh(
            tris.reshape(-1, 3), np.arange(3 * len(tris), dtype=np.int32), normals=np.repeat(n, 3, axis=0),
            compute_inertia=False,
        )  # fmt: skip
        builder.shape_flags[s] &= ~int(newton.ShapeFlags.VISIBLE)
        builder.add_shape_mesh(
            builder.shape_body[s], xform=builder.shape_transform[s], mesh=flat, scale=builder.shape_scale[s],
            cfg=visual, color=builder.shape_color[s], label=f"{builder.shape_label[s]}/flat",
        )  # fmt: skip


@wp.kernel
def _gather_corners(q: wp.array[wp.vec3], tri: wp.array[wp.int32], out: wp.array[wp.vec3]):
    i = wp.tid()
    out[i] = q[tri[i]]


class FlatTriangles:
    """Draws the model's particle triangles (cloth, soft grid surfaces) flat-shaded in place of the viewer's smooth
    surface. Call :meth:`log` each frame after ``viewer.log_state``."""

    def __init__(self, viewer, model: newton.Model, color=None):
        self.viewer, self.model = viewer, model
        self.tri = model.tri_indices.flatten()
        n = self.tri.shape[0]
        self.points = wp.zeros(n, dtype=wp.vec3, device=model.device)
        self.indices = wp.array(np.arange(n, dtype=np.int32), device=model.device)
        if color is None and model.tri_color is not None:
            color = tuple(float(c) for c in model.tri_color.numpy()[0][:3])
        self.color = color
        viewer.show_triangles = False  # the smooth surface is replaced, not drawn under this one

    def log(self, state: newton.State) -> None:
        if not self.model.tri_count or state.particle_q is None:
            return
        wp.launch(_gather_corners, dim=self.points.shape[0], inputs=[state.particle_q, self.tri, self.points])
        self.viewer.log_mesh(
            "/flat/triangles", self.points, self.indices, backface_culling=False, color=self.color, dynamic=True
        )
