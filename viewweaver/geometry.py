"""Depth unprojection, consistent camera normalization and RGB point rendering."""
import math

import torch
import torch.nn.functional as F


def unproject_depth(depth, extrinsics, intrinsics):
    """OpenCV cameras: depth [S,H,W], w2c [S,3,4], K [S,3,3]."""
    height, width = depth.shape[-2:]
    y, x = torch.meshgrid(torch.arange(height, device=depth.device),
                          torch.arange(width, device=depth.device), indexing="ij")
    x = (x - intrinsics[:, None, None, 0, 2]) / intrinsics[:, None, None, 0, 0]
    y = (y - intrinsics[:, None, None, 1, 2]) / intrinsics[:, None, None, 1, 1]
    camera_points = torch.stack((x * depth, y * depth, depth), dim=-1)
    return torch.einsum("shwc,scd->shwd",
                       camera_points - extrinsics[:, None, None, :, 3], extrinsics[:, :, :3])


def normalize_scene(points, source_w2c):
    lower, upper = points.amin(dim=0), points.amax(dim=0)
    center = (lower + upper) / 2
    scale = (upper - lower).amax() / 2
    if not torch.isfinite(scale) or scale < 1e-6:
        raise ValueError("Degenerate reconstructed point cloud")
    normalized = source_w2c.clone()
    normalized[:, :, :3] = source_w2c[:, :, :3] * scale
    normalized[:, :, 3] = source_w2c[:, :, :3] @ center + source_w2c[:, :, 3]
    return (points - center) / scale, normalized, center, scale


def angle_pairs(azimuth, elevation):
    """Pair angles in degrees, broadcasting a singleton; never a Cartesian product."""
    azimuth, elevation = list(azimuth), list(elevation)
    if not azimuth or not elevation or not all(math.isfinite(v) for v in azimuth + elevation):
        raise ValueError("Angles must be nonempty finite lists")
    if any(not -89 < v < 89 for v in elevation):
        raise ValueError("Elevation must be between -89 and 89 degrees (exclusive)")
    count = max(len(azimuth), len(elevation))
    if len(azimuth) not in (1, count) or len(elevation) not in (1, count):
        raise ValueError("Angle lists must have equal lengths, or one list must contain a single value")
    return azimuth * (count if len(azimuth) == 1 else 1), elevation * (count if len(elevation) == 1 else 1)


def orbit_cameras(source_w2c, scale, azimuth=(0.0,), elevation=(15.0,), distance=3.5,
                  fov=55.0, principal_y=0.0, image_size=512, reference_view=0, principal_x=0.0):
    """Generate targets in the reconstructed frame, without benchmark cameras."""
    if not 0 <= reference_view < len(source_w2c):
        raise ValueError(f"reference-view must be between 0 and {len(source_w2c) - 1}")
    reference = source_w2c[reference_view]
    source_rotation = reference[:, :3] / scale
    up = F.normalize(source_rotation.T @ source_w2c.new_tensor([0, -1, 0]), dim=0)
    front = -torch.linalg.solve(reference[:, :3], reference[:, 3])
    front = front - (front @ up) * up
    if front.norm() < 1e-6:
        front = -(source_rotation.T @ source_w2c.new_tensor([0, 0, 1]))
        front = front - (front @ up) * up
    front = F.normalize(front, dim=0)
    side = F.normalize(torch.linalg.cross(up, front), dim=0)
    azimuth, elevation = angle_pairs(azimuth, elevation)
    angles = source_w2c.new_tensor(azimuth).deg2rad()
    horizontal = angles.cos()[:, None] * front + angles.sin()[:, None] * side
    elev = source_w2c.new_tensor(elevation).deg2rad()[:, None]
    positions = distance * (elev.cos() * horizontal + elev.sin() * up)
    forward = F.normalize(-positions, dim=-1)
    right = F.normalize(torch.linalg.cross(forward, up.expand_as(forward)), dim=-1)
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack((right, down, forward), dim=1)
    translation = -torch.einsum("vij,vj->vi", rotation, positions)
    # Keep the same camera-space scale as the VGGT source cameras.
    targets = torch.cat((rotation, translation[:, :, None]), dim=-1) * scale
    focal = image_size / (2 * math.tan(math.radians(fov) / 2))
    intrinsic = source_w2c.new_tensor([
        [focal, 0, image_size * (0.5 + principal_x)],
        [0, focal, image_size * (0.5 + principal_y)],
        [0, 0, 1],
    ])
    return targets, intrinsic.expand(len(azimuth), -1, -1).clone()


@torch.no_grad()
def render_points(points, colors, w2c, intrinsic, image_size=512, radius=1.5):
    """Opaque RGB splats with a z-buffer, implemented with native torch scatter."""
    camera = points @ w2c[:, :3].T + w2c[:, 3]
    valid = torch.isfinite(camera).all(-1) & (camera[:, 2] > 1e-6)
    camera, rgb = camera[valid], colors[valid]
    uv = camera @ intrinsic.T
    xy = (uv[:, :2] / uv[:, 2:]).round().long()
    depth = camera[:, 2]
    limit = math.ceil(radius)
    offsets = [(dx, dy) for dy in range(-limit, limit + 1)
               for dx in range(-limit, limit + 1) if dx * dx + dy * dy <= radius * radius]
    # ponytail: hard opaque discs, O(N*r^2); use an alpha rasterizer if soft splats are needed.
    indices, point_ids = [], []
    ids = torch.arange(len(xy), device=points.device)
    for dx, dy in offsets:
        x, y = xy[:, 0] + dx, xy[:, 1] + dy
        keep = (x >= 0) & (x < image_size) & (y >= 0) & (y < image_size)
        indices.append(y[keep] * image_size + x[keep])
        point_ids.append(ids[keep])
    indices, point_ids = torch.cat(indices), torch.cat(point_ids)
    if indices.numel() == 0:
        raise ValueError("Target camera sees no reconstructed foreground; adjust the orbit")
    zbuffer = points.new_full((image_size * image_size,), float("inf"))
    zbuffer.scatter_reduce_(0, indices, depth[point_ids], reduce="amin", include_self=True)
    nearest = depth[point_ids] == zbuffer[indices]
    winners = torch.full_like(zbuffer, len(rgb), dtype=torch.long)
    winners.scatter_reduce_(0, indices[nearest], point_ids[nearest], reduce="amin", include_self=True)
    palette = torch.cat((rgb, torch.ones((1, 3), device=rgb.device)), dim=0)
    image = palette[winners].reshape(image_size, image_size, 3)
    mask = torch.isfinite(zbuffer).reshape(image_size, image_size)
    return image, mask
