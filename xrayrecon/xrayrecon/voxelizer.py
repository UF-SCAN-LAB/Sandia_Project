
import numpy as np
import cv2 as cv
from typing import List, Tuple
from .gerber_utils import GerberLayer
from .materials import MATERIAL_NAMES, MaterialProperties

_MATERIAL_ID_BY_NAME = {name: idx for idx, name in enumerate(MATERIAL_NAMES)}


def _finite_bounds(gl: GerberLayer) -> bool:
    return all(np.isfinite(v) for v in gl.bounds) and (
        gl.bounds[2] > gl.bounds[0] and gl.bounds[3] > gl.bounds[1]
    )


class Voxelizer:
    def __init__(self, resolution_xy: float = 0.01, resolution_z: float = 0.01):
        """
        resolution_xy: mm/pixel
        resolution_z: mm/slice
        """
        self.res_xy = resolution_xy
        self.res_z = resolution_z

    def voxelize_stackup(self,
                         layers: List[Tuple[GerberLayer, float, MaterialProperties, MaterialProperties]],
                         subplot_material: MaterialProperties,
                         return_material_ids: bool = False): # subplot arg kept for compatibility but not used if layers have bg
        
        # 1. Determine Stackup Dimensions
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        has_bounds = False
        for gl, _, _, _ in layers:
            if isinstance(gl, GerberLayer) and _finite_bounds(gl):
                min_x = min(min_x, gl.bounds[0])
                min_y = min(min_y, gl.bounds[1])
                max_x = max(max_x, gl.bounds[2])
                max_y = max(max_y, gl.bounds[3])
                has_bounds = True

        if not has_bounds:
            # Fallback size
            min_x, min_y, max_x, max_y = 0, 0, 10, 10

        # Add margin
        margin = 1.0 # mm
        min_x -= margin
        min_y -= margin
        max_x += margin
        max_y += margin
        
        width_mm = max_x - min_x
        height_mm = max_y - min_y
        
        if width_mm <= 0: width_mm = 10
        if height_mm <= 0: height_mm = 10
        
        w_px = int(np.ceil(width_mm / self.res_xy))
        h_px = int(np.ceil(height_mm / self.res_xy))
        
        print(f"Voxel Grid Size: {w_px}x{h_px} (XY) - using res {self.res_xy} mm/px")

        volume_slices = []
        label_slices = []

        for gl, thickness, feat_mat, bg_mat in layers:
            num_slices = int(np.round(thickness / self.res_z))
            if num_slices < 1: num_slices = 1
            
            # --- Rasterization / Mask Generation ---
            
            # If PyGerber raster mask exists, use it/resize it
            if gl and gl.raster_mask is not None:
                mask = np.zeros((h_px, w_px), dtype=np.uint8)

                if _finite_bounds(gl):
                    layer_min_x, layer_min_y, layer_max_x, layer_max_y = gl.bounds
                    target_w = max(
                        1, int(np.ceil((layer_max_x - layer_min_x) / self.res_xy))
                    )
                    target_h = max(
                        1, int(np.ceil((layer_max_y - layer_min_y) / self.res_xy))
                    )
                    resized = cv.resize(
                        gl.raster_mask,
                        (target_w, target_h),
                        interpolation=cv.INTER_NEAREST,
                    )

                    x0 = int(np.round((layer_min_x - min_x) / self.res_xy))
                    y0 = int(np.round((layer_min_y - min_y) / self.res_xy))
                    x1 = x0 + target_w
                    y1 = y0 + target_h

                    src_x0 = max(0, -x0)
                    src_y0 = max(0, -y0)
                    dst_x0 = max(0, x0)
                    dst_y0 = max(0, y0)
                    dst_x1 = min(w_px, x1)
                    dst_y1 = min(h_px, y1)

                    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
                        mask[dst_y0:dst_y1, dst_x0:dst_x1] = resized[
                            src_y0 : src_y0 + (dst_y1 - dst_y0),
                            src_x0 : src_x0 + (dst_x1 - dst_x0),
                        ]
                else:
                    mask = cv.resize(
                        gl.raster_mask,
                        (w_px, h_px),
                        interpolation=cv.INTER_NEAREST,
                    )
                
            elif gl and gl.objects:
                # Custom Simple Vector Rasterizer
                mask = np.zeros((h_px, w_px), dtype=np.uint8)
                
                def to_px(pt):
                    return (int((pt[0] - min_x) / self.res_xy),
                            int((pt[1] - min_y) / self.res_xy))
                
                for obj in gl.objects:
                    p1 = to_px(obj.start)
                    p2 = to_px(obj.end)
                    
                    if obj.aperture.type == 'C' or obj.aperture.type == 'R':
                        # Use first param as size/diameter
                         thickness_px = int(obj.aperture.params[0] / self.res_xy)
                         if thickness_px < 1: thickness_px = 1
                         radius = thickness_px // 2
                    else:
                        thickness_px = 1
                        radius = 1
                    
                    if obj.type == 'LINE':
                        cv.line(mask, p1, p2, 255, thickness_px)
                        cv.circle(mask, p1, radius, 255, -1)
                        cv.circle(mask, p2, radius, 255, -1)
                        
                    elif obj.type == 'FLASH':
                        if obj.aperture.type == 'C': 
                             cv.circle(mask, p1, radius, 255, -1)
                        elif obj.aperture.type == 'R': 
                             w_dim = int(obj.aperture.params[0] / self.res_xy)
                             h_dim = int(obj.aperture.params[1] / self.res_xy) if len(obj.aperture.params)>1 else w_dim
                             top_left = (p1[0] - w_dim//2, p1[1] - h_dim//2)
                             bottom_right = (p1[0] + w_dim//2, p1[1] + h_dim//2)
                             cv.rectangle(mask, top_left, bottom_right, 255, -1)
            else:
                 # Empty layer
                 mask = np.zeros((h_px, w_px), dtype=np.uint8)

            # Apply Materials
            slice_z = np.full((h_px, w_px), bg_mat.z_eff, dtype=np.float32)
            slice_rho = np.full((h_px, w_px), bg_mat.electron_density, dtype=np.float32)

            feat_mask_bool = (mask > 128)
            slice_z[feat_mask_bool] = feat_mat.z_eff
            slice_rho[feat_mask_bool] = feat_mat.electron_density

            combined_slice = np.stack([slice_z, slice_rho], axis=-1)

            if return_material_ids:
                bg_id = self._material_id(bg_mat)
                feat_id = self._material_id(feat_mat)
                label_slice = np.full((h_px, w_px), bg_id, dtype=np.int32)
                label_slice[feat_mask_bool] = feat_id
            else:
                label_slice = None

            for _ in range(num_slices):
                volume_slices.append(combined_slice)
                if return_material_ids:
                    label_slices.append(label_slice)

        volume = np.stack(volume_slices, axis=0)
        if not return_material_ids:
            return volume
        material_id_volume = np.stack(label_slices, axis=0)
        return volume, material_id_volume

    @staticmethod
    def _material_id(mat: MaterialProperties) -> int:
        idx = _MATERIAL_ID_BY_NAME.get(mat.name.upper())
        if idx is None:
            available = ", ".join(MATERIAL_NAMES)
            raise ValueError(
                f"Material '{mat.name}' is not a MATERIALS_DB entry -- ground-truth "
                f"material-ID labeling requires one of: {available}."
            )
        return idx
