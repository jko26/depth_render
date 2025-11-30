import open3d as o3d
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pydicom
import os
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import SimpleITK as sitk
from scipy.ndimage import map_coordinates
import image_utils
import pdb

# Clean png files 
prefixes = [
    "ct_window",
    "ct_slice_with_raypoint",
    "ct_slice_HU_colormap"
]

cwd = Path(".")

for f in cwd.iterdir():
    if f.is_file() and f.suffix.lower() == ".png":
        if any(f.name.startswith(p) for p in prefixes):
            print(f"Deleting {f}")
            f.unlink()  # deletes the file

def display_ct_window(ct_volume, z_idx, x_idx, y_idx, window_size=20):
    """
    Display a zoomed window of CT slice with HU values overlaid.
    
    Parameters:
    -----------
    ct_volume : np.ndarray
        3D CT volume with shape (Z, X, Y)
    z_idx, x_idx, y_idx : int
        Center coordinates of the window
    window_size : int
        Size of window to display (default 20x20)
    """
    # Clamp indices
    z_int = np.clip(z_idx, 0, ct_volume.shape[0] - 1)
    x_int = np.clip(x_idx, 0, ct_volume.shape[1] - 1)
    y_int = np.clip(y_idx, 0, ct_volume.shape[2] - 1)
    
    # Extract full slice (X, Y)
    ct_slice = ct_volume[z_int, :, :]
    
    # Window bounds
    #pdb.set_trace()
    row_idx = y_int
    col_idx = x_int
    half_window = window_size // 2
    row_start = max(0, row_idx - half_window)
    row_end = min(ct_slice.shape[0], row_idx + half_window)
    col_start = max(0, col_idx - half_window)
    col_end = min(ct_slice.shape[1], col_idx + half_window)
    
    # Extract window
    window = ct_slice[row_start:row_end, col_start:col_end]

    fig, ax = plt.subplots(figsize=(10, 10))

    # *** X on horizontal axis, Y on vertical axis ***
    # extent = [xmin, xmax, ymin, ymax]
    im = ax.imshow(
        window,                
        cmap='gray',
        origin='lower',
    )

    ax.set_title(f"CT slice z={z_int}, center (x={x_int}, y={y_int})")
    ax.set_xlabel("X (voxel index)")
    ax.set_ylabel("Y (voxel index)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("HU")

    # Mark the center point (using swapped axes)
    ax.plot(half_window, half_window, 'b+', markersize=15, markeredgewidth=2)

    plt.tight_layout()
    out = f"ct_window_z{z_int}_x{x_int}_y{y_int}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Saved {out}")


def overlay_on_ct_slice(ct_volume, mesh, point_cloud, slice_idx, ct_spacing, 
                        axis='axial', output_file='ct_overlay.png'):
    """
    Overlay mesh edges and point cloud on a CT slice.
    
    Parameters:
    - ct_volume: 3D numpy array (Z, Y, X) in HU
    - mesh: open3d.geometry.TriangleMesh
    - point_cloud: numpy array (N, 3) in physical coordinates (mm)
    - slice_idx: which slice to visualize
    - ct_spacing: [pixel_spacing_x, pixel_spacing_y, slice_thickness]
    - axis: 'axial', 'sagittal', or 'coronal'
    """
    
    # Extract CT slice based on axis
    if axis == 'axial':
        ct_slice = ct_volume[slice_idx, :, :]
        slice_pos = slice_idx * ct_spacing[2]  # Z position in mm
        coord_idx = 2  # Z coordinate
        extent = [0, ct_slice.shape[1]*ct_spacing[0],  # X range
                  0, ct_slice.shape[0]*ct_spacing[1]]   # Y range
        x_label, y_label = 'X (mm)', 'Y (mm)'
    elif axis == 'sagittal':
        ct_slice = ct_volume[:, :, slice_idx]  # (Z, Y) slice at X=slice_idx
        slice_pos = slice_idx * ct_spacing[0]  # X position in mm
        coord_idx = 0  # X coordinate (not 1!)
        extent = [0, ct_slice.shape[1]*ct_spacing[1],  # Y range
                  0, ct_slice.shape[0]*ct_spacing[2]]   # Z range
        x_label, y_label = 'Y (mm)', 'Z (mm)'
    elif axis == 'coronal':
        ct_slice = ct_volume[:, slice_idx, :]  # (Z, X) slice at Y=slice_idx
        slice_pos = slice_idx * ct_spacing[1]  # Y position in mm
        coord_idx = 1  # Y coordinate (not 0!)
        extent = [0, ct_slice.shape[1]*ct_spacing[0],  # X range
                  0, ct_slice.shape[0]*ct_spacing[2]]   # Z range
        x_label, y_label = 'X (mm)', 'Z (mm)'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display CT slice
    ax.imshow(ct_slice, cmap='gray', origin='lower', extent=extent)
    
    # Project mesh vertices onto slice plane
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Find triangles that intersect the slice
    tolerance = ct_spacing[coord_idx] * 2
    for tri in triangles:
        tri_verts = vertices[tri]
        if axis == 'axial':
            # Check if triangle intersects Z plane
            if np.any(np.abs(tri_verts[:, 2] + slice_pos) < tolerance):
                ax.plot(-tri_verts[:, 0], -tri_verts[:, 1], 'b-', linewidth=0.5, alpha=0.6)
        elif axis == 'sagittal':
            # Check if triangle intersects X plane
            if np.any(np.abs(tri_verts[:, 0] + slice_pos) < tolerance):
                ax.plot(-tri_verts[:, 1], tri_verts[:, 2] + 159.103, 'b-', linewidth=0.5, alpha=0.6)
        elif axis == 'coronal':
            # Check if triangle intersects Y plane
            if np.any(np.abs(tri_verts[:, 1] + slice_pos) < tolerance):
                ax.plot(-tri_verts[:, 0], tri_verts[:, 2] + 159.103, 'b-', linewidth=0.5, alpha=0.6)
    
    # Project point cloud onto slice
    if axis == 'axial':
        pc_mask = np.abs(point_cloud[:, 2] + slice_pos) < tolerance
        pc_proj = point_cloud[pc_mask][:, [0, 1]]
        if len(pc_proj) > 0:
            ax.scatter(-pc_proj[:, 0], -pc_proj[:, 1], c='red', s=2, alpha=0.8, label='Point Cloud')
    elif axis == 'sagittal':
        pc_mask = np.abs(point_cloud[:, 0] + slice_pos) < tolerance
        pc_proj = point_cloud[pc_mask][:, [1, 2]]
        if len(pc_proj) > 0:
            ax.scatter(-pc_proj[:, 0], pc_proj[:, 1] + 159.103, c='red', s=2, alpha=0.8, label='Point Cloud')
    elif axis == 'coronal':
        pc_mask = np.abs(point_cloud[:, 1] + slice_pos) < tolerance
        pc_proj = point_cloud[pc_mask][:, [0, 2]]
        if len(pc_proj) > 0:
            ax.scatter(-pc_proj[:, 0], pc_proj[:, 1] + 159.103, c='red', s=2, alpha=0.8, label='Point Cloud')
    
    
    
    ax.set_title(f'{axis.capitalize()} Slice {slice_idx} - Mesh (blue) & Points (red)')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_file}")


def world_to_ct(P_world_xyz, origin, ct_spacing):
    """Convert world points assumed RAS to CT LPS coordinates (mm)."""
    P = np.asarray(P_world_xyz, float).reshape(-1, 3) 
    P = P - origin
    P = P / ct_spacing
    return np.array([P[:, 2], -P[:, 0], -P[:, 1]]) # z, x, y

def world_to_ct_row_idx(P_world_xyz, origin, ct_spacing):
    P = np.asarray(P_world_xyz, float).reshape(-1, 3) 
    P = P - origin
    P = P / ct_spacing
    return np.array([P[:, 2], -P[:, 1], -P[:, 0]]) # z, y, x

# ==================== MAIN CODE ====================

patient01_datapath = "sample_data/Patient01/Processed/"

p01_json_data = json.load(open(patient01_datapath + "p01_left_preop_scope01.json"))
mesh = o3d.io.read_triangle_mesh(patient01_datapath + "segmentations/p01_left_preop_scope01_mesh.stl")
mesh.compute_vertex_normals()

intrinsic_matrix = np.array(p01_json_data["intrinsics"])
fx = intrinsic_matrix[0][0]
fy = intrinsic_matrix[1][1]
cx = intrinsic_matrix[0][2]
cy = intrinsic_matrix[1][2]

width, height = 1920, 1080
intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# Mesh render
render = o3d.visualization.rendering.OffscreenRenderer(width, height)
render.scene.set_background([0, 0, 0, 0])
render.scene.add_geometry("segmentation", mesh, o3d.visualization.rendering.MaterialRecord())

# load CT (SimpleITK returns physical LPS, mm; intensities already in HU)
reader = sitk.ImageSeriesReader()
files = reader.GetGDCMSeriesFileNames(patient01_datapath + "/dicom")
reader.SetFileNames(files)
ct_img = reader.Execute()
ct_volume = sitk.GetArrayFromImage(ct_img) 

# CT metadata
origin    = np.array(ct_img.GetOrigin(), dtype=float)       # (ox,oy,oz)
ct_spacing   = np.array(ct_img.GetSpacing(), dtype=float)      # (sx,sy,sz)
direction = np.array(ct_img.GetDirection(), dtype=float).reshape(3,3)

# print(f"CT shape (Z,Y,X): {ct_volume.shape}")
# print("CT origin (mm):", origin, "spacing (mm):", ct_spacing)
# print("CT direction:\n", direction)

for img_name in os.listdir(patient01_datapath + "correspondences/"):
    
    extrinsic_matrix_c_to_w = np.array(p01_json_data["poses"][img_name[7:]])  
    extrinsic_matrix_w_to_c = np.linalg.inv(extrinsic_matrix_c_to_w)

    render.setup_camera(intrinsics, extrinsic_matrix_w_to_c)

    # Pixels range from 0 (near plane) to 1 (far plane). If z_in_view_space is set to True then pixels are pre-transformed into view space (i.e., distance from camera).
    depth_image = np.asarray(render.render_to_depth_image(z_in_view_space = True))
    depth_img_obj = o3d.geometry.Image(depth_image.astype(np.float32))
    
    # pc_world is in world coordinate system (extrinsic converts it to world)
    pc_world = o3d.geometry.PointCloud.create_from_depth_image(
        depth_img_obj, intrinsics, extrinsic=extrinsic_matrix_w_to_c, depth_scale=1.0
    )
    pts = np.asarray(pc_world.points)
    
    # Initialize HU image obtained from CT volume at the point cloud coordinates
    hu_image = np.zeros((height, width), dtype=np.float32)
    hu_values = map_coordinates(
        ct_volume,
        world_to_ct_row_idx(pts, origin, ct_spacing),
        order=1, 
        mode='constant',
        cval=np.nan 
    )
    print("Hu_values.shape", hu_values.shape)
    hu_image.ravel()[:] = hu_values

    midpoint_x = width // 2
    midpoint_y = height // 2

    midpoint_depth = depth_image[midpoint_y, midpoint_x]  # index image by row, column
    hu_vals = []

    depths = np.linspace(0, 1.4 * midpoint_depth, 40)


    for i, z in enumerate(depths):
        # Back-project pixel (midpoint_x, midpoint_y) at depth z
        projection_point = image_utils.project_2d_to_3d(
            midpoint_x,
            midpoint_y,
            z,
            pose=extrinsic_matrix_w_to_c,
            intrinsics=intrinsic_matrix,
            world_to_camera=False  # For some reason it works with world_to_camera=False...?  
        ).reshape(1, 3)  # (1, 3) world coords
        projection_point_in_ct = world_to_ct(projection_point, origin, ct_spacing)
        projection_point_in_ct = np.asarray(projection_point_in_ct).reshape(3, -1)

        # Unpack continuous indices (z, x, y)
        z_idx, x_idx, y_idx = projection_point_in_ct[:, 0]

        in_bounds = (
            0 <= z_idx < ct_volume.shape[0] and
            0 <= y_idx < ct_volume.shape[1] and
            0 <= x_idx < ct_volume.shape[2]
        )

        z_int = int(round(z_idx))
        y_int = int(round(y_idx))
        x_int = int(round(x_idx))
        row_idx = y_int
        col_idx = x_int

        row_idx_point = np.array([[z_idx], [y_idx], [x_idx]])
        #print(f"z={z}: idx=({z_idx}, {y_idx}, {x_idx}), in_bounds={in_bounds}")
        #print("row_idx_point", row_idx_point)
        #Sample HU at this point
        hu_value_proj_pt = map_coordinates(
            ct_volume,
            row_idx_point,
            order=1, 
            mode='constant',
            cval=np.nan 
        )[0]  # scalar
    
        
        #hu_value_proj_pt = ct_volume[z_int][x_int][y_int]
        #hu_value_proj_pt = ct_volume[z_int][row_idx][col_idx]

        #print("hu_value_proj_pt", hu_value_proj_pt)
        hu_vals.append(hu_value_proj_pt)

        # --- NEW: visualize the CT slice with the sampled point ---
        if in_bounds and img_name[7:-4] == "00001994":
            # Nearest voxel indices for visualization
            z_int = int(round(z_idx))
            y_int = int(round(y_idx))
            x_int = int(round(x_idx))

            # Clamp just in case rounding lands on edge
            z_int = np.clip(z_int, 0, ct_volume.shape[0] - 1)
            y_int = np.clip(y_int, 0, ct_volume.shape[1] - 1)
            x_int = np.clip(x_int, 0, ct_volume.shape[2] - 1)

            ct_slice = ct_volume[z_int, :, :]  # (X, Y)

            display_ct_window(ct_volume, z_idx=z_int, x_idx=x_int, y_idx=y_int, window_size=100)

            fig, ax = plt.subplots(figsize=(8, 8))

            vmin, vmax = -1030, 1000 

            im = ax.imshow(
                ct_slice,
                cmap='viridis',     # or 'jet', 'viridis', etc.
                origin='lower',
                vmin=vmin,
                vmax=vmax
            )
            # Red dot at sampling point
            ax.scatter(x_int, y_int, c='red', s=10, alpha=0.5)
            ax.set_title(f"CT slice z_idx={z_int}, HU value reading={hu_value_proj_pt}")
            ax.set_xlabel("x (voxel)")
            ax.set_ylabel("y (voxel)")

            # Add colorbar showing HU scale
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("HU (Hounsfield Units)")

            out_slice_path = f"ct_slice_HU_colormap_{img_name[7:-4]}_{i:03d}.png"
            plt.tight_layout()
            plt.savefig(out_slice_path, dpi=150)
            plt.close(fig)
            print(f"Saved {out_slice_path}")
        # -----------------------------------------------------------

    #print(hu_vals)

    plt.figure()
    plt.plot(depths, hu_vals)
    plt.axvline(x=midpoint_depth, color='red', linestyle='--', linewidth=2,
                label=f'Midpoint Depth: {midpoint_depth:.2f}')
    plt.xlabel('Depth (camera z)')
    plt.ylabel('HU Value')
    plt.title(f"hu_values_vs_depth_{img_name[7:-4]}.png")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"hu_values_vs_depth_{img_name[7:-4]}.png")
    plt.close()



    fig, ax = plt.subplots(figsize=(8, 8))

    # Show the HU image
    im = ax.imshow(hu_image, cmap='viridis', origin='lower')

    # Add a colorbar that shows HU values
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("HU")

    # Turn off axes if you want cleaner export
    ax.axis('off')

    # Save the figure instead of using imsave
    outname = f"hu_values_{img_name[7:-4]}.png"
    plt.savefig(outname, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {outname}")
    
    # ==================== NEW SANITY CHECK VISUALIZATIONS ====================
    
    # Single slice overlay at middle of CT volume
    # mid_slice = ct_volume.shape[0] // 2
    # mid_slice_sagittal = int(ct_volume.shape[1]*0.75)
    # mid_slice_coronal = ct_volume.shape[2] // 2
    # overlay_on_ct_slice(
    #     ct_volume, 
    #     mesh, 
    #     pts,
    #     slice_idx=mid_slice,
    #     ct_spacing=ct_spacing,
    #     axis='axial',
    #     output_file=f'ct_overlay_axial_{img_name[7:-4]}.png'
    # )

    # overlay_on_ct_slice(
    #     ct_volume, 
    #     mesh, 
    #     pts,
    #     slice_idx=mid_slice_sagittal,
    #     ct_spacing=ct_spacing,
    #     axis='sagittal',
    #     output_file=f'ct_overlay_sagittal_{img_name[7:-4]}.png'
    # )

    # overlay_on_ct_slice(
    #     ct_volume, 
    #     mesh, 
    #     pts,
    #     slice_idx=mid_slice_coronal,
    #     ct_spacing=ct_spacing,
    #     axis='coronal',
    #     output_file=f'ct_overlay_coronal_{img_name[7:-4]}.png'
    # )
    
    print(f"Sanity check visualizations saved for {img_name}")