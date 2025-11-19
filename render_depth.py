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


def scatter3d_with_mesh_save(points, mesh, outfile="pc_world_with_mesh.png",
                             pt_size=0.5, mesh_alpha=0.15, max_faces=50000,
                             elev=20, azim=-60, dpi=200):
    """
    Save a 3D view of points overlaid on the mesh.
    - points: (N,3) array
    - mesh:   open3d.geometry.TriangleMesh
    - max_faces: randomly subsample triangles for speed if mesh is large
    """
    pts = np.asarray(points)

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)

    # build triangle collection
    tris = V[F]  # (M,3,3)
    poly = Poly3DCollection(tris, facecolor=(0.7, 0.7, 0.7, mesh_alpha),
                            edgecolor=(0.5, 0.5, 0.5, 0.2), linewidths=0.1)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # mesh first, then points on top
    ax.add_collection3d(poly)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=pt_size, depthshade=False, c='r')

    # equal aspect across both mesh + points
    mins = np.minimum(pts.min(axis=0), V.min(axis=0))
    maxs = np.maximum(pts.max(axis=0), V.max(axis=0))
    centers = (mins + maxs) / 2.0
    span = (maxs - mins).max() / 2.0 + 1e-9
    ax.set_xlim(centers[0]-span, centers[0]+span)
    ax.set_ylim(centers[1]-span, centers[1]+span)
    ax.set_zlim(centers[2]-span, centers[2]+span)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout()
    plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


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
        slice_pos = slice_idx * ct_spacing[2] # convert pixel to mm 
        coord_idx = 2  # Z coordinate
    elif axis == 'sagittal':
        ct_slice = ct_volume[:, :, slice_idx]
        slice_pos = slice_idx * ct_spacing[0]
        coord_idx = 0  # X coordinate
    elif axis == 'coronal':
        ct_slice = ct_volume[:, slice_idx, :]
        slice_pos = slice_idx * ct_spacing[1] 
        coord_idx = 1  # Y coordinate
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display CT slice
    ax.imshow(ct_slice, cmap='gray', origin='lower', 
              extent=[0, ct_slice.shape[1]*ct_spacing[0], 
                     0, ct_slice.shape[0]*ct_spacing[1] if axis=='axial' else ct_slice.shape[0]*ct_spacing[2]])
    
    # Project mesh vertices onto slice plane
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Find triangles that intersect the slice
    tolerance = ct_spacing[coord_idx] * 2
    for tri in triangles:
        tri_verts = vertices[tri]
        if axis == 'axial':
            if np.any(np.abs(tri_verts[:, 2] + slice_pos) < tolerance): # this makes sense since tri_verts is negative, and slice_pos is positive
                coords = tri_verts[:, [0, 1]] # get x and y of the triangle
                ax.plot(-coords[:, 0], -coords[:, 1], 'b-', linewidth=0.5, alpha=0.6)
        elif axis == 'sagittal':
            if np.any(np.abs(tri_verts[:, 0] + slice_pos) < tolerance):
                coords = tri_verts[:, [1, 2]]
                ax.plot(-coords[:, 0], -coords[:, 1], 'b-', linewidth=0.5, alpha=0.6)
        elif axis == 'coronal':
            if np.any(np.abs(tri_verts[:, 1] + slice_pos) < tolerance):
                coords = tri_verts[:, [0, 2]]
                ax.plot(-coords[:, 0], -coords[:, 1], 'b-', linewidth=0.5, alpha=0.6)
    
    # Project point cloud onto slice
    if axis == 'axial':
        pc_mask = np.abs(point_cloud[:, 2] + slice_pos) < tolerance # makes sense since point_cloud is negative, and slice_pos is positive
        pc_proj = point_cloud[pc_mask][:, [0, 1]]
    elif axis == 'sagittal':
        pc_mask = np.abs(point_cloud[:, 0] + slice_pos) < tolerance
        pc_proj = point_cloud[pc_mask][:, [1, 2]]
    elif axis == 'coronal':
        pc_mask = np.abs(point_cloud[:, 1] + slice_pos) < tolerance
        pc_proj = point_cloud[pc_mask][:, [0, 2]]
    
    if len(pc_proj) > 0:
        ax.scatter(-pc_proj[:, 0], -pc_proj[:, 1], c='red', s=2, alpha=0.8, label='Point Cloud')
    
    ax.set_title(f'{axis.capitalize()} Slice {slice_idx} - Mesh (blue) & Points (red)')
    ax.set_xlabel('X (mm)' if axis != 'sagittal' else 'Y (mm)')
    ax.set_ylabel('Y (mm)' if axis == 'axial' else 'Z (mm)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_file}")

    


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

render = o3d.visualization.rendering.OffscreenRenderer(width, height)
render.scene.set_background([0, 0, 0, 0])
render.scene.add_geometry("segmentation", mesh, o3d.visualization.rendering.MaterialRecord())

# Load mask
mask_img = Image.open(patient01_datapath + "undistorted_mask.bmp").convert("L")
mask = np.array(mask_img)
binary_mask = (mask > 0).astype(np.uint8)

# load CT (SimpleITK returns physical LPS, mm; intensities already in HU)
reader = sitk.ImageSeriesReader()
files = reader.GetGDCMSeriesFileNames(patient01_datapath + "/dicom")
reader.SetFileNames(files)
ct_img = reader.Execute()
ct_volume = sitk.GetArrayFromImage(ct_img) 


origin    = np.array(ct_img.GetOrigin(), dtype=float)       # (ox,oy,oz)
spacing   = np.array(ct_img.GetSpacing(), dtype=float)      # (sx,sy,sz)
direction = np.array(ct_img.GetDirection(), dtype=float).reshape(3,3)

print(f"CT shape (Z,Y,X): {ct_volume.shape}")
print("CT origin (mm):", origin, "spacing (mm):", spacing)
print("CT direction:\n", direction)

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(ct_volume[ct_volume.shape[0]//4, :, :], cmap='gray')
# axes[1].imshow(ct_volume[ct_volume.shape[0]//2, :, :], cmap='gray')
# axes[2].imshow(ct_volume[ct_volume.shape[0]//8, :, :], cmap='gray')
# plt.savefig("ct_slices.png", dpi=150, bbox_inches='tight')
# plt.close()

# Define CT spacing
slice_thickness = 0.833  
pixel_spacing = 0.415    
ct_spacing = np.array([pixel_spacing, pixel_spacing, slice_thickness])

for img_name in os.listdir(patient01_datapath + "correspondences/"):
    
    extrinsic_matrix_w_to_c = np.array(p01_json_data["poses"][img_name[7:]])  
    
    # For rendering, setup_camera expects C→W transform
    extrinsic_matrix_c_to_w = np.linalg.inv(extrinsic_matrix_w_to_c)
    render.setup_camera(intrinsics, extrinsic_matrix_c_to_w)

    depth_image = np.asarray(render.render_to_depth_image(z_in_view_space = True))

    # Apply mask
    depth_image_masked = np.where(binary_mask, depth_image, 0)
    
    # Visualization

    # circle = (depth_image_masked > 0)
    # dmin, dmax = depth_image[circle].min(), depth_image[circle].max()
    # normalized_depth = np.zeros_like(depth_image)
    # normalized_depth[circle] = (depth_image_masked[circle] - dmin) / (dmax - dmin)
    # plt.imsave(f"depth_image_{img_name[7:-4]}.png", normalized_depth, cmap="gray")

    depth_img_obj = o3d.geometry.Image(depth_image.astype(np.float32))
    
    pc_world = o3d.geometry.PointCloud.create_from_depth_image(
        depth_img_obj, intrinsics, extrinsic=extrinsic_matrix_c_to_w, depth_scale=1.0
    )
    
    print(f"\nProcessing {img_name}")
    pts = np.asarray(pc_world.points)
    
    # scatter3d_with_mesh_save(
    #     pts, mesh,
    #     outfile=f"pc_world_with_mesh_{img_name[7:-4]}.png",
    #     pt_size=0.5,     
    #     mesh_alpha=0.18, 
    #     max_faces=40000   
    # )

    voxel_coords_continuous = pts / ct_spacing
    
    hu_image = np.zeros((height, width), dtype=np.float32)

    hu_values = map_coordinates(
        ct_volume,
        np.array([-voxel_coords_continuous[:, 2], # Negative sign to end up with positive indices
                -voxel_coords_continuous[:, 0], 
                -voxel_coords_continuous[:, 1]]),
        order=1, 
        mode='constant',
        cval=np.nan 
    )
    #print(np.unique(hu_values))
    hu_image.ravel()[:] = hu_values
    #plt.imsave(f"hu_values_{img_name[7:-4]}.png", hu_image)
    
    # ==================== NEW SANITY CHECK VISUALIZATIONS ====================
    
    # Single slice overlay at middle of CT volume
    mid_slice = ct_volume.shape[0] // 2
    overlay_on_ct_slice(
        ct_volume, 
        mesh, 
        pts,
        slice_idx=mid_slice,
        ct_spacing=ct_spacing,
        axis='axial',
        output_file=f'ct_overlay_axial_{img_name[7:-4]}.png'
    )

    overlay_on_ct_slice(
        ct_volume, 
        mesh, 
        pts,
        slice_idx=mid_slice,
        ct_spacing=ct_spacing,
        axis='sagittal',
        output_file=f'ct_overlay_sagittal_{img_name[7:-4]}.png'
    )

    overlay_on_ct_slice(
        ct_volume, 
        mesh, 
        pts,
        slice_idx=mid_slice,
        ct_spacing=ct_spacing,
        axis='coronal',
        output_file=f'ct_overlay_sagittal_{img_name[7:-4]}.png'
    )
    
    print(f"Sanity check visualizations saved for {img_name}")