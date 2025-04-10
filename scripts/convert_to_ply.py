import os
import torch
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

# Define constants
MAX_NUM_LEVELS = 16

def get_voxel_size(scene_extent, octlevel):
    '''The voxel size at the given levels.'''
    return np.ldexp(scene_extent, -octlevel)

def octpath_to_coords_numpy(octpath, octlevel):
    """Convert octree path to coordinates using numpy"""
    P = len(octpath)
    coords = np.zeros((P, 3), dtype=np.int64)
    
    for idx in range(P):
        path = int(octpath[idx])
        lv = int(octlevel[idx])
        
        # Shift right to remove irrelevant bits based on octlevel
        path >>= (3 * (MAX_NUM_LEVELS - lv))
        
        i, j, k = 0, 0, 0
        # Process each 3-bit group
        for l in range(lv):
            bits = path & 0b111
            i |= ((bits & 0b100) >> 2) << l
            j |= ((bits & 0b010) >> 1) << l
            k |= (bits & 0b001) << l
            
            path >>= 3
        
        coords[idx, 0] = i
        coords[idx, 1] = j
        coords[idx, 2] = k
    
    return coords

def octpath_decoding_numpy(octpath, octlevel, scene_center, scene_extent):
    '''Compute world-space voxel center positions using numpy'''
    octpath = octpath.reshape(-1)
    octlevel = octlevel.reshape(-1)

    scene_min_xyz = scene_center - 0.5 * scene_extent
    vox_size = get_voxel_size(scene_extent, octlevel.reshape(-1, 1))
    vox_ijk = octpath_to_coords_numpy(octpath, octlevel)
    vox_center = scene_min_xyz + (vox_ijk + 0.5) * vox_size
    
    return vox_center

def build_grid_pts_link_numpy(octpath, octlevel):
    '''Build link between voxel and grid_pts using numpy unique function'''
    # Binary encoding of the eight octants
    subtree_shift = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ], dtype=np.int64)
    
    # Get all voxel coordinates
    vox_ijk = octpath_to_coords_numpy(octpath, octlevel)
    
    # Calculate shift for each voxel based on octlevel
    lv2max = (MAX_NUM_LEVELS - octlevel).reshape(-1, 1)
    
    # Create grid points array (similar to the original CUDA code)
    N = len(octpath)
    gridpts = np.zeros((N, 8, 3), dtype=np.int64)
    
    for i in range(N):
        shift = lv2max[i, 0]
        base = np.expand_dims(vox_ijk[i] << shift, axis=0)  # Shape: (1, 3)
        gridpts[i] = base + (subtree_shift * (1 << shift))
    
    # Reshape and find unique grid points
    gridpts_flat = gridpts.reshape(-1, 3)
    unique_coords, vox_key = np.unique(gridpts_flat, axis=0, return_inverse=True)
    
    vox_key = vox_key.reshape(N, 8)
    
    print(f"Total unique grid points: {len(unique_coords)}")
    return vox_key

def convert_to_ply(input_path, output_path, use_cpu=False):
    """Convert a model.pt file to PLY format using numpy for memory efficiency"""
    print(f"Loading model from {input_path}...")
    
    # Load the state dictionary
    device = 'cpu' if use_cpu else 'cuda'
    state_dict = torch.load(input_path, map_location=device)
    
    # Convert to numpy immediately
    if state_dict.get('quantized', False):
        print("Dequantizing data...")
        if isinstance(state_dict['_geo_grid_pts'], dict):
            _geo_grid_pts = state_dict['_geo_grid_pts']['codebook'][state_dict['_geo_grid_pts']['index'].long()]
        else:
            _geo_grid_pts = state_dict['_geo_grid_pts']
        
        if isinstance(state_dict['_sh0'], list):
            _sh0 = torch.cat([v['codebook'][v['index'].long()] for v in state_dict['_sh0']], dim=1)
        else:
            _sh0 = state_dict['_sh0']
        
        if isinstance(state_dict['_shs'], list):
            _shs = torch.cat([v['codebook'][v['index'].long()] for v in state_dict['_shs']], dim=1)
        else:
            _shs = state_dict['_shs']
    else:
        _geo_grid_pts = state_dict['_geo_grid_pts']
        _sh0 = state_dict['_sh0']
        _shs = state_dict['_shs']
    
    # Convert everything to numpy
    scene_center = state_dict['scene_center'].cpu().numpy()
    scene_extent = float(state_dict['scene_extent'].item())
    active_sh_degree = state_dict['active_sh_degree']
    
    octpath = state_dict['octpath'].cpu().numpy().squeeze()
    octlevel = state_dict['octlevel'].cpu().numpy().squeeze()
    
    _geo_grid_pts = _geo_grid_pts.cpu().numpy().squeeze()
    sh0_np = _sh0.cpu().numpy()
    shs_np = _shs.reshape(_shs.shape[0], -1).cpu().numpy()
    
    # Calculate derived values
    print("Computing voxel centers...")
    vox_center = octpath_decoding_numpy(octpath, octlevel, scene_center, scene_extent)
    
    print("Mapping grid points...")
    vox_key = build_grid_pts_link_numpy(octpath, octlevel)
    
    # Define attribute list
    attributes = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),          # positions
        ('octpath', 'u4'), ('octlevel', 'u1'),    # octree data
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')  # DC components
    ]
    
    # Add rest of SH coefficients
    for i in range(shs_np.shape[1]):
        attributes.append((f'f_rest_{i}', 'f4'))
    
    # Add grid point values for each corner
    for i in range(8):
        attributes.append((f'grid{i}_value', 'f4'))
    
    # Create elements array
    elements = np.empty(len(vox_center), dtype=attributes)
    
    # Fill position and octree data
    elements['x'] = vox_center[:, 0]
    elements['y'] = vox_center[:, 1]
    elements['z'] = vox_center[:, 2]
    elements['octpath'] = octpath.astype(np.uint32)
    elements['octlevel'] = octlevel.astype(np.uint8)
    
    # Fill DC components
    elements['f_dc_0'] = sh0_np[:, 0] 
    elements['f_dc_1'] = sh0_np[:, 1]
    elements['f_dc_2'] = sh0_np[:, 2]
    
    # Fill higher-order SH coefficients
    for i in range(shs_np.shape[1]):
        elements[f'f_rest_{i}'] = shs_np[:, i]
    
    # Fill grid point values for each voxel
    print("Filling grid point values...")
    for i in range(8):
        for vox_idx in range(len(vox_center)):
            grid_idx = vox_key[vox_idx, i]
            if grid_idx < len(_geo_grid_pts):
                elements[f'grid{i}_value'][vox_idx] = _geo_grid_pts[grid_idx]
            else:
                elements[f'grid{i}_value'][vox_idx] = 0.0
    
    # Add comments to the PLY file
    header_comments = []
    header_comments.append(f"scene_center {scene_center[0]} {scene_center[1]} {scene_center[2]}")
    header_comments.append(f"scene_extent {scene_extent}")
    header_comments.append(f"active_sh_degree {active_sh_degree}")
    
    # Write PLY file
    print(f"Writing PLY file to {output_path}...")
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el], comments=header_comments).write(output_path)
    print(f"PLY file saved to: {output_path} with {len(vox_center)} points")

def main():
    parser = argparse.ArgumentParser(description='Convert sparse voxel model PT file to PLY format')
    parser.add_argument('input_path', type=str, help='Path to the model.pt file')
    parser.add_argument('output_path', type=str, help='Path where to save the PLY file')
    parser.add_argument('--cpu', action='store_true', help='Use CPU for processing')
    
    args = parser.parse_args()
    
    convert_to_ply(args.input_path, args.output_path, args.cpu)

if __name__ == "__main__":
    main()