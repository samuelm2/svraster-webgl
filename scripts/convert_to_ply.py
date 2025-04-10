import os
import torch
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from functools import partial

# Define constants
MAX_NUM_LEVELS = 16

def get_voxel_size(scene_extent, octlevel):
    '''The voxel size at the given levels.'''
    return np.ldexp(scene_extent, -octlevel)

# Process a single octree path
def process_single_path(params):
    """Process a single octree path to coordinates"""
    idx, path, lv = params
    
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
    
    return (i, j, k)

def octpath_to_coords(octpath, octlevel, desc="Processing octree paths"):
    """Convert octree path to coordinates using parallel processing"""
    P = len(octpath)
    
    # Prepare parameters for each path
    params = [(idx, int(octpath[idx]), int(octlevel[idx])) for idx in range(P)]
    
    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    
    # Process in parallel
    with Pool(processes=num_cores) as pool:
        # Use imap to process in chunks for better progress bar
        results = list(tqdm(pool.imap(process_single_path, params, chunksize=max(1, P//num_cores)), 
                            total=P, desc=desc))
    
    # Convert results to numpy array
    coords = np.array(results, dtype=np.int64)
    return coords

def octpath_decoding(octpath, octlevel, scene_center, scene_extent):
    '''Compute world-space voxel center positions using numpy'''
    octpath = octpath.reshape(-1)
    octlevel = octlevel.reshape(-1)

    scene_min_xyz = scene_center - 0.5 * scene_extent
    vox_size = get_voxel_size(scene_extent, octlevel.reshape(-1, 1))
    vox_ijk = octpath_to_coords(octpath, octlevel, desc="Computing voxel centers")
    vox_center = scene_min_xyz + (vox_ijk + 0.5) * vox_size
    
    return vox_center

# Process a batch of grid points
def process_grid_batch(batch_idx, vox_ijk, lv2max, subtree_shift):
    """Process a batch of grid points"""
    i = batch_idx
    shift = lv2max[i, 0]
    base = np.expand_dims(vox_ijk[i] << shift, axis=0)
    return base + (subtree_shift * (1 << shift))

def link_grid_pts(octpath, octlevel):
    '''Build link between voxel and grid_pts using numpy unique'''
    # Binary encoding of the eight octants
    subtree_shift = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ], dtype=np.int64)
    
    # Get all voxel coordinates
    vox_ijk = octpath_to_coords(octpath, octlevel, desc="Generating voxel coordinates")
    
    # Calculate shift for each voxel based on octlevel
    lv2max = (MAX_NUM_LEVELS - octlevel).reshape(-1, 1)
    
    # Create grid points array
    N = len(octpath)
    
    num_cores = multiprocessing.cpu_count()
    # Prepare the worker function with fixed arguments
    process_func = partial(process_grid_batch, vox_ijk=vox_ijk, lv2max=lv2max, subtree_shift=subtree_shift)
    
    #Process in parallel
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(process_func, range(N), chunksize=max(1, N//num_cores)), 
                            total=N, desc="Generating grid point coordinates"))
    
    gridpts = np.array(results)
    
    # print("Processing grid points...")
    # gridpts = np.zeros((N, 8, 3), dtype=np.int64)
    # for i in tqdm(range(N), desc="Building grid points"):
    #     shift = lv2max[i, 0]
    #     base = np.expand_dims(vox_ijk[i] << shift, axis=0)
    #     gridpts[i] = base + (subtree_shift * (1 << shift))
    
    # Reshape and find unique grid points
    gridpts_flat = gridpts.reshape(-1, 3)
    print("Mapping grid points to voxels ...")
    
    unique_coords, vox_key = np.unique(gridpts_flat, axis=0, return_inverse=True)
    vox_key = vox_key.reshape(N, 8)
    
    print(f"Total unique grid points: {len(unique_coords)}")
    return vox_key

# Process grid values for a specific corner and range of voxels
def process_grid_values_batch(params):
    """Process a batch of grid values"""
    i, start_idx, end_idx, vox_key, geo_grid_pts_len = params
    
    result = np.zeros(end_idx - start_idx, dtype=np.float32)
    
    for idx, vox_idx in enumerate(range(start_idx, end_idx)):
        grid_idx = vox_key[vox_idx, i]
        if grid_idx < geo_grid_pts_len:
            result[idx] = geo_grid_pts[grid_idx]
        else:
            result[idx] = 0.0
            
    return (start_idx, result)

def convert_to_ply(input_path, output_path, use_cpu=False):
    """Convert a model.pt file to PLY format using numpy for memory efficiency"""
    print(f"Loading model from {input_path}...")
    
    # Load the state dictionary
    device = 'cpu'
    state_dict = torch.load(input_path, map_location=device)
    
    # Convert to numpy immediately
    if state_dict.get('quantized', False):
        raise NotImplementedError("Quantized models are not supported yet")
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
    vox_center = octpath_decoding(octpath, octlevel, scene_center, scene_extent)
    vox_key = link_grid_pts(octpath, octlevel)
    
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
    
    # Shared variable for multiprocessing
    global geo_grid_pts
    geo_grid_pts = _geo_grid_pts
    
    # Parallel grid filling for large datasets
    N = len(vox_center)
    for i in tqdm(range(8), desc="Writing grid points to PLY data structure"):
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
    
    args = parser.parse_args()
    
    if __name__ == '__main__':
        # This is important for multiprocessing on Windows
        multiprocessing.freeze_support()
    
    convert_to_ply(args.input_path, args.output_path, args.cpu)

if __name__ == "__main__":
    # This is important for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()