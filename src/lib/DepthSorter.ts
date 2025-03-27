export class DepthSorter {
  /**
   * Sorts voxels based on their distance from the camera position
   * @param positions Float32Array of voxel positions [x1,y1,z1,x2,y2,z2,...]
   * @param cameraPosition [x,y,z] position of the camera
   * @returns Uint32Array of sorted indices
   */
  static sortVoxels(
    positions: Float32Array,
    cameraPosition: [number, number, number]
  ): Uint32Array {
    const numVoxels = positions.length / 3;
    
    // Create array of [index, distance] pairs
    const indexDistancePairs: Array<[number, number]> = [];
    
    for (let i = 0; i < numVoxels; i++) {
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      
      // Calculate squared distance (faster than using Math.sqrt)
      const dx = x - cameraPosition[0];
      const dy = y - cameraPosition[1];
      const dz = z - cameraPosition[2];
      const distanceSquared = dx * dx + dy * dy + dz * dz;
      
      indexDistancePairs.push([i, distanceSquared]);
    }
    
    // Sort pairs by distance (descending for back-to-front rendering)
    // Change to ascending for front-to-back rendering if needed
    indexDistancePairs.sort((a, b) => b[1] - a[1]);
    
    // Extract just the indices
    const indices = new Uint32Array(numVoxels);
    for (let i = 0; i < numVoxels; i++) {
      indices[i] = indexDistancePairs[i][0];
    }
    
    return indices;
  }
} 