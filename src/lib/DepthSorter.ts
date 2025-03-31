export class DepthSorter {
  // Static arrays that will be reused between calls
  private static BUCKET_COUNT = 256 * 256;
  private static counts: Uint32Array | null = null;
  private static starts: Uint32Array | null = null;
  private static lastCameraPosition: [number, number, number] | null = null;
  
  /**
   * Sorts voxels based on their distance from the camera position
   * @param positions Float32Array of voxel positions [x1,y1,z1,x2,y2,z2,...]
   * @param cameraPosition [x,y,z] position of the camera
   * @param outIndices Optional pre-allocated array to store the result
   * @returns Uint32Array of sorted indices
   */
  static sortVoxels(
    positions: Float32Array,
    cameraPosition: [number, number, number],
    outIndices?: Uint32Array
  ): Uint32Array {
    const numVoxels = positions.length / 3;
    
    // Early exit if camera hasn't moved significantly and number of voxels hasn't changed
    if (this.lastCameraPosition && outIndices && outIndices.length === numVoxels) {
      const [lastX, lastY, lastZ] = this.lastCameraPosition;
      const [x, y, z] = cameraPosition;
      const dot = lastX * x + lastY * y + lastZ * z;
      const lenSq1 = lastX * lastX + lastY * lastY + lastZ * lastZ;
      const lenSq2 = x * x + y * y + z * z;
      
      // If camera direction hasn't changed significantly, reuse last sort
      if (Math.abs(dot / Math.sqrt(lenSq1 * lenSq2) - 1) < 0.01) {
        return outIndices;
      }
    }
    
    // Initialize or reuse output array
    const indices = (outIndices && outIndices.length === numVoxels) 
      ? outIndices 
      : new Uint32Array(numVoxels);
    
    // Initialize static arrays if needed
    if (!this.counts || !this.starts || this.counts.length !== this.BUCKET_COUNT) {
      this.counts = new Uint32Array(this.BUCKET_COUNT);
      this.starts = new Uint32Array(this.BUCKET_COUNT);
    } else {
      // Clear counts array
      this.counts.fill(0);
    }
    
    // Find min/max distances
    let minDist = Infinity;
    let maxDist = -Infinity;
    
    // Calculate squared distances and find min/max
    const distances = new Float32Array(numVoxels);
    
    for (let i = 0; i < numVoxels; i++) {
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      
      const dx = x - cameraPosition[0];
      const dy = y - cameraPosition[1];
      const dz = z - cameraPosition[2];
      const distanceSquared = dx * dx + dy * dy + dz * dz;

      distances[i] = distanceSquared;
      
      if (distanceSquared < minDist) minDist = distanceSquared;
      if (distanceSquared > maxDist) maxDist = distanceSquared;
    }
    
    // Use logarithmic scale to give more buckets to nearby objects
    const logMin = Math.log(minDist);
    const logMax = Math.log(maxDist);
    const logRange = logMax - logMin;
    
    // Scale factor for logarithmic mapping
    const distInv = (this.BUCKET_COUNT - 1) / (logRange || 1);
    
    // Count occurrences of each bucket using log scale (first pass)
    for (let i = 0; i < numVoxels; i++) {
      // Use logarithmic mapping for better distribution
      const logDist = Math.log(distances[i]);
      const bucketIndex = Math.min(
        this.BUCKET_COUNT - 1,
        ((logDist - logMin) * distInv) | 0
      );
      this.counts![bucketIndex]++;
    }
    
    // Calculate starting positions (prefix sum)
    this.starts![0] = 0;
    for (let i = 1; i < this.BUCKET_COUNT; i++) {
      this.starts![i] = this.starts![i - 1] + this.counts![i - 1];
    }
    
    // Copy starts to a working array that will be modified during distribution
    const startsCopy = new Uint32Array(this.starts!);
    
    // Distribute indices to final positions using log scale (second pass)
    for (let i = 0; i < numVoxels; i++) {
      const logDist = Math.log(distances[i]);
      const bucketIndex = Math.min(
        this.BUCKET_COUNT - 1,
        ((logDist - logMin) * distInv) | 0
      );
      // Place voxel index in the correct position
      indices[startsCopy[bucketIndex]++] = i;
    }
    
    // Store camera position for next call
    this.lastCameraPosition = [...cameraPosition];
    
    // Since we want back-to-front (farther objects first), we need to reverse the result
    this.reverseIndices(indices);
    
    return indices;
  }
  
  /**
   * Reverses an array in-place
   */
  private static reverseIndices(arr: Uint32Array): void {
    let left = 0;
    let right = arr.length - 1;
    
    while (left < right) {
      const temp = arr[left];
      arr[left] = arr[right];
      arr[right] = temp;
      left++;
      right--;
    }
  }
} 