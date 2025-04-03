export class DistanceSorter {
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
    
    // Calculate and store log distances directly instead of squared distances
    const logDistances = new Float32Array(numVoxels);
    
    // Compute squared distances and their logarithms
    for (let i = 0; i < numVoxels; i++) {
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      
      const dx = x - cameraPosition[0];
      const dy = y - cameraPosition[1];
      const dz = z - cameraPosition[2];
      const distanceSquared = dx * dx + dy * dy + dz * dz;

      // Store the log of the distance directly
      const logDist = Math.log(distanceSquared);
      logDistances[i] = logDist;
      
      if (logDist < minDist) minDist = logDist;
      if (logDist > maxDist) maxDist = logDist;
    }
    
    const logRange = maxDist - minDist;
    
    // Scale factor for logarithmic mapping
    const distInv = (this.BUCKET_COUNT - 1) / (logRange || 1);
    
    // Count occurrences of each bucket using log scale (first pass)
    for (let i = 0; i < numVoxels; i++) {
      // Reuse the pre-calculated log distance
      const bucketIndex = Math.min(
        this.BUCKET_COUNT - 1,
        ((logDistances[i] - minDist) * distInv) | 0
      );
      this.counts![bucketIndex]++;
    }
    
    // Calculate bucket positions with farthest buckets first
    let position = 0;
    const bucketStarts = new Uint32Array(this.BUCKET_COUNT);
    for (let i = this.BUCKET_COUNT - 1; i >= 0; i--) {
      bucketStarts[i] = position;
      position += this.counts![i];
    }
    
    // Reset counts for the second pass
    this.counts!.fill(0);
    
    // Distribute indices in back-to-front order
    for (let i = 0; i < numVoxels; i++) {
      // Reuse the pre-calculated log distance again
      const bucketIndex = Math.min(
        this.BUCKET_COUNT - 1,
        ((logDistances[i] - minDist) * distInv) | 0
      );
      
      // Calculate the position for this voxel
      const pos = bucketStarts[bucketIndex] + this.counts![bucketIndex];
      indices[pos] = i;
      this.counts![bucketIndex]++;
    }
    
    // Store camera position for next call
    this.lastCameraPosition = [...cameraPosition];
    
    return indices;
  }
} 