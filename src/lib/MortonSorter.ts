/**
 * 
 * NOTE: This file is complete AI SLOP right now. It doesn't work and doesn't follow the paper.
 * 
 * MortonSorter class for sorting voxels in correct back-to-front order
 * Based on direction-dependent Morton order sorting as described in the paper
 */
export class MortonSorter {
  // Morton order bit masks for interleaving
  private static readonly MASKS = [
    0x9249249249249249n, // bits: 0 00 1 00 1 00 1 00 1 ...
    0x2492492492492492n, // bits: 0 01 0 01 0 01 0 01 0 ...
    0x4924924924924924n  // bits: 0 10 0 10 0 10 0 10 0 ...
  ];

  // Maps ray sign bits to the appropriate permutation mapping
  // The 8 permutations of Morton order based on ray direction
  private static readonly PERMUTATION_MAPS = [
    [0, 1, 2, 3, 4, 5, 6, 7], // [+x, +y, +z] -> [0b000]
    [1, 0, 3, 2, 5, 4, 7, 6], // [+x, +y, -z] -> [0b001]
    [2, 3, 0, 1, 6, 7, 4, 5], // [+x, -y, +z] -> [0b010]
    [3, 2, 1, 0, 7, 6, 5, 4], // [+x, -y, -z] -> [0b011]
    [4, 5, 6, 7, 0, 1, 2, 3], // [-x, +y, +z] -> [0b100]
    [5, 4, 7, 6, 1, 0, 3, 2], // [-x, +y, -z] -> [0b101]
    [6, 7, 4, 5, 2, 3, 0, 1], // [-x, -y, +z] -> [0b110]
    [7, 6, 5, 4, 3, 2, 1, 0]  // [-x, -y, -z] -> [0b111]
  ];

  /**
   * Compute the ray sign bits (3 bits) from the ray direction
   * @param rayDirection The normalized ray direction vector [x, y, z]
   * @returns A number 0-7 representing the ray sign bits
   */
  public static getRaySignBits(rayDirection: [number, number, number]): number {
    const [x, y, z] = rayDirection;
    let signBits = 0;
    
    // Create a 3-bit code where each bit represents the sign of x, y, z
    // Negative: 1, Positive: 0
    if (x < 0) signBits |= 0b100;
    if (y < 0) signBits |= 0b010;
    if (z < 0) signBits |= 0b001;
    
    return signBits;
  }

  /**
   * Extracts a morton code from an octpath value and level
   * @param octpath The octpath value storing the path through the octree
   * @param octlevel The octree level (depth)
   * @returns The morton code as a BigInt
   */
  public static octpathToMorton(octpath: number, octlevel: number): bigint {
    let result = 0n;
    
    // Each level is represented by 3 bits in the octpath
    // We extract these bits from the octpath for each level
    for (let level = 0; level < octlevel; level++) {
      // Shift bits to get the 3 bits for this level (from most to least significant)
      // i.e., first level is top 3 bits, second level is next 3 bits, etc.
      const shift = 3 * (octlevel - 1 - level);
      const levelBits = (octpath >> shift) & 0b111;
      
      // Add these bits to our morton code
      result |= BigInt(levelBits) << BigInt(level * 3);
    }
    
    return result;
  }

  /**
   * Computes a Morton code from 3D coordinates
   * @param x X coordinate (normalized 0-1)
   * @param y Y coordinate (normalized 0-1)
   * @param z Z coordinate (normalized 0-1)
   * @param bits Number of bits per dimension
   * @returns A Morton code as a BigInt
   */
  public static mortonCode(x: number, y: number, z: number, bits: number = 16): bigint {
    // Normalize coordinates to integers (0 to 2^bits-1)
    const scale = (1 << bits) - 1;
    const ix = Math.min(Math.max(Math.floor(x * scale), 0), scale);
    const iy = Math.min(Math.max(Math.floor(y * scale), 0), scale);
    const iz = Math.min(Math.max(Math.floor(z * scale), 0), scale);
    
    // Convert to BigInt for operations
    let bx = BigInt(ix);
    let by = BigInt(iy);
    let bz = BigInt(iz);
    
    // Spread the bits
    bx = this.spreadBits(bx, bits);
    by = this.spreadBits(by, bits);
    bz = this.spreadBits(bz, bits);
    
    // Interleave bits (x in bit positions 2, 5, 8...)
    // y in positions 1, 4, 7..., and z in 0, 3, 6...
    return (bx << 2n) | (by << 1n) | bz;
  }
  
  /**
   * Spreads the bits of a value apart to make room for interleaving
   * @param val The value to spread
   * @param bits Number of bits in the original value
   * @returns The value with bits spread apart
   */
  private static spreadBits(val: bigint, bits: number): bigint {
    // For a 21-bit number, we need to process in 3 groups of 7 bits
    const groupBits = 7;
    const groups = Math.ceil(bits / groupBits);
    
    let result = 0n;
    for (let i = 0; i < groups; i++) {
      // Extract a group of bits
      const mask = (1n << BigInt(groupBits)) - 1n;
      const group = (val >> BigInt(i * groupBits)) & mask;
      
      // Spread the bits - putting each bit 3 positions apart
      let spreadGroup = 0n;
      for (let b = 0; b < groupBits; b++) {
        if ((group & (1n << BigInt(b))) !== 0n) {
          spreadGroup |= (1n << BigInt(b * 3));
        }
      }
      
      // Add to result at the appropriate position
      result |= (spreadGroup << BigInt(i * groupBits * 3));
    }
    
    return result;
  }

  /**
   * Apply the direction-dependent permutation to a Morton code
   * @param mortonCode The original Morton code
   * @param raySignBits The ray sign bits (0-7)
   * @param maxLevel The maximum octree level
   * @returns The direction-adjusted Morton code
   */
  public static applyDirectionPermutation(
    mortonCode: bigint, 
    raySignBits: number, 
    maxLevel: number = 16
  ): bigint {
    // Get the permutation mapping for these ray sign bits
    const permutation = this.PERMUTATION_MAPS[raySignBits];
    
    // For each level, extract the 3-bit code and remap it
    let result = 0n;
    for (let level = 0; level < maxLevel; level++) {
      // Get the 3 bits at this level (each level is 3 bits in the Morton code)
      const shift = BigInt(level * 3);
      const levelBits = Number((mortonCode >> shift) & 0b111n);
      
      // Apply the permutation
      const remappedBits = permutation[levelBits];
      
      // Add back to the result
      result |= BigInt(remappedBits) << shift;
    }
    
    return result;
  }

  /**
   * Sort voxels based on direction-dependent Morton codes using octree information
   * @param positions Array of voxel positions [x1,y1,z1, x2,y2,z2, ...]
   * @param cameraPosition Camera position [x,y,z]
   * @param cameraTarget Camera target [x,y,z]
   * @param octlevels Optional octree levels per voxel
   * @param octpaths Optional octree paths per voxel
   * @returns Indices array with sorted order
   */
  public static sortVoxels(
    positions: Float32Array,
    cameraPosition: [number, number, number],
    cameraTarget: [number, number, number],
    octlevels?: Uint8Array,
    octpaths?: Uint32Array
  ): Uint32Array {
    // Create a ray direction from camera to target
    const rayDirection: [number, number, number] = [
      cameraTarget[0] - cameraPosition[0],
      cameraTarget[1] - cameraPosition[1],
      cameraTarget[2] - cameraPosition[2]
    ];
    
    // Normalize the ray direction
    const length = Math.sqrt(
      rayDirection[0] * rayDirection[0] +
      rayDirection[1] * rayDirection[1] +
      rayDirection[2] * rayDirection[2]
    );
    
    rayDirection[0] /= length;
    rayDirection[1] /= length;
    rayDirection[2] /= length;
    
    // Get ray sign bits
    const raySignBits = this.getRaySignBits(rayDirection);
    
    // Find min/max for normalization (only needed if we don't have octpath info)
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    
    if (!octpaths || !octlevels) {
      for (let i = 0; i < positions.length; i += 3) {
        minX = Math.min(minX, positions[i]);
        minY = Math.min(minY, positions[i+1]);
        minZ = Math.min(minZ, positions[i+2]);
        
        maxX = Math.max(maxX, positions[i]);
        maxY = Math.max(maxY, positions[i+1]);
        maxZ = Math.max(maxZ, positions[i+2]);
      }
    }
    
    // Range for normalization
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const rangeZ = maxZ - minZ || 1;
    
    // Create an array of indices and their Morton codes
    const voxelCount = positions.length / 3;
    const indexMortonPairs: { index: number; mortonCode: bigint }[] = [];
    
    for (let i = 0; i < voxelCount; i++) {
      let mortonCode: bigint;
      
      // Use octpath if available
      if (octpaths && octlevels && i < octpaths.length && i < octlevels.length) {
        mortonCode = this.octpathToMorton(octpaths[i], octlevels[i]);
      } else {
        // Fall back to computing from position
        const x = (positions[i*3] - minX) / rangeX;
        const y = (positions[i*3+1] - minY) / rangeY;
        const z = (positions[i*3+2] - minZ) / rangeZ;
        
        mortonCode = this.mortonCode(x, y, z);
      }
      
      // Apply direction-dependent permutation
      mortonCode = this.applyDirectionPermutation(mortonCode, raySignBits);
      
      indexMortonPairs.push({ index: i, mortonCode });
    }
    
    // Sort by Morton code
    indexMortonPairs.sort((a, b) => {
      // Compare with the correct ordering based on ray direction
      // For back-to-front rendering, we want furthest first
      if (a.mortonCode < b.mortonCode) return -1;
      if (a.mortonCode > b.mortonCode) return 1;
      return 0;
    });
    
    // Return sorted indices
    const sortedIndices = new Uint32Array(voxelCount);
    for (let i = 0; i < voxelCount; i++) {
      sortedIndices[i] = indexMortonPairs[i].index;
    }
    
    return sortedIndices;
  }
}