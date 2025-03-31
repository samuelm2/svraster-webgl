// Import DepthSorter instead of MortonSorter
import { DepthSorter } from '../lib/DepthSorter';

// Define message types
interface SortRequest {
  type: 'sort';
  positions: Float32Array;
  cameraPosition: [number, number, number];
  cameraTarget: [number, number, number];
  octlevels?: Uint8Array;   // Add octree levels information
  octpaths?: Uint32Array;   // Add octree paths information
}

interface SortResponse {
  type: 'sorted';
  indices: Uint32Array;
  sortTime: number;
}

// Web Worker context
const ctx: Worker = self as any;

// Handle incoming messages
ctx.addEventListener('message', (event: MessageEvent) => {
  const data = event.data as SortRequest;
  
  if (data.type === 'sort') {
    console.log(`SortWorker: Starting depth-based sort for ${data.positions.length / 3} voxels`);
    
    const startTime = performance.now();
    
    // Use DepthSorter to sort the voxels by distance from camera
    const indices = DepthSorter.sortVoxels(
      data.positions,
      data.cameraPosition
      // No longer need camera target for depth-based sorting
      // Octree data may still be useful for optimization but is optional
    );
    
    const sortTime = performance.now() - startTime;
    console.log(`SortWorker: Depth-based sort complete in ${sortTime.toFixed(2)}ms, returning ${indices.length} indices`);
    
    // Send back sorted indices
    const response: SortResponse = {
      type: 'sorted',
      indices,
      sortTime
    };
    
    ctx.postMessage(response, [response.indices.buffer]);
  }
});

// Let the main thread know we're ready
console.log('SortWorker: Initialized and ready');
ctx.postMessage({ type: 'ready' });