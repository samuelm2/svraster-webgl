// Import DepthSorter instead of MortonSorter
import { DepthSorter } from '../lib/DepthSorter';

// Define message types
interface SortRequest {
  type: 'sort';
  positions: Float32Array;
  cameraPosition: [number, number, number];
  cameraTarget: [number, number, number];
  sceneTransformMatrix: Float32Array; // Add transform matrix
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
    
    // Transform positions before sorting
    const transformedPositions = new Float32Array(data.positions.length);
    
    // Apply the scene transformation to each position
    for (let i = 0; i < data.positions.length / 3; i++) {
      const x = data.positions[i * 3];
      const y = data.positions[i * 3 + 1];
      const z = data.positions[i * 3 + 2];
      
      // Apply transformation matrix
      transformedPositions[i * 3]     = x * data.sceneTransformMatrix[0] + 
                                         y * data.sceneTransformMatrix[4] + 
                                         z * data.sceneTransformMatrix[8] + 
                                         data.sceneTransformMatrix[12];
      transformedPositions[i * 3 + 1] = x * data.sceneTransformMatrix[1] + 
                                         y * data.sceneTransformMatrix[5] + 
                                         z * data.sceneTransformMatrix[9] + 
                                         data.sceneTransformMatrix[13];
      transformedPositions[i * 3 + 2] = x * data.sceneTransformMatrix[2] + 
                                         y * data.sceneTransformMatrix[6] + 
                                         z * data.sceneTransformMatrix[10] + 
                                         data.sceneTransformMatrix[14];
    }
    
    // Use DepthSorter to sort the transformed voxels by distance from camera
    const indices = DepthSorter.sortVoxels(
      transformedPositions,
      data.cameraPosition
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