import './style.css';
import { Viewer } from './lib/Viewer';
import { Camera } from './lib/Camera';
import { LoadPLY } from './lib/LoadPLY';

document.addEventListener('DOMContentLoaded', () => {
  // Create the WebGL viewer
  const viewer = new Viewer('app');
  
  // Get direct access to the camera
  const camera = viewer.getCamera();
  
  // Configure the camera for a nice view
  camera.setPosition(0, 0, 5);   // Position away from the scene
  camera.setTarget(0, 0, 0);     // Look at the center
  

  // Handle window resize events
  window.addEventListener('resize', () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    viewer.resize(width, height);
  });
  
  // Add UI controls
  addControls(viewer, camera);
  
  // Add PLY upload UI
  addPLYUploadUI(viewer, camera);
});



/**
 * Add some basic UI controls for the demo
 */
function addControls(viewer: Viewer, camera: Camera) {
  // Create a simple control panel
  const controls = document.createElement('div');
  controls.style.position = 'absolute';
  controls.style.top = '10px';
  controls.style.left = '10px';
  controls.style.padding = '10px';
  controls.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
  controls.style.color = 'white';
  controls.style.fontFamily = 'sans-serif';
  controls.style.borderRadius = '5px';
  
  // Add simple camera control instructions
  controls.innerHTML = `
    <h3>Camera Controls</h3>
    <ul style="padding-left: 20px; margin: 5px 0;">
      <li>Drag mouse to orbit camera</li>
      <li>Scroll to zoom in/out</li>
    </ul>
  `;
  
  // Add the controls to the document
  document.body.appendChild(controls);
  
  // No event listeners needed for the descriptive controls
}

/**
 * Add PLY file upload UI
 */
function addPLYUploadUI(viewer: Viewer, camera: Camera) {
  // Create a file upload container
  const uploadContainer = document.createElement('div');
  uploadContainer.style.position = 'absolute';
  uploadContainer.style.top = '10px';
  uploadContainer.style.right = '10px';
  uploadContainer.style.padding = '10px';
  uploadContainer.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
  uploadContainer.style.color = 'white';
  uploadContainer.style.fontFamily = 'sans-serif';
  uploadContainer.style.borderRadius = '5px';
  
  uploadContainer.innerHTML = `
    <h3>Load PLY Model</h3>
    <input type="file" id="ply-upload" accept=".ply">
    <div id="ply-info" style="margin-top: 10px; font-size: 12px;"></div>
  `;
  
  document.body.appendChild(uploadContainer);
  
  // Handle file uploads
  const fileInput = document.getElementById('ply-upload') as HTMLInputElement;
  fileInput.addEventListener('change', async (event) => {
    const target = event.target as HTMLInputElement;
    const files = target.files;
    
    if (files && files.length > 0) {
      const file = files[0];
      const infoElement = document.getElementById('ply-info');
      
      try {
        infoElement!.textContent = 'Loading PLY file...';
        
        // Show loading progress
        const startTime = performance.now();
        
        // Load the PLY file
        const plyData = await LoadPLY.loadFromFile(file);
        
        const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
        console.log(`Loaded ${plyData.vertexCount} vertices in ${loadTime} seconds`);
        
        // Set scene parameters if available
        if (plyData.sceneCenter && plyData.sceneExtent) {
          viewer.setSceneParameters(plyData.sceneCenter, plyData.sceneExtent);
        }
        
        // Load the point cloud with octlevels for scaling
        viewer.loadPointCloud(
          plyData.vertices, 
          plyData.sh0Values, 
          plyData.octlevels, 
          plyData.octpaths, 
          plyData.gridValues,
          plyData.shRestValues
        );
        
        // Update info display to include octlevel range if available
        let octlevelInfo = '';
        if (plyData.octlevels && plyData.octlevels.length > 0) {
          const minOct = plyData.octlevels.reduce((min, val) => val < min ? val : min, plyData.octlevels[0]);
          const maxOct = plyData.octlevels.reduce((max, val) => val > max ? val : max, plyData.octlevels[0]);
          octlevelInfo = `\nOctlevels: ${minOct} to ${maxOct}`;
        }
        
        infoElement!.textContent = `Loaded: ${file.name}
          Vertices: ${plyData.vertexCount.toLocaleString()}
          Size: ${(file.size / (1024 * 1024)).toFixed(2)} MB
          Load time: ${loadTime}s${octlevelInfo}`;
        
        // Adjust camera to fit the point cloud based on scene center and extent
        if (plyData.sceneCenter && plyData.sceneExtent) {
          // Use the scene extent to position the camera
          const viewDistance = plyData.sceneExtent * 2;
          
          camera.setPosition(
            plyData.sceneCenter[0], 
            plyData.sceneCenter[1] + viewDistance * 0.5,  
            plyData.sceneCenter[2] + viewDistance
          );
          
          camera.setTarget(
            plyData.sceneCenter[0], 
            plyData.sceneCenter[1], 
            plyData.sceneCenter[2]
          );
          
        } else {
          // Fallback to default positioning
          camera.setPosition(0, 0, 5);
          camera.setTarget(0, 0, 0);
        }
      } catch (error: any) {
        infoElement!.textContent = `Error loading PLY: ${error.message}`;
        console.error('PLY loading error:', error);
      }
    }
  });
  
}