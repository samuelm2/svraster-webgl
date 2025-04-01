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
  
  // Update the UI to have buttons for both inputs
  uploadContainer.innerHTML = `
    <h3>Load PLY Model</h3>
    <div style="display: flex; gap: 5px; margin-bottom: 10px;">
      <input type="file" id="ply-upload" accept=".ply" 
        style="flex-grow: 1; padding: 5px; border-radius: 3px; border: 1px solid #ccc;">
      <button id="load-file" style="padding: 5px 10px; border-radius: 3px; border: 1px solid #ccc; cursor: pointer;">
        Load File
      </button>
    </div>
    <div style="display: flex; gap: 5px; margin-bottom: 10px;">
      <input type="text" id="ply-url" placeholder="Enter PLY URL" 
        style="padding: 5px; border-radius: 3px; border: 1px solid #ccc; flex-grow: 1;">
      <button id="load-url" style="padding: 5px 10px; border-radius: 3px; border: 1px solid #ccc; cursor: pointer;">
        Load URL
      </button>
    </div>
    <div id="ply-info" style="margin-top: 10px; font-size: 12px;"></div>
  `;
  
  document.body.appendChild(uploadContainer);
  
  // Get UI elements
  const fileInput = document.getElementById('ply-upload') as HTMLInputElement;
  const fileButton = document.getElementById('load-file') as HTMLButtonElement;
  const urlInput = document.getElementById('ply-url') as HTMLInputElement;
  const urlButton = document.getElementById('load-url') as HTMLButtonElement;
  const infoElement = document.getElementById('ply-info')!;

  // Helper function to update camera position based on PLY data
  const updateCameraPosition = (plyData: any) => {
    if (plyData.sceneCenter && plyData.sceneExtent) {
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
      camera.setPosition(0, 0, 5);
      camera.setTarget(0, 0, 0);
    }
  };

  // Helper function to load PLY data into viewer
  const loadPLYData = (plyData: any, loadTime: string, fileName: string, fileSize?: number) => {
    if (plyData.sceneCenter && plyData.sceneExtent) {
      viewer.setSceneParameters(plyData.sceneCenter, plyData.sceneExtent);
    }

    viewer.loadPointCloud(
      plyData.vertices, 
      plyData.sh0Values, 
      plyData.octlevels, 
      plyData.octpaths, 
      plyData.gridValues,
      plyData.shRestValues
    );

    let octlevelInfo = '';
    if (plyData.octlevels && plyData.octlevels.length > 0) {
      const minOct = plyData.octlevels.reduce((min: number, val: number) => val < min ? val : min, plyData.octlevels[0]);
      const maxOct = plyData.octlevels.reduce((max: number, val: number) => val > max ? val : max, plyData.octlevels[0]);
      octlevelInfo = `\nOctlevels: ${minOct} to ${maxOct}`;
    }

    const sizeInfo = fileSize ? `\nSize: ${(fileSize / (1024 * 1024)).toFixed(2)} MB` : '';
    infoElement.textContent = `Loaded: ${fileName}
      Vertices: ${plyData.vertexCount.toLocaleString()}${sizeInfo}
      Load time: ${loadTime}s${octlevelInfo}`;

    updateCameraPosition(plyData);
  };

  // Handle file input with button click
  fileButton.addEventListener('click', async () => {
    if (!fileInput.files || fileInput.files.length === 0) {
      alert('Please select a file first');
      return;
    }

    const file = fileInput.files[0];
    try {
      fileButton.disabled = true;
      fileButton.textContent = 'Loading...';
      infoElement.textContent = 'Loading PLY file...';

      const startTime = performance.now();
      const plyData = await LoadPLY.loadFromFile(file);
      const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
      
      loadPLYData(plyData, loadTime, file.name, file.size);
    } catch (error: any) {
      infoElement.textContent = `Error loading PLY: ${error.message}`;
      console.error('PLY loading error:', error);
    } finally {
      fileButton.disabled = false;
      fileButton.textContent = 'Load File';
    }
  });

  // Handle URL input
  urlButton.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    if (!url) {
      alert('Please enter a URL');
      return;
    }

    try {
      urlButton.disabled = true;
      urlButton.textContent = 'Loading...';
      infoElement.textContent = 'Loading PLY from URL...';

      const startTime = performance.now();
      const plyData = await LoadPLY.loadFromUrl(url);
      const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);

      const fileName = url.split('/').pop() || 'remote-ply';
      loadPLYData(plyData, loadTime, fileName);
    } catch (error: any) {
      infoElement.textContent = `Error loading PLY: ${error.message}`;
      console.error('PLY loading error:', error);
    } finally {
      urlButton.disabled = false;
      urlButton.textContent = 'Load URL';
    }
  });
}