import './style.css';
import { Viewer } from './lib/Viewer';
import { Camera } from './lib/Camera';
import { LoadPLY } from './lib/LoadPLY';

// Add these at the top of the file, after imports
let progressContainer: HTMLDivElement;
let progressBarInner: HTMLDivElement;
let progressText: HTMLDivElement;

// Create a function to initialize the progress bar
function createProgressBar() {
  progressContainer = document.createElement('div');
  progressContainer.style.position = 'absolute';
  progressContainer.style.top = '50%';
  progressContainer.style.left = '50%';
  progressContainer.style.transform = 'translate(-50%, -50%)';
  progressContainer.style.padding = '20px';
  progressContainer.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
  progressContainer.style.color = 'white';
  progressContainer.style.fontFamily = 'sans-serif';
  progressContainer.style.borderRadius = '10px';
  progressContainer.style.textAlign = 'center';
  progressContainer.style.display = 'none';

  progressText = document.createElement('div');
  progressText.style.marginBottom = '10px';
  progressText.textContent = 'Loading PLY file...';

  const progressBarOuter = document.createElement('div');
  progressBarOuter.style.width = '200px';
  progressBarOuter.style.height = '20px';
  progressBarOuter.style.backgroundColor = 'rgba(255, 255, 255, 0.2)';
  progressBarOuter.style.borderRadius = '10px';
  progressBarOuter.style.overflow = 'hidden';

  progressBarInner = document.createElement('div');
  progressBarInner.style.width = '0%';
  progressBarInner.style.height = '100%';
  progressBarInner.style.backgroundColor = '#4CAF50';

  progressBarOuter.appendChild(progressBarInner);
  progressContainer.appendChild(progressText);
  progressContainer.appendChild(progressBarOuter);
  document.body.appendChild(progressContainer);
}

// Helper function to update progress
function updateProgress(progress: number) {
  const percentage = Math.round(progress * 100);
  // Remove transition for 100% to ensure it completes
  if (percentage === 100) {
    progressBarInner.style.transition = 'none';
  }
  
  // Update both the width and text in the same frame
  const width = `${percentage}%`;
  const text = `Loading PLY file... ${percentage}%`;
  
  // Ensure both updates happen in the same frame
  requestAnimationFrame(() => {
    progressBarInner.style.width = width;
    progressText.textContent = text;
  });
}

// Also, let's reset the progress bar when starting a new load
function resetProgress() {
  progressBarInner.style.width = '0%';
  progressText.textContent = 'Loading PLY file... 0%';
}

document.addEventListener('DOMContentLoaded', async () => {
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
  
  // Get URL parameters
  const urlParams = new URLSearchParams(window.location.search);
  const plyUrl = urlParams.get('url') || 'https://huggingface.co/samuelm2/voxel-data/resolve/main/pumpkin_600k.ply';
  const showLoadingUI = urlParams.get('showLoadingUI') === 'true';

  // Add UI controls and get info element
  const infoDisplay = addControls();
  
  // Replace the progress bar creation code with:
  createProgressBar();

  // Only add PLY upload UI if showLoadingUI is true
  if (showLoadingUI) {
    addPLYUploadUI(viewer, camera);
  }

  // Auto-load the PLY file
  try {
    const startTime = performance.now();
    progressContainer.style.display = 'block';
    resetProgress();

    const plyData = await LoadPLY.loadFromUrl(plyUrl, (progress) => {
      updateProgress(progress);
    });

    const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
    progressContainer.style.display = 'none';

    const fileName = plyUrl.split('/').pop() || 'remote-ply';
    
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

    // Update both info displays if they exist
    const infoText = `Loaded: ${fileName}
      Voxels: ${plyData.vertexCount.toLocaleString()}
      Load time: ${loadTime}s${octlevelInfo}`;
    
    infoDisplay.textContent = infoText;
    
    if (showLoadingUI) {
      const uploadInfoElement = document.getElementById('ply-info');
      if (uploadInfoElement) {
        uploadInfoElement.textContent = infoText;
      }
    }

    viewer.setSceneTransformMatrix([0.9964059591293335,0.07686585187911987,0.03559183329343796,0,0.06180455908179283,-0.9470552206039429,0.3150659501552582,0,0.05792524665594101,-0.3117338716983795,-0.9484022259712219,0,0,0,0,1]);
    
    if (plyData.sceneCenter && plyData.sceneExtent) {
      camera.setPosition(-5.3627543449401855,-0.40146273374557495,3.546692371368408);      
      camera.setTarget(
        plyData.sceneCenter[0],
        plyData.sceneCenter[1],
        plyData.sceneCenter[2]
      );
    }
  } catch (error: any) {
    progressContainer.style.display = 'none';
    console.error('Error loading initial PLY:', error);
    const errorText = `Error loading PLY: ${error.message}`;
    infoDisplay.textContent = errorText;
    
    if (showLoadingUI) {
      const uploadInfoElement = document.getElementById('ply-info');
      if (uploadInfoElement) {
        uploadInfoElement.textContent = errorText;
      }
    }
  }
});

/**
 * Add some basic UI controls for the demo
 */
function addControls() {
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
  controls.style.fontSize = '14px';
  
  // Check if device is mobile
  const isMobile = window.matchMedia('(max-width: 768px)').matches;
  
  // Different instructions based on device type
  const controlInstructions = isMobile ? `
    <h3 style="margin: 0 0 5px 0; font-size: 1em;">Controls</h3>
    <ul style="padding-left: 15px; margin: 0;">
      <li style="margin: 2px 0;">1 finger drag: orbit</li>
      <li style="margin: 2px 0;">2 finger drag: pan/zoom</li>
    </ul>
  ` : `
    <h3 style="margin: 0 0 5px 0; font-size: 1em;">Controls</h3>
    <ul style="padding-left: 15px; margin: 0;">
      <li style="margin: 2px 0;">LClick + drag: orbit</li>
      <li style="margin: 2px 0;">RClick + drag: pan</li>
      <li style="margin: 2px 0;">Scroll: zoom</li>
      <li style="margin: 2px 0;">WASD/Arrow Keys: move</li>
    </ul>
  `;
  
  controls.innerHTML = `
    <h2 style="margin: 0 0 5px 0; font-size: 1.2em;">WebGL SVRaster Viewer</h2>
    <a href="https://github.com/samuelm2/svraster-webgl" style="text-decoration: underline; color: white; font-size: 0.85em; display: block; margin-bottom: 10px;">GitHub</a>
    ${controlInstructions}
    <div id="model-info" style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.3); font-size: 0.85em;"></div>
  `;

  // Add media query for mobile devices
  if (isMobile) {
    controls.style.fontSize = '12px';
    controls.style.padding = '8px';
    controls.style.maxWidth = '150px';
  }
  
  // Add the controls to the document
  document.body.appendChild(controls);
  
  return document.getElementById('model-info')!;
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
      
      camera.setPosition(-5.3627543449401855,-0.40146273374557495,3.546692371368408);      
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
      Voxels: ${plyData.vertexCount.toLocaleString()}${sizeInfo}
      Load time: ${loadTime}s${octlevelInfo}`;

    viewer.setSceneTransformMatrix([0.9964059591293335,0.07686585187911987,0.03559183329343796,0,0.06180455908179283,-0.9470552206039429,0.3150659501552582,0,0.05792524665594101,-0.3117338716983795,-0.9484022259712219,0,0,0,0,1]);
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
      progressContainer.style.display = 'block';
      resetProgress();

      const startTime = performance.now();
      const plyData = await LoadPLY.loadFromUrl(url, (progress) => {
        updateProgress(progress);
      });
      const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);

      progressContainer.style.display = 'none';
      const fileName = url.split('/').pop() || 'remote-ply';
      loadPLYData(plyData, loadTime, fileName);
    } catch (error: any) {
      progressContainer.style.display = 'none';
      infoElement.textContent = `Error loading PLY: ${error.message}`;
      console.error('PLY loading error:', error);
    } finally {
      urlButton.disabled = false;
      urlButton.textContent = 'Load URL';
    }
  });
}