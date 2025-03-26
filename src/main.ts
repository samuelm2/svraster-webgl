import './style.css'
import { Viewer } from './lib/Viewer'
import { Camera } from './lib/Camera'
import { LoadPLY } from './lib/LoadPLY'

document.addEventListener('DOMContentLoaded', () => {
  // Create the WebGL viewer
  const viewer = new Viewer('app')
  
  // Get direct access to the camera
  const camera = viewer.getCamera()
  
  // Configure the camera for a nice view
  camera.setPosition(0, 0, 5)   // Position away from the scene
  camera.setTarget(0, 0, 0)     // Look at the center
  
  // Set up manual camera animation
  setupCameraAnimation(camera)
  
  // Handle window resize events
  window.addEventListener('resize', () => {
    const width = window.innerWidth
    const height = window.innerHeight
    viewer.resize(width, height)
  })
  
  // Add UI controls
  addControls(viewer, camera)
  
  // Add PLY upload UI
  addPLYUploadUI(viewer, camera)
})

/**
 * Set up manual camera animation
 */
function setupCameraAnimation(camera: Camera) {
  let orbitSpeed = 0.3
  let isOrbiting = true
  let lastTime = 0
  
  // Animation parameters
  let orbitRadius = 15
  let orbitHeight = 5
  let orbitAngle = 0
  
  // Create animation loop
  function animate(timestamp: number) {
    // Calculate delta time in seconds
    const deltaTime = (timestamp - lastTime) / 1000
    lastTime = timestamp
    
    if (isOrbiting) {
      // Update orbit angle based on speed
      orbitAngle += deltaTime * orbitSpeed
      
      // Calculate new camera position in a circle
      const x = Math.sin(orbitAngle) * orbitRadius
      const z = Math.cos(orbitAngle) * orbitRadius
      
      // Update camera position
      camera.setPosition(x, orbitHeight, z)
      
      // Keep looking at the center
      camera.setTarget(0, 0, 0)
    }
    
    // Continue animation loop
    requestAnimationFrame(animate)
  }
  
  // Add controls to window for direct access
  (window as any).cameraControls = {
    setOrbitSpeed: (speed: number) => { orbitSpeed = speed },
    setOrbitRadius: (radius: number) => { orbitRadius = radius },
    setOrbitHeight: (height: number) => { orbitHeight = height },
    toggleOrbiting: () => { isOrbiting = !isOrbiting },
    getOrbitSpeed: () => orbitSpeed,
    getOrbitRadius: () => orbitRadius,
    getOrbitHeight: () => orbitHeight,
    isOrbiting: () => isOrbiting
  }
  
  // Start animation
  requestAnimationFrame(animate)
}

/**
 * Add some basic UI controls for the demo
 */
function addControls(viewer: Viewer, camera: Camera) {
  // Create a simple control panel
  const controls = document.createElement('div')
  controls.style.position = 'absolute'
  controls.style.top = '10px'
  controls.style.left = '10px'
  controls.style.padding = '10px'
  controls.style.backgroundColor = 'rgba(0, 0, 0, 0.5)'
  controls.style.color = 'white'
  controls.style.fontFamily = 'sans-serif'
  controls.style.borderRadius = '5px'
  
  // Add a toggle for animation
  const animationToggle = document.createElement('div')
  animationToggle.innerHTML = `
    <label>
      <input type="checkbox" id="animation-toggle" checked>
      Enable orbit animation
    </label>
  `
  controls.appendChild(animationToggle)
  
  // Add a slider for rotation speed
  const speedControl = document.createElement('div')
  speedControl.innerHTML = `
    <label for="speed">Orbit Speed: </label>
    <input type="range" id="speed" min="0" max="1" step="0.05" value="0.3">
  `
  controls.appendChild(speedControl)
  
  // Add a slider for orbit radius
  const radiusControl = document.createElement('div')
  radiusControl.innerHTML = `
    <label for="radius">Orbit Radius: </label>
    <input type="range" id="radius" min="5" max="30" step="0.5" value="15">
  `
  controls.appendChild(radiusControl)
  
  // Add a slider for camera height
  const heightControl = document.createElement('div')
  heightControl.innerHTML = `
    <label for="height">Camera Height: </label>
    <input type="range" id="height" min="0" max="15" step="0.5" value="5">
  `
  controls.appendChild(heightControl)
  
  // Add a slider for number of cubes
  const countControl = document.createElement('div')
  countControl.innerHTML = `
    <label for="count">Cube Count: </label>
    <input type="range" id="count" min="1" max="100" step="1" value="25">
  `
  controls.appendChild(countControl)
  
  // Add the controls to the document
  document.body.appendChild(controls)
  
  // Add event listeners for the controls
  document.getElementById('animation-toggle')?.addEventListener('change', (e) => {
    (window as any).cameraControls.toggleOrbiting()
  })
  
  document.getElementById('speed')?.addEventListener('input', (e) => {
    const target = e.target as HTMLInputElement
    (window as any).cameraControls.setOrbitSpeed(parseFloat(target.value))
  })
  
  document.getElementById('radius')?.addEventListener('input', (e) => {
    const target = e.target as HTMLInputElement
    (window as any).cameraControls.setOrbitRadius(parseFloat(target.value))
  })
  
  document.getElementById('height')?.addEventListener('input', (e) => {
    const target = e.target as HTMLInputElement
    (window as any).cameraControls.setOrbitHeight(parseFloat(target.value))
  })
  
  document.getElementById('count')?.addEventListener('input', (e) => {
    const target = e.target as HTMLInputElement
    viewer.setInstanceCount(parseInt(target.value))
  })
}

/**
 * Add PLY file upload UI
 */
function addPLYUploadUI(viewer: Viewer, camera: Camera) {
  // Create a file upload container
  const uploadContainer = document.createElement('div')
  uploadContainer.style.position = 'absolute'
  uploadContainer.style.top = '10px'
  uploadContainer.style.right = '10px'
  uploadContainer.style.padding = '10px'
  uploadContainer.style.backgroundColor = 'rgba(0, 0, 0, 0.5)'
  uploadContainer.style.color = 'white'
  uploadContainer.style.fontFamily = 'sans-serif'
  uploadContainer.style.borderRadius = '5px'
  
  uploadContainer.innerHTML = `
    <h3>Load PLY Model</h3>
    <input type="file" id="ply-upload" accept=".ply">
    <div id="ply-info" style="margin-top: 10px; font-size: 12px;"></div>
    <div style="margin-top: 10px;">
      <button id="sample-ply">Load Sample PLY</button>
    </div>
  `
  
  document.body.appendChild(uploadContainer)
  
  // Handle file uploads
  const fileInput = document.getElementById('ply-upload') as HTMLInputElement
  fileInput.addEventListener('change', async (event) => {
    const target = event.target as HTMLInputElement
    const files = target.files
    
    if (files && files.length > 0) {
      const file = files[0]
      const infoElement = document.getElementById('ply-info')
      
      try {
        infoElement!.textContent = 'Loading PLY file...'
        
        // Show loading progress
        const startTime = performance.now()
        
        // Load the PLY file
        const plyData = await LoadPLY.loadFromFile(file)
        
        const loadTime = ((performance.now() - startTime) / 1000).toFixed(2)
        console.log(`Loaded ${plyData.vertexCount} vertices in ${loadTime} seconds`)
        
        // Set scene parameters if available
        if (plyData.sceneCenter && plyData.sceneExtents) {
          viewer.setSceneParameters(plyData.sceneCenter, plyData.sceneExtents)
        }
        
        // Load the point cloud with octlevels for scaling
        viewer.loadPointCloud(plyData.vertices, plyData.colors, plyData.octlevels)
        
        // Update info display to include octlevel range if available
        let octlevelInfo = ''
        if (plyData.octlevels) {
          const minOct = Math.min(...plyData.octlevels)
          const maxOct = Math.max(...plyData.octlevels)
          octlevelInfo = `\nOctlevels: ${minOct} to ${maxOct}`
        }
        
        infoElement!.textContent = `Loaded: ${file.name}
          Vertices: ${plyData.vertexCount.toLocaleString()}
          Size: ${(file.size / (1024 * 1024)).toFixed(2)} MB
          Load time: ${loadTime}s${octlevelInfo}`
        
        // Adjust camera to fit the point cloud
        if (plyData.sceneCenter && plyData.sceneExtents) {
          // Calculate a good camera position based on scene center and extents
          const maxExtent = Math.max(...plyData.sceneExtents) * 2
          camera.setPosition(
            plyData.sceneCenter[0], 
            plyData.sceneCenter[1] + maxExtent * 0.5, 
            plyData.sceneCenter[2] + maxExtent
          )
          camera.setTarget(plyData.sceneCenter[0], plyData.sceneCenter[1], plyData.sceneCenter[2])
          
          // Update orbit controls
          (window as any).cameraControls.setOrbitRadius(maxExtent)
          (window as any).cameraControls.setOrbitHeight(maxExtent * 0.5)
        } else {
          // Fallback to default positioning
          camera.setPosition(0, 0, 5)
          camera.setTarget(0, 0, 0)
          
          (window as any).cameraControls.setOrbitRadius(5)
          (window as any).cameraControls.setOrbitHeight(0)
        }
      } catch (error: any) {
        infoElement!.textContent = `Error loading PLY: ${error.message}`
        console.error('PLY loading error:', error)
      }
    }
  })
  
  // Handle sample PLY loading
  const sampleButton = document.getElementById('sample-ply')
  sampleButton?.addEventListener('click', async () => {
    const infoElement = document.getElementById('ply-info')
    
    infoElement!.textContent = 'Please use the file upload for your specific PLY format.'
    
    // Optionally, provide a sample PLY file if you have one hosted somewhere
    // const sampleUrl = 'your-sample-url.ply'
    // const plyData = await LoadPLY.loadFromUrl(sampleUrl)
    // ...
  })
}
