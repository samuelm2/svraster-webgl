/**
 * Viewer class for WebGL2 rendering with instanced cubes
 * Specialized for binary PLY point cloud visualization
 */
import { Camera } from './Camera';

export class Viewer {
  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext | null;
  private program: WebGLProgram | null;
  private positionBuffer: WebGLBuffer | null = null;
  private colorBuffer: WebGLBuffer | null = null;
  private indexBuffer: WebGLBuffer | null = null;
  private instanceBuffer: WebGLBuffer | null = null;
  private instanceColorBuffer: WebGLBuffer | null = null;
  private instanceCount: number = 10; // Number of instances to render
  private indexCount: number = 0;     // Number of indices in the cube geometry
  private vao: WebGLVertexArrayObject | null = null;
  private container: HTMLElement;
  private resizeObserver: ResizeObserver;
  
  // Camera
  private camera: Camera;
  
  // Animation
  private rotationSpeed: number = 0.001;
  private lastFrameTime: number = 0;
  
  // Rendering flags

  // Scene properties for scaling calculation
  private sceneCenter: [number, number, number] = [0, 0, 0];
  private sceneExtent: number = 1.0;
  private instanceScaleBuffer: WebGLBuffer | null = null;
  private baseVoxelSize: number = 0.01;
  private instanceGridValuesBuffer: WebGLBuffer | null = null; // New buffer for grid density values

  private lastCameraPosition: [number, number, number] = [0, 0, 0];
  private resortThreshold: number = 0.1; // Threshold for camera movement to trigger resort

  // Add these properties to the Viewer class definition at the top
  private sortWorker: Worker | null = null;
  private pendingSortRequest: boolean = false;
  private originalPositions: Float32Array | null = null;
  private originalColors: Float32Array | null = null;
  private originalScales: Float32Array | null = null;
  private originalGridValues1: Float32Array | null = null;
  private originalGridValues2: Float32Array | null = null;
  private sortedIndices: Uint32Array | null = null;

  // Add these properties to store the original octree data
  private originalOctlevels: Uint8Array | null = null;
  private originalOctpaths: Uint32Array | null = null;

  // Add a flag for this debug feature

  // Add these properties to the Viewer class definition at the top
  private isDragging: boolean = false;
  private isPanning: boolean = false;
  private lastMouseX: number = 0;
  private lastMouseY: number = 0;
  private orbitSpeed: number = 0.005;
  private panSpeed: number = 0.01;
  private zoomSpeed: number = 0.1;

  // Add this property to the Viewer class
  private sceneTransformMatrix: Float32Array = new Float32Array([
    1, 0, 0, 0,   // First row
    0, -1, 0, 0,  // Second row - negate Y to flip the scene vertically
    0, 0, 1, 0,   // Third row
    0, 0, 0, 1    // Fourth row
  ]);

  // Add these properties to the Viewer class definition
  private instanceDensityBuffer: WebGLBuffer | null = null;

  constructor(containerId: string) {
    // Create canvas element
    this.canvas = document.createElement('canvas');
    
    // Get container
    this.container = document.getElementById(containerId)!;
    if (!this.container) {
      throw new Error(`Container element with id "${containerId}" not found`);
    }
    
    // Set canvas to fill container completely
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.canvas.style.display = 'block'; // Remove any extra space beneath the canvas
    
    // Append to container
    this.container.appendChild(this.canvas);
    
    // Initialize WebGL2 context
    this.gl = this.canvas.getContext('webgl2');
    if (!this.gl) {
      throw new Error('WebGL2 not supported in this browser');
    }
    
    // Disable depth testing and enable alpha blending
    this.gl.disable(this.gl.DEPTH_TEST);
    this.gl.enable(this.gl.BLEND);
    this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
    
    // Initialize camera - revert to original positive Z position
    this.camera = new Camera();
    this.camera.setPosition(0, 0, 15); // Back to positive Z
    this.camera.setTarget(0, 0, 0);    // Still look at origin
    
    // Fix: Initialize last camera position using explicit array
    const pos = this.camera.getPosition();
    this.lastCameraPosition = [pos[0], pos[1], pos[2]]; 
    
    // Set initial size
    this.updateCanvasSize();
    
    // Create a resize observer to handle container size changes
    this.resizeObserver = new ResizeObserver(() => {
      this.updateCanvasSize();
    });
    this.resizeObserver.observe(this.container);
    
    // Also handle window resize
    window.addEventListener('resize', () => {
      this.updateCanvasSize();
    });
    
    // Setup program and buffers
    this.program = null;
    this.initShaders();
    this.initBuffers();
    
    // Initialize the sort worker
    this.initSortWorker();
    
    // Add mouse event listeners for orbital controls
    this.initOrbitControls();
    
    this.render(0);
  }
  
  /**
   * Updates canvas size to match container dimensions
   */
  private updateCanvasSize(): void {
    // Get container dimensions (using getBoundingClientRect for true pixel dimensions)
    const rect = this.container.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    
    // Set canvas dimensions to match container
    this.canvas.width = width;
    this.canvas.height = height;
    
    // Update camera aspect ratio
    this.camera.setAspectRatio(width / height);
    
    // Update WebGL viewport
    if (this.gl) {
      this.gl.viewport(0, 0, width, height);
    }
  }

  private initShaders(): void {
    const gl = this.gl!;
    
    // Updated vertex shader to pass density values to fragment shader
    const vsSource = `#version 300 es
      in vec4 aVertexPosition;
      in vec4 aVertexColor;
      in vec4 aInstanceOffset;
      in vec4 aInstanceColor;
      in float aInstanceScale;
      in vec4 aInstanceDensity0;   // Density values for 4 corners
      in vec4 aInstanceDensity1;   // Density values for other 4 corners
      
      uniform mat4 uProjectionMatrix;
      uniform mat4 uViewMatrix;
      uniform mat4 uSceneTransformMatrix;
      uniform bool uUseInstanceColors;
      
      out vec4 vColor;
      out vec3 vWorldPos;
      out float vScale;
      out vec3 vVoxelCenter;
      out vec4 vDensity0;          // Pass density values to fragment shader
      out vec4 vDensity1;
      
      void main() {
        // Scale the vertex position by instance scale
        vec4 scaledPosition = vec4(aVertexPosition.xyz * aInstanceScale, aVertexPosition.w);
        
        // Transform instance offset by scene transform matrix
        vec4 transformedOffset = uSceneTransformMatrix * vec4(aInstanceOffset.xyz, 1.0);
        
        // Position is scaled vertex position + transformed instance offset
        vec4 instancePosition = scaledPosition + vec4(transformedOffset.xyz, 0.0);
        
        // Calculate final position
        gl_Position = uProjectionMatrix * uViewMatrix * instancePosition;
        
        // Pass world position of the vertex
        vWorldPos = instancePosition.xyz;
        
        // Calculate and pass voxel center
        vVoxelCenter = transformedOffset.xyz;
        
        // Pass scale to fragment shader
        vScale = aInstanceScale;
        
        // Pass density values to fragment shader
        vDensity0 = aInstanceDensity0;
        vDensity1 = aInstanceDensity1;
        
        // Use instance color if enabled, otherwise use vertex color
        vColor = uUseInstanceColors ? aInstanceColor : aVertexColor;
      }
    `;
    
    // Updated fragment shader with debug output
    const fsSource = `#version 300 es
      precision mediump float;
      
      in vec4 vColor;
      in vec3 vWorldPos;
      in float vScale;
      in vec3 vVoxelCenter;
      in vec4 vDensity0;       // Density values for corners 0-3
      in vec4 vDensity1;       // Density values for corners 4-7
      
      uniform vec3 uCameraPosition;
      
      out vec4 fragColor;
      
      // Ray-box intersection function - returns entry and exit t values
      vec2 rayBoxIntersection(vec3 rayOrigin, vec3 rayDir, vec3 boxCenter, float boxScale) {
        vec3 boxMin = boxCenter - vec3(boxScale * 0.5);
        vec3 boxMax = boxCenter + vec3(boxScale * 0.5);
        
        vec3 invDir = 1.0 / rayDir;
        vec3 tMin = (boxMin - rayOrigin) * invDir;
        vec3 tMax = (boxMax - rayOrigin) * invDir;
        vec3 t1 = min(tMin, tMax);
        vec3 t2 = max(tMin, tMax);
        float tNear = max(max(t1.x, t1.y), t1.z);
        float tFar = min(min(t2.x, t2.y), t2.z);
        return vec2(tNear, tFar);
      }
      
      // Updated trilinear interpolation function that accounts for y-axis flip
      float trilinearInterpolation(vec3 pos, vec3 boxMin, vec3 boxMax, vec4 density0, vec4 density1) {
        // Normalize position within the box [0,1]
        vec3 normPos = (pos - boxMin) / (boxMax - boxMin);
        
        // For Y axis, we need to invert the interpolation factor since our
        // world coordinates are flipped (e.g., Y in shader is -Y in original data)
        normPos.y = 1.0 - normPos.y;
        
        // Rest of corner mapping stays the same
        float c000 = density0.x; // Corner [0,0,0]
        float c001 = density0.y; // Corner [0,0,1]
        float c010 = density0.z; // Corner [0,1,0]
        float c011 = density0.w; // Corner [0,1,1]
        float c100 = density1.x; // Corner [1,0,0]
        float c101 = density1.y; // Corner [1,0,1]
        float c110 = density1.z; // Corner [1,1,0]
        float c111 = density1.w; // Corner [1,1,1]
        
        // Linear interpolation factors
        float fx = normPos.x;
        float fy = normPos.y;  // This is now inverted
        float fz = normPos.z;
        float fx1 = 1.0 - fx;
        float fy1 = 1.0 - fy;
        float fz1 = 1.0 - fz;
        
        // First interpolate along x axis
        float c00 = fx1 * c000 + fx * c100; // Edge [0,0,0] to [1,0,0]
        float c01 = fx1 * c001 + fx * c101; // Edge [0,0,1] to [1,0,1]
        float c10 = fx1 * c010 + fx * c110; // Edge [0,1,0] to [1,1,0]
        float c11 = fx1 * c011 + fx * c111; // Edge [0,1,1] to [1,1,1]
        
        // Then interpolate along y axis
        float c0 = fy1 * c00 + fy * c10; // Edge [x,0,0] to [x,1,0]
        float c1 = fy1 * c01 + fy * c11; // Edge [x,0,1] to [x,1,1]
        
        // Finally interpolate along z axis
        return fz1 * c0 + fz * c1; // Edge [x,y,0] to [x,y,1]
      }
      
      void main() {
        // Calculate ray from camera to this fragment
        vec3 rayOrigin = uCameraPosition;
        vec3 rayDir = normalize(vWorldPos - uCameraPosition);
        
        // Get ray-box intersection
        vec2 tIntersect = rayBoxIntersection(rayOrigin, rayDir, vVoxelCenter, vScale);
        float tNear = max(0.0, tIntersect.x);
        float tFar = min(tIntersect.y, 1000.0);
        
        if (tNear < tFar) {
          // Calculate box min and max
          vec3 boxMin = vVoxelCenter - vec3(vScale * 0.5);
          vec3 boxMax = vVoxelCenter + vec3(vScale * 0.5);
          
          // Get intersection length
          float rayLength = tFar - tNear;
          
          // Sample at three points within the cube
          // First sample at 1/4 along the ray path
          vec3 samplePoint1 = rayOrigin + rayDir * (tNear + rayLength * 0.25);
          float density1 = trilinearInterpolation(samplePoint1, boxMin, boxMax, vDensity0, vDensity1);
          
          // Second sample at 2/4 (halfway) along the ray path
          vec3 samplePoint2 = rayOrigin + rayDir * (tNear + rayLength * 0.5);
          float density2 = trilinearInterpolation(samplePoint2, boxMin, boxMax, vDensity0, vDensity1);
          
          // Third sample at 3/4 along the ray path
          vec3 samplePoint3 = rayOrigin + rayDir * (tNear + rayLength * 0.75);
          float density3 = trilinearInterpolation(samplePoint3, boxMin, boxMax, vDensity0, vDensity1);
          
          // Average the densities from all three samples
          float avgDensity = (density1 + density2 + density3) / 3.0;
          
          // Calculate opacity based on average density and intersection length
          float alpha = avgDensity;
          
          // Output color with calculated alpha
          fragColor = vec4(vColor.rgb, alpha);
        } else {
          discard;
        }
      }
    `;
    
    // Create shaders with better error handling
    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    if (!vertexShader) {
      console.error('Failed to create vertex shader');
      return;
    }
    
    gl.shaderSource(vertexShader, vsSource);
    gl.compileShader(vertexShader);
    
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(vertexShader);
      console.error('Vertex shader compilation failed:', info);
      gl.deleteShader(vertexShader);
      return;
    } else {
      console.log('Vertex shader compiled successfully');
    }
    
    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    if (!fragmentShader) {
      console.error('Failed to create fragment shader');
      return;
    }
    
    gl.shaderSource(fragmentShader, fsSource);
    gl.compileShader(fragmentShader);
    
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(fragmentShader);
      console.error('Fragment shader compilation failed:', info);
      gl.deleteShader(fragmentShader);
      return;
    } else {
      console.log('Fragment shader compiled successfully');
    }
    
    // Create and link program
    this.program = gl.createProgram();
    if (!this.program) {
      console.error('Failed to create shader program');
      return;
    }
    
    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.linkProgram(this.program);
    
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(this.program);
      console.error('Shader program linking failed:', info);
      return;
    } else {
      console.log('Shader program linked successfully');
    }
  }
  
  private createShader(type: number, source: string): WebGLShader {
    const gl = this.gl!;
    const shader = gl.createShader(type);
    
    if (!shader) {
      throw new Error('Failed to create shader');
    }
    
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Could not compile shader: ${info}`);
    }
    
    return shader;
  }
  
  private initBuffers(): void {
    const gl = this.gl!;
    
    // Create a vertex array object (VAO)
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    
    // Initialize cube geometry for instancing
    // Use a smaller cube size by default (better for point clouds)
    this.initCubeGeometry(0.01); 
    
    // Example: Create a simple grid of instances
    const instanceOffsets = [];
    const gridSize = Math.ceil(Math.sqrt(this.instanceCount));
    const spacing = 2.0; // Space between instances
    
    for (let i = 0; i < this.instanceCount; i++) {
      const x = (i % gridSize) * spacing - (gridSize * spacing / 2) + spacing / 2;
      const z = Math.floor(i / gridSize) * spacing - (gridSize * spacing / 2) + spacing / 2;
      // Position each instance with a different offset
      instanceOffsets.push(x, 0.0, z, 0.0);
    }
    
    // Create and fill the instance buffer
    this.instanceBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(instanceOffsets), gl.STATIC_DRAW);
    
    // Set up instance position attribute
    const instanceAttributeLocation = gl.getAttribLocation(this.program!, 'aInstanceOffset');
    gl.enableVertexAttribArray(instanceAttributeLocation);
    gl.vertexAttribPointer(
      instanceAttributeLocation,
      4,          // 4 components per instance offset (x, y, z, w)
      gl.FLOAT,   // data type
      false,      // no normalization
      0,          // stride
      0           // offset
    );
    
    // Enable instancing for positions
    gl.vertexAttribDivisor(instanceAttributeLocation, 1);
    
    // Unbind the VAO
    gl.bindVertexArray(null);
  }
  
  /**
   * Initialize cube geometry for instance rendering
   */
  private initCubeGeometry(size: number): void {
    const gl = this.gl!;
    
    // Create a simplified cube with the specified size
    const halfSize = size / 2;
    
    // Simplify: Just use a cube with 8 vertices and 12 triangles
    const positions = [
      // Front face
      -halfSize, -halfSize,  halfSize,
       halfSize, -halfSize,  halfSize,
       halfSize,  halfSize,  halfSize,
      -halfSize,  halfSize,  halfSize,
      
      // Back face
      -halfSize, -halfSize, -halfSize,
      -halfSize,  halfSize, -halfSize,
       halfSize,  halfSize, -halfSize,
       halfSize, -halfSize, -halfSize,
    ];
    
    // Create position buffer
    this.positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
    
    // Set up the position attribute
    const positionAttributeLocation = gl.getAttribLocation(this.program!, 'aVertexPosition');
    gl.enableVertexAttribArray(positionAttributeLocation);
    gl.vertexAttribPointer(
      positionAttributeLocation,
      3,        // 3 components per vertex
      gl.FLOAT, // data type
      false,    // no normalization
      0,        // stride
      0         // offset
    );
    
    // Simplify: Solid white colors for all vertices
    const colors = [];
    for (let i = 0; i < 8; i++) {
      colors.push(1.0, 1.0, 1.0, 1.0);
    }
    
    // Create color buffer
    this.colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
    
    // Set up the color attribute
    const colorAttributeLocation = gl.getAttribLocation(this.program!, 'aVertexColor');
    gl.enableVertexAttribArray(colorAttributeLocation);
    gl.vertexAttribPointer(
      colorAttributeLocation,
      4,        // 4 components per color (RGBA)
      gl.FLOAT, // data type
      false,    // no normalization
      0,        // stride
      0         // offset
    );
    
    // Simplify: Use just front and back faces for a total of 12 triangles
    const indices = [
      0, 1, 2,    0, 2, 3,    // Front face
      4, 5, 6,    4, 6, 7,    // Back face
      0, 3, 5,    0, 5, 4,    // Left face
      1, 7, 6,    1, 6, 2,    // Right face
      3, 2, 6,    3, 6, 5,    // Top face
      0, 4, 7,    0, 7, 1     // Bottom face
    ];
    
    // Create index buffer
    this.indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    
    // Store the number of indices
    this.indexCount = indices.length;
    
    console.log(`Initialized cube geometry with ${this.indexCount} indices`);
  }
  
  /**
   * Main render function with time-based animation
   */
  private render(timestamp: number): void {
    const gl = this.gl!;
    
    // Calculate delta time in seconds
    const deltaTime = (timestamp - this.lastFrameTime) / 1000;
    this.lastFrameTime = timestamp;
    
    // Check if camera has moved enough to trigger a resort
    const cameraPos = this.camera.getPosition();
    const dx = cameraPos[0] - this.lastCameraPosition[0];
    const dy = cameraPos[1] - this.lastCameraPosition[1];
    const dz = cameraPos[2] - this.lastCameraPosition[2];
    const cameraMoveDistance = Math.sqrt(dx*dx + dy*dy + dz*dz);
    
    if (cameraMoveDistance > this.resortThreshold && !this.pendingSortRequest) {
      this.lastCameraPosition = [cameraPos[0], cameraPos[1], cameraPos[2]];
      this.requestSort();
    }
    
    // Debug: Ensure we have valid data to render
    if (this.instanceCount === 0) {
      console.warn('No instances to render');
    }
    
    // Clear the canvas with a slightly visible color to see if rendering is happening
    gl.clearColor(0.1, 0.1, 0.1, 1.0); // Dark gray instead of black
    gl.clear(gl.COLOR_BUFFER_BIT);
    
    // Ensure blending is properly set up
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    
    // Use our shader program
    gl.useProgram(this.program);
    
    // Debug: Check if program is valid
    if (!this.program) {
      console.error('Shader program is null');
      requestAnimationFrame((time) => this.render(time));
      return;
    }
    
    // Bind the VAO
    gl.bindVertexArray(this.vao);
    
    // Debug: Check if VAO is valid
    if (!this.vao) {
      console.error('VAO is null');
      requestAnimationFrame((time) => this.render(time));
      return;
    }
    
    // Set uniforms with camera matrices
    const projectionMatrixLocation = gl.getUniformLocation(this.program, 'uProjectionMatrix');
    const viewMatrixLocation = gl.getUniformLocation(this.program, 'uViewMatrix');
    const sceneTransformMatrixLocation = gl.getUniformLocation(this.program, 'uSceneTransformMatrix');
    const useInstanceColorsLocation = gl.getUniformLocation(this.program, 'uUseInstanceColors');
    const cameraPositionLocation = gl.getUniformLocation(this.program, 'uCameraPosition');
    
    // Pass matrices to shader
    gl.uniformMatrix4fv(projectionMatrixLocation, false, this.camera.getProjectionMatrix());
    gl.uniformMatrix4fv(viewMatrixLocation, false, this.camera.getViewMatrix());
    gl.uniformMatrix4fv(sceneTransformMatrixLocation, false, this.sceneTransformMatrix);
    gl.uniform1i(useInstanceColorsLocation, 1);
    
    // Pass camera position to the shader
    gl.uniform3f(cameraPositionLocation, cameraPos[0], cameraPos[1], cameraPos[2]);
    
    // Draw instanced geometry
    gl.drawElementsInstanced(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0, this.instanceCount);
    
    // Check for GL errors
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
      console.error(`WebGL error: ${error}`);
    }
    
    // Unbind the VAO
    gl.bindVertexArray(null);
    
    // Request animation frame for continuous rendering
    requestAnimationFrame((time) => this.render(time));
  }
  
  /**
   * Load a point cloud from positions and colors
   */
  public loadPointCloud(
    positions: Float32Array, 
    colors?: Float32Array,
    octlevels?: Uint8Array,
    octpaths?: Uint32Array,
    gridValues?: Float32Array
  ): void {
    console.log(`Loading point cloud with ${positions.length / 3} points`);
    
    // Save original data for sorting
    this.originalPositions = new Float32Array(positions);
    
    // Save colors
    if (colors) {
      this.originalColors = new Float32Array(colors);
    } else {
      this.originalColors = new Float32Array(positions.length / 3 * 4);
      for (let i = 0; i < positions.length / 3; i++) {
        this.originalColors[i * 4 + 0] = 1.0; // R
        this.originalColors[i * 4 + 1] = 1.0; // G
        this.originalColors[i * 4 + 2] = 1.0; // B
        this.originalColors[i * 4 + 3] = 1.0; // A
      }
    }
    
    // Save octree data
    if (octlevels) {
      this.originalOctlevels = new Uint8Array(octlevels);
      
      this.originalScales = new Float32Array(octlevels.length);
      for (let i = 0; i < octlevels.length; i++) {
        this.originalScales[i] = this.baseVoxelSize * Math.pow(2, -octlevels[i]);
      }
    }
    
    if (octpaths) {
      this.originalOctpaths = new Uint32Array(octpaths);
    }
    
    // Save grid values if available
    if (gridValues) {
      console.log(`Saving ${gridValues.length} grid values`);
      this.originalGridValues1 = new Float32Array(positions.length / 3 * 4);
      this.originalGridValues2 = new Float32Array(positions.length / 3 * 4);
      
      // Check if gridValues contains non-1.0 values
      let hasNonOneValues = false;
      let minVal = 1.0, maxVal = 1.0;
      
      for (let i = 0; i < Math.min(gridValues.length, 100); i++) {
        if (gridValues[i] !== 1.0) {
          hasNonOneValues = true;
          minVal = Math.min(minVal, gridValues[i]);
          maxVal = Math.max(maxVal, gridValues[i]);
        }
      }
      
      console.log(`Grid values check: Has values != 1.0: ${hasNonOneValues}, Min: ${minVal}, Max: ${maxVal}`);
      console.log(`First few grid values:`, gridValues.slice(0, 24));
      
      // Use original, non-swapped grid values order - this will work with our shader fix
      for (let i = 0; i < positions.length / 3; i++) {
        // First 4 corners in density0 (following the specified ordering)
        this.originalGridValues1[i * 4 + 0] = gridValues[i * 8 + 0]; // Corner [0,0,0]
        this.originalGridValues1[i * 4 + 1] = gridValues[i * 8 + 1]; // Corner [0,0,1]
        this.originalGridValues1[i * 4 + 2] = gridValues[i * 8 + 2]; // Corner [0,1,0]
        this.originalGridValues1[i * 4 + 3] = gridValues[i * 8 + 3]; // Corner [0,1,1]
        
        // Next 4 corners in density1 (following the specified ordering)
        this.originalGridValues2[i * 4 + 0] = gridValues[i * 8 + 4]; // Corner [1,0,0]
        this.originalGridValues2[i * 4 + 1] = gridValues[i * 8 + 5]; // Corner [1,0,1]
        this.originalGridValues2[i * 4 + 2] = gridValues[i * 8 + 6]; // Corner [1,1,0]
        this.originalGridValues2[i * 4 + 3] = gridValues[i * 8 + 7]; // Corner [1,1,1]
      }
    } else {
      // If no grid values, use default density of 1.0 for all corners
      this.originalGridValues1 = new Float32Array(positions.length / 3 * 4);
      this.originalGridValues2 = new Float32Array(positions.length / 3 * 4);
      
      for (let i = 0; i < positions.length / 3; i++) {
        for (let j = 0; j < 4; j++) {
          this.originalGridValues1[i * 4 + j] = 1.0;
          this.originalGridValues2[i * 4 + j] = 1.0;
        }
      }
    }
    
    // Initialize WebGL resources
    const gl = this.gl!;
    
    // Create VAO
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    
    // Initialize cube geometry
    this.initCubeGeometry(1.0);
    
    // Set up instance positions
    this.instanceBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    
    const instanceAttributeLocation = gl.getAttribLocation(this.program!, 'aInstanceOffset');
    gl.enableVertexAttribArray(instanceAttributeLocation);
    gl.vertexAttribPointer(
      instanceAttributeLocation,
      3,
      gl.FLOAT,
      false,
      0,
      0
    );
    gl.vertexAttribDivisor(instanceAttributeLocation, 1);
    
    // Set up instance colors
    this.instanceColorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.originalColors, gl.STATIC_DRAW);
    
    const instanceColorLocation = gl.getAttribLocation(this.program!, 'aInstanceColor');
    gl.enableVertexAttribArray(instanceColorLocation);
    gl.vertexAttribPointer(
      instanceColorLocation,
      4,
      gl.FLOAT,
      false,
      0,
      0
    );
    gl.vertexAttribDivisor(instanceColorLocation, 1);
    
    // Set the instance count
    this.instanceCount = positions.length / 3;
    
    // Set up instance scales
    if (this.originalScales) {
      this.instanceScaleBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceScaleBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, this.originalScales, gl.STATIC_DRAW);
      
      const instanceScaleLocation = gl.getAttribLocation(this.program!, 'aInstanceScale');
      if (instanceScaleLocation !== -1) {
        gl.enableVertexAttribArray(instanceScaleLocation);
        gl.vertexAttribPointer(
          instanceScaleLocation,
          1,
          gl.FLOAT,
          false,
          0,
          0
        );
        gl.vertexAttribDivisor(instanceScaleLocation, 1);
      }
    } else {
      // Use default scale
      const scales = new Float32Array(this.instanceCount);
      for (let i = 0; i < this.instanceCount; i++) {
        scales[i] = this.baseVoxelSize;
      }
      
      this.instanceScaleBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceScaleBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, scales, gl.STATIC_DRAW);
      
      const instanceScaleLocation = gl.getAttribLocation(this.program!, 'aInstanceScale');
      if (instanceScaleLocation !== -1) {
        gl.enableVertexAttribArray(instanceScaleLocation);
        gl.vertexAttribPointer(
          instanceScaleLocation,
          1,
          gl.FLOAT,
          false,
          0,
          0
        );
        gl.vertexAttribDivisor(instanceScaleLocation, 1);
      }
    }
    
    // Set up density values for the first 4 corners
    const instanceDensity0Location = gl.getAttribLocation(this.program!, 'aInstanceDensity0');
    console.log('Density0 attribute location:', instanceDensity0Location);
    if (instanceDensity0Location !== -1 && this.originalGridValues1) {
      this.instanceGridValuesBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceGridValuesBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, this.originalGridValues1, gl.STATIC_DRAW);
      
      gl.enableVertexAttribArray(instanceDensity0Location);
      gl.vertexAttribPointer(
        instanceDensity0Location,
        4,
        gl.FLOAT,
        false,
        0,
        0
      );
      gl.vertexAttribDivisor(instanceDensity0Location, 1);
    }
    
    // Set up density values for the other 4 corners
    const instanceDensity1Location = gl.getAttribLocation(this.program!, 'aInstanceDensity1');
    console.log('Density1 attribute location:', instanceDensity1Location);
    if (instanceDensity1Location !== -1 && this.originalGridValues2) {
      this.instanceDensityBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceDensityBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, this.originalGridValues2, gl.STATIC_DRAW);
      
      gl.enableVertexAttribArray(instanceDensity1Location);
      gl.vertexAttribPointer(
        instanceDensity1Location,
        4,
        gl.FLOAT,
        false,
        0,
        0
      );
      gl.vertexAttribDivisor(instanceDensity1Location, 1);
    }
    
    // Unbind the VAO
    gl.bindVertexArray(null);
    
    // Request initial sorting
    this.requestSort();
  }
  
  /**
   * Set camera position
   */
  public setCameraPosition(x: number, y: number, z: number): void {
    this.camera.setPosition(x, y, z);
  }
  
  /**
   * Set camera target
   */
  public setCameraTarget(x: number, y: number, z: number): void {
    this.camera.setTarget(x, y, z);
  }
  
  /**
   * Set rotation speed
   */
  public setRotationSpeed(speed: number): void {
    this.rotationSpeed = speed;
  }
  
  /**
   * Resize the canvas to specified dimensions
   */
  public resize(width: number, height: number): void {
    if (this.canvas) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.gl!.viewport(0, 0, width, height);
      this.camera.setAspectRatio(width / height);
    }
  }
  
  /**
   * Get direct access to the camera
   */
  public getCamera(): Camera {
    return this.camera;
  }
  
  /**
   * Clean up resources when the viewer is no longer needed
   */
  public dispose(): void {
    // Stop observing resize events
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
    }
    
    // Remove event listeners
    window.removeEventListener('resize', this.updateCanvasSize);
    
    // Clean up WebGL resources
    const gl = this.gl;
    if (gl) {
      // Delete buffers
      if (this.positionBuffer) gl.deleteBuffer(this.positionBuffer);
      if (this.colorBuffer) gl.deleteBuffer(this.colorBuffer);
      if (this.indexBuffer) gl.deleteBuffer(this.indexBuffer);
      if (this.instanceBuffer) gl.deleteBuffer(this.instanceBuffer);
      if (this.instanceColorBuffer) gl.deleteBuffer(this.instanceColorBuffer);
      if (this.instanceScaleBuffer) gl.deleteBuffer(this.instanceScaleBuffer);
      if (this.instanceGridValuesBuffer) gl.deleteBuffer(this.instanceGridValuesBuffer);
      
      // Delete VAO
      if (this.vao) gl.deleteVertexArray(this.vao);
      
      // Delete program and shaders
      if (this.program) gl.deleteProgram(this.program);
    }
    
    // Clear reference data
    this.originalPositions = null;
    this.originalColors = null;
    this.originalScales = null;
    this.originalGridValues1 = null;
    this.originalGridValues2 = null;
    this.originalOctlevels = null;
    this.originalOctpaths = null;
    this.sortedIndices = null;
    
    // Terminate the sort worker
    if (this.sortWorker) {
      this.sortWorker.terminate();
      this.sortWorker = null;
    }
  }

  /**
   * Set scene parameters from PLY file header
   */
  public setSceneParameters(center: [number, number, number], extent: number): void {
    this.sceneCenter = center;
    this.sceneExtent = extent;
    this.baseVoxelSize = extent; // Use the extent as the base voxel size
    
    console.log(`Scene center: [${center}], extent: ${extent}, base voxel size: ${this.baseVoxelSize}`);
  }

  /**
   * Initialize the worker for sorting voxels
   */
  private initSortWorker(): void {
    try {
      console.log('Initializing sort worker...');
      
      // Create worker
      this.sortWorker = new Worker(new URL('../workers/SortWorker.ts', import.meta.url), { type: 'module' });
      
      // Log initialization
      console.log('Sort worker created, setting up event handlers');
      
      // Set up message handler
      this.sortWorker.onmessage = (event) => {
        console.log('Received message from sort worker:', event.data.type);
        
        const data = event.data;
        
        if (data.type === 'ready') {
          console.log('Sort worker initialized and ready');
        } else if (data.type === 'sorted') {
          console.log('Received sorted indices from worker');
          
          // Store the sorted indices for rendering
          this.sortedIndices = data.indices;
          this.pendingSortRequest = false;
          
          if (this.sortedIndices) {
            console.log(`Received ${this.sortedIndices.length} sorted indices from worker`);
            
            // Apply the sorted order to the buffers
            this.applySortedOrder();
          } else {
            console.error('Received null indices from worker');
          }
        }
      };
      
      this.sortWorker.onerror = (error) => {
        console.error('Sort worker error:', error);
        this.pendingSortRequest = false;
      };
      
      console.log('Sort worker event handlers configured');
    } catch (error) {
      console.error('Failed to initialize sort worker:', error);
    }
  }

  /**
   * Apply sorted indices to reorder instance data
   */
  private applySortedOrder(): void {
    if (!this.sortedIndices || this.sortedIndices.length === 0 || !this.originalPositions) {
      return;
    }
    
    const gl = this.gl!;
    
    // Create sorted arrays
    const sortedPositions = new Float32Array(this.instanceCount * 3);
    let sortedColors: Float32Array | null = null;
    let sortedScales: Float32Array | null = null;
    let sortedGridValues1: Float32Array | null = null;
    let sortedGridValues2: Float32Array | null = null;
    
    if (this.originalColors) {
      sortedColors = new Float32Array(this.instanceCount * 4);
    }
    
    if (this.originalScales) {
      sortedScales = new Float32Array(this.instanceCount);
    }
    
    if (this.originalGridValues1) {
      sortedGridValues1 = new Float32Array(this.instanceCount * 4);
    }
    
    if (this.originalGridValues2) {
      sortedGridValues2 = new Float32Array(this.instanceCount * 4);
    }
    
    // Reorder based on indices
    for (let i = 0; i < this.instanceCount; i++) {
      const srcIdx = this.sortedIndices[i];
      
      // Reorder positions
      sortedPositions[i * 3] = this.originalPositions[srcIdx * 3];
      sortedPositions[i * 3 + 1] = this.originalPositions[srcIdx * 3 + 1];
      sortedPositions[i * 3 + 2] = this.originalPositions[srcIdx * 3 + 2];
      
      // Reorder colors
      if (sortedColors && this.originalColors) {
        sortedColors[i * 4] = this.originalColors[srcIdx * 4];
        sortedColors[i * 4 + 1] = this.originalColors[srcIdx * 4 + 1];
        sortedColors[i * 4 + 2] = this.originalColors[srcIdx * 4 + 2];
        sortedColors[i * 4 + 3] = this.originalColors[srcIdx * 4 + 3];
      }
      
      // Reorder scales
      if (sortedScales && this.originalScales) {
        sortedScales[i] = this.originalScales[srcIdx];
      }
      
      // Reorder grid values 1 (first 4 corners)
      if (sortedGridValues1 && this.originalGridValues1) {
        sortedGridValues1[i * 4] = this.originalGridValues1[srcIdx * 4];
        sortedGridValues1[i * 4 + 1] = this.originalGridValues1[srcIdx * 4 + 1];
        sortedGridValues1[i * 4 + 2] = this.originalGridValues1[srcIdx * 4 + 2];
        sortedGridValues1[i * 4 + 3] = this.originalGridValues1[srcIdx * 4 + 3];
      }
      
      // Reorder grid values 2 (other 4 corners)
      if (sortedGridValues2 && this.originalGridValues2) {
        sortedGridValues2[i * 4] = this.originalGridValues2[srcIdx * 4];
        sortedGridValues2[i * 4 + 1] = this.originalGridValues2[srcIdx * 4 + 1];
        sortedGridValues2[i * 4 + 2] = this.originalGridValues2[srcIdx * 4 + 2];
        sortedGridValues2[i * 4 + 3] = this.originalGridValues2[srcIdx * 4 + 3];
      }
    }
    
    // Update GPU buffers
    gl.bindVertexArray(this.vao);
    
    // Update positions
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, sortedPositions, gl.STATIC_DRAW);
    
    // Update colors
    if (sortedColors && this.instanceColorBuffer) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, sortedColors, gl.STATIC_DRAW);
    }
    
    // Update scales
    if (sortedScales && this.instanceScaleBuffer) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceScaleBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, sortedScales, gl.STATIC_DRAW);
    }
    
    // Update grid values 1
    if (sortedGridValues1 && this.instanceGridValuesBuffer) {
      console.log('Updating grid values 1:', 
        sortedGridValues1[0], sortedGridValues1[1], 
        sortedGridValues1[2], sortedGridValues1[3]);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceGridValuesBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, sortedGridValues1, gl.STATIC_DRAW);
    }
    
    // Update grid values 2
    if (sortedGridValues2 && this.instanceDensityBuffer) {
      console.log('Updating grid values 2:', 
        sortedGridValues2[0], sortedGridValues2[1], 
        sortedGridValues2[2], sortedGridValues2[3]);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceDensityBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, sortedGridValues2, gl.STATIC_DRAW);
    }
    
    gl.bindVertexArray(null);
  }

  /**
   * Request voxel sorting based on current camera position
   */
  private requestSort(): void {
    if (!this.sortWorker || this.pendingSortRequest || this.instanceCount === 0 || !this.originalPositions) {
      return;
    }
    
    // Get camera position and target
    const cameraPos = this.camera.getPosition();
    const cameraTarget = this.camera.getTarget();
    
    // Send data to worker for sorting
    this.pendingSortRequest = true;
    
    // Clone positions to send to worker
    const positions = new Float32Array(this.originalPositions);
    
    // Apply the scene transformation to the positions - flip Y axis
    for (let i = 0; i < positions.length / 3; i++) {
      // Extract position
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      
      // Apply transformation (this just flips Y)
      positions[i * 3 + 1] = -y;
    }
    
    // Create copies of octree data to send to worker
    let octlevels: Uint8Array | undefined = undefined;
    let octpaths: Uint32Array | undefined = undefined;
    
    // Use the original octree data if available
    if (this.originalOctlevels) {
      octlevels = new Uint8Array(this.originalOctlevels);
      console.log(`Sending ${octlevels.length} octlevels to sort worker`);
    }
    
    if (this.originalOctpaths) {
      octpaths = new Uint32Array(this.originalOctpaths);
      console.log(`Sending ${octpaths.length} octpaths to sort worker`);
    }
    
    // Debug check of grid values
    if (this.originalGridValues1 && this.originalGridValues2) {
      console.log('Original grid values check:',
        this.originalGridValues1[0], this.originalGridValues1[1],
        this.originalGridValues2[0], this.originalGridValues2[1]);
    }
    
    // Send the data to the worker
    this.sortWorker.postMessage({
      type: 'sort',
      positions: positions,
      cameraPosition: cameraPos,
      cameraTarget: cameraTarget,
      octlevels: octlevels,
      octpaths: octpaths
    }, [positions.buffer]);
    
    // Transfer buffers to avoid copying large data
    if (octlevels) {
      this.sortWorker.postMessage({}, [octlevels.buffer]);
    }
    
    if (octpaths) {
      this.sortWorker.postMessage({}, [octpaths.buffer]);
    }
  }

  /**
   * React to camera movement
   */
  public handleCameraChange(): void {
    // Request a new sort when the camera changes
    this.requestSort();
  }

  /**
   * Initialize orbital camera controls
   */
  private initOrbitControls(): void {
    // Mouse down event
    this.canvas.addEventListener('mousedown', (event: MouseEvent) => {
      if (event.button === 0) { // Left click
        this.isDragging = true;
        this.isPanning = false;
      } else if (event.button === 2) { // Right click
        this.isPanning = true;
        this.isDragging = false;
        // Prevent context menu on right click
        event.preventDefault();
      }
      this.lastMouseX = event.clientX;
      this.lastMouseY = event.clientY;
    });
    
    // Mouse move event
    this.canvas.addEventListener('mousemove', (event: MouseEvent) => {
      if (!this.isDragging && !this.isPanning) return;
      
      const deltaX = event.clientX - this.lastMouseX;
      const deltaY = event.clientY - this.lastMouseY;
      
      if (this.isDragging) {
        // Orbit the camera
        this.orbit(deltaX, deltaY);
      } else if (this.isPanning) {
        // Pan the camera
        this.pan(deltaX, deltaY);
      }
      
      this.lastMouseX = event.clientX;
      this.lastMouseY = event.clientY;
      this.handleCameraChange();
    });
    
    // Mouse up event
    window.addEventListener('mouseup', () => {
      this.isDragging = false;
      this.isPanning = false;
    });
    
    // Mouse wheel event for zooming
    this.canvas.addEventListener('wheel', (event: WheelEvent) => {
      event.preventDefault();
      // Zoom in or out
      const zoomAmount = event.deltaY * this.zoomSpeed * 0.01;
      this.zoom(zoomAmount);
      this.handleCameraChange();
    });
    
    // Prevent context menu on right click
    this.canvas.addEventListener('contextmenu', (event) => {
      event.preventDefault();
    });
  }
  
  /**
   * Orbit the camera around the target
   */
  private orbit(deltaX: number, deltaY: number): void {
    const pos = this.camera.getPosition();
    const target = this.camera.getTarget();
    
    // Calculate the camera's current position relative to the target
    const relX = pos[0] - target[0];
    const relY = pos[1] - target[1];
    const relZ = pos[2] - target[2];
    
    // Calculate distance from target
    const distance = Math.sqrt(relX * relX + relY * relY + relZ * relZ);
    
    // Calculate current spherical coordinates
    let theta = Math.atan2(relX, relZ);
    let phi = Math.acos(relY / distance);
    
    // Update angles based on mouse movement - revert back to original approach
    theta -= deltaX * this.orbitSpeed;
    phi = Math.max(0.1, Math.min(Math.PI - 0.1, phi + deltaY * this.orbitSpeed)); // Back to original plus sign
    
    // Convert back to Cartesian coordinates
    const newRelX = distance * Math.sin(phi) * Math.sin(theta);
    const newRelY = distance * Math.cos(phi);
    const newRelZ = distance * Math.sin(phi) * Math.cos(theta);
    
    // Update camera position
    this.camera.setPosition(
      target[0] + newRelX,
      target[1] + newRelY,
      target[2] + newRelZ
    );
  }
  
  /**
   * Pan the camera (move target and camera together)
   */
  private pan(deltaX: number, deltaY: number): void {
    const pos = this.camera.getPosition();
    const target = this.camera.getTarget();
    
    // Calculate forward vector (from camera to target)
    const forwardX = target[0] - pos[0];
    const forwardY = target[1] - pos[1];
    const forwardZ = target[2] - pos[2];
    const forwardLength = Math.sqrt(forwardX * forwardX + forwardY * forwardY + forwardZ * forwardZ);
    
    // Normalize forward vector
    const forwardNormX = forwardX / forwardLength;
    const forwardNormY = forwardY / forwardLength;
    const forwardNormZ = forwardZ / forwardLength;
    
    // Calculate right vector (cross product of forward and up)
    // Back to original up vector
    const upX = 0, upY = 1, upZ = 0; // Standard world up vector
    const rightX = forwardNormZ * upY - forwardNormY * upZ;
    const rightY = forwardNormX * upZ - forwardNormZ * upX;
    const rightZ = forwardNormY * upX - forwardNormX * upY;
    
    // Calculate normalized up vector (cross product of right and forward)
    const upNormX = rightY * forwardNormZ - rightZ * forwardNormY;
    const upNormY = rightZ * forwardNormX - rightX * forwardNormZ;
    const upNormZ = rightX * forwardNormY - rightY * forwardNormX;
    
    // Calculate pan amounts based on delta
    const panAmount = this.panSpeed * Math.max(1, forwardLength / 10);
    const panX = -(rightX * -deltaX + upNormX * deltaY) * panAmount;
    const panY = -(rightY * -deltaX + upNormY * deltaY) * panAmount;
    const panZ = -(rightZ * -deltaX + upNormZ * deltaY) * panAmount;
    
    // Move both camera and target
    this.camera.setPosition(pos[0] + panX, pos[1] + panY, pos[2] + panZ);
    this.camera.setTarget(target[0] + panX, target[1] + panY, target[2] + panZ);
  }
  
  /**
   * Zoom the camera by adjusting distance to target
   */
  private zoom(zoomAmount: number): void {
    const pos = this.camera.getPosition();
    const target = this.camera.getTarget();
    
    // Calculate direction vector from target to camera
    const dirX = pos[0] - target[0];
    const dirY = pos[1] - target[1];
    const dirZ = pos[2] - target[2];
    
    // Get current distance
    const distance = Math.sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
    
    // Calculate new distance with zoom factor
    const newDistance = Math.max(0.1, distance * (1 + zoomAmount));
    
    // Calculate zoom ratio
    const ratio = newDistance / distance;
    
    // Update camera position
    this.camera.setPosition(
      target[0] + dirX * ratio,
      target[1] + dirY * ratio,
      target[2] + dirZ * ratio
    );
  }
}
