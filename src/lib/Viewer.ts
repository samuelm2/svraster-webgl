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
    
    // Updated vertex shader to include scene transformation matrix
    const vsSource = `#version 300 es
      in vec4 aVertexPosition;
      in vec4 aVertexColor;
      in vec4 aInstanceOffset;
      in vec4 aInstanceColor;
      in float aInstanceScale;   // Scale for the voxel
      
      uniform mat4 uProjectionMatrix;
      uniform mat4 uViewMatrix;
      uniform mat4 uSceneTransformMatrix; // Add scene transform matrix uniform
      uniform bool uUseInstanceColors;
      
      out vec4 vColor;
      out vec3 vWorldPos;     // World position of the vertex
      out float vScale;       // Pass scale to fragment shader
      out vec3 vVoxelCenter;  // Center of the voxel in world space
      
      void main() {
        // Scale the vertex position by instance scale
        vec4 scaledPosition = vec4(aVertexPosition.xyz * aInstanceScale, aVertexPosition.w);
        
        // Transform instance offset by scene transform matrix
        vec4 transformedOffset = uSceneTransformMatrix * vec4(aInstanceOffset.xyz, 1.0);
        
        // Position is scaled vertex position + transformed instance offset
        vec4 instancePosition = scaledPosition + vec4(transformedOffset.xyz, 0.0);
        
        // Calculate final position
        gl_Position = uProjectionMatrix * uViewMatrix * instancePosition;
        
        // Pass world position of the vertex - use transformed position
        vWorldPos = instancePosition.xyz;
        
        // Calculate and pass transformed voxel center
        vVoxelCenter = transformedOffset.xyz;
        
        // Pass scale to fragment shader
        vScale = aInstanceScale;
        
        // Use instance color if enabled, otherwise use vertex color
        vColor = uUseInstanceColors ? aInstanceColor : aVertexColor;
      }
    `;
    
    // Fragment shader with proper ray-box intersection
    const fsSource = `#version 300 es
      precision mediump float;
      
      in vec4 vColor;
      in vec3 vWorldPos;      // World position of the vertex
      in float vScale;        // Scale of the voxel
      in vec3 vVoxelCenter;   // Center of the voxel in world space
      
      uniform vec3 uCameraPosition; // Camera position in world space
      
      out vec4 fragColor;
      
      // Ray-box intersection function in world space
      // Returns entry and exit t values
      vec2 rayBoxIntersection(vec3 rayOrigin, vec3 rayDir, vec3 boxCenter, float boxScale) {
        // Calculate box min and max in world space
        vec3 boxMin = boxCenter - vec3(boxScale * 0.5);
        vec3 boxMax = boxCenter + vec3(boxScale * 0.5);
        
        // Standard ray-box intersection algorithm
        vec3 invDir = 1.0 / rayDir;
        vec3 tMin = (boxMin - rayOrigin) * invDir;
        vec3 tMax = (boxMax - rayOrigin) * invDir;
        vec3 t1 = min(tMin, tMax);
        vec3 t2 = max(tMin, tMax);
        float tNear = max(max(t1.x, t1.y), t1.z);
        float tFar = min(min(t2.x, t2.y), t2.z);
        return vec2(tNear, tFar);
      }
      
      void main() {
        // Calculate ray from camera to this fragment in world space
        vec3 rayOrigin = uCameraPosition;
        vec3 rayDir = normalize(vWorldPos - uCameraPosition);
        
        // Get ray-box intersection with the voxel in world space
        vec2 tIntersect = rayBoxIntersection(rayOrigin, rayDir, vVoxelCenter, vScale);
        float tNear = max(0.0, tIntersect.x);
        float tFar = min(tIntersect.y, 1000.0);  // Limit far intersection for safety
        
        // If ray intersects the box
        if (tNear < tFar) {
          // Calculate intersection length (affects opacity)
          float intersectionLength = tFar - tNear;
          
          // Scale opacity based on intersection length and voxel scale
          // Smaller voxels need higher opacity to be visible
          float baseOpacity = 0.8;
          float relativeSize = vScale / 0.1; // Compare to a reference size
          float sizeScale = clamp(1.0 / relativeSize, 0.2, 5.0);
          
          // Calculate alpha based on both intersection length and voxel size
          float alpha = baseOpacity * min(intersectionLength * sizeScale, 1.0);
          
          // Output color with calculated alpha
          fragColor = vec4(vColor.rgb, alpha);
        } else {
          // No intersection, discard the fragment
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
    
    // Always save colors from PLY
    if (colors) {
      console.log("Saving colors from PLY file");
      this.originalColors = new Float32Array(colors);
    } else {
      // If no colors provided, create white voxels
      console.log("No colors in PLY, using white");
      this.originalColors = new Float32Array(positions.length / 3 * 4);
      for (let i = 0; i < positions.length / 3; i++) {
        this.originalColors[i * 4 + 0] = 1.0; // R
        this.originalColors[i * 4 + 1] = 1.0; // G
        this.originalColors[i * 4 + 2] = 1.0; // B
        this.originalColors[i * 4 + 3] = 1.0; // A
      }
    }
    
    // IMPORTANT: Save the original octree data
    if (octlevels) {
      console.log(`Saving ${octlevels.length} octlevels`);
      this.originalOctlevels = new Uint8Array(octlevels);
      
      // Also derive scales for rendering
      this.originalScales = new Float32Array(octlevels.length);
      for (let i = 0; i < octlevels.length; i++) {
        this.originalScales[i] = this.baseVoxelSize * Math.pow(2, -octlevels[i]);
      }
    }
    
    if (octpaths) {
      console.log(`Saving ${octpaths.length} octpaths`);
      this.originalOctpaths = new Uint32Array(octpaths);
    }
    
    // Call the original implementation
    const gl = this.gl!;
    
    // Create a VAO for the point cloud
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    
    // Initialize cube geometry
    const cubeSize = 1.0; // Base size that will be scaled
    this.initCubeGeometry(cubeSize);
    
    // Store the instance positions
    this.instanceBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    
    // Set up instance position attribute
    const instanceAttributeLocation = gl.getAttribLocation(this.program!, 'aInstanceOffset');
    gl.enableVertexAttribArray(instanceAttributeLocation);
    gl.vertexAttribPointer(
      instanceAttributeLocation,
      3,        // 3 components per position (x, y, z)
      gl.FLOAT, // data type
      false,    // no normalization
      0,        // stride
      0         // offset
    );
    
    // Enable instancing
    gl.vertexAttribDivisor(instanceAttributeLocation, 1);
    
    // Always setup the instance color buffer
    this.instanceColorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.originalColors, gl.STATIC_DRAW);
    
    // Set up instance color attribute
    const instanceColorLocation = gl.getAttribLocation(this.program!, 'aInstanceColor');
    gl.enableVertexAttribArray(instanceColorLocation);
    gl.vertexAttribPointer(
      instanceColorLocation,
      4,        // 4 components per color (RGBA)
      gl.FLOAT, // data type
      false,    // no normalization
      0,        // stride
      0         // offset
    );
    
    // Enable instancing for colors
    gl.vertexAttribDivisor(instanceColorLocation, 1);
    
    // Set the instance count to the number of points
    this.instanceCount = positions.length / 3;
    
    // IMPORTANT: Create and set up scale buffer for varied voxel sizes
    if (this.originalOctlevels) {
      console.log("Setting up scale buffer from octlevels");
      
      // Get scales from octlevels
      const scales = new Float32Array(this.instanceCount);
      for (let i = 0; i < this.instanceCount; i++) {
        scales[i] = this.baseVoxelSize * Math.pow(2, -this.originalOctlevels[i]);
      }
      
      // Create scale buffer
      this.instanceScaleBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceScaleBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, scales, gl.STATIC_DRAW);
      
      // Set up instance scale attribute
      const instanceScaleLocation = gl.getAttribLocation(this.program!, 'aInstanceScale');
      if (instanceScaleLocation !== -1) {
        gl.enableVertexAttribArray(instanceScaleLocation);
        gl.vertexAttribPointer(
          instanceScaleLocation,
          1,        // 1 component per scale
          gl.FLOAT, // data type
          false,    // no normalization
          0,        // stride
          0         // offset
        );
        
        // Enable instancing for scales
        gl.vertexAttribDivisor(instanceScaleLocation, 1);
      } else {
        console.error("Could not get aInstanceScale attribute location");
      }
    } else {
      // If no octlevels, use a default scale
      console.warn("No octlevels available, using default scale");
      
      // Create a buffer with all the same scale
      const scales = new Float32Array(this.instanceCount);
      for (let i = 0; i < this.instanceCount; i++) {
        scales[i] = this.baseVoxelSize;
      }
      
      // Create scale buffer
      this.instanceScaleBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceScaleBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, scales, gl.STATIC_DRAW);
      
      // Set up instance scale attribute
      const instanceScaleLocation = gl.getAttribLocation(this.program!, 'aInstanceScale');
      if (instanceScaleLocation !== -1) {
        gl.enableVertexAttribArray(instanceScaleLocation);
        gl.vertexAttribPointer(
          instanceScaleLocation,
          1,        // 1 component per scale
          gl.FLOAT, // data type
          false,    // no normalization
          0,        // stride
          0         // offset
        );
        
        // Enable instancing for scales
        gl.vertexAttribDivisor(instanceScaleLocation, 1);
      } else {
        console.error("Could not get aInstanceScale attribute location");
      }
    }
    
    // Unbind the VAO
    gl.bindVertexArray(null);
    
    // Debug: Check the final state
    console.log('Point cloud loaded, requesting initial sort');
    console.log({
      vao: !!this.vao,
      program: !!this.program,
      instanceBuffer: !!this.instanceBuffer,
      instanceCount: this.instanceCount,
      sortWorker: !!this.sortWorker
    });
    
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
    
    // Create sorted arrays for all instance attributes
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
    
    // Reorder the data based on indices
    for (let i = 0; i < this.instanceCount; i++) {
      const srcIdx = this.sortedIndices[i];
      
      // Reorder positions
      sortedPositions[i * 3] = this.originalPositions[srcIdx * 3];
      sortedPositions[i * 3 + 1] = this.originalPositions[srcIdx * 3 + 1];
      sortedPositions[i * 3 + 2] = this.originalPositions[srcIdx * 3 + 2];
      
      // Reorder colors if present
      if (sortedColors && this.originalColors) {
        sortedColors[i * 4] = this.originalColors[srcIdx * 4];
        sortedColors[i * 4 + 1] = this.originalColors[srcIdx * 4 + 1];
        sortedColors[i * 4 + 2] = this.originalColors[srcIdx * 4 + 2];
        sortedColors[i * 4 + 3] = this.originalColors[srcIdx * 4 + 3];
      }
      
      // Reorder scales if present
      if (sortedScales && this.originalScales) {
        sortedScales[i] = this.originalScales[srcIdx];
      }
      
      // Reorder grid values if present
      if (sortedGridValues1 && this.originalGridValues1) {
        sortedGridValues1[i * 4] = this.originalGridValues1[srcIdx * 4];
        sortedGridValues1[i * 4 + 1] = this.originalGridValues1[srcIdx * 4 + 1];
        sortedGridValues1[i * 4 + 2] = this.originalGridValues1[srcIdx * 4 + 2];
        sortedGridValues1[i * 4 + 3] = this.originalGridValues1[srcIdx * 4 + 3];
      }
      
      if (sortedGridValues2 && this.originalGridValues2) {
        sortedGridValues2[i * 4] = this.originalGridValues2[srcIdx * 4];
        sortedGridValues2[i * 4 + 1] = this.originalGridValues2[srcIdx * 4 + 1];
        sortedGridValues2[i * 4 + 2] = this.originalGridValues2[srcIdx * 4 + 2];
        sortedGridValues2[i * 4 + 3] = this.originalGridValues2[srcIdx * 4 + 3];
      }
    }
    
    // Update the GPU buffers with the sorted data
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
    if (sortedScales && this.instanceScaleBuffer && this.originalScales) {
      console.log("Updating scale buffer with sorted values");
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceScaleBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, sortedScales, gl.STATIC_DRAW);
    }
    
    // Update grid values
    if (sortedGridValues1 && this.instanceGridValuesBuffer) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceGridValuesBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, sortedGridValues1, gl.STATIC_DRAW);
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
