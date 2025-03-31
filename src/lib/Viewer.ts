/**
 * Viewer class for WebGL2 rendering with instanced cubes
 * Specialized for binary PLY point cloud visualization
 */
import { Camera } from './Camera';

enum TextureType {
  MainAttributes,
  GridValues,
  ShCoefficients
}


export class Viewer {
  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext | null;
  private program: WebGLProgram | null;
  private positionBuffer: WebGLBuffer | null = null;
  private indexBuffer: WebGLBuffer | null = null;
  private instanceBuffer: WebGLBuffer | null = null;
  private instanceCount: number = 10; // Number of instances to render
  private indexCount: number = 0;     // Number of indices in the cube geometry
  private vao: WebGLVertexArrayObject | null = null;
  private container: HTMLElement;
  private resizeObserver: ResizeObserver;
  
  // Camera
  private camera: Camera;
  
  // Animation
  private lastFrameTime: number = 0;
  
  // Rendering flags

  // Scene properties for scaling calculation
  private sceneCenter: [number, number, number] = [0, 0, 0];
  private sceneExtent: number = 1.0;
  private baseVoxelSize: number = 0.01;
  
  private lastCameraPosition: [number, number, number] = [0, 0, 0];
  private resortThreshold: number = 0.1; // Threshold for camera movement to trigger resort
  
  // Add these properties to the Viewer class definition at the top
  private sortWorker: Worker | null = null;
  private pendingSortRequest: boolean = false;
  private originalPositions: Float32Array | null = null;
  private originalSH0Values: Float32Array | null = null;
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
  
  // Add these properties to the Viewer class

  private originalSH1Values: Float32Array | null = null;

  private instanceIndexBuffer: WebGLBuffer | null = null;
  private sortedIndicesArray: Uint32Array | null = null;

  private textureWidth: number = 0;
  private textureHeight: number = 0;

  // Update class properties
  private posScaleTexture: WebGLTexture | null = null;      // pos + scale (4 values)
  private gridValuesTexture: WebGLTexture | null = null;    // grid values (8 values)
  private shTexture: WebGLTexture | null = null;            // sh0 + sh1 (4+8 = 12 values)

  // Add these properties to the class
  private posScaleWidth: number = 0;
  private posScaleHeight: number = 0;
  private gridValuesWidth: number = 0;
  private gridValuesHeight: number = 0;
  private shWidth: number = 0;
  private shHeight: number = 0;

  // Then modify the createDataTexture function to use an enum for texture types

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
    
    if (!this.gl.getExtension('EXT_color_buffer_float')) {
      console.error('EXT_color_buffer_float extension not supported');
    }
    
    // Initialize camera - revert to original positive Z position
    this.camera = new Camera();
    this.camera.setPosition(0, 0, 1); // Back to positive Z
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
    
    // Updated vertex shader to use texture fetches
    const vsSource = `#version 300 es
      precision highp float;
      precision highp sampler2D;
      
      // Core attributes
      in vec4 aVertexPosition;
      in uint aInstanceIndex;
      
      // Uniforms for matrices and camera
      uniform mat4 uProjectionMatrix;
      uniform mat4 uViewMatrix;
      uniform mat4 uSceneTransformMatrix;
      uniform vec3 uCameraPosition;

      // Uniforms for textures
      uniform sampler2D uPosScaleTexture;
      uniform sampler2D uGridValuesTexture;
      uniform sampler2D uShTexture;

      // Texture dimensions
      uniform ivec2 uPosScaleDims;
      uniform ivec2 uGridValuesDims;
      uniform ivec2 uShDims;
      
      // Outputs to fragment shader
      out vec3 vWorldPos;
      out float vScale;
      out vec3 vVoxelCenter;     
      out vec4 vDensity0;        
      out vec4 vDensity1;
      out vec3 vColor;           
      
      // Helper function to calculate texture coordinate
      vec2 getTexCoord(int instanceIdx, int offsetIdx, ivec2 dims, int vec4sPerInstance) {
        int texelIdx = instanceIdx * vec4sPerInstance + offsetIdx;
        int x = texelIdx % dims.x;
        int y = texelIdx / dims.x;
        return (vec2(x, y) + 0.5) / vec2(dims);
      }

      // Helper functions to fetch data
      vec4 fetch4(sampler2D tex, int idx, int offsetIdx, ivec2 dims, int vec4sPerInstance) {
        vec2 coord = getTexCoord(idx, offsetIdx, dims, vec4sPerInstance);
        return texture(tex, coord);
      }

      // Spherical Harmonics evaluation with full 9 coefficients (moved from fragment shader)
      vec3 evaluateSH(vec4 sh0, vec3 sh1_0, vec3 sh1_1, vec3 sh1_2, vec3 direction) {
        // Normalize direction vector
        vec3 dir = normalize(direction);

        // TODO: Remove this once we have a proper coordinate system
        dir.y = -dir.y;
        
        // SH0
        vec3 color = sh0.rgb * 0.28209479177387814;
        
        // Calculate basis functions for SH1 (first order terms only)
        // Y_1,-1 = 0.488603 * y
        // Y_1,0  = 0.488603 * z
        // Y_1,1  = 0.488603 * x
        float basis_y = 0.488603 * dir.y;
        float basis_z = 0.488603 * dir.z;
        float basis_x = 0.488603 * dir.x;
        
        vec3 sh1_contrib = vec3(0);
        // Apply SH1 coefficients per color channel
        sh1_contrib += sh1_0 * basis_x;
        sh1_contrib += sh1_1 * basis_y;
        sh1_contrib += sh1_2 * basis_z;
        
        color += sh1_contrib;
        color += 0.5;
        
        
        return max(color, 0.0);
      }
      

      void main() {
        // Get the index for this instance
        int idx = int(aInstanceIndex);
        
        // Fetch position and scale (1 vec4 per instance)
        vec4 posAndScale = fetch4(uPosScaleTexture, idx, 0, uPosScaleDims, 1);
        vec3 instancePosition = posAndScale.xyz;
        float instanceScale = posAndScale.w;
        
        // Fetch grid values (2 vec4s per instance)
        vec4 gridValues1 = fetch4(uGridValuesTexture, idx, 0, uGridValuesDims, 2);
        vec4 gridValues2 = fetch4(uGridValuesTexture, idx, 1, uGridValuesDims, 2);
        
        // Fetch SH values (3 vec4s per instance)
        vec4 sh0 = fetch4(uShTexture, idx, 0, uShDims, 3);
        vec4 sh1_part1 = fetch4(uShTexture, idx, 1, uShDims, 3);
        vec4 sh1_part2 = fetch4(uShTexture, idx, 2, uShDims, 3);
        
        // Extract SH1 coefficients
        vec3 sh1_0 = sh1_part1.xyz;
        vec3 sh1_1 = vec3(sh1_part1.w, sh1_part2.xy);
        vec3 sh1_2 = vec3(sh1_part2.zw, 0.0); // Fix syntax error and only use 8/9 SH values
    
        // Scale the vertex position by instance scale
        vec4 scaledPosition = vec4(aVertexPosition.xyz * instanceScale, aVertexPosition.w);
        
        // Transform instance offset by scene transform matrix
        vec4 transformedOffset = uSceneTransformMatrix * vec4(instancePosition, 1.0);
        
        // Position is scaled vertex position + transformed instance offset
        vec4 instancePos = scaledPosition + vec4(transformedOffset.xyz, 0.0);
        
        // Calculate final position
        gl_Position = uProjectionMatrix * uViewMatrix * instancePos;
        
        // Pass world position of the vertex
        vWorldPos = instancePos.xyz;
        
        // Calculate and pass voxel center
        vVoxelCenter = transformedOffset.xyz;
        
        // Pass scale to fragment shader
        vScale = instanceScale;
        
        // Calculate viewing direction from voxel center to camera
        vec3 viewDir = normalize(vVoxelCenter - uCameraPosition);
        
        // Calculate color using SH and pass to fragment shader
        vColor = evaluateSH(sh0, sh1_0, sh1_1, sh1_2, viewDir);
        
        // Pass density values to fragment shader
        vDensity0 = gridValues1;
        vDensity1 = gridValues2;
      }
    `;
    
    // Updated fragment shader with debug output
    const fsSource = `#version 300 es

      precision highp float;  // Changed from mediump to highp to match vertex shader
      
      in vec3 vWorldPos;
      in float vScale;
      in vec3 vVoxelCenter;
      in vec4 vDensity0;         // Density values for corners 0-3
      in vec4 vDensity1;         // Density values for corners 4-7
      in vec3 vColor;            // Pre-calculated color from vertex shader
      
      uniform vec3 uCameraPosition;
      uniform mat4 uViewMatrix;
      
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
        
        // If camera is inside the box, tNear will be negative
        // In this case, we should start sampling from the camera position
        tNear = max(0.0, tNear);
        
        return vec2(tNear, tFar);
      }
      
      // Updated trilinear interpolation function
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
        float fy = normPos.y;
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
      
      float explin(float x) {
        float threshold = 1.1;
        if (x > threshold) {
          return x;
        } else {
          float ln1_1 = 0.0953101798043; // pre-computed ln(1.1)
          return exp((x / threshold) - 1.0 + ln1_1);
        }
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
          
          // Calculate entry and exit points in world space
          vec3 entryPoint = rayOrigin + rayDir * tNear;
          vec3 exitPoint = rayOrigin + rayDir * tFar;
          
          // Transform to view space
          vec4 entryPointView = uViewMatrix * vec4(entryPoint, 1.0);
          vec4 exitPointView = uViewMatrix * vec4(exitPoint, 1.0);
          
          // Calculate ray length in view space
          float viewSpaceRayLength = distance(entryPointView.xyz, exitPointView.xyz);
          float stepLength = viewSpaceRayLength / 3.0;
          
          // Get raw interpolated densities
          vec3 samplePoint1 = rayOrigin + rayDir * (tNear + (tFar - tNear) * 0.25);
          float rawDensity1 = trilinearInterpolation(samplePoint1, boxMin, boxMax, vDensity0, vDensity1);
          
          vec3 samplePoint2 = rayOrigin + rayDir * (tNear + (tFar - tNear) * 0.5);
          float rawDensity2 = trilinearInterpolation(samplePoint2, boxMin, boxMax, vDensity0, vDensity1);
          
          vec3 samplePoint3 = rayOrigin + rayDir * (tNear + (tFar - tNear) * 0.75);
          float rawDensity3 = trilinearInterpolation(samplePoint3, boxMin, boxMax, vDensity0, vDensity1);
          
          
          // Apply explin after interpolation
          // I'm not sure why, but the CUDA reference has a 100x scale factor.
          const float STEP_SCALE = 100.0;
          float density1 = STEP_SCALE * stepLength * explin(rawDensity1);
          float density2 = STEP_SCALE * stepLength * explin(rawDensity2);
          float density3 = STEP_SCALE * stepLength * explin(rawDensity3);

          float totalDensity = density1 + density2 + density3;
                    
          // Use view space ray length for Beer-Lambert law
          float alpha = 1.0 - exp(-totalDensity);
          
          // Premultiply the color by alpha
          vec3 premultipliedColor = vColor * alpha;
          fragColor = vec4(premultipliedColor, alpha);
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

  
  private initBuffers(): void {
    const gl = this.gl!;
    
    // Create a vertex array object (VAO)
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    
    // Initialize cube geometry for instancing
    this.initCubeGeometry(1); 
    
    // Create and initialize the instance index buffer (this is all we need now for instancing)
    const instanceIndices = new Uint32Array(this.instanceCount);
    for (let i = 0; i < this.instanceCount; i++) {
      instanceIndices[i] = i;
    }
    
    // Create and fill the instance index buffer
    this.instanceIndexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceIndexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, instanceIndices, gl.DYNAMIC_DRAW);
    
    // Get attribute location for instance index
    const instanceIndexLocation = gl.getAttribLocation(this.program!, 'aInstanceIndex');
    console.log('Instance index attribute location:', instanceIndexLocation);
    
    // Only set up instance index attribute if it exists in the shader
    if (instanceIndexLocation !== -1) {
      gl.enableVertexAttribArray(instanceIndexLocation);
      gl.vertexAttribIPointer(
        instanceIndexLocation,
        1,               // 1 component per instance index
        gl.UNSIGNED_INT, // data type
        0,               // stride
        0                // offset
      );
      gl.vertexAttribDivisor(instanceIndexLocation, 1);
    } else {
      console.error('Could not find aInstanceIndex attribute in shader');
    }
    
    // Unbind the VAO
    gl.bindVertexArray(null);
  }
  
  /**
   * Initialize cube geometry for instance rendering
   */
  private initCubeGeometry(size: number): void {
    const gl = this.gl!;
    
    // Create cube geometry
    const halfSize = size / 2;
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
    
    // Get attribute location for vertex position
    const positionAttributeLocation = gl.getAttribLocation(this.program!, 'aVertexPosition');
    
    if (positionAttributeLocation !== -1) {
      gl.enableVertexAttribArray(positionAttributeLocation);
      gl.vertexAttribPointer(
        positionAttributeLocation,
        3,        // 3 components per vertex
        gl.FLOAT, // data type
        false,    // no normalization
        0,        // stride
        0         // offset
      );
    } else {
      console.error('Could not find aVertexPosition attribute in shader');
    }
    
    // Indices as before
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
    // gl.clearColor(1.0 / 255.0, 121.0 / 255.0, 51.0 / 255.0, 1.0); 
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    
    // Ensure blending is properly set up
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);


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
    const cameraPositionLocation = gl.getUniformLocation(this.program, 'uCameraPosition');
    
    // Pass matrices to shader
    gl.uniformMatrix4fv(projectionMatrixLocation, false, this.camera.getProjectionMatrix());
    gl.uniformMatrix4fv(viewMatrixLocation, false, this.camera.getViewMatrix());
    gl.uniformMatrix4fv(sceneTransformMatrixLocation, false, this.sceneTransformMatrix);
    
    // Pass camera position to the shader
    gl.uniform3f(cameraPositionLocation, cameraPos[0], cameraPos[1], cameraPos[2]);
    
    // Set texture uniforms
    const textureWidthLocation = gl.getUniformLocation(this.program, 'uTextureWidth');
    const textureHeightLocation = gl.getUniformLocation(this.program, 'uTextureHeight');
    gl.uniform1i(textureWidthLocation, this.textureWidth);
    gl.uniform1i(textureHeightLocation, this.textureHeight);
    
    // Bind textures and set uniforms
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.posScaleTexture);
    gl.uniform1i(gl.getUniformLocation(this.program!, 'uPosScaleTexture'), 0);
    gl.uniform2i(gl.getUniformLocation(this.program!, 'uPosScaleDims'), 
      this.posScaleWidth, this.posScaleHeight);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.gridValuesTexture);
    gl.uniform1i(gl.getUniformLocation(this.program!, 'uGridValuesTexture'), 1);
    gl.uniform2i(gl.getUniformLocation(this.program!, 'uGridValuesDims'), 
      this.gridValuesWidth, this.gridValuesHeight);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.shTexture);
    gl.uniform1i(gl.getUniformLocation(this.program!, 'uShTexture'), 2);
    gl.uniform2i(gl.getUniformLocation(this.program!, 'uShDims'), 
      this.shWidth, this.shHeight);
    
    // Draw instanced geometry
    console.log('Drawing with instance count:', this.instanceCount);
    if (this.instanceCount <= 0) {
        console.warn('No instances to draw');
        requestAnimationFrame((time) => this.render(time));
        return;
    }
    
    gl.drawElementsInstanced(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0, this.instanceCount);
    
    // Check for GL errors
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
      console.error(`WebGL error: ${error}`);
    }
    
    // Unbind the VAO
    gl.bindVertexArray(null);
    
    // Debug: Log SH1 values periodically to check if they're being used
    if (this.lastFrameTime === 0 && this.originalSH1Values) {
      const nonZeroCount = Array.from(this.originalSH1Values).filter(v => Math.abs(v) > 0.0001).length;
      console.log(`SH1 values check: ${nonZeroCount} non-zero values out of ${this.originalSH1Values.length}`);
      if (nonZeroCount > 0) {
        console.log('SH1 values are present and will affect rendering');
      } else {
        console.log('Warning: All SH1 values are near zero, no directional lighting effect will be visible');
      }
    }
    
    // Request animation frame for continuous rendering
    requestAnimationFrame((time) => this.render(time));
  }
  
  /**
   * Load a point cloud from positions and colors
   */
  public loadPointCloud(
    positions: Float32Array, 
    sh0Values?: Float32Array,
    octlevels?: Uint8Array,
    octpaths?: Uint32Array,
    gridValues?: Float32Array,
    shRestValues?: Float32Array
  ): void {
    console.log(`Loading point cloud with ${positions.length / 3} points`);
    
    // Save original data (we still need it for the sort worker)
    this.originalPositions = new Float32Array(positions);
    
    // Save SH0 (base colors)
    if (sh0Values) {
      this.originalSH0Values = new Float32Array(sh0Values);
    } else {
      this.originalSH0Values = new Float32Array(positions.length / 3 * 4);
      for (let i = 0; i < positions.length / 3; i++) {
        this.originalSH0Values[i * 4 + 0] = 1.0; // R
        this.originalSH0Values[i * 4 + 1] = 1.0; // G
        this.originalSH0Values[i * 4 + 2] = 1.0; // B
        this.originalSH0Values[i * 4 + 3] = 1.0; // A
      }
    }
    
    // Extract SH1 coefficients from shRestValues
    if (shRestValues && shRestValues.length > 0) {
      // Each vertex has multiple rest values, we need to extract 9 values for SH1
      const restPerVertex = shRestValues.length / positions.length * 3;
      console.log(`Found ${restPerVertex} rest values per vertex, extracting SH1 (9 values per vertex)`);
      
      // We need space for 9 values per vertex
      this.originalSH1Values = new Float32Array(positions.length / 3 * 9);
      
      for (let i = 0; i < positions.length / 3; i++) {
        // Extract 9 values from shRestValues for each vertex
        for (let j = 0; j < 9; j++) {
          // Only extract if we have enough values
          if (j < restPerVertex) {
            this.originalSH1Values[i * 9 + j] = shRestValues[i * restPerVertex + j];
          } else {
            // If not enough rest values, set to 0
            this.originalSH1Values[i * 9 + j] = 0.0;
          }
        }
      }
      
      console.log(`Extracted ${this.originalSH1Values.length / 9} SH1 sets with 9 values each`);
      console.log('SH1 sample values (first vertex):', 
                  this.originalSH1Values.slice(0, 9));
    } else {
      // If no rest values, use zeros (no directional lighting)
      this.originalSH1Values = new Float32Array(positions.length / 3 * 9);
      console.log('No SH1 values found, using default (no directional lighting)');
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
    
    // Set the instance count
    this.instanceCount = positions.length / 3;
    
    // Initialize initial indices (sequential ordering)
    this.sortedIndicesArray = new Uint32Array(this.instanceCount);
    for (let i = 0; i < this.instanceCount; i++) {
      this.sortedIndicesArray[i] = i;
    }
    
    // Initialize WebGL resources
    const gl = this.gl!;
    
    // Create VAO
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    
    // Initialize cube geometry
    this.initCubeGeometry(1.0);
    
    // Create and upload index buffer
    this.instanceIndexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceIndexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.sortedIndicesArray, gl.DYNAMIC_DRAW);
    
    const instanceIndexLocation = gl.getAttribLocation(this.program!, 'aInstanceIndex');
    gl.enableVertexAttribArray(instanceIndexLocation);
    gl.vertexAttribIPointer(
      instanceIndexLocation,
      1,
      gl.UNSIGNED_INT,
      0,
      0
    );
    gl.vertexAttribDivisor(instanceIndexLocation, 1);
    
    // Create optimized textures instead of individual texture creation
    this.createOptimizedTextures();
    
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
      // Delete textures
      if (this.positionsTexture) gl.deleteTexture(this.positionsTexture);
      if (this.sh0Texture) gl.deleteTexture(this.sh0Texture);
      if (this.scalesTexture) gl.deleteTexture(this.scalesTexture);
      if (this.gridValues1Texture) gl.deleteTexture(this.gridValues1Texture);
      if (this.gridValues2Texture) gl.deleteTexture(this.gridValues2Texture);
      if (this.sh1_0Texture) gl.deleteTexture(this.sh1_0Texture);
      if (this.sh1_1Texture) gl.deleteTexture(this.sh1_1Texture);
      if (this.sh1_2Texture) gl.deleteTexture(this.sh1_2Texture);
      
      // Delete buffers
      if (this.positionBuffer) gl.deleteBuffer(this.positionBuffer);
      if (this.indexBuffer) gl.deleteBuffer(this.indexBuffer);
      if (this.instanceBuffer) gl.deleteBuffer(this.instanceBuffer);

      if (this.instanceIndexBuffer) gl.deleteBuffer(this.instanceIndexBuffer);
      
      // Delete VAO
      if (this.vao) gl.deleteVertexArray(this.vao);
      
      // Delete program and shaders
      if (this.program) gl.deleteProgram(this.program);
    }
    
    // Clear reference data
    this.originalPositions = null;
    this.originalSH0Values = null;
    this.originalScales = null;
    this.originalGridValues1 = null;
    this.originalGridValues2 = null;
    this.originalOctlevels = null;
    this.originalOctpaths = null;
    this.originalSH1Values = null;
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
    if (!this.sortedIndices || this.sortedIndices.length === 0) {
      console.warn('Missing indices for sorting');
      return;
    }
    
    console.log('Applying sort order with indices');

    const gl = this.gl!;
    
    // Simply update the index buffer with the new sorted indices
    this.sortedIndicesArray = new Uint32Array(this.sortedIndices);
    
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceIndexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.sortedIndicesArray, gl.DYNAMIC_DRAW);
    gl.bindVertexArray(null);
    
    // Check for GL errors
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
      console.error('WebGL error during index buffer update:', error);
    }
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

  /**
   * Creates a texture from float data
   */
  private createDataTexture(
    data: Float32Array, 
    componentsPerElement: number, 
    textureType: TextureType
  ): WebGLTexture | null {
    const gl = this.gl!;
    
    // Calculate dimensions that won't exceed hardware limits
    const numElements = data.length / componentsPerElement;
    const maxTextureSize = Math.min(4096, gl.getParameter(gl.MAX_TEXTURE_SIZE));
    const width = Math.min(maxTextureSize, Math.ceil(Math.sqrt(numElements)));
    const height = Math.ceil(numElements / width);
    
    // Store dimensions based on texture type
    const paddedWidth = Math.ceil(width * componentsPerElement / 4) * 4;
    const finalWidth = paddedWidth / 4;
    
    switch (textureType) {
      case TextureType.MainAttributes:
        this.posScaleWidth = finalWidth;
        this.posScaleHeight = height;
        console.log(`Creating main attributes texture: ${finalWidth}x${height}`);
        break;
      case TextureType.GridValues:
        this.gridValuesWidth = finalWidth;
        this.gridValuesHeight = height;
        console.log(`Creating grid values texture: ${finalWidth}x${height}`);
        break;
      case TextureType.ShCoefficients:
        this.shWidth = finalWidth;
        this.shHeight = height;
        console.log(`Creating SH coefficients texture: ${finalWidth}x${height}`);
        break;
    }
    
    // Create and set up texture
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    
    // Create padded data array
    const paddedData = new Float32Array(paddedWidth * height);
    paddedData.set(data);
    
    // Upload data to texture
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA32F,
      finalWidth,
      height,
      0,
      gl.RGBA,
      gl.FLOAT,
      paddedData
    );
    
    // Set texture parameters
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    
    return texture;
  }

  private createOptimizedTextures(): void {
    // First, check if all the data is available
    if (!this.originalPositions || !this.originalSH0Values || !this.originalScales || 
        !this.originalGridValues1 || !this.originalGridValues2 || !this.originalSH1Values) {
      console.error('Missing required data for texture creation');
      return;
    }
    
    // Get the WebGL context
    const gl = this.gl!;
    if (!gl) {
      console.error('WebGL context is null');
      return;
    }
    
    // 1. Create position + scale texture (4 values per instance)
    const posScaleData = new Float32Array(this.instanceCount * 4);
    for (let i = 0; i < this.instanceCount; i++) {
      // xyz position
      posScaleData[i * 4 + 0] = this.originalPositions[i * 3 + 0];
      posScaleData[i * 4 + 1] = this.originalPositions[i * 3 + 1];
      posScaleData[i * 4 + 2] = this.originalPositions[i * 3 + 2];
      
      // scale
      posScaleData[i * 4 + 3] = this.originalScales ? 
        this.originalScales[i] : this.baseVoxelSize;
    }
    
    // 2. Create grid values texture (8 values per instance = 2 vec4s)
    const gridValuesData = new Float32Array(this.instanceCount * 8);
    for (let i = 0; i < this.instanceCount; i++) {
      // First 4 corners
      gridValuesData[i * 8 + 0] = this.originalGridValues1[i * 4 + 0];
      gridValuesData[i * 8 + 1] = this.originalGridValues1[i * 4 + 1];
      gridValuesData[i * 8 + 2] = this.originalGridValues1[i * 4 + 2];
      gridValuesData[i * 8 + 3] = this.originalGridValues1[i * 4 + 3];
      
      // Second 4 corners
      gridValuesData[i * 8 + 4] = this.originalGridValues2[i * 4 + 0];
      gridValuesData[i * 8 + 5] = this.originalGridValues2[i * 4 + 1];
      gridValuesData[i * 8 + 6] = this.originalGridValues2[i * 4 + 2];
      gridValuesData[i * 8 + 7] = this.originalGridValues2[i * 4 + 3];
    }
    
    // 3. Create SH0 + SH1 texture (4+8=12 values per instance = 3 vec4s)
    const shData = new Float32Array(this.instanceCount * 12);
    for (let i = 0; i < this.instanceCount; i++) {
      // SH0 (rgba)
      shData[i * 12 + 0] = this.originalSH0Values[i * 4 + 0];
      shData[i * 12 + 1] = this.originalSH0Values[i * 4 + 1];
      shData[i * 12 + 2] = this.originalSH0Values[i * 4 + 2];
      shData[i * 12 + 3] = this.originalSH0Values[i * 4 + 3];
      
      // SH1 (first 8 values out of 9, we'll drop the last one or use 0)
      for (let j = 0; j < 8; j++) {
        if (j < this.originalSH1Values.length / this.instanceCount) {
          shData[i * 12 + 4 + j] = this.originalSH1Values[i * 9 + j];
        } else {
          shData[i * 12 + 4 + j] = 0.0;
        }
      }
    }
    
    // Create textures using the existing createDataTexture method with the correct parameters
    this.posScaleTexture = this.createDataTexture(
      posScaleData, 
      4,  // 4 components per element (exactly one vec4)
      TextureType.MainAttributes
    );
    
    this.gridValuesTexture = this.createDataTexture(
      gridValuesData, 
      8,  // 8 components per element (2 vec4s)
      TextureType.GridValues
    );
    
    this.shTexture = this.createDataTexture(
      shData, 
      12, // 12 components per element (3 vec4s)
      TextureType.ShCoefficients
    );
  }
}
