/**
 * Viewer class for WebGL2 rendering with instanced cubes
 * Specialized for binary PLY point cloud visualization
 */
import { Camera } from './Camera';
import { mat4, vec3 } from 'gl-matrix';

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

  // Scene properties for scaling calculation

  private baseVoxelSize: number = 0.01;
  
  private lastCameraPosition: vec3 = vec3.create();
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
  
  
  // Add these properties to the Viewer class definition at the top
  private isDragging: boolean = false;
  private isPanning: boolean = false;
  private lastMouseX: number = 0;
  private lastMouseY: number = 0;
  private orbitSpeed: number = 0.005;
  private panSpeed: number = 0.01;
  private zoomSpeed: number = 0.1;
  
  // Add this property to the Viewer class
  private sceneTransformMatrix: mat4 = mat4.fromValues(
    1, 0, 0, 0,   // First row
    0, -1, 0, 0,  // Second row
    0, 0, -1, 0,   // Third row
    0, 0, 0, 1    // Fourth row
  );
  
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

  private fpsUpdateInterval: number = 500; // Update FPS display every 500ms
  private lastFpsUpdateTime: number = 0;
  private fpsElement: HTMLElement | null = null;
  private currentFps: number = 0;

  // Update these properties
  private lastRafTime: number = 0;
  private currentFrameTime: number = 0; // in milliseconds

  // Add these properties to the class
  private frameTimeHistory: number[] = [];
  private frameTimeHistoryMaxLength: number = 10;

  // Add these properties to the class
  private sortTimeElement: HTMLElement | null = null;
  private lastSortTime: number = 0;

  private isIntelGPU: boolean = false;
  private customPixelRatio: number = 1.0;

  // Add these properties to the class
  private touchStartPositions: { [key: number]: { x: number; y: number } } = {};
  private lastTouchDistance: number = 0;
  private isTouchOrbit: boolean = false;

  constructor(containerId: string) {
    // Create canvas element
    this.canvas = document.createElement('canvas');
    
    // Get container
    this.container = document.getElementById(containerId)!;
    if (!this.container) {
      throw new Error(`Container element with id "${containerId}" not found`);
    }
    
    // Set canvas to fill container completely
    this.canvas.style.width = '100vw';
    this.canvas.style.height = '100vh';
    this.canvas.style.display = 'block';
    
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
    this.camera.setPosition(0, 0, 1);
    this.camera.setTarget(0, 0, 0);
    
    // Get camera position and copy it to lastCameraPosition
    const pos = this.camera.getPosition();
    vec3.set(this.lastCameraPosition, pos[0], pos[1], pos[2]);
    
    // Detect GPU vendor and set appropriate pixel ratio
    this.detectGPUVendor();
    
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
    
    // Initialize the FPS counter
    this.initFpsCounter();

    this.initWebGLConstants();

    // Add keyboard controls for scene rotation
    this.initKeyboardControls();

    this.render(0);

  }
  
  private initWebGLConstants(): void {
    const gl = this.gl!;
    // Ensure blending is properly set up
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    // Enable backface culling
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
  }

  private detectGPUVendor(): void {
    const gl = this.gl!;
    const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
    
    if (debugInfo) {
      const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
      console.log('GPU detected:', renderer);
      
      // Check if this is an Intel GPU
      this.isIntelGPU = renderer.toLowerCase().includes('intel');
      
      if (this.isIntelGPU) {
        console.log('Intel GPU detected - reducing resolution for better performance');
        // Use a lower pixel ratio for Intel GPUs (0.75 = 75% of normal resolution)
        this.customPixelRatio = 0.75;
      } else {
        // Use device pixel ratio for non-Intel GPUs, but cap it for high-DPI displays
        this.customPixelRatio = Math.min(window.devicePixelRatio, 2.0);
      }
    }
  }

  /**
   * Updates canvas size to match container dimensions
   */
  private updateCanvasSize(): void {
    // Get container dimensions (using getBoundingClientRect for true pixel dimensions)
    const rect = this.container.getBoundingClientRect();
    
    // Apply custom pixel ratio
    const width = Math.floor(rect.width * this.customPixelRatio);
    const height = Math.floor(rect.height * this.customPixelRatio);
    
    // Set canvas dimensions while keeping display size
    this.canvas.width = width;
    this.canvas.height = height;
    
    // Make sure canvas still appears at the browser-reported size
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    
    // Update camera aspect ratio
    this.camera.setAspectRatio(width / height);
    
    // Update WebGL viewport
    if (this.gl) {
      this.gl.viewport(0, 0, width, height);
    }
  }

  private initShaders(): void {
    const gl = this.gl!;
    
    // Get sample count from URL
    const sampleCount = this.getSampleCountFromURL();
    console.log(`Using ${sampleCount} samples for rendering`);
    
    // Updated vertex shader to use texture fetches
    const vsSource = `#version 300 es
      precision mediump float;
      precision mediump sampler2D;
      
      // Core attributes
      in vec4 aVertexPosition;
      in uint aInstanceIndex;
      
      // Uniforms for matrices and camera
      uniform mat4 uProjectionMatrix;
      uniform mat4 uViewMatrix;
      uniform mat4 uSceneTransformMatrix;
      uniform mat4 uInverseTransformMatrix;
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

      // Modified SH evaluation with proper transform handling
      vec3 evaluateSH(vec3 sh0, vec3 sh1_0, vec3 sh1_1, vec3 sh1_2, vec3 direction) {
        // Transform the direction vector using the inverse transform matrix
        // This handles rotations correctly in the shader space
        vec4 transformedDir = uInverseTransformMatrix * vec4(direction, 0.0);
        
        // Normalize the transformed direction
        vec3 dir = normalize(transformedDir.xyz);
        
        // Rest of the SH evaluation remains the same
        // SH0
        vec3 color = sh0 * 0.28209479177387814;
        
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
        vec4 sh0_vec4 = fetch4(uShTexture, idx, 0, uShDims, 3);
        vec4 sh1_part1 = fetch4(uShTexture, idx, 1, uShDims, 3);
        vec4 sh1_part2 = fetch4(uShTexture, idx, 2, uShDims, 3);
        
        // Extract SH0 (rgb only - first 3 components)
        vec3 sh0 = sh0_vec4.rgb;
        
        // Extract SH1 values (all 9 components)
        // The 4th value of sh0_vec4 is the first SH1 value
        vec3 sh1_0 = vec3(sh0_vec4.a, sh1_part1.xy);
        vec3 sh1_1 = vec3(sh1_part1.zw, sh1_part2.x);
        vec3 sh1_2 = sh1_part2.yzw;
    
        // Scale the vertex position for this instance
        vec3 scaledVertexPos = aVertexPosition.xyz * instanceScale;
        
        // Position vertex relative to instance position
        vec3 worldVertexPos = scaledVertexPos + instancePosition;
        
        // Apply scene transform to the entire positioned vertex
        vec4 transformedPos = uSceneTransformMatrix * vec4(worldVertexPos, 1.0);
        
        // Calculate final position
        gl_Position = uProjectionMatrix * uViewMatrix * transformedPos;
        
        // Pass transformed world position of the vertex to fragment shader
        vWorldPos = transformedPos.xyz;
        
        // Calculate voxel center in transformed space
        vVoxelCenter = (uSceneTransformMatrix * vec4(instancePosition, 1.0)).xyz;
        
        // Pass scale to fragment shader
        vScale = instanceScale;
        
        // Calculate viewing direction from voxel center to camera in world space
        vec3 viewDir = normalize(vVoxelCenter - uCameraPosition);
        
        // Calculate color using SH and pass to fragment shader
        vColor = evaluateSH(sh0, sh1_0, sh1_1, sh1_2, viewDir);
        
        // Pass density values to fragment shader
        vDensity0 = gridValues1;
        vDensity1 = gridValues2;
      }
    `;
    
    // Updated fragment shader with sampling loop
    const fsSource = `#version 300 es
      precision mediump float;
      
      // Define the sample count as a constant
      const int SAMPLE_COUNT = ${sampleCount};
      
      in vec3 vWorldPos;
      in float vScale;
      in vec3 vVoxelCenter;
      in vec4 vDensity0;         // Density values for corners 0-3
      in vec4 vDensity1;         // Density values for corners 4-7
      in vec3 vColor;            // Pre-calculated color from vertex shader
      
      uniform vec3 uCameraPosition;
      uniform mat4 uViewMatrix;
      uniform mat4 uInverseTransformMatrix;
      uniform vec3 uTransformFlips; // x, y, z components will be -1 for flipped axes, 1 for unchanged
      
      out vec4 fragColor;
      
      // Ray-box intersection function - returns entry and exit t values
      vec2 rayBoxIntersection(vec3 rayOrigin, vec3 rayDir, vec3 boxCenter, float boxScale) {
        const float EPSILON = 1e-3;
        
        // Get box dimensions
        vec3 halfExtent = vec3(boxScale * 0.5);
        
        // For non-axis aligned boxes, we should transform the ray into box space
        // rather than transforming the box into world space
        
        // 1. Create a coordinate system for the box (this is the inverse transform)
        // This moves ray into the box's local space where it's axis-aligned
        vec3 localRayOrigin = rayOrigin - boxCenter;
        
        // Apply inverse rotation (would be done by multiplying by inverse matrix)
        // Since we're in the fragment shader, we can use the uniform
        vec4 transformedOrigin = uInverseTransformMatrix * vec4(localRayOrigin, 0.0);
        vec4 transformedDir = uInverseTransformMatrix * vec4(rayDir, 0.0);
        
        // Now perform standard AABB intersection in this space
        vec3 invDir = 1.0 / transformedDir.xyz;
        vec3 boxMin = -halfExtent;
        vec3 boxMax = halfExtent;
        
        vec3 tMin = (boxMin - transformedOrigin.xyz) * invDir;
        vec3 tMax = (boxMax - transformedOrigin.xyz) * invDir;
        vec3 t1 = min(tMin, tMax);
        vec3 t2 = max(tMin, tMax);
        float tNear = max(max(t1.x, t1.y), t1.z);
        float tFar = min(min(t2.x, t2.y), t2.z);
        
        // If camera is inside the box, tNear will be negative
        tNear = max(0.0, tNear);
        
        return vec2(tNear, tFar);
      }
      
      // Modified trilinear interpolation for arbitrary transforms
      float trilinearInterpolation(vec3 pos, vec3 boxMin, vec3 boxMax, vec4 density0, vec4 density1) {

        // TODO: This is very unoptimized. Need to optimize this so we don't have to transform the position
        // back and forth between different spaces.
        
        // Calculate the size of the box
        vec3 boxSize = boxMax - boxMin;
        
        // First, transform the sample position back to the original data space
        // 1. Convert the position to a normalized position in the box [0,1]
        vec3 normalizedPos = (pos - boxMin) / boxSize;
        
        // 2. Convert to box-local coordinates [-0.5, 0.5]
        vec3 localPos = normalizedPos - 0.5;
        
        // 3. Transform this position back to the original data space using inverse transform
        vec4 originalLocalPos = uInverseTransformMatrix * vec4(localPos, 0.0);
        
        // 4. Convert back to normalized [0,1] range
        vec3 originalNormalizedPos = originalLocalPos.xyz + 0.5;
        
        // 5. Clamp to ensure we're in the valid range [0,1]
        originalNormalizedPos = clamp(originalNormalizedPos, 0.0, 1.0);
        
        // Now use these coordinates to sample the grid values in their original orientation
        float fx = originalNormalizedPos.x;
        float fy = originalNormalizedPos.y;
        float fz = originalNormalizedPos.z;
        float fx1 = 1.0 - fx;
        float fy1 = 1.0 - fy;
        float fz1 = 1.0 - fz;
        
        // Standard grid corner ordering
        float c000 = density0.x; // Corner [0,0,0]
        float c001 = density0.y; // Corner [0,0,1]
        float c010 = density0.z; // Corner [0,1,0]
        float c011 = density0.w; // Corner [0,1,1]
        float c100 = density1.x; // Corner [1,0,0]
        float c101 = density1.y; // Corner [1,0,1]
        float c110 = density1.z; // Corner [1,1,0]
        float c111 = density1.w; // Corner [1,1,1]
        
        // Trilinear interpolation using original-space coordinates
        float c00 = fx1 * c000 + fx * c100;
        float c01 = fx1 * c001 + fx * c101;
        float c10 = fx1 * c010 + fx * c110;
        float c11 = fx1 * c011 + fx * c111;
        
        float c0 = fy1 * c00 + fy * c10;
        float c1 = fy1 * c01 + fy * c11;
        
        return fz1 * c0 + fz * c1;
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
          float stepLength = viewSpaceRayLength / float(SAMPLE_COUNT);
          
          // Use a loop to calculate total density
          float totalDensity = 0.0;
          
          // Apply explin after interpolation
          // The CUDA reference has a 100x scale factor
          const float STEP_SCALE = 100.0;
          
          for (int i = 0; i < SAMPLE_COUNT; i++) {
            // Calculate sample position - evenly distribute samples
            float t = tNear + (tFar - tNear) * (float(i) + 0.5) / float(SAMPLE_COUNT);
            vec3 samplePoint = rayOrigin + rayDir * t;
            
            // Get density at sample point
            float rawDensity = trilinearInterpolation(samplePoint, boxMin, boxMax, vDensity0, vDensity1);
            
            // Apply explin and accumulate
            float density = STEP_SCALE * stepLength * explin(rawDensity);
            totalDensity += density;
          }
          
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

    if (this.positionBuffer) {
      gl.deleteBuffer(this.positionBuffer);
    }
    
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
    

    if (this.indexBuffer) {
      gl.deleteBuffer(this.indexBuffer);
    }
    
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
    
    // Calculate actual frame time in milliseconds
    if (this.lastRafTime > 0) {
      const frameTime = timestamp - this.lastRafTime;
      
      // Add to frame time history
      this.frameTimeHistory.push(frameTime);
      
      // Keep only last N frames
      if (this.frameTimeHistory.length > this.frameTimeHistoryMaxLength) {
        this.frameTimeHistory.shift(); // Remove oldest frame time
      }
      
      // Calculate average frame time
      const avgFrameTime = this.frameTimeHistory.reduce((sum, time) => sum + time, 0) / 
                           this.frameTimeHistory.length;
      
      this.currentFrameTime = avgFrameTime;
      this.currentFps = 1000 / avgFrameTime; // FPS = 1000ms / frame time
    }
    this.lastRafTime = timestamp;
    
    // Update display at specified intervals
    if (timestamp - this.lastFpsUpdateTime > this.fpsUpdateInterval) {
      if (this.fpsElement) {
        const fps = Math.round(this.currentFps);
        const frameTime = this.currentFrameTime.toFixed(1);
        this.fpsElement.textContent = `FPS: ${fps} | Frame: ${frameTime}ms`;
      }
      
      // Update sort time display
      if (this.sortTimeElement) {
        this.sortTimeElement.textContent = `Sort: ${this.lastSortTime.toFixed(1)}ms`;
      }
      
      this.lastFpsUpdateTime = timestamp;
    }
    
    // Check if camera has moved enough to trigger a resort
    const cameraPos = this.camera.getPosition();

    // Use vec3.distance for a cleaner distance calculation
    const cameraMoveDistance = vec3.distance(
      this.lastCameraPosition,
      vec3.fromValues(cameraPos[0], cameraPos[1], cameraPos[2])
    );

    if (cameraMoveDistance > this.resortThreshold && !this.pendingSortRequest) {
      // Update lastCameraPosition with current position
      vec3.set(this.lastCameraPosition, cameraPos[0], cameraPos[1], cameraPos[2]);
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
    
    // Calculate the inverse transform matrix
    const inverseTransformMatrix = this.getInverseTransformMatrix();
    
    // Set uniforms with camera matrices
    const projectionMatrixLocation = gl.getUniformLocation(this.program, 'uProjectionMatrix');
    const viewMatrixLocation = gl.getUniformLocation(this.program, 'uViewMatrix');
    const sceneTransformMatrixLocation = gl.getUniformLocation(this.program, 'uSceneTransformMatrix');
    const inverseTransformMatrixLocation = gl.getUniformLocation(this.program, 'uInverseTransformMatrix');
    const cameraPositionLocation = gl.getUniformLocation(this.program, 'uCameraPosition');
    
    // Pass matrices to shader
    gl.uniformMatrix4fv(projectionMatrixLocation, false, this.camera.getProjectionMatrix());
    gl.uniformMatrix4fv(viewMatrixLocation, false, this.camera.getViewMatrix());
    gl.uniformMatrix4fv(sceneTransformMatrixLocation, false, this.sceneTransformMatrix);
    gl.uniformMatrix4fv(inverseTransformMatrixLocation, false, inverseTransformMatrix);
    
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
    
    // Add a uniform to pass transformation information to the fragment shader
    const flipsX = this.sceneTransformMatrix[0] < 0 ? -1 : 1;
    const flipsY = this.sceneTransformMatrix[5] < 0 ? -1 : 1;
    const flipsZ = this.sceneTransformMatrix[10] < 0 ? -1 : 1;

    const transformFlipsLocation = gl.getUniformLocation(this.program, 'uTransformFlips');
    gl.uniform3f(transformFlipsLocation, flipsX, flipsY, flipsZ);
    
    // Draw instanced geometry
    if (this.instanceCount <= 0) {
        console.warn('No instances to draw');
        requestAnimationFrame((time) => this.render(time));
        return;
    }
    
    gl.drawElementsInstanced(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0, this.instanceCount);
    
    // Check for GL errors
    // const error = gl.getError();
    // if (error !== gl.NO_ERROR) {
    //   console.error(`WebGL error: ${error}`);
    // }
    
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
    sh0Values: Float32Array,
    octlevels: Uint8Array,
    octpaths: Uint32Array,
    gridValues: Float32Array,
    shRestValues?: Float32Array
  ): void {
    console.log(`Loading point cloud with ${positions.length / 3} points`);
    
    // Save original data (we still need it for the sort worker)
    this.originalPositions = new Float32Array(positions);
    
    // Save SH0 (base colors)
    this.originalSH0Values = new Float32Array(sh0Values);
    
    // We need space for 9 values per vertex
    
    // Extract SH1 coefficients from shRestValues if provided
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
      // If no rest values provided, use zeros (no directional lighting)
      this.originalSH1Values = new Float32Array(positions.length / 3 * 9);
      console.log('No SH1 values provided, using default (no directional lighting)');
    }
    
    // Save octree data
    this.originalOctlevels = new Uint8Array(octlevels);
    
    this.originalScales = new Float32Array(octlevels.length);
    for (let i = 0; i < octlevels.length; i++) {
      this.originalScales[i] = this.baseVoxelSize * Math.pow(2, -octlevels[i]);
    }
    
    this.originalOctpaths = new Uint32Array(octpaths);
    
    // Save grid values
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
          
          // Store the sort time
          this.lastSortTime = data.sortTime || 0;
          
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
    // const error = gl.getError();
    // if (error !== gl.NO_ERROR) {
    //   console.error('WebGL error during index buffer update:', error);
    // }
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
    
    // Create copies of octree data to send to worker
    let octlevels: Uint8Array | undefined = undefined;
    let octpaths: Uint32Array | undefined = undefined;
    
    // Use the original octree data if available
    if (this.originalOctlevels) {
      octlevels = new Uint8Array(this.originalOctlevels);
    }
    
    if (this.originalOctpaths) {
      octpaths = new Uint32Array(this.originalOctpaths);
    }
    
    // Create a copy of the scene transform matrix
    const transformMatrix = new Float32Array(this.sceneTransformMatrix);
    
    // Send the data to the worker
    this.sortWorker.postMessage({
      type: 'sort',
      positions: positions,
      cameraPosition: cameraPos,
      cameraTarget: cameraTarget,
      sceneTransformMatrix: transformMatrix,
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
    
    this.sortWorker.postMessage({}, [transformMatrix.buffer]);
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
        this.orbit(deltaX, -deltaY);
      } else if (this.isPanning) {
        // Pan the camera
        this.pan(deltaX, deltaY);
      }
      
      this.lastMouseX = event.clientX;
      this.lastMouseY = event.clientY;
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
    });
    
    // Prevent context menu on right click
    this.canvas.addEventListener('contextmenu', (event) => {
      event.preventDefault();
    });

    // Add touch event listeners
    this.canvas.addEventListener('touchstart', (event: TouchEvent) => {
        event.preventDefault();
        
        // Reset touch state
        this.isTouchOrbit = false;
        
        // Store initial touch positions
        for (let i = 0; i < event.touches.length; i++) {
            const touch = event.touches[i];
            this.touchStartPositions[touch.identifier] = {
                x: touch.clientX,
                y: touch.clientY
            };
        }
        
        if (event.touches.length === 1) {
            // Single touch = orbit
            this.isTouchOrbit = true;
            this.lastMouseX = event.touches[0].clientX;
            this.lastMouseY = event.touches[0].clientY;
        } else if (event.touches.length === 2) {
            // Two finger touch - initialize for both zoom and pan
            const touch1 = event.touches[0];
            const touch2 = event.touches[1];
            this.lastTouchDistance = Math.hypot(
                touch2.clientX - touch1.clientX,
                touch2.clientY - touch1.clientY
            );
            this.lastMouseX = (touch1.clientX + touch2.clientX) / 2;
            this.lastMouseY = (touch1.clientY + touch2.clientY) / 2;
        }
    });

    this.canvas.addEventListener('touchmove', (event: TouchEvent) => {
        event.preventDefault();
        
        if (event.touches.length === 1 && this.isTouchOrbit) {
            // Single touch orbit
            const touch = event.touches[0];
            const deltaX = touch.clientX - this.lastMouseX;
            const deltaY = touch.clientY - this.lastMouseY;
            
            this.orbit(deltaX, -deltaY);
            
            this.lastMouseX = touch.clientX;
            this.lastMouseY = touch.clientY;
        } else if (event.touches.length === 2) {
            const touch1 = event.touches[0];
            const touch2 = event.touches[1];
            
            // Calculate new touch distance for zoom
            const newTouchDistance = Math.hypot(
                touch2.clientX - touch1.clientX,
                touch2.clientY - touch1.clientY
            );
            
            // Calculate center point of the two touches
            const centerX = (touch1.clientX + touch2.clientX) / 2;
            const centerY = (touch1.clientY + touch2.clientY) / 2;
            
            // Handle both zoom and pan simultaneously
            // Handle zoom
            const zoomDelta = (this.lastTouchDistance - newTouchDistance) * 0.01;
            this.zoom(zoomDelta);
            
            // Handle pan
            const deltaX = centerX - this.lastMouseX;
            const deltaY = centerY - this.lastMouseY;
            this.pan(deltaX, deltaY);
            
            this.lastTouchDistance = newTouchDistance;
            this.lastMouseX = centerX;
            this.lastMouseY = centerY;
        }
    });

    this.canvas.addEventListener('touchend', (event: TouchEvent) => {
        event.preventDefault();
        
        // Remove ended touches from tracking
        for (let i = 0; i < event.changedTouches.length; i++) {
            delete this.touchStartPositions[event.changedTouches[i].identifier];
        }
        
        // Reset state if no touches remain
        if (event.touches.length === 0) {
            this.isTouchOrbit = false;
        }
    });
  }
  
  /**
   * Orbit the camera around the target
   */
  private orbit(deltaX: number, deltaY: number): void {
    const pos = this.camera.getPosition();
    const target = this.camera.getTarget();
    
    // Create vec3 objects from camera position and target
    const position = vec3.fromValues(pos[0], pos[1], pos[2]);
    const targetVec = vec3.fromValues(target[0], target[1], target[2]);
    
    // Calculate the camera's current position relative to the target
    const eyeDir = vec3.create();
    vec3.subtract(eyeDir, position, targetVec);
    
    // Calculate distance from target
    const distance = vec3.length(eyeDir);
    
    // Calculate current spherical coordinates
    let theta = Math.atan2(eyeDir[0], eyeDir[2]);
    let phi = Math.acos(eyeDir[1] / distance);
    
    // Update angles based on mouse movement
    theta -= deltaX * this.orbitSpeed;
    phi = Math.max(0.1, Math.min(Math.PI - 0.1, phi + deltaY * this.orbitSpeed));
    
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
    
    // 3. Create SH0 + SH1 texture (3+9=12 values per instance = 3 vec4s)
    const shData = new Float32Array(this.instanceCount * 12);
    for (let i = 0; i < this.instanceCount; i++) {
      // SH0 (rgb) - first 3 values
      shData[i * 12 + 0] = this.originalSH0Values[i * 3 + 0]; // R
      shData[i * 12 + 1] = this.originalSH0Values[i * 3 + 1]; // G
      shData[i * 12 + 2] = this.originalSH0Values[i * 3 + 2]; // B
      
      // SH1 (all 9 values) - starting from the 4th position
      for (let j = 0; j < 9; j++) {
        if (j < this.originalSH1Values.length / this.instanceCount) {
          shData[i * 12 + 3 + j] = this.originalSH1Values[i * 9 + j];
        } else {
          shData[i * 12 + 3 + j] = 0.0;
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

  private initFpsCounter(): void {
    // Create container for performance metrics
    const perfContainer = document.createElement('div');
    perfContainer.style.position = 'absolute';
    perfContainer.style.bottom = '10px';
    perfContainer.style.right = '10px';
    perfContainer.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    perfContainer.style.padding = '5px';
    perfContainer.style.borderRadius = '3px';
    perfContainer.style.fontFamily = 'monospace';
    perfContainer.style.fontSize = '14px';
    perfContainer.style.color = 'white';
    
    // Create FPS counter element
    this.fpsElement = document.createElement('div');
    this.fpsElement.textContent = 'FPS: --';
    perfContainer.appendChild(this.fpsElement);
    
    // Create sort time element
    this.sortTimeElement = document.createElement('div');
    this.sortTimeElement.textContent = 'Sort: -- ms';
    perfContainer.appendChild(this.sortTimeElement);
    
    // Append container to document
    document.body.appendChild(perfContainer);
  }

  // Add this method to Viewer class
  public setSceneTransformMatrix(matrix: Float32Array | number[]): void {
    if (matrix.length !== 16) {
      throw new Error('Transform matrix must be a 4x4 matrix with 16 elements');
    }
    
    // Create a new mat4 from the input
    mat4.copy(this.sceneTransformMatrix, matrix as Float32Array);
    
    // Request a resort to update the view with the new transform
    this.requestSort();
  }

  // Add this method to get the inverse transform matrix for use in direction calculations
  private getInverseTransformMatrix(): Float32Array {
    // Create a new matrix to store the inverse
    const inverse = mat4.create();
    
    // Calculate the inverse of the scene transform matrix
    mat4.invert(inverse, this.sceneTransformMatrix);
    
    // Return as Float32Array for WebGL
    return inverse as Float32Array;
  }

  /**
   * Rotates the scene around the camera's forward axis (view direction)
   * @param angleInRadians Angle to rotate in radians (positive = clockwise, negative = counterclockwise)
   */
  public rotateSceneAroundViewDirection(angleInRadians: number): void {
    // Get camera position and target to determine view direction
    const pos = this.camera.getPosition();
    const target = this.camera.getTarget();
    
    // Calculate the forward vector
    const forward = vec3.create();
    vec3.subtract(forward, 
      vec3.fromValues(target[0], target[1], target[2]), 
      vec3.fromValues(pos[0], pos[1], pos[2])
    );
    vec3.normalize(forward, forward);
    
    // Create rotation matrix around the forward vector
    const rotationMatrix = mat4.create();
    mat4.fromRotation(rotationMatrix, angleInRadians, forward);
    
    // Create a new matrix for the result
    const newTransform = mat4.create();
    mat4.multiply(newTransform, rotationMatrix, this.sceneTransformMatrix);
    
    // Update the scene transform
    this.setSceneTransformMatrix(newTransform as Float32Array);
  }

  /**
   * Initialize keyboard controls
   */
  private initKeyboardControls(): void {
    // Rotation amount per keypress in radians
    const rotationAmount = 0.1; // About 5.7 degrees
    
    // Movement speed (adjust this value to change movement sensitivity)
    const moveSpeed = 0.1;
    
    // Track which keys are currently pressed
    const pressedKeys = new Set<string>();
    
    // Add event listener for keydown
    window.addEventListener('keydown', (event) => {
        pressedKeys.add(event.key.toLowerCase());
        
        switch (event.key.toLowerCase()) {
            case 'q':
                this.rotateSceneAroundViewDirection(-rotationAmount);
                break;
            case 'e':
                this.rotateSceneAroundViewDirection(rotationAmount);
                break;
        }
    });
    
    // Add event listener for keyup
    window.addEventListener('keyup', (event) => {
        pressedKeys.delete(event.key.toLowerCase());
    });
    
    // Add continuous movement update
    const updateMovement = () => {
        const pos = this.camera.getPosition();
        const target = this.camera.getTarget();
        
        // Calculate forward vector (from camera to target)
        const forward = vec3.create();
        vec3.subtract(forward, 
            vec3.fromValues(target[0], target[1], target[2]),
            vec3.fromValues(pos[0], pos[1], pos[2])
        );
        vec3.normalize(forward, forward);
        
        // Calculate right vector (cross product of forward and up)
        const up = vec3.fromValues(0, 1, 0);
        const right = vec3.create();
        vec3.cross(right, forward, up);
        vec3.normalize(right, right);
        
        let moveX = 0;
        let moveY = 0;
        let moveZ = 0;
        
        // Check for WASD and arrow keys
        if (pressedKeys.has('w') || pressedKeys.has('arrowup')) {
            // Move in forward direction
            moveX += forward[0] * moveSpeed;
            moveY += forward[1] * moveSpeed;
            moveZ += forward[2] * moveSpeed;
        }
        if (pressedKeys.has('s') || pressedKeys.has('arrowdown')) {
            // Move in backward direction
            moveX -= forward[0] * moveSpeed;
            moveY -= forward[1] * moveSpeed;
            moveZ -= forward[2] * moveSpeed;
        }
        if (pressedKeys.has('a') || pressedKeys.has('arrowleft')) {
            // Move left
            moveX -= right[0] * moveSpeed;
            moveY -= right[1] * moveSpeed;
            moveZ -= right[2] * moveSpeed;
        }
        if (pressedKeys.has('d') || pressedKeys.has('arrowright')) {
            // Move right
            moveX += right[0] * moveSpeed;
            moveY += right[1] * moveSpeed;
            moveZ += right[2] * moveSpeed;
        }
        
        // Space and Shift for up/down movement
        if (pressedKeys.has(' ')) {
            moveY += moveSpeed;
        }
        if (pressedKeys.has('shift')) {
            moveY -= moveSpeed;
        }
        
        // Apply movement if any keys were pressed
        if (moveX !== 0 || moveY !== 0 || moveZ !== 0) {
            this.camera.setPosition(pos[0] + moveX, pos[1] + moveY, pos[2] + moveZ);
            this.camera.setTarget(target[0] + moveX, target[1] + moveY, target[2] + moveZ);
        }
        
        // Continue the update loop
        requestAnimationFrame(updateMovement);
    };
    
    // Start the movement update loop
    updateMovement();
  }

  private getSampleCountFromURL(): number {
    const urlParams = new URLSearchParams(window.location.search);
    const sampleCount = parseInt(urlParams.get('samples') || '3', 10);
    // Ensure the count is at least 1 and not too high for performance
    return Math.max(1, Math.min(sampleCount, 64));
  }
}
