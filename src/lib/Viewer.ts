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
  private useInstanceColors: boolean = false;

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
    
    // Enable depth testing
    this.gl.enable(this.gl.DEPTH_TEST);
    
    // Initialize camera
    this.camera = new Camera();
    this.camera.setPosition(0, 0, 15); // Set initial camera position
    this.camera.setTarget(0, 0, 0);    // Look at origin
    
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
    
    // Updated vertex shader optimized for instanced rendering
    const vsSource = `#version 300 es
      in vec4 aVertexPosition;
      in vec4 aVertexColor;
      in vec4 aInstanceOffset;
      in vec4 aInstanceColor;
      
      uniform mat4 uProjectionMatrix;
      uniform mat4 uViewMatrix;
      uniform bool uUseInstanceColors;
      
      out vec4 vColor;
      
      void main() {
        // Position is vertex position + instance offset (xyz only)
        vec4 instancePosition = aVertexPosition + vec4(aInstanceOffset.xyz, 0.0);
        gl_Position = uProjectionMatrix * uViewMatrix * instancePosition;
        
        // Use instance color if enabled, otherwise use vertex color
        vColor = uUseInstanceColors ? aInstanceColor : aVertexColor;
      }
    `;
    
    // Fragment shader
    const fsSource = `#version 300 es
      precision mediump float;
      in vec4 vColor;
      out vec4 fragColor;
      
      void main() {
        fragColor = vColor;
      }
    `;
    
    // Create shaders
    const vertexShader = this.createShader(gl.VERTEX_SHADER, vsSource);
    const fragmentShader = this.createShader(gl.FRAGMENT_SHADER, fsSource);
    
    // Create and link program
    this.program = gl.createProgram();
    if (!this.program) {
      throw new Error('Failed to create shader program');
    }
    
    gl.attachShader(this.program, vertexShader);
    gl.attachShader(this.program, fragmentShader);
    gl.linkProgram(this.program);
    
    // Check if program linked successfully
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(this.program);
      throw new Error(`Could not compile WebGL program: ${info}`);
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
    
    // Create a cube with the specified size
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
      
      // Top face
      -halfSize,  halfSize, -halfSize,
      -halfSize,  halfSize,  halfSize,
       halfSize,  halfSize,  halfSize,
       halfSize,  halfSize, -halfSize,
      
      // Bottom face
      -halfSize, -halfSize, -halfSize,
       halfSize, -halfSize, -halfSize,
       halfSize, -halfSize,  halfSize,
      -halfSize, -halfSize,  halfSize,
      
      // Right face
       halfSize, -halfSize, -halfSize,
       halfSize,  halfSize, -halfSize,
       halfSize,  halfSize,  halfSize,
       halfSize, -halfSize,  halfSize,
      
      // Left face
      -halfSize, -halfSize, -halfSize,
      -halfSize, -halfSize,  halfSize,
      -halfSize,  halfSize,  halfSize,
      -halfSize,  halfSize, -halfSize,
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
    
    // Default cube vertex colors (white)
    const colors = [];
    for (let i = 0; i < 24; i++) {
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
    
    // Create indices for the cube
    const indices = [
      0,  1,  2,    0,  2,  3,    // Front face
      4,  5,  6,    4,  6,  7,    // Back face
      8,  9,  10,   8,  10, 11,   // Top face
      12, 13, 14,   12, 14, 15,   // Bottom face
      16, 17, 18,   16, 18, 19,   // Right face
      20, 21, 22,   20, 22, 23,   // Left face
    ];
    
    // Create index buffer
    this.indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    
    // Store the number of indices
    this.indexCount = indices.length;
  }
  
  /**
   * Main render function with time-based animation
   */
  private render(timestamp: number): void {
    const gl = this.gl!;
    
    // Calculate delta time in seconds
    const deltaTime = (timestamp - this.lastFrameTime) / 1000;
    this.lastFrameTime = timestamp;
    
    // Clear the canvas
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clearDepth(1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    
    // Orbit the camera (this is now handled by the external controls)
    // this.camera.orbit(deltaTime * this.rotationSpeed * 10);
    
    // Use our shader program
    gl.useProgram(this.program);
    
    // Bind the VAO
    gl.bindVertexArray(this.vao);
    
    // Set uniforms with camera matrices
    const projectionMatrixLocation = gl.getUniformLocation(this.program!, 'uProjectionMatrix');
    const viewMatrixLocation = gl.getUniformLocation(this.program!, 'uViewMatrix');
    const useInstanceColorsLocation = gl.getUniformLocation(this.program!, 'uUseInstanceColors');
    
    gl.uniformMatrix4fv(projectionMatrixLocation, false, this.camera.getProjectionMatrix());
    gl.uniformMatrix4fv(viewMatrixLocation, false, this.camera.getViewMatrix());
    gl.uniform1i(useInstanceColorsLocation, this.useInstanceColors ? 1 : 0);
    
    // Draw instanced geometry
    gl.drawElementsInstanced(gl.TRIANGLES, this.indexCount, gl.UNSIGNED_SHORT, 0, this.instanceCount);
    
    // Unbind the VAO
    gl.bindVertexArray(null);
    
    // Request animation frame for continuous rendering
    requestAnimationFrame((time) => this.render(time));
  }
  
  /**
   * Load a point cloud from positions and colors
   */
  public loadPointCloud(positions: Float32Array, colors?: Float32Array): void {
    const gl = this.gl!;
    
    // Create a VAO for the point cloud
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    
    // For very large point clouds, use smaller cubes
    const cubeSize = 0.005; // Smaller cube size for dense point clouds
    this.initCubeGeometry(cubeSize);
    
    // For very large point clouds, limit the number of instances
    const maxInstances = 500000; // Limit for better performance
    
    if (positions.length / 3 > maxInstances) {
      console.warn(`Point cloud has ${positions.length / 3} points, limiting to ${maxInstances} for performance`);
      
      // Create a downsampled version
      const stride = Math.ceil(positions.length / 3 / maxInstances);
      const sampledPositions = new Float32Array(maxInstances * 3);
      const sampledColors = colors ? new Float32Array(maxInstances * 4) : undefined;
      
      let j = 0;
      for (let i = 0; i < positions.length; i += 3 * stride) {
        if (j >= maxInstances) break;
        
        sampledPositions[j * 3] = positions[i];
        sampledPositions[j * 3 + 1] = positions[i + 1];
        sampledPositions[j * 3 + 2] = positions[i + 2];
        
        if (sampledColors && colors) {
          sampledColors[j * 4] = colors[Math.floor(i / 3) * 4];
          sampledColors[j * 4 + 1] = colors[Math.floor(i / 3) * 4 + 1];
          sampledColors[j * 4 + 2] = colors[Math.floor(i / 3) * 4 + 2];
          sampledColors[j * 4 + 3] = colors[Math.floor(i / 3) * 4 + 3];
        }
        
        j++;
      }
      
      // Use the downsampled data
      positions = sampledPositions;
      colors = sampledColors || colors;
    }
    
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
    
    // Store instance colors if provided
    if (colors) {
      this.instanceColorBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
      
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
      
      this.useInstanceColors = true;
    } else {
      this.useInstanceColors = false;
    }
    
    // Set the instance count to the number of points
    this.instanceCount = positions.length / 3;
    
    // Unbind the VAO
    gl.bindVertexArray(null);
    
    console.log(`Loaded point cloud with ${this.instanceCount} points`);
  }
  
  /**
   * Set the number of instances to display
   */
  public setInstanceCount(count: number): void {
    if (count < 1) count = 1;
    this.instanceCount = count;
    
    // Re-initialize buffers to update instance data
    this.initBuffers();
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
      
      // Delete VAO
      if (this.vao) gl.deleteVertexArray(this.vao);
      
      // Delete program and shaders
      if (this.program) gl.deleteProgram(this.program);
    }
  }
}
