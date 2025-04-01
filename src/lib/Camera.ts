/**
 * Camera class for WebGL rendering
 * Handles view and projection matrices
 */
import { mat4, vec3, quat } from 'gl-matrix';

export class Camera {
  // Camera matrices
  private viewMatrix: mat4 = mat4.create();
  private projectionMatrix: mat4 = mat4.create();
  
  // Camera properties
  private position: vec3 = vec3.fromValues(0, 0, 0);
  private target: vec3 = vec3.fromValues(0, 0, -1);
  private up: vec3 = vec3.fromValues(0, 1, 0);
  
  // Control-specific properties
  private controlUp: vec3 = vec3.fromValues(0, 1, 0);
  
  // Projection parameters
  private fieldOfView: number = 45 * Math.PI / 180; // in radians
  private aspectRatio: number = 1.0;
  private nearClip: number = 0.1;
  private farClip: number = 1000.0;
  
  constructor() {
    this.updateViewMatrix();
    this.updateProjectionMatrix();
  }
  
  /**
   * Updates the view matrix based on current camera properties
   */
  private updateViewMatrix(): void {
    mat4.lookAt(this.viewMatrix, this.position, this.target, this.up);
  }
  
  /**
   * Updates the projection matrix based on current camera properties
   */
  private updateProjectionMatrix(): void {
    mat4.perspective(
      this.projectionMatrix, 
      this.fieldOfView, 
      this.aspectRatio, 
      this.nearClip, 
      this.farClip
    );
  }
  
  /**
   * Sets the camera position
   */
  public setPosition(x: number, y: number, z: number): void {
    vec3.set(this.position, x, y, z);
    this.updateViewMatrix();
  }
  
  /**
   * Sets the camera target/look-at point
   */
  public setTarget(x: number, y: number, z: number): void {
    vec3.set(this.target, x, y, z);
    this.updateViewMatrix();
  }
  
  /**
   * Sets the camera's up vector
   */
  public setUp(x: number, y: number, z: number): void {
    vec3.set(this.up, x, y, z);
    this.updateViewMatrix();
  }
  
  /**
   * Sets the camera's field of view
   * @param fovInRadians Field of view in radians
   */
  public setFieldOfView(fovInRadians: number): void {
    this.fieldOfView = fovInRadians;
    this.updateProjectionMatrix();
  }
  
  /**
   * Sets the aspect ratio (width/height)
   */
  public setAspectRatio(aspect: number): void {
    this.aspectRatio = aspect;
    this.updateProjectionMatrix();
  }
  
  /**
   * Sets the near and far clipping planes
   */
  public setClippingPlanes(near: number, far: number): void {
    this.nearClip = near;
    this.farClip = far;
    this.updateProjectionMatrix();
  }
  
  /**
   * Get the view matrix
   */
  public getViewMatrix(): mat4 {
    return this.viewMatrix;
  }
  
  /**
   * Get the projection matrix
   */
  public getProjectionMatrix(): mat4 {
    return this.projectionMatrix;
  }

  public getPosition(): vec3 {
    return this.position;
  }

  public getTarget(): vec3 {
    return this.target;
  }
  
  /**
   * Get the camera's up vector
   */
  public getUp(): vec3 {
    return this.up;
  }

  /**
   * Get the standard world up vector used for controls
   */
  public getControlUp(): vec3 {
    return this.controlUp;
  }
  
  /**
   * Sets the standard control up vector
   */
  public setControlUp(x: number, y: number, z: number): void {
    vec3.set(this.controlUp, x, y, z);
    vec3.normalize(this.controlUp, this.controlUp);
  }
  
  /**
   * Apply an orbit rotation using standard control coordinates
   */
  public orbit(deltaX: number, deltaY: number, speed: number): void {
    // Calculate the camera's current position relative to the target
    const relPosition = vec3.create();
    vec3.subtract(relPosition, this.position, this.target);
    
    // Calculate distance from target
    const distance = vec3.length(relPosition);
    
    // Create a rotation around the world up axis for left/right movement
    const horizontalRotation = quat.create();
    quat.setAxisAngle(horizontalRotation, this.controlUp, -deltaX * speed);
    
    // Apply the horizontal rotation to our position
    vec3.transformQuat(relPosition, relPosition, horizontalRotation);
    
    // Calculate the right vector (perpendicular to both view direction and world up)
    const forward = vec3.create();
    vec3.normalize(forward, relPosition);
    vec3.negate(forward, forward); // Negate because camera looks at negative of rel position
    
    const right = vec3.create();
    vec3.cross(right, this.controlUp, forward);
    vec3.normalize(right, right);
    
    // Create a rotation around the right axis for up/down movement
    const verticalRotation = quat.create();
    quat.setAxisAngle(verticalRotation, right, -deltaY * speed);
    
    // Apply the vertical rotation to our position
    vec3.transformQuat(relPosition, relPosition, verticalRotation);
    
    // Update the position based on the new relative position
    vec3.add(this.position, this.target, relPosition);
    
    // Recalculate the camera's up vector to be perpendicular to the view direction
    // This keeps the camera's orientation stable during orbiting
    vec3.cross(this.up, right, forward);
    vec3.normalize(this.up, this.up);
    
    this.updateViewMatrix();
  }
  
  /**
   * Pan the camera (move position and target together)
   */
  public pan(deltaX: number, deltaY: number, speed: number): void {
    // Calculate view direction and right vector
    const viewDir = vec3.create();
    vec3.subtract(viewDir, this.target, this.position);
    vec3.normalize(viewDir, viewDir);
    
    const right = vec3.create();
    vec3.cross(right, viewDir, this.controlUp);
    vec3.normalize(right, right);
    
    const worldUp = vec3.create();
    vec3.cross(worldUp, right, viewDir);
    vec3.normalize(worldUp, worldUp);
    
    // Calculate movement amount
    const panAmount = speed * Math.max(1, vec3.distance(this.position, this.target) / 10);
    
    // Calculate pan offset
    const panOffset = vec3.create();
    vec3.scale(right, right, -deltaX * panAmount);
    vec3.scale(worldUp, worldUp, deltaY * panAmount);
    vec3.add(panOffset, right, worldUp);
    
    // Move both position and target
    vec3.add(this.position, this.position, panOffset);
    vec3.add(this.target, this.target, panOffset);
    
    this.updateViewMatrix();
  }
}
