/**
 * Camera class for WebGL rendering
 * Handles view and projection matrices
 */
import { mat4, vec3 } from 'gl-matrix';

export class Camera {
  // Camera matrices
  private viewMatrix: mat4 = mat4.create();
  private projectionMatrix: mat4 = mat4.create();
  
  // Camera properties
  private position: vec3 = vec3.fromValues(0, 0, 0);
  private target: vec3 = vec3.fromValues(0, 0, -1);
  private up: vec3 = vec3.fromValues(0, 1, 0);
  
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
  
  /**
   * Orbit the camera around the target
   * @param angleY Angle in radians to rotate around Y axis
   */
  public orbit(angleY: number): void {
    // Calculate direction vector from target to position
    const direction = vec3.create();
    vec3.subtract(direction, this.position, this.target);
    
    // Rotate around Y axis
    const rotatedDirection = vec3.create();
    vec3.rotateY(rotatedDirection, direction, [0, 0, 0], angleY);
    
    // Update position based on rotated direction
    vec3.add(this.position, this.target, rotatedDirection);
    
    this.updateViewMatrix();
  }
}
