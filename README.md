# SVRaster WebGL Viewer

A WebGL-based viewer for visualizing sparse voxel scenes from the [Sparse Voxels Rasterization paper](https://svraster.github.io/). This viewer provides an interactive way to explore and visualize the voxel radiance field results. You can try the viewer at [vid2scene.com/voxel](https://vid2scene.com/voxel)

## Features

- Real-time rendering of sparse voxel scenes using WebGL2
- Adaptive level-of-detail visualization
- Interactive camera controls:
  - Left-click + drag: Orbit camera
  - Right-click + drag: Pan camera
  - Mouse wheel: Zoom
  - WASD/Arrow keys: Move camera
  - Q/E: Rotate scene around view direction
  - Space/Shift: Move up/down
- Touch controls for mobile devices:
  - 1 finger drag: Orbit
  - 2 finger drag: Pan/zoom
- Performance metrics display (FPS counter)

## Implementation Notes

- This viewer uses a simpler depth-based sorting approach rather than the ray direction-dependent Morton ordering described in the paper
- The current implementation has only basic optimizations applied - there's significant room for performance improvements. Only the lowest-hanging fruit optimizations have been implemented so far. If you have a perf improvement suggestion, please feel free to submit a PR! Right now, the fragment shader is the bottleneck.

