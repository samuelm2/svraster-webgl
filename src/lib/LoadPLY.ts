/**
 * LoadPLY utility class for loading specialized binary PLY files with f_dc color data
 */
export interface PLYData {
  vertices: Float32Array;    // Position data (x, y, z)
  colors: Float32Array;      // Color data from f_dc fields
  octlevels?: Uint8Array;    // Optional octlevel data for scaling
  vertexCount: number;
  sceneCenter?: [number, number, number]; // Optional scene center
  sceneExtent?: number;      // Optional scene extent as a single value
}

export class LoadPLY {
  /**
   * Load a PLY file from a URL
   */
  public static async loadFromUrl(url: string): Promise<PLYData> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load PLY file: ${response.statusText}`);
    }
    
    const arrayBuffer = await response.arrayBuffer();
    return LoadPLY.parse(arrayBuffer);
  }
  
  /**
   * Load a PLY file from a File object
   */
  public static async loadFromFile(file: File): Promise<PLYData> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (event) => {
        if (!event.target || !event.target.result) {
          reject(new Error('Failed to read file'));
          return;
        }
        
        try {
          const arrayBuffer = event.target.result as ArrayBuffer;
          const plyData = LoadPLY.parse(arrayBuffer);
          resolve(plyData);
        } catch (error: any) {
          reject(new Error(`Failed to parse PLY: ${error.message}`));
        }
      };
      
      reader.onerror = () => {
        reject(new Error('Error reading file'));
      };
      
      reader.readAsArrayBuffer(file);
    });
  }
  
  /**
   * Parse a binary PLY file with the specified format
   */
  private static parse(arrayBuffer: ArrayBuffer): PLYData {
    // First, get the header as text
    const textDecoder = new TextDecoder();
    
    // Read the first chunk of data to get the header
    const headerView = new Uint8Array(arrayBuffer, 0, Math.min(10000, arrayBuffer.byteLength));
    const headerText = textDecoder.decode(headerView);
    
    // Find the end of the header
    const headerEndIndex = headerText.indexOf('end_header');
    if (headerEndIndex === -1) {
      throw new Error('Invalid PLY file: Missing end_header');
    }
    
    const header = headerText.substring(0, headerEndIndex);
    const lines = header.split('\n');
    
    // Extract scene center and extent from comments
    let sceneCenter: [number, number, number] | undefined;
    let sceneExtent: number | undefined;
    
    for (const line of lines) {
      const trimmed = line.trim();
      
      // Look for scene center in comments
      if (trimmed.startsWith('comment scene_center ')) {
        const parts = trimmed.substring('comment scene_center '.length).trim().split(/\s+/);
        if (parts.length >= 3) {
          sceneCenter = [
            parseFloat(parts[0]),
            parseFloat(parts[1]),
            parseFloat(parts[2])
          ];
          console.log(`Found scene center: [${sceneCenter}]`);
        }
      }
      
      // Look for scene extent in comments (single value)
      if (trimmed.startsWith('comment scene_extent ')) {
        const value = parseFloat(trimmed.substring('comment scene_extent '.length).trim());
        if (!isNaN(value)) {
          sceneExtent = value;
          console.log(`Found scene extent: ${sceneExtent}`);
        }
      }
    }
    
    // Check that this is a valid PLY file
    if (lines[0].trim() !== 'ply') {
      throw new Error('Invalid PLY file: Does not start with "ply"');
    }
    
    // Verify binary format
    const formatLine = lines.find(line => line.trim().startsWith('format'));
    if (!formatLine || !formatLine.includes('binary_little_endian')) {
      throw new Error('Only binary_little_endian format is supported');
    }
    
    // Get vertex count
    const vertexLine = lines.find(line => line.trim().startsWith('element vertex'));
    if (!vertexLine) {
      throw new Error('Invalid PLY file: Missing element vertex');
    }
    
    const vertexCount = parseInt(vertexLine.split(/\s+/)[2], 10);
    console.log(`Vertex count: ${vertexCount}`);
    
    // Collect property information
    const propertyLines = lines.filter(line => line.trim().startsWith('property') && !line.includes('list'));
    const properties = propertyLines.map(line => {
      const parts = line.trim().split(/\s+/);
      return {
        type: parts[1],
        name: parts[2]
      };
    });
    
    // Check for required properties
    if (!properties.some(p => p.name === 'x') || 
        !properties.some(p => p.name === 'y') || 
        !properties.some(p => p.name === 'z')) {
      throw new Error('PLY file missing required position properties (x, y, z)');
    }
    
    if (!properties.some(p => p.name === 'f_dc_0') || 
        !properties.some(p => p.name === 'f_dc_1') || 
        !properties.some(p => p.name === 'f_dc_2')) {
      throw new Error('PLY file missing required color properties (f_dc_0, f_dc_1, f_dc_2)');
    }
    
    // Check if the file has octlevel property
    const hasOctlevel = properties.some(p => p.name === 'octlevel');
    let octlevels: Uint8Array | undefined;
    
    if (hasOctlevel) {
      octlevels = new Uint8Array(vertexCount);
      console.log('File has octlevel property');
    }
    
    // Calculate data offsets for binary reading
    const propertyOffsets: { [key: string]: number } = {};
    const propertySizes: { [key: string]: number } = {};
    let currentOffset = 0;
    
    for (const prop of properties) {
      propertyOffsets[prop.name] = currentOffset;
      
      switch (prop.type) {
        case 'char':
        case 'uchar':
          propertySizes[prop.name] = 1;
          break;
        case 'short':
        case 'ushort':
          propertySizes[prop.name] = 2;
          break;
        case 'int':
        case 'uint':
        case 'float':
          propertySizes[prop.name] = 4;
          break;
        case 'double':
          propertySizes[prop.name] = 8;
          break;
        default:
          propertySizes[prop.name] = 4; // Default to 4 bytes
      }
      
      currentOffset += propertySizes[prop.name];
    }
    
    const vertexSize = currentOffset;
    console.log(`Vertex size: ${vertexSize} bytes`);
    
    // Calculate the start of the data section
    const dataOffset = headerEndIndex + 'end_header'.length + 1; // +1 for the newline
    
    // Prepare arrays for the data
    const vertices = new Float32Array(vertexCount * 3); // x, y, z for each vertex
    const colors = new Float32Array(vertexCount * 4);   // r, g, b, a for each vertex
    
    // Create a DataView for binary reading
    const dataView = new DataView(arrayBuffer);
    
    // Process each vertex
    for (let i = 0; i < vertexCount; i++) {
      const vertexOffset = dataOffset + (i * vertexSize);
      const vertexIndex = i * 3;
      const colorIndex = i * 4;
      
      // Read position (x, y, z)
      vertices[vertexIndex] = dataView.getFloat32(vertexOffset + propertyOffsets['x'], true);
      vertices[vertexIndex + 1] = dataView.getFloat32(vertexOffset + propertyOffsets['y'], true);
      vertices[vertexIndex + 2] = dataView.getFloat32(vertexOffset + propertyOffsets['z'], true);
      
      // Read colors (f_dc_0, f_dc_1, f_dc_2)
      colors[colorIndex] = dataView.getFloat32(vertexOffset + propertyOffsets['f_dc_0'], true);
      colors[colorIndex + 1] = dataView.getFloat32(vertexOffset + propertyOffsets['f_dc_1'], true);
      colors[colorIndex + 2] = dataView.getFloat32(vertexOffset + propertyOffsets['f_dc_2'], true);
      colors[colorIndex + 3] = 1.0; // Alpha channel (full opacity)
      
      // Read octlevel if present
      if (hasOctlevel && octlevels && propertyOffsets['octlevel'] !== undefined) {
        // Octlevel is typically a uchar/char (1 byte)
        octlevels[i] = dataView.getUint8(vertexOffset + propertyOffsets['octlevel']);
      }
      
      // For debugging, log a few vertices
      if (i < 5) {
        console.log(`Vertex ${i}: (${vertices[vertexIndex]}, ${vertices[vertexIndex + 1]}, ${vertices[vertexIndex + 2]})`);
        console.log(`Color ${i}: (${colors[colorIndex]}, ${colors[colorIndex + 1]}, ${colors[colorIndex + 2]})`);
      }
    }
    
    return {
      vertices,
      colors,
      octlevels,
      vertexCount,
      sceneCenter,
      sceneExtent
    };
  }
} 