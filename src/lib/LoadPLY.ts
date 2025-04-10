/**
 * LoadPLY utility class for loading specialized binary PLY files with f_dc color data
 */
export interface PLYData {
  vertices: Float32Array;    // Position data (x, y, z)
  sh0Values: Float32Array;      // Color data from f_dc fields
  octlevels: Uint8Array;    // Octlevel data for scaling
  octpaths: Uint32Array;    // Octpath data
  shRestValues: Float32Array | undefined; // f_rest values (0-8)
  gridValues: Float32Array; // grid point density values (0-7)
  vertexCount: number;
  sceneCenter: [number, number, number]; // scene center
  sceneExtent: number;      // scene extent as a single value
}

export class LoadPLY {
  /**
   * Load a PLY file from a URL
   */
  public static async loadFromUrl(url: string, onProgress?: (progress: number) => void): Promise<PLYData> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load PLY file: ${response.statusText}`);
    }
    
    const contentLength = parseInt(response.headers.get('Content-Length') || '0');
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Failed to get response reader');
    }

    // Read the response as chunks
    const chunks: Uint8Array[] = [];
    let receivedLength = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      receivedLength += value.length;

      if (contentLength && onProgress) {
        onProgress(receivedLength / contentLength);
      }
    }

    // Combine all chunks into a single array buffer
    const arrayBuffer = new ArrayBuffer(receivedLength);
    const view = new Uint8Array(arrayBuffer);
    let position = 0;
    for (const chunk of chunks) {
      view.set(chunk, position);
      position += chunk.length;
    }
    
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
    let sceneCenter: [number, number, number] = [0, 0, 0];
    let sceneExtent: number = 3.0;
    
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
    let octlevels: Uint8Array;

    if (hasOctlevel) {
      octlevels = new Uint8Array(vertexCount);
      console.log('File has octlevel property');
    } else {
      throw new Error('PLY file missing required octlevel property');
    }
    
    // Check if the file has octpath property
    const hasOctpath = properties.some(p => p.name === 'octpath');
    let octpaths: Uint32Array;
    
    if (hasOctpath) {
      octpaths = new Uint32Array(vertexCount);
      console.log('File has octpath property');
    } else {
      throw new Error('PLY file missing required octpath property');
    }
    
    // Check if the file has f_rest properties
    const hasRestValues = properties.some(p => p.name.startsWith('f_rest_'));
    let restValues: Float32Array | undefined;
    
    if (hasRestValues) {
      // Count how many f_rest properties are present (should be 9 from f_rest_0 to f_rest_8)
      const restCount = properties.filter(p => p.name.startsWith('f_rest_')).length;
      restValues = new Float32Array(vertexCount * restCount);
      console.log(`File has ${restCount} f_rest properties`);
    } else {
      console.log('PLY file missing f_rest values. No directional lighting will be visible.');
      restValues = undefined;
    }
    
    // Check if the file has grid value properties
    const hasGridValues = properties.some(p => p.name.includes('grid') && p.name.includes('_value'));
    let gridValues: Float32Array;
    
    if (hasGridValues) {
      // Count grid properties (should be 8: grid0_value to grid7_value)
      const gridCount = properties.filter(p => p.name.includes('grid') && p.name.includes('_value')).length;
      gridValues = new Float32Array(vertexCount * gridCount);
      console.log(`File has ${gridCount} grid value properties`);
    } else {
      throw new Error('PLY file missing required grid value properties');
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
    const sh0s = new Float32Array(vertexCount * 3);   // r, g, b for each vertex
    
    // Create a DataView for binary reading
    const dataView = new DataView(arrayBuffer);
    
    // Process each vertex
    for (let i = 0; i < vertexCount; i++) {
      const vertexOffset = dataOffset + (i * vertexSize);
      const vertexIndex = i * 3;
      const colorIndex = i * 3;
      
      // Read position (x, y, z)
      vertices[vertexIndex] = dataView.getFloat32(vertexOffset + propertyOffsets['x'], true);
      vertices[vertexIndex + 1] = dataView.getFloat32(vertexOffset + propertyOffsets['y'], true);
      vertices[vertexIndex + 2] = dataView.getFloat32(vertexOffset + propertyOffsets['z'], true);
      
      // Read sh0s (f_dc_0, f_dc_1, f_dc_2)
      sh0s[colorIndex] = dataView.getFloat32(vertexOffset + propertyOffsets['f_dc_0'], true);
      sh0s[colorIndex + 1] = dataView.getFloat32(vertexOffset + propertyOffsets['f_dc_1'], true);
      sh0s[colorIndex + 2] = dataView.getFloat32(vertexOffset + propertyOffsets['f_dc_2'], true);
      
      // Read octlevel if present
      if (hasOctlevel && octlevels && propertyOffsets['octlevel'] !== undefined) {
        octlevels[i] = dataView.getUint8(vertexOffset + propertyOffsets['octlevel']);
      }
      
      // Read octpath if present
      if (hasOctpath && octpaths && propertyOffsets['octpath'] !== undefined) {
        octpaths[i] = dataView.getUint32(vertexOffset + propertyOffsets['octpath'], true);
      }
      
      // Read grid values if present
      if (hasGridValues && gridValues) {
        const gridCount = properties.filter(p => p.name.includes('grid') && p.name.includes('_value')).length;
        for (let g = 0; g < gridCount; g++) {
          const propName = `grid${g}_value`;
          if (propertyOffsets[propName] !== undefined) {
            gridValues[i * gridCount + g] = dataView.getFloat32(vertexOffset + propertyOffsets[propName], true);
          }
        }
      }

      // Read f_rest values if present
      if (hasRestValues && restValues) {
        const restCount = properties.filter(p => p.name.startsWith('f_rest_')).length;
        for (let r = 0; r < restCount; r++) {
          const propName = `f_rest_${r}`;
          if (propertyOffsets[propName] !== undefined) {
            restValues[i * restCount + r] = dataView.getFloat32(vertexOffset + propertyOffsets[propName], true);
          }
        }
      }
            
      
      // For debugging, log a few vertices
      if (i < 5) {
        console.log(`Vertex ${i}: (${vertices[vertexIndex]}, ${vertices[vertexIndex + 1]}, ${vertices[vertexIndex + 2]})`);
        console.log(`SH0 ${i}: (${sh0s[colorIndex]}, ${sh0s[colorIndex + 1]}, ${sh0s[colorIndex + 2]})`);
      }
    }
    
    return {
      vertices,
      sh0Values: sh0s,
      octlevels,
      octpaths,
      shRestValues: restValues,
      gridValues,
      vertexCount,
      sceneCenter,
      sceneExtent
    };
  }
} 