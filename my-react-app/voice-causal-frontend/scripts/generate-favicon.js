const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

// Create the scripts directory if it doesn't exist
const scriptsDir = path.join(__dirname);
if (!fs.existsSync(scriptsDir)) {
  fs.mkdirSync(scriptsDir, { recursive: true });
}

// Path to the source image
const sourcePath = path.join(__dirname, '../../..', 'SAF-MAIN-LOGO (2).png');
const outputPath = path.join(__dirname, '../public/favicon.ico');

// Generate favicon sizes
const sizes = [16, 32, 48, 64];

async function generateFavicon() {
  try {
    console.log('Generating favicon from:', sourcePath);
    
    // Create a buffer for each size
    const buffers = await Promise.all(
      sizes.map(async (size) => {
        return await sharp(sourcePath)
          .resize(size, size)
          .toBuffer();
      })
    );
    
    // Write the ICO file
    const ico = require('to-ico');
    const icoBuffer = await ico(buffers);
    fs.writeFileSync(outputPath, icoBuffer);
    
    console.log('Favicon generated successfully at:', outputPath);
  } catch (error) {
    console.error('Error generating favicon:', error);
  }
}

generateFavicon();
