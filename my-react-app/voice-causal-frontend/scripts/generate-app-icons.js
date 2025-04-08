const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

// Path to the source image
const sourcePath = path.join(__dirname, '../public', 'saf-main-logo.png');
const outputDir = path.join(__dirname, '../public');

// Generate app icon sizes
const sizes = [192, 512];

async function generateAppIcons() {
  try {
    console.log('Generating app icons from:', sourcePath);
    
    // Create each size
    for (const size of sizes) {
      const outputPath = path.join(outputDir, `logo${size}.png`);
      
      await sharp(sourcePath)
        .resize(size, size)
        .toFile(outputPath);
      
      console.log(`Generated ${size}x${size} icon at: ${outputPath}`);
    }
    
    console.log('App icons generated successfully!');
  } catch (error) {
    console.error('Error generating app icons:', error);
  }
}

generateAppIcons();
