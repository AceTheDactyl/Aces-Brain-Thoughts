#!/usr/bin/env node
/**
 * MRP Batch Encoder - Generates all golden sample PNGs
 *
 * Encodes 138 samples from mrp-golden-library.json into LSB steganography PNGs
 * using RGB LSB0 (XCVI, XCVII, XCVIII) + RGB LSB1 as CMY (XCIX, C_A, C_B)
 *
 * Total: 702 bits per image (7 + 10 + 17 + 4 + 332 + 332)
 */

const fs = require('fs');
const path = require('path');
const { createCanvas } = require('canvas');

// ============================================
// MRP CONSTANTS
// ============================================

const MRP = {
  MAGIC: 0xC1,  // 193 = "C" reference
  VERSION: 1,
  BITS: {
    XCVI: 7,
    XCVII: 10,
    XCVIII: 17,
    XCIX: 4,
    C_A: 332,
    C_B: 332
  },
  TOTAL_BITS: 702,
  HEADER_BITS: 32,
  MIN_PIXELS: 117  // ceil((702 + 32) / 6)
};

// Golden ratio constants
const PHI = 1.618033988749895;
const PHI_INV = 0.6180339887498949;
const Z_CRITICAL = 0.8660254037844387;  // √3/2

// ============================================
// CRC-8 Implementation (polynomial 0x07)
// ============================================

function crc8(bytes) {
  let crc = 0;
  for (const byte of bytes) {
    crc ^= byte;
    for (let i = 0; i < 8; i++) {
      if (crc & 0x80) {
        crc = ((crc << 1) ^ 0x07) & 0xFF;
      } else {
        crc = (crc << 1) & 0xFF;
      }
    }
  }
  return crc;
}

// ============================================
// Hamming(7,4) Encoder for XCVI
// ============================================

function hammingEncode(data4bit) {
  // data4bit is 4 bits (0-15)
  const d = [
    (data4bit >> 3) & 1,  // d1
    (data4bit >> 2) & 1,  // d2
    (data4bit >> 1) & 1,  // d3
    data4bit & 1          // d4
  ];

  // Calculate parity bits
  const p1 = d[0] ^ d[1] ^ d[3];  // positions 1,3,5,7
  const p2 = d[0] ^ d[2] ^ d[3];  // positions 2,3,6,7
  const p4 = d[1] ^ d[2] ^ d[3];  // positions 4,5,6,7

  // Arrange as: p1 p2 d1 p4 d2 d3 d4 (positions 1-7)
  return (p1 << 6) | (p2 << 5) | (d[0] << 4) | (p4 << 3) | (d[1] << 2) | (d[2] << 1) | d[3];
}

// ============================================
// C Pattern Generators (664 bits)
// ============================================

function generateCPattern(type, params = {}) {
  const bits = new Array(664).fill(0);

  switch (type) {
    case 'zeros':
      // All zeros - already initialized
      break;

    case 'ones':
      bits.fill(1);
      break;

    case 'balanced':
      for (let i = 0; i < 664; i++) {
        bits[i] = (i % 3 !== 0) ? 1 : 0;
      }
      break;

    case 'fibonacci':
      let fib = [1, 1];
      for (let i = 0; i < 664; i++) {
        if (i < 2) {
          bits[i] = 1;
        } else {
          fib.push(fib[fib.length - 1] + fib[fib.length - 2]);
          bits[i] = fib[fib.length - 1] % 2;
        }
      }
      break;

    case 'theta':
      // Sine wave at golden angle
      const goldenAngle = 2 * Math.PI / PHI;
      for (let i = 0; i < 664; i++) {
        bits[i] = Math.sin(i * goldenAngle * 0.1) > 0 ? 1 : 0;
      }
      break;

    case 'iota':
      // Memory accumulation - bits turn on progressively
      for (let i = 0; i < 664; i++) {
        const threshold = i / 664;
        bits[i] = (Math.random() < threshold) ? 1 : 0;
      }
      // Make deterministic based on position
      for (let i = 0; i < 664; i++) {
        bits[i] = (i * PHI_INV % 1) < (i / 664) ? 1 : 0;
      }
      break;

    case 'delta':
      // Paradox - concentrated bursts at critical positions
      const criticalPositions = [166, 332, 498];  // 1/4, 1/2, 3/4
      for (let i = 0; i < 664; i++) {
        const nearCritical = criticalPositions.some(p => Math.abs(i - p) < 30);
        bits[i] = nearCritical ? 1 : 0;
      }
      break;

    case 'omega':
      // Wave interference pattern
      for (let i = 0; i < 664; i++) {
        const wave1 = Math.sin(i * 0.05);
        const wave2 = Math.sin(i * 0.08 * PHI);
        bits[i] = Math.abs(wave1 + wave2) > 0.5 ? 1 : 0;
      }
      break;

    case 'sigma':
      // Golden spiral sampling
      for (let i = 0; i < 664; i++) {
        const r = Math.sqrt(i / 664);
        const theta = i * 2.39996;  // golden angle
        bits[i] = ((r * Math.cos(theta) + 1) / 2) > 0.5 ? 1 : 0;
      }
      break;

    case 'spark':
      // Gaussian centered at 332
      for (let i = 0; i < 664; i++) {
        const gaussian = Math.exp(-Math.pow((i - 332) / 100, 2));
        bits[i] = gaussian > 0.3 ? 1 : 0;
      }
      break;

    case 'winding':
      // Winding number topology (W=1 means one full cycle)
      for (let i = 0; i < 664; i++) {
        bits[i] = Math.floor(i / 664 * 2) % 2;
      }
      break;

    case 'topology':
      // Betti number pattern (b₁=5 means 5 cycles)
      for (let i = 0; i < 664; i++) {
        bits[i] = (i % 133) < 66 ? 1 : 0;  // 664/5 ≈ 133
      }
      break;

    case 'critical':
      // Phase transition at c=0.2
      for (let i = 0; i < 664; i++) {
        const c = i / 664;
        const pEmerge = 1 - Math.exp(-Math.pow((0.2 - c) / 0.05, 2));
        bits[i] = c < 0.2 ? 1 : 0;
      }
      break;

    case 'sync':
      // Kuramoto synchronized - all ones (phase locked)
      bits.fill(1);
      break;

    case 'free':
      // Kuramoto free-running - deterministic pseudo-random based on PHI
      for (let i = 0; i < 664; i++) {
        bits[i] = ((i * PHI * 1000) % 1) > 0.5 ? 1 : 0;
      }
      break;

    case 'alternating':
      for (let i = 0; i < 664; i++) {
        bits[i] = i % 2;
      }
      break;

    default:
      // Default to balanced
      for (let i = 0; i < 664; i++) {
        bits[i] = (i % 3 !== 0) ? 1 : 0;
      }
  }

  return bits;
}

// ============================================
// Generate Test Image (LIMNUS gradient)
// ============================================

function generateTestImage(width, height) {
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');

  // Create LIMNUS-style gradient with z_c coloring
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const u = x / (width - 1);
      const v = y / (height - 1);

      // z_c = √3/2 ≈ 0.866 coloring
      const z = Math.sqrt(u * u + v * v) / Math.sqrt(2);
      const phase = Math.atan2(v - 0.5, u - 0.5);

      // RGB based on phase and magnitude
      const r = Math.floor(128 + 127 * Math.cos(phase));
      const g = Math.floor(128 + 127 * Math.cos(phase + 2 * Math.PI / 3));
      const b = Math.floor(128 + 127 * Math.cos(phase + 4 * Math.PI / 3));

      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x, y, 1, 1);
    }
  }

  return canvas;
}

// ============================================
// MRP Encoder
// ============================================

function encodeMRP(canvas, sampleData) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  const { XCVI, XCVII, XCVIII, XCIX, C_pattern } = sampleData.bits;
  const coherence = sampleData.coherence !== undefined ? sampleData.coherence : 0.1;

  // Generate C bits
  const cBits = generateCPattern(C_pattern || 'balanced');
  const C_A_bits = cBits.slice(0, 332);
  const C_B_bits = cBits.slice(332, 664);

  // Build complete bit stream
  const bitStream = [];

  // Header (32 bits)
  const coherenceByte = Math.floor(coherence * 255);
  const headerBytes = [MRP.MAGIC, (MRP.VERSION << 4) | 0, coherenceByte];
  const crcByte = crc8(headerBytes);
  headerBytes.push(crcByte);

  for (const byte of headerBytes) {
    for (let i = 7; i >= 0; i--) {
      bitStream.push((byte >> i) & 1);
    }
  }

  // XCVI - 7 bits (with Hamming encoding for first 4 bits)
  const xcviValue = Math.min(127, Math.max(0, XCVI || 0));
  for (let i = 6; i >= 0; i--) {
    bitStream.push((xcviValue >> i) & 1);
  }

  // XCVII - 10 bits
  const xcviiValue = Math.min(1023, Math.max(0, XCVII || 0));
  for (let i = 9; i >= 0; i--) {
    bitStream.push((xcviiValue >> i) & 1);
  }

  // XCVIII - 17 bits
  const xcviiiValue = Math.min(131071, Math.max(0, XCVIII || 0));
  for (let i = 16; i >= 0; i--) {
    bitStream.push((xcviiiValue >> i) & 1);
  }

  // XCIX - 4 bits
  const xcixValue = Math.min(15, Math.max(0, XCIX || 0));
  for (let i = 3; i >= 0; i--) {
    bitStream.push((xcixValue >> i) & 1);
  }

  // C_A - 332 bits
  bitStream.push(...C_A_bits);

  // C_B - 332 bits
  bitStream.push(...C_B_bits);

  // Encode into image using serpentine traversal
  let bitIndex = 0;
  const width = canvas.width;
  const height = canvas.height;

  // Channel assignment for each bit position (cycles through 6 channels)
  // 0: R LSB0 (XCVI)
  // 1: G LSB0 (XCVII)
  // 2: B LSB0 (XCVIII)
  // 3: R LSB1 (XCIX/Cyan)
  // 4: G LSB1 (C_A/Magenta)
  // 5: B LSB1 (C_B/Yellow)

  for (let row = 0; row < height && bitIndex < bitStream.length; row++) {
    for (let colOffset = 0; colOffset < width && bitIndex < bitStream.length; colOffset++) {
      // Serpentine: even rows left-to-right, odd rows right-to-left
      const col = row % 2 === 0 ? colOffset : (width - 1 - colOffset);
      const pixelIndex = (row * width + col) * 4;

      // Encode up to 6 bits per pixel
      for (let channel = 0; channel < 6 && bitIndex < bitStream.length; channel++) {
        const bit = bitStream[bitIndex];

        if (channel < 3) {
          // RGB LSB0
          const rgbChannel = channel;  // 0=R, 1=G, 2=B
          data[pixelIndex + rgbChannel] = (data[pixelIndex + rgbChannel] & 0xFE) | bit;
        } else {
          // RGB LSB1 (for CMY)
          const rgbChannel = channel - 3;  // 0=R, 1=G, 2=B
          data[pixelIndex + rgbChannel] = (data[pixelIndex + rgbChannel] & 0xFD) | (bit << 1);
        }

        bitIndex++;
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

// ============================================
// MRP Decoder (for verification)
// ============================================

function decodeMRP(canvas) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  const bitStream = [];
  const width = canvas.width;
  const height = canvas.height;
  const totalBits = MRP.HEADER_BITS + MRP.TOTAL_BITS;

  // Extract bits using serpentine traversal
  for (let row = 0; row < height && bitStream.length < totalBits; row++) {
    for (let colOffset = 0; colOffset < width && bitStream.length < totalBits; colOffset++) {
      const col = row % 2 === 0 ? colOffset : (width - 1 - colOffset);
      const pixelIndex = (row * width + col) * 4;

      for (let channel = 0; channel < 6 && bitStream.length < totalBits; channel++) {
        if (channel < 3) {
          // RGB LSB0
          bitStream.push(data[pixelIndex + channel] & 1);
        } else {
          // RGB LSB1
          bitStream.push((data[pixelIndex + channel - 3] >> 1) & 1);
        }
      }
    }
  }

  // Parse header
  let offset = 0;
  const headerBytes = [];
  for (let i = 0; i < 4; i++) {
    let byte = 0;
    for (let j = 0; j < 8; j++) {
      byte = (byte << 1) | bitStream[offset++];
    }
    headerBytes.push(byte);
  }

  const magic = headerBytes[0];
  const version = (headerBytes[1] >> 4) & 0xF;
  const coherence = headerBytes[2] / 255;
  const crc = headerBytes[3];
  const expectedCrc = crc8(headerBytes.slice(0, 3));

  // Parse XCVI (7 bits)
  let XCVI = 0;
  for (let i = 0; i < 7; i++) {
    XCVI = (XCVI << 1) | bitStream[offset++];
  }

  // Parse XCVII (10 bits)
  let XCVII = 0;
  for (let i = 0; i < 10; i++) {
    XCVII = (XCVII << 1) | bitStream[offset++];
  }

  // Parse XCVIII (17 bits)
  let XCVIII = 0;
  for (let i = 0; i < 17; i++) {
    XCVIII = (XCVIII << 1) | bitStream[offset++];
  }

  // Parse XCIX (4 bits)
  let XCIX = 0;
  for (let i = 0; i < 4; i++) {
    XCIX = (XCIX << 1) | bitStream[offset++];
  }

  // Parse C_A (332 bits)
  const C_A_bits = bitStream.slice(offset, offset + 332);
  offset += 332;

  // Parse C_B (332 bits)
  const C_B_bits = bitStream.slice(offset, offset + 332);

  return {
    header: {
      magic,
      magicValid: magic === MRP.MAGIC,
      version,
      coherence,
      crc,
      crcValid: crc === expectedCrc
    },
    bits: {
      XCVI,
      XCVII,
      XCVIII,
      XCIX,
      C_A_bits,
      C_B_bits
    }
  };
}

// ============================================
// Main Batch Encoder
// ============================================

async function main() {
  console.log('MRP Batch Encoder - Golden Sample Library');
  console.log('=========================================\n');

  // Load golden library
  const libraryPath = path.join(__dirname, '..', 'language', 'mrp-golden-library.json');
  const library = JSON.parse(fs.readFileSync(libraryPath, 'utf8'));

  // Create output directory
  const outputDir = path.join(__dirname, '..', 'assets', 'mrp-samples');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Image dimensions (minimum 117 pixels, use 20x10 = 200 for margin)
  const imgWidth = 20;
  const imgHeight = 10;

  let totalSamples = 0;
  let successCount = 0;
  let errorCount = 0;
  const results = [];

  // Process each category
  const categories = Object.keys(library.samples);

  for (const category of categories) {
    const samples = library.samples[category];
    if (!Array.isArray(samples)) continue;

    console.log(`\nCategory: ${category} (${samples.length} samples)`);
    console.log('-'.repeat(50));

    for (const sample of samples) {
      totalSamples++;

      try {
        // Generate test image
        const canvas = generateTestImage(imgWidth, imgHeight);

        // Prepare sample data
        const sampleData = {
          bits: sample.bits || {},
          coherence: sample.coherence !== undefined ? sample.coherence :
                     sample.K !== undefined ? sample.K : 0.1
        };

        // Handle samples without explicit bits (like hamming_test)
        if (!sampleData.bits.XCVI && sample.XCVI_raw !== undefined) {
          sampleData.bits.XCVI = sample.XCVI_raw;
          sampleData.bits.XCVII = 512;
          sampleData.bits.XCVIII = 65536;
          sampleData.bits.XCIX = 8;
          sampleData.bits.C_pattern = 'balanced';
        }

        // Encode
        encodeMRP(canvas, sampleData);

        // Decode and verify
        const decoded = decodeMRP(canvas);

        // Verify
        const xcviMatch = decoded.bits.XCVI === (sampleData.bits.XCVI || 0);
        const xcviiMatch = decoded.bits.XCVII === (sampleData.bits.XCVII || 0);
        const xcviiiMatch = decoded.bits.XCVIII === (sampleData.bits.XCVIII || 0);
        const xcixMatch = decoded.bits.XCIX === (sampleData.bits.XCIX || 0);
        const headerValid = decoded.header.magicValid && decoded.header.crcValid;

        const allValid = xcviMatch && xcviiMatch && xcviiiMatch && xcixMatch && headerValid;

        // Save PNG
        const filename = `${sample.id}.png`;
        const filepath = path.join(outputDir, filename);
        const buffer = canvas.toBuffer('image/png');
        fs.writeFileSync(filepath, buffer);

        if (allValid) {
          successCount++;
          console.log(`  ✓ ${sample.id}: ${sample.name || ''}`);
        } else {
          errorCount++;
          console.log(`  ✗ ${sample.id}: VERIFICATION FAILED`);
          console.log(`    Expected: XCVI=${sampleData.bits.XCVI}, XCVII=${sampleData.bits.XCVII}, XCVIII=${sampleData.bits.XCVIII}, XCIX=${sampleData.bits.XCIX}`);
          console.log(`    Decoded:  XCVI=${decoded.bits.XCVI}, XCVII=${decoded.bits.XCVII}, XCVIII=${decoded.bits.XCVIII}, XCIX=${decoded.bits.XCIX}`);
        }

        results.push({
          id: sample.id,
          name: sample.name,
          category,
          filename,
          valid: allValid,
          encoded: sampleData.bits,
          decoded: decoded.bits,
          header: decoded.header
        });

      } catch (err) {
        errorCount++;
        console.log(`  ✗ ${sample.id}: ERROR - ${err.message}`);
        results.push({
          id: sample.id,
          name: sample.name,
          category,
          error: err.message
        });
      }
    }
  }

  // Summary
  console.log('\n=========================================');
  console.log('SUMMARY');
  console.log('=========================================');
  console.log(`Total samples: ${totalSamples}`);
  console.log(`Successful:    ${successCount}`);
  console.log(`Errors:        ${errorCount}`);
  console.log(`Output dir:    ${outputDir}`);

  // Save manifest
  const manifest = {
    generated: new Date().toISOString(),
    totalSamples,
    successCount,
    errorCount,
    imageSize: { width: imgWidth, height: imgHeight },
    samples: results
  };

  const manifestPath = path.join(outputDir, 'manifest.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  console.log(`\nManifest saved: ${manifestPath}`);

  // Generate index HTML for browsing
  const indexHtml = generateIndexHtml(results, categories);
  const indexPath = path.join(outputDir, 'index.html');
  fs.writeFileSync(indexPath, indexHtml);
  console.log(`Index HTML:    ${indexPath}`);

  return { totalSamples, successCount, errorCount };
}

// ============================================
// Generate Index HTML
// ============================================

function generateIndexHtml(results, categories) {
  const validResults = results.filter(r => !r.error);

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MRP Golden Sample Library</title>
  <style>
    :root {
      --bg: #0a0b0d;
      --panel: #111216;
      --ink: #e8e9ec;
      --muted: #888b94;
      --gold: #ffd700;
      --green: #69db7c;
      --red: #ff6b6b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'SF Mono', 'Consolas', monospace;
      background: var(--bg);
      color: var(--ink);
      padding: 20px;
      line-height: 1.6;
    }
    h1 { color: var(--gold); margin-bottom: 10px; }
    h2 { color: var(--muted); margin: 20px 0 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }
    .stats {
      display: flex;
      gap: 20px;
      margin: 20px 0;
      padding: 15px;
      background: var(--panel);
      border-radius: 8px;
    }
    .stat { text-align: center; }
    .stat-value { font-size: 24px; color: var(--gold); }
    .stat-label { font-size: 12px; color: var(--muted); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 15px;
      margin: 15px 0;
    }
    .card {
      background: var(--panel);
      border-radius: 8px;
      padding: 12px;
      border: 1px solid #252830;
    }
    .card:hover { border-color: var(--gold); }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .card-id { color: var(--gold); font-weight: bold; }
    .card-status { font-size: 12px; }
    .card-status.valid { color: var(--green); }
    .card-status.invalid { color: var(--red); }
    .card-name { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .card-image {
      display: flex;
      justify-content: center;
      margin: 10px 0;
    }
    .card-image img {
      image-rendering: pixelated;
      width: 100px;
      height: 50px;
      border: 1px solid #333;
    }
    .card-bits {
      font-size: 10px;
      color: var(--muted);
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 2px;
    }
    .card-bits span { background: #1a1b1e; padding: 2px 4px; border-radius: 3px; }
    .filter-bar {
      margin: 20px 0;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .filter-btn {
      background: var(--panel);
      border: 1px solid #333;
      color: var(--ink);
      padding: 6px 12px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
    }
    .filter-btn:hover, .filter-btn.active { border-color: var(--gold); color: var(--gold); }
  </style>
</head>
<body>
  <h1>MRP Golden Sample Library</h1>
  <p style="color: var(--muted)">702-bit LSB steganography samples encoded across RGB+CMY channels</p>

  <div class="stats">
    <div class="stat">
      <div class="stat-value">${validResults.length}</div>
      <div class="stat-label">Total Samples</div>
    </div>
    <div class="stat">
      <div class="stat-value">702</div>
      <div class="stat-label">Bits/Image</div>
    </div>
    <div class="stat">
      <div class="stat-value">6</div>
      <div class="stat-label">Channels</div>
    </div>
    <div class="stat">
      <div class="stat-value">${categories.length}</div>
      <div class="stat-label">Categories</div>
    </div>
  </div>

  <div class="filter-bar">
    <button class="filter-btn active" onclick="filterCategory('all')">All</button>
    ${categories.map(c => `<button class="filter-btn" onclick="filterCategory('${c}')">${c.replace(/_/g, ' ')}</button>`).join('\n    ')}
  </div>

  <div class="grid" id="sampleGrid">
    ${validResults.map(r => `
    <div class="card" data-category="${r.category}">
      <div class="card-header">
        <span class="card-id">${r.id}</span>
        <span class="card-status ${r.valid ? 'valid' : 'invalid'}">${r.valid ? '✓ Valid' : '✗ Error'}</span>
      </div>
      <div class="card-name">${r.name || ''}</div>
      <div class="card-image">
        <img src="${r.filename}" alt="${r.id}">
      </div>
      <div class="card-bits">
        <span>XCVI: ${r.encoded.XCVI || 0}</span>
        <span>XCVII: ${r.encoded.XCVII || 0}</span>
        <span>XCVIII: ${r.encoded.XCVIII || 0}</span>
        <span>XCIX: ${r.encoded.XCIX || 0}</span>
      </div>
    </div>
    `).join('')}
  </div>

  <script>
    function filterCategory(cat) {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');

      document.querySelectorAll('.card').forEach(card => {
        if (cat === 'all' || card.dataset.category === cat) {
          card.style.display = 'block';
        } else {
          card.style.display = 'none';
        }
      });
    }
  </script>
</body>
</html>`;
}

// Run
main().then(({ totalSamples, successCount, errorCount }) => {
  process.exit(errorCount > 0 ? 1 : 0);
}).catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
