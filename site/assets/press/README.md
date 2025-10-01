# Press Assets README

## Overview
This directory contains placeholder assets for the FBA-Bench press kit. Replace these with official brand materials when available.

## Required Assets
- **og-image.png**: Open Graph image (recommended: 1200x630 pixels, PNG format). Used for social sharing.
- **logo-light.svg**: Light mode logo (SVG for scalability; provide transparent background).
- **logo-dark.svg**: Dark mode logo (SVG; adapt to dark themes).
- **logo.png**: Fallback raster logo (PNG, 512x512 or similar square for versatility).

## Additional Files
- **brand-colors.md**: Brand color palette with hex codes and usage notes.
- **brand-typography.md**: Typography guidelines (font families, sizes, weights).
- **brand-guidelines.md**: Overall brand usage rules.

## Replacement Instructions
1. Obtain official assets from the design team.
2. Ensure formats and dimensions match recommendations above.
3. For images: Optimize for web (under 100KB where possible); use tools like ImageOptim or TinyPNG.
4. Update references in `site/press.html` if paths change.
5. Regenerate the press kit ZIP using the command below.

## Generating press-kit.zip
Run this PowerShell command from the repository root to create the ZIP:

```powershell
Compress-Archive -Path "repos/fba-bench-core/site/assets/press/*" -DestinationPath "repos/fba-bench-core/site/assets/press/press-kit.zip" -Force
```

This includes all files in the directory (excluding the ZIP itself). Re-run after replacing assets.

## Formats and Dimensions
- SVGs: Vector, scalable; no fixed dimensions.
- PNGs: Raster, 72-96 DPI; avoid JPEG for logos to preserve transparency.
- Expected sizes:
  - Logos: Minimum 256x256 px.
  - OG Image: Exactly 1200x630 px for optimal display.