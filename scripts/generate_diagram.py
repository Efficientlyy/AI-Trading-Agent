#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to convert ASCII art diagrams to PNG images.
"""

import os
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def create_image_from_ascii(ascii_file, output_file, font_size=14, padding=20):
    """
    Convert ASCII art to a PNG image.
    
    Args:
        ascii_file: Path to the file containing ASCII art
        output_file: Path to save the PNG image
        font_size: Font size to use
        padding: Padding around the text in pixels
    """
    # Read the ASCII art file
    with open(ascii_file, 'r') as f:
        ascii_text = f.read()
    
    # Remove the markdown code block markers if present
    ascii_text = ascii_text.replace('```', '').strip()
    
    # Split into lines
    lines = ascii_text.split('\n')
    
    # Find the maximum line length
    max_line_length = max(len(line) for line in lines)
    
    # Create a monospace font
    try:
        font = ImageFont.truetype("Courier New", font_size)
    except IOError:
        # Fallback to default
        font = ImageFont.load_default()
    
    # Estimate text dimensions
    char_width = font_size * 0.6  # Approximate width of a monospace character
    char_height = font_size * 1.2  # Approximate height including line spacing
    
    # Calculate image dimensions
    width = int(max_line_length * char_width + 2 * padding)
    height = int(len(lines) * char_height + 2 * padding)
    
    # Create a white image
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw the text
    for i, line in enumerate(lines):
        draw.text(
            (padding, padding + i * char_height),
            line,
            font=font,
            fill='black'
        )
    
    # Save the image
    image.save(output_file)
    print(f"Image saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert ASCII art to PNG image')
    parser.add_argument('input', help='ASCII art file path')
    parser.add_argument('--output', '-o', help='Output PNG file path')
    parser.add_argument('--font-size', '-f', type=int, default=14, help='Font size')
    parser.add_argument('--padding', '-p', type=int, default=20, help='Padding in pixels')
    
    args = parser.parse_args()
    
    # If output path is not specified, use input filename with .png extension
    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.with_suffix('.png')
        args.output = str(output_path)
    
    create_image_from_ascii(args.input, args.output, args.font_size, args.padding)

if __name__ == "__main__":
    main() 