#!/usr/bin/env python3
"""
Automatically fix all color violations to follow Kearney standards
"""

import re
import os
from pathlib import Path

# Color mappings
COLOR_MAPPINGS = {
    # Non-standard purples to Kearney purple
    "#7823dc": "#6f42c1",
    "#9b4ae3": "#6f42c1",
    
    # Non-standard charcoals/blacks to Kearney charcoal
    "#1e1e1e": "#272b30",
    "#323232": "#52575c",
    "#1f2937": "#272b30",
    "#303030": "#3a3f44",
    
    # Non-standard grays to Kearney grays
    "#f5f5f5": "#e9ecef",
    "#e6e6e6": "#e9ecef",
    "#a5a5a5": "#999999",
    "#5f5f5f": "#7a8288",
    
    # Forbidden colors to Kearney alternatives
    # Reds to dark gray
    "#ff0000": "#52575c",
    "#ef4444": "#52575c",
    "#dc3545": "#52575c",
    "#ff6347": "#52575c",
    "#ff6b6b": "#52575c",
    
    # Greens to purple
    "#00ff00": "#6f42c1",
    "#10b981": "#6f42c1",
    "#28a745": "#6f42c1",
    "#32cd32": "#6f42c1",
    "#4ecdc4": "#6f42c1",
    
    # Blues to purple
    "#0000ff": "#6f42c1",
    "#3b82f6": "#6f42c1",
    "#007bff": "#6f42c1",
    "#17a2b8": "#6f42c1",
    
    # Yellows/oranges to medium gray
    "#ffff00": "#999999",
    "#f59e0b": "#999999",
    "#ffa500": "#999999",
    "#ffe66d": "#999999",
    "#ffc107": "#999999",
    
    # Other colors
    "#00ffff": "#6f42c1",  # Cyan to purple
    "#ff00ff": "#999999",  # Magenta to gray
    "#8b5cf6": "#6f42c1",  # Violet to purple
    "#ec4899": "#999999",  # Pink to gray
    "#6c757d": "#7a8288",  # Bootstrap secondary to border gray
}

def fix_colors_in_file(file_path: Path) -> int:
    """Fix color violations in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            original_content = content
    except Exception as e:
        print(f"⚠️  Could not read {file_path}: {str(e)}")
        return 0
    
    replacements = 0
    
    # Fix each color mapping
    for old_color, new_color in COLOR_MAPPINGS.items():
        # Case insensitive replacement
        pattern = re.compile(re.escape(old_color), re.IGNORECASE)
        matches = pattern.findall(content)
        if matches:
            content = pattern.sub(new_color, content)
            replacements += len(matches)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed {replacements} colors in {file_path}")
    
    return replacements

def fix_all_colors(directory: str) -> int:
    """Fix all color violations in directory"""
    total_fixed = 0
    
    # Process Python files
    for file_path in Path(directory).rglob("*.py"):
        if any(skip in str(file_path) for skip in ['__pycache__', 'venv', '.git']):
            continue
        total_fixed += fix_colors_in_file(file_path)
    
    # Process HTML files
    for file_path in Path(directory).rglob("*.html"):
        total_fixed += fix_colors_in_file(file_path)
    
    # Process CSS files
    for file_path in Path(directory).rglob("*.css"):
        total_fixed += fix_colors_in_file(file_path)
    
    # Process JS files
    for file_path in Path(directory).rglob("*.js"):
        total_fixed += fix_colors_in_file(file_path)
    
    return total_fixed

if __name__ == "__main__":
    print("🎨 Fixing all color violations to match Kearney standards...\n")
    
    # Fix src directory
    src_fixed = fix_all_colors("/Users/pfay01/Projects/cocoa-market-signals-v3/src")
    
    # Fix templates directory
    templates_fixed = fix_all_colors("/Users/pfay01/Projects/cocoa-market-signals-v3/templates")
    
    # Fix static directory
    static_fixed = fix_all_colors("/Users/pfay01/Projects/cocoa-market-signals-v3/static")
    
    total = src_fixed + templates_fixed + static_fixed
    
    print(f"\n✅ TOTAL COLORS FIXED: {total}")
    print("\nNow running validation to confirm...")
    
    # Run validation
    os.system("python3 /Users/pfay01/Projects/cocoa-market-signals-v3/src/validation/color_validator.py")