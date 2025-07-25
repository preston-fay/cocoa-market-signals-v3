"""
Color Validation Script
Ensures all colors in the codebase follow Kearney standards
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# OFFICIAL KEARNEY COLORS from preston-dev-setup/kearney_design_system.py
ALLOWED_COLORS = {
    # Primary Brand Colors
    "#6f42c1",  # primary_purple
    "#9955bb",  # secondary_purple  
    "#6610f2",  # indigo_accent
    
    # Charcoal & Gray Scheme
    "#272b30",  # primary_charcoal
    "#3a3f44",  # secondary_charcoal
    "#52575c",  # medium_charcoal
    "#7a8288",  # light_charcoal
    
    # Extended Gray Scale
    "#ffffff",  # white
    "#f8f9fa",  # light_gray_100
    "#e9ecef",  # light_gray_200
    "#dee2e6",  # light_gray_300
    "#ced4da",  # light_gray_400
    "#999999",  # medium_gray_500
    "#7a8288",  # medium_gray_600 (duplicate, that's ok)
    "#52575c",  # dark_gray_700 (duplicate, that's ok)
    "#3a3f44",  # dark_gray_800 (duplicate, that's ok)
    "#272b30",  # dark_gray_900 (duplicate, that's ok)
    "#000000",  # black
    
    # Professional Blues (limited use)
    "#004466",  # professional_blue
    "#b3fff0",  # accent_teal
    "#004d99",  # link_blue
    
    # Medical/Healthcare (from Coach app)
    "#007f6d",  # medical_systolic
    "#996124",  # medical_diastolic
}

# FORBIDDEN COLORS
FORBIDDEN_COLORS = [
    # Pure RGB
    "#ff0000", "#00ff00", "#0000ff",
    # Common web colors  
    "#ffa500", "#ff6347", "#32cd32",  # Orange, Tomato, Green
    "#00ffff", "#ff00ff", "#ffff00",  # Cyan, Magenta, Yellow
    # Bootstrap colors
    "#dc3545", "#28a745", "#17a2b8",  # danger, success, info
    "#007bff", "#6c757d", "#f8f9fa",  # primary, secondary, light
    # Tailwind colors
    "#ef4444", "#10b981", "#3b82f6",  # red, green, blue
    "#f59e0b", "#8b5cf6", "#ec4899",  # amber, violet, pink
    # Other common non-standard colors
    "#ff6b6b", "#4ecdc4", "#ffe66d",  # Custom reds, teals, yellows
    "#5f5f5f",  # Non-standard gray
]

def find_color_hex(text: str) -> List[Tuple[str, int]]:
    """Find all hex color codes in text"""
    # Pattern matches #RGB and #RRGGBB formats
    pattern = r'#[0-9A-Fa-f]{3}(?:[0-9A-Fa-f]{3})?'
    matches = []
    for match in re.finditer(pattern, text):
        color = match.group().lower()
        matches.append((color, match.start()))
    return matches

def validate_file(file_path: Path) -> Dict[str, List]:
    """Validate colors in a single file"""
    violations = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        return {'violations': [], 'error': f"Could not read {file_path}: {str(e)}"}
    
    # Find all colors
    colors = find_color_hex(content)
    
    for color, position in colors:
        color_lower = color.lower()
        
        # Check if color is allowed
        if color_lower not in [c.lower() for c in ALLOWED_COLORS]:
            # Find line number
            line_num = content[:position].count('\n') + 1
            line_content = lines[line_num - 1].strip()
            
            # Check if it's a known forbidden color
            is_forbidden = color_lower in [c.lower() for c in FORBIDDEN_COLORS]
            
            violations.append({
                'color': color,
                'line': line_num,
                'content': line_content,
                'type': 'FORBIDDEN' if is_forbidden else 'NON_STANDARD',
                'position': position
            })
    
    return {'violations': violations}

def validate_directory(directory: str, extensions: List[str] = ['.py', '.js', '.jsx', '.ts', '.tsx', '.css', '.html']) -> Dict[str, Dict]:
    """Validate all files in directory"""
    results = {}
    path = Path(directory)
    
    for ext in extensions:
        for file_path in path.rglob(f'*{ext}'):
            # Skip node_modules, __pycache__, etc
            if any(part in str(file_path) for part in ['node_modules', '__pycache__', '.git', 'venv', '.pytest_cache']):
                continue
                
            result = validate_file(file_path)
            if result['violations']:
                results[str(file_path)] = result
    
    return results

def print_validation_report(results: Dict[str, Dict]):
    """Print a formatted validation report"""
    if not results:
        logger.info("✅ SUCCESS: All colors follow Kearney standards!")
        return
    
    logger.warning("❌ COLOR STANDARD VIOLATIONS FOUND:\n")
    
    total_violations = 0
    
    for file_path, data in results.items():
        violations = data['violations']
        if violations:
            logger.warning(f"\n📄 {file_path}")
            logger.warning(f"   Found {len(violations)} violations:")
            
            for v in violations:
                total_violations += 1
                logger.warning(f"   Line {v['line']}: {v['type']} color {v['color']}")
                logger.warning(f"   > {v['content'][:80]}...")
    
    logger.warning(f"\n❌ Total violations: {total_violations}")
    logger.info("\n✅ ALLOWED COLORS:")
    logger.info("Primary: #6f42c1 (purple)")
    logger.info("Background: #272b30 (charcoal)")
    logger.info("Grays: #999999, #7a8288, #52575c, #e9ecef")
    logger.info("Black/White: #000000, #FFFFFF")
    
    return total_violations

if __name__ == "__main__":
    # Validate the entire src directory
    print("🎨 Validating Kearney Color Standards...\n")
    
    src_results = validate_directory("/Users/pfay01/Projects/cocoa-market-signals-v3/src")
    templates_results = validate_directory("/Users/pfay01/Projects/cocoa-market-signals-v3/templates", ['.html'])
    static_results = validate_directory("/Users/pfay01/Projects/cocoa-market-signals-v3/static", ['.css', '.js'])
    
    all_results = {**src_results, **templates_results, **static_results}
    
    violation_count = print_validation_report(all_results)
    
    # Exit with error code if violations found
    import sys
    sys.exit(1 if violation_count > 0 else 0)