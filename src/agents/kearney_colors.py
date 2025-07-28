#!/usr/bin/env python3
"""
Official Kearney Color Palette
Exact colors from the Kearney Design System
"""

# Primary Colors
KEARNEY_COLORS = {
    # Core colors
    'charcoal': '#1E1E1E',  # RGB 30 30 30
    'white': '#FFFFFF',      # RGB 255 255 255
    'primary_purple': '#7823DC',  # RGB 120 35 220
    
    # Chart colors (in order)
    'chart_1': '#D2D2D2',  # RGB 210 210 210
    'chart_2': '#A5A5A5',  # RGB 165 165 165
    'chart_3': '#787878',  # RGB 120 120 120
    'chart_4': '#E6D2FA',  # RGB 230 210 250
    'chart_5': '#C8A5F0',  # RGB 200 165 240
    'chart_6': '#AF7DEB',  # RGB 175 125 235
    
    # Additional colors (use sparingly)
    'gray_1': '#F5F5F5',   # RGB 245 245 245
    'gray_2': '#E6E6E6',   # RGB 230 230 230
    'gray_3': '#B9B9B9',   # RGB 185 185 185
    'gray_4': '#8C8C8C',   # RGB 140 140 140
    'gray_5': '#5F5F5F',   # RGB 95 95 95
    'gray_6': '#323232',   # RGB 50 50 50
    'gray_7': '#4B4B4B',   # RGB 75 75 75
    'gray_8': '#1E1E1E',   # RGB 30 30 30 (same as charcoal)
    
    # Purple variations
    'purple_1': '#8737E1',  # RGB 135 55 225
    'purple_2': '#A064E6',  # RGB 160 100 230
    'purple_3': '#9150E1',  # RGB 145 80 225
    'purple_4': '#7823DC',  # RGB 120 35 220 (same as primary)
    'purple_5': '#B991EB',  # RGB 185 145 235
    'purple_6': '#D7BEF5',  # RGB 215 190 245
}

# Quick access groups
CHART_COLORS = [
    KEARNEY_COLORS['chart_1'],
    KEARNEY_COLORS['chart_2'],
    KEARNEY_COLORS['chart_3'],
    KEARNEY_COLORS['chart_4'],
    KEARNEY_COLORS['chart_5'],
    KEARNEY_COLORS['chart_6']
]

GRAY_SCALE = [
    KEARNEY_COLORS['white'],
    KEARNEY_COLORS['gray_1'],
    KEARNEY_COLORS['gray_2'],
    KEARNEY_COLORS['gray_3'],
    KEARNEY_COLORS['gray_4'],
    KEARNEY_COLORS['gray_5'],
    KEARNEY_COLORS['gray_6'],
    KEARNEY_COLORS['gray_7'],
    KEARNEY_COLORS['charcoal']
]

PURPLE_SCALE = [
    KEARNEY_COLORS['purple_6'],
    KEARNEY_COLORS['purple_5'],
    KEARNEY_COLORS['chart_4'],
    KEARNEY_COLORS['chart_5'],
    KEARNEY_COLORS['chart_6'],
    KEARNEY_COLORS['purple_3'],
    KEARNEY_COLORS['purple_2'],
    KEARNEY_COLORS['purple_1'],
    KEARNEY_COLORS['primary_purple']
]

# CSS Variables
def get_css_variables():
    """Generate CSS variables for all Kearney colors"""
    css = ":root {\n"
    for name, color in KEARNEY_COLORS.items():
        css_var_name = name.replace('_', '-')
        css += f"    --kearney-{css_var_name}: {color};\n"
    css += "}\n"
    return css

# Validation function
def is_valid_kearney_color(color: str) -> bool:
    """Check if a color is in the official Kearney palette"""
    color = color.upper().strip()
    return color in [c.upper() for c in KEARNEY_COLORS.values()]

# Get complementary colors for data visualization
def get_chart_palette(n_colors: int = 6):
    """Get n colors from the chart palette"""
    if n_colors <= len(CHART_COLORS):
        return CHART_COLORS[:n_colors]
    else:
        # If more colors needed, add purple variations
        extended = CHART_COLORS + [
            KEARNEY_COLORS['purple_3'],
            KEARNEY_COLORS['purple_2'],
            KEARNEY_COLORS['purple_5']
        ]
        return extended[:n_colors]