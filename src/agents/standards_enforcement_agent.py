#!/usr/bin/env python3
"""
Standards Enforcement Agent - Ensures 100% Kearney Design System Compliance
This agent validates all dashboard components against standards
"""
import re
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StandardsEnforcementAgent:
    """
    Enforces Kearney design standards across all dashboard components
    """
    
    # Kearney color palette - ONLY these colors allowed
    ALLOWED_COLORS = {
        'primary_purple': '#6f42c1',
        'primary_charcoal': '#272b30',
        'white': '#FFFFFF',
        'light_gray': '#e9ecef',
        'medium_gray': '#999999',
        'dark_gray': '#52575c',
        'border_gray': '#7a8288'
    }
    
    # Forbidden colors - NEVER use these
    FORBIDDEN_COLORS = [
        '#ff0000', '#ef4444', '#dc3545',  # Reds
        '#00ff00', '#10b981', '#28a745',  # Greens
        '#ffff00', '#f59e0b', '#ffa500',  # Yellows/Oranges
        '#0000ff', '#3b82f6', '#007bff'   # Blues
    ]
    
    # Required dashboard elements
    REQUIRED_ELEMENTS = {
        'dark_theme': True,
        'feather_icons': True,
        'responsive_design': True,
        'accessibility': True
    }
    
    def validate_color(self, color: str) -> Tuple[bool, Optional[str]]:
        """Validate a single color against standards"""
        color = color.lower().strip()
        
        # Check if it's an allowed color
        if color in [c.lower() for c in self.ALLOWED_COLORS.values()]:
            return True, None
            
        # Check if it's a forbidden color
        if color in self.FORBIDDEN_COLORS:
            return False, f"FORBIDDEN COLOR: {color}. Use Kearney palette only!"
            
        # Check for RGB/RGBA formats
        if 'rgb' in color:
            return False, f"Non-standard color: {color}. Use hex values from Kearney palette!"
            
        return False, f"Unknown color: {color}. Only use approved Kearney colors!"
    
    def validate_html_file(self, filepath: str) -> Dict[str, any]:
        """Validate an HTML file for standards compliance"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            results = {
                'file': filepath,
                'valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Check for forbidden colors
            color_pattern = r'#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b|rgb\([^)]+\)|rgba\([^)]+\)'
            colors_found = re.findall(color_pattern, content)
            
            for color in colors_found:
                valid, error = self.validate_color(color)
                if not valid:
                    results['valid'] = False
                    results['errors'].append(error)
            
            # Check for dark theme
            if 'bg-dark' not in content and 'background-color: #272b30' not in content:
                results['warnings'].append("Dark theme not detected")
            
            # Check for Feather icons
            if 'feather' not in content:
                results['warnings'].append("Feather icons not detected")
                
            return results
            
        except Exception as e:
            return {
                'file': filepath,
                'valid': False,
                'errors': [f"Error reading file: {str(e)}"],
                'warnings': []
            }
    
    def generate_compliant_css(self) -> str:
        """Generate CSS that follows all Kearney standards"""
        return f"""
/* Kearney Design System - Dark Theme Only */
:root {{
    --primary-purple: {self.ALLOWED_COLORS['primary_purple']};
    --primary-charcoal: {self.ALLOWED_COLORS['primary_charcoal']};
    --white: {self.ALLOWED_COLORS['white']};
    --light-gray: {self.ALLOWED_COLORS['light_gray']};
    --medium-gray: {self.ALLOWED_COLORS['medium_gray']};
    --dark-gray: {self.ALLOWED_COLORS['dark_gray']};
    --border-gray: {self.ALLOWED_COLORS['border_gray']};
}}

body {{
    background-color: var(--primary-charcoal);
    color: var(--white);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}

.card {{
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-gray);
    border-radius: 8px;
}}

.text-primary {{
    color: var(--primary-purple) !important;
}}

.btn-primary {{
    background-color: var(--primary-purple);
    border-color: var(--primary-purple);
    color: var(--white);
}}

.btn-primary:hover {{
    background-color: var(--dark-gray);
    border-color: var(--dark-gray);
}}

/* Charts must use only approved colors */
.chart-line-actual {{
    stroke: var(--white);
}}

.chart-line-predicted {{
    stroke: var(--primary-purple);
}}

.signal-marker {{
    fill: var(--primary-purple);
    stroke: var(--white);
}}
"""
    
    def enforce_standards(self, component: str) -> str:
        """Enforce standards on a component and return corrected version"""
        # Replace forbidden colors
        for forbidden in self.FORBIDDEN_COLORS:
            if forbidden in component:
                # Map to appropriate Kearney color
                if 'ff0000' in forbidden or 'ef4444' in forbidden:
                    component = component.replace(forbidden, self.ALLOWED_COLORS['primary_purple'])
                elif '00ff00' in forbidden or '10b981' in forbidden:
                    component = component.replace(forbidden, self.ALLOWED_COLORS['light_gray'])
                else:
                    component = component.replace(forbidden, self.ALLOWED_COLORS['medium_gray'])
                    
        return component
    
    def create_standards_report(self, results: List[Dict]) -> str:
        """Create a comprehensive standards compliance report"""
        total_files = len(results)
        compliant_files = sum(1 for r in results if r['valid'])
        
        report = f"""
# Kearney Standards Compliance Report

## Summary
- Total Files Checked: {total_files}
- Compliant Files: {compliant_files}
- Compliance Rate: {(compliant_files/total_files)*100:.1f}%

## Allowed Colors (Kearney Palette)
- Primary Purple: #6f42c1
- Primary Charcoal: #272b30
- White: #FFFFFF
- Light Gray: #e9ecef
- Medium Gray: #999999
- Dark Gray: #52575c
- Border Gray: #7a8288

## Validation Results
"""
        for result in results:
            report += f"\n### {result['file']}\n"
            report += f"- Status: {'✅ Compliant' if result['valid'] else '❌ Non-compliant'}\n"
            
            if result['errors']:
                report += f"- Errors:\n"
                for error in result['errors']:
                    report += f"  - {error}\n"
                    
            if result['warnings']:
                report += f"- Warnings:\n"
                for warning in result['warnings']:
                    report += f"  - {warning}\n"
                    
        return report


if __name__ == "__main__":
    # Test the agent
    agent = StandardsEnforcementAgent()
    
    # Generate compliant CSS
    css = agent.generate_compliant_css()
    print("Generated Kearney-compliant CSS")
    
    # Test color validation
    test_colors = ['#6f42c1', '#ff0000', '#272b30', 'rgb(255,0,0)']
    for color in test_colors:
        valid, error = agent.validate_color(color)
        print(f"Color {color}: {'Valid' if valid else error}")