#!/usr/bin/env python3
"""
Dashboard Standards Compliance Tests
ENFORCES dark theme standards and requirements
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
import re

class DashboardStandardsTests:
    """
    Tests that ENFORCE standards compliance
    """
    
    def __init__(self):
        self.violations = []
        self.passed = []
        
    def test_no_emojis(self, html_content: str):
        """NO EMOJIS ALLOWED"""
        print("\nTEST: No Emojis")
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"  # dingbats
            "\u3030"
            "]+", 
            re.UNICODE
        )
        
        emojis_found = emoji_pattern.findall(html_content)
        if emojis_found:
            self.violations.append(f"FOUND {len(emojis_found)} EMOJIS: {emojis_found}")
            print(f"  ❌ FAILED: Found emojis in dashboard")
        else:
            self.passed.append("No emojis found")
            print(f"  ✓ PASSED: No emojis")
            
    def test_no_gridlines(self, html_content: str):
        """NO GRIDLINES IN CHARTS"""
        print("\nTEST: No Gridlines")
        
        # Check for grid configurations
        grid_patterns = [
            r"grid:\s*{[^}]*display:\s*true",
            r"grid:\s*{[^}]*color:",
            r"drawOnChartArea:\s*true"
        ]
        
        violations = []
        for pattern in grid_patterns:
            if re.search(pattern, html_content):
                violations.append(f"Found grid pattern: {pattern}")
        
        if violations:
            self.violations.extend(violations)
            print(f"  ❌ FAILED: Found gridlines configurations")
        else:
            self.passed.append("No gridlines found")
            print(f"  ✓ PASSED: No gridlines")
            
    def test_dark_theme_colors(self, html_content: str):
        """ONLY APPROVED COLORS"""
        print("\nTEST: Color Compliance")
        
        # Approved colors
        approved_colors = {
            '#6f42c1': 'Primary Purple',
            '#272b30': 'Primary Charcoal', 
            '#FFFFFF': 'White',
            '#ffffff': 'White',
            '#e9ecef': 'Light Gray',
            '#999999': 'Medium Gray',
            '#52575c': 'Dark Gray',
            '#7a8288': 'Border Gray'
        }
        
        # Forbidden colors
        forbidden_patterns = [
            r'#[fF][fF]0000',  # Red
            r'#[0-9a-fA-F]{0,2}[fF][fF][0-9a-fA-F]{0,2}',  # Greenish
            r'#[fF][fF][fF][fF]00',  # Yellow
            r'#0000[fF][fF]',  # Blue
            r'\bcolor:\s*red\b|\bbackground-color:\s*red\b|\bbackgroundColor:\s*["\']red["\']',  # Color property red
            r'\bcolor:\s*green\b|\bbackground-color:\s*green\b|\bbackgroundColor:\s*["\']green["\']',  # Color property green
            r'\bcolor:\s*yellow\b|\bbackground-color:\s*yellow\b|\bbackgroundColor:\s*["\']yellow["\']',  # Color property yellow
            r'\bcolor:\s*blue\b|\bbackground-color:\s*blue\b|\bbackgroundColor:\s*["\']blue["\']',  # Color property blue
            r'\bcolor:\s*orange\b|\bbackground-color:\s*orange\b|\bbackgroundColor:\s*["\']orange["\']',  # Color property orange
            r'rgb\s*\(\s*255\s*,\s*0\s*,\s*0',  # RGB red
            r'rgb\s*\(\s*0\s*,\s*255\s*,\s*0',  # RGB green
        ]
        
        violations = []
        for pattern in forbidden_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            if matches:
                violations.append(f"Found forbidden color pattern: {matches}")
        
        if violations:
            self.violations.extend(violations)
            print(f"  ❌ FAILED: Found forbidden colors")
        else:
            self.passed.append("Color compliance verified")
            print(f"  ✓ PASSED: Only approved colors")
            
    def test_actual_vs_predicted_chart(self, html_content: str):
        """ACTUAL VS PREDICTED MUST BE IN CHART"""
        print("\nTEST: Actual vs Predicted Chart")
        
        required_elements = [
            "label: 'Actual Price'",
            "label: 'Predicted Price'",
            "actualPrices",
            "predictedPrices"
        ]
        
        missing = []
        for element in required_elements:
            if element not in html_content:
                missing.append(element)
        
        if missing:
            self.violations.append(f"Missing required chart elements: {missing}")
            print(f"  ❌ FAILED: Actual vs Predicted chart incomplete")
        else:
            self.passed.append("Actual vs Predicted chart present")
            print(f"  ✓ PASSED: Chart has both actual and predicted")
            
    def test_chart_styling(self, html_content: str):
        """CHARTS MUST FOLLOW DARK THEME"""
        print("\nTEST: Chart Dark Theme Styling")
        
        required_styles = [
            "backgroundColor: '#000000'",  # Black background for tooltips
            "borderColor: '#7a8288'",  # Border gray
            "color: '#e9ecef'",  # Light gray text
            "grid: {",  # Grid config must exist
            "display: false"  # But be turned off
        ]
        
        missing = []
        for style in required_styles:
            if style not in html_content:
                missing.append(style)
        
        if missing:
            self.violations.append(f"Missing required chart styles: {missing}")
            print(f"  ❌ FAILED: Chart styling non-compliant")
        else:
            self.passed.append("Chart styling compliant")
            print(f"  ✓ PASSED: Charts follow dark theme")
            
    def generate_compliance_report(self):
        """Generate compliance report"""
        print("\n" + "="*60)
        print("STANDARDS COMPLIANCE REPORT")
        print("="*60)
        
        total_tests = len(self.passed) + len(self.violations)
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {len(self.passed)}")
        print(f"Failed: {len(self.violations)}")
        
        if self.violations:
            print("\n🚨 VIOLATIONS FOUND:")
            for i, violation in enumerate(self.violations, 1):
                print(f"  {i}. {violation}")
            
            print("\n⚠️  DASHBOARD IS NOT COMPLIANT!")
            return False
        else:
            print("\n✅ ALL STANDARDS TESTS PASSED")
            return True
            
    def test_dashboard_file(self, filepath: str):
        """Test a dashboard file for compliance"""
        print(f"\nTesting: {filepath}")
        print("-" * 40)
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Run all tests
            self.test_no_emojis(content)
            self.test_no_gridlines(content)
            self.test_dark_theme_colors(content)
            self.test_actual_vs_predicted_chart(content)
            self.test_chart_styling(content)
            
            return self.generate_compliance_report()
            
        except Exception as e:
            print(f"Error reading file: {e}")
            return False

def main():
    """Run standards compliance tests"""
    print("\n" + "#"*60)
    print("# DASHBOARD STANDARDS COMPLIANCE TESTS")
    print("#"*60)
    
    tester = DashboardStandardsTests()
    
    # Test the compliant dashboard
    dashboard_file = "src/dashboard/app_zen_compliant.py"
    
    if os.path.exists(dashboard_file):
        compliant = tester.test_dashboard_file(dashboard_file)
        
        if not compliant:
            print("\n⚠️  FIX ALL VIOLATIONS BEFORE DEPLOYMENT!")
            sys.exit(1)
    else:
        print(f"\nError: Dashboard file not found: {dashboard_file}")
        sys.exit(1)

if __name__ == "__main__":
    main()