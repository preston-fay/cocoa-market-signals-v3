#!/usr/bin/env python3
"""
Validate the showcase dashboard against Kearney standards
"""
import sys
sys.path.append('.')

from src.agents.standards_enforcement_agent import StandardsEnforcementAgent
from src.validation.color_validator import validate_file

def validate_showcase():
    """Validate the showcase dashboard"""
    print("🔍 Validating Showcase Dashboard Against Kearney Standards")
    print("=" * 60)
    
    # Initialize agents
    standards_agent = StandardsEnforcementAgent()
    
    # Files to validate
    files_to_check = [
        'templates/dashboard_showcase.html',
        'src/dashboard/app_showcase.py'
    ]
    
    all_valid = True
    results = []
    
    for file in files_to_check:
        print(f"\n📄 Checking: {file}")
        
        # Use standards agent
        result = standards_agent.validate_html_file(file)
        results.append(result)
        
        if result['valid']:
            print("   ✅ Standards compliant")
        else:
            print("   ❌ Violations found:")
            for error in result['errors']:
                print(f"      - {error}")
            all_valid = False
            
        if result['warnings']:
            print("   ⚠️  Warnings:")
            for warning in result['warnings']:
                print(f"      - {warning}")
    
    # Additional Python file validation
    print(f"\n📄 Checking Python colors in: src/dashboard/app_showcase.py")
    try:
        with open('src/dashboard/app_showcase.py', 'r') as f:
            content = f.read()
            
        # Check for forbidden colors in Python
        forbidden_found = False
        for color in standards_agent.FORBIDDEN_COLORS:
            if color in content:
                print(f"   ❌ Forbidden color found: {color}")
                forbidden_found = True
                all_valid = False
                
        if not forbidden_found:
            print("   ✅ No forbidden colors in Python code")
            
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        all_valid = False
    
    # Generate compliance report
    report = standards_agent.create_standards_report(results)
    
    print("\n" + "=" * 60)
    print("📊 COMPLIANCE SUMMARY")
    print("=" * 60)
    
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED - Dashboard is 100% Kearney compliant!")
    else:
        print("❌ Validation failed - See errors above")
        
    # Save report
    with open('STANDARDS_COMPLIANCE_SHOWCASE.md', 'w') as f:
        f.write(report)
    print(f"\n📄 Full report saved to: STANDARDS_COMPLIANCE_SHOWCASE.md")
    
    return all_valid


if __name__ == "__main__":
    valid = validate_showcase()
    sys.exit(0 if valid else 1)