"""
Standards Enforcement System
Ensures ALL preston-dev-setup standards are followed
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re
import logging

# Import color validator
from color_validator import validate_directory as validate_colors

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class StandardsEnforcer:
    """Comprehensive standards enforcement following preston-dev-setup"""
    
    def __init__(self):
        self.violations = []
        self.standards_path = "/Users/pfay01/Projects/preston-dev-setup"
        
    def validate_all_standards(self, project_path: str) -> bool:
        """Validate ALL standards from preston-dev-setup"""
        print("🔍 VALIDATING ALL PRESTON-DEV-SETUP STANDARDS...\n")
        
        # 1. Color Standards
        print("1️⃣ Checking Kearney Color Standards...")
        color_violations = self._check_color_standards(project_path)
        
        # 2. Development Standards
        print("\n2️⃣ Checking Development Standards...")
        dev_violations = self._check_development_standards(project_path)
        
        # 3. Data Science Standards
        print("\n3️⃣ Checking Data Science Standards...")
        ds_violations = self._check_data_science_standards(project_path)
        
        # 4. No Fake Data
        print("\n4️⃣ Checking for Fake/Synthetic Data...")
        fake_data_violations = self._check_no_fake_data(project_path)
        
        # 5. Documentation Standards
        print("\n5️⃣ Checking Documentation Standards...")
        doc_violations = self._check_documentation_standards(project_path)
        
        # Summary
        total_violations = (
            color_violations + dev_violations + ds_violations + 
            fake_data_violations + doc_violations
        )
        
        print("\n" + "="*60)
        print("STANDARDS VALIDATION SUMMARY")
        print("="*60)
        print(f"✅ Color Standards: {color_violations} violations")
        print(f"✅ Development Standards: {dev_violations} violations")
        print(f"✅ Data Science Standards: {ds_violations} violations")
        print(f"✅ No Fake Data: {fake_data_violations} violations")
        print(f"✅ Documentation: {doc_violations} violations")
        print(f"\n{'✅ ALL STANDARDS FOLLOWED!' if total_violations == 0 else f'❌ TOTAL VIOLATIONS: {total_violations}'}")
        
        return total_violations == 0
    
    def _check_color_standards(self, project_path: str) -> int:
        """Check Kearney color standards"""
        from color_validator import validate_directory, print_validation_report
        
        src_results = validate_directory(f"{project_path}/src")
        templates_results = validate_directory(f"{project_path}/templates", ['.html'])
        static_results = validate_directory(f"{project_path}/static", ['.css', '.js'])
        
        all_results = {**src_results, **templates_results, **static_results}
        
        violation_count = sum(len(r['violations']) for r in all_results.values())
        
        if violation_count > 0:
            print(f"   ❌ Found {violation_count} color violations")
        else:
            print("   ✅ All colors follow Kearney standards")
            
        return violation_count
    
    def _check_development_standards(self, project_path: str) -> int:
        """Check development standards from DEVELOPMENT_STANDARDS.md"""
        violations = 0
        
        # Check for proper error handling
        py_files = list(Path(project_path).rglob("*.py"))
        
        for file_path in py_files:
            if any(skip in str(file_path) for skip in ['__pycache__', 'venv', '.git']):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Check for bare except statements
                if re.search(r'except\s*:', content):
                    violations += 1
                    print(f"   ❌ Bare except in {file_path}")
                    
                # Check for print debugging (should use logging)
                if 'print(' in content and '__main__' not in content:
                    # Allow prints in main execution
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'print(' in line and not line.strip().startswith('#'):
                            violations += 1
                            print(f"   ❌ Print statement (use logging) in {file_path}:{i+1}")
                            break
            except Exception as e:
                print(f"   ⚠️  Error reading {file_path}: {str(e)}")
                
        if violations == 0:
            print("   ✅ Development standards followed")
            
        return violations
    
    def _check_data_science_standards(self, project_path: str) -> int:
        """Check data science standards from DATA_SCIENCE_STANDARDS.md"""
        violations = 0
        
        # Check for reproducibility (random seeds)
        model_files = list(Path(project_path).rglob("*model*.py"))
        model_files.extend(list(Path(project_path).rglob("*backtest*.py")))
        
        for file_path in model_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Check for random_state in models
                if 'RandomForest' in content or 'train_test_split' in content:
                    if 'random_state=' not in content:
                        violations += 1
                        print(f"   ❌ Missing random_state for reproducibility in {file_path}")
                        
                # Check for proper train/test split
                if 'train_test_split' in content:
                    if 'TimeSeriesSplit' not in content and 'time' in str(file_path).lower():
                        violations += 1
                        print(f"   ❌ Should use TimeSeriesSplit for time series in {file_path}")
            except Exception as e:
                print(f"   ⚠️  Error reading {file_path}: {str(e)}")
                
        if violations == 0:
            print("   ✅ Data science standards followed")
            
        return violations
    
    def _check_no_fake_data(self, project_path: str) -> int:
        """Check for synthetic/fake data usage"""
        violations = 0
        
        suspicious_patterns = [
            r'np\.random\.normal\(',
            r'np\.random\.rand\(',
            r'generate.*fake',
            r'synthetic.*data',
            r'dummy.*data',
            r'fake.*price',
            r'mock.*data'
        ]
        
        py_files = list(Path(project_path).rglob("*.py"))
        
        for file_path in py_files:
            if any(skip in str(file_path) for skip in ['test_', '__pycache__', 'venv']):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for pattern in suspicious_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        # Check if it's for ML reproducibility (allowed)
                        if 'random_state' in lines[line_num-1] or 'seed' in lines[line_num-1]:
                            continue
                        violations += 1
                        print(f"   ❌ Potential fake data in {file_path}:{line_num}")
                        break
            except Exception as e:
                print(f"   ⚠️  Error reading {file_path}: {str(e)}")
                
        if violations == 0:
            print("   ✅ No fake/synthetic data found")
            
        return violations
    
    def _check_documentation_standards(self, project_path: str) -> int:
        """Check documentation standards"""
        violations = 0
        
        # Check for CLAUDE.md
        if not Path(f"{project_path}/CLAUDE.md").exists():
            violations += 1
            print("   ❌ Missing CLAUDE.md file")
        else:
            print("   ✅ CLAUDE.md exists")
            
        # Check for proper docstrings
        py_files = list(Path(project_path).rglob("*.py"))
        
        for file_path in py_files[:5]:  # Sample check
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Check for class/function docstrings
                if 'class ' in content or 'def ' in content:
                    if '"""' not in content:
                        violations += 1
                        print(f"   ❌ Missing docstrings in {file_path}")
                        break
            except Exception as e:
                print(f"   ⚠️  Error reading {file_path}: {str(e)}")
                
        return violations

def create_standards_config():
    """Create a standards configuration file"""
    config = {
        "project": "cocoa-market-signals-v3",
        "standards_version": "1.0",
        "enforced_standards": {
            "colors": {
                "source": "preston-dev-setup/kearney_design_system.py",
                "primary_purple": "#6f42c1",
                "primary_charcoal": "#272b30",
                "allowed_grays": ["#999999", "#7a8288", "#52575c", "#e9ecef"]
            },
            "development": {
                "source": "preston-dev-setup/DEVELOPMENT_STANDARDS.md",
                "no_bare_except": True,
                "use_logging": True,
                "type_hints": True
            },
            "data_science": {
                "source": "preston-dev-setup/DATA_SCIENCE_STANDARDS.md",
                "reproducibility": True,
                "random_seeds": True,
                "time_series_split": True
            },
            "data_integrity": {
                "no_fake_data": True,
                "real_data_only": True,
                "data_validation": True
            }
        },
        "validation_schedule": "pre-commit",
        "last_validated": None
    }
    
    with open("/Users/pfay01/Projects/cocoa-market-signals-v3/.standards.json", "w") as f:
        json.dump(config, f, indent=2)
        
    print("✅ Created .standards.json configuration file")

if __name__ == "__main__":
    # Create standards config
    create_standards_config()
    
    # Run enforcement
    enforcer = StandardsEnforcer()
    project_path = "/Users/pfay01/Projects/cocoa-market-signals-v3"
    
    all_standards_followed = enforcer.validate_all_standards(project_path)
    
    # Exit with appropriate code
    sys.exit(0 if all_standards_followed else 1)