#!/usr/bin/env python3
"""
Dashboard Testing Agent - Validates the showcase dashboard functionality
Tests all endpoints, validates responses, and ensures everything works
"""
import requests
import time
import json
from typing import Dict, List, Tuple
import subprocess
import sys
import os

class DashboardTestingAgent:
    """Agent responsible for testing the showcase dashboard"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.dashboard_process = None
        self.test_results = []
        
    def start_dashboard(self) -> bool:
        """Start the dashboard server"""
        print("🚀 Starting dashboard server...")
        try:
            # Start dashboard in subprocess
            self.dashboard_process = subprocess.Popen(
                [sys.executable, "src/dashboard/app_showcase.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(3)
            
            # Check if process is running
            if self.dashboard_process.poll() is not None:
                stdout, stderr = self.dashboard_process.communicate()
                print(f"❌ Dashboard failed to start")
                print(f"Error: {stderr.decode()}")
                return False
                
            print("✅ Dashboard server started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")
            return False
    
    def test_endpoint(self, endpoint: str, expected_keys: List[str] = None) -> Tuple[bool, str]:
        """Test a single endpoint"""
        try:
            response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
            
            if response.status_code != 200:
                return False, f"Status code {response.status_code}"
            
            # For HTML endpoints
            if endpoint == "/":
                if "Cocoa Market Signals" in response.text:
                    return True, "HTML loaded successfully"
                else:
                    return False, "HTML missing expected content"
            
            # For API endpoints
            data = response.json()
            
            if expected_keys:
                missing_keys = [key for key in expected_keys if key not in data]
                if missing_keys:
                    return False, f"Missing keys: {missing_keys}"
            
            return True, "Endpoint working correctly"
            
        except requests.exceptions.ConnectionError:
            return False, "Connection refused - server not running"
        except Exception as e:
            return False, str(e)
    
    def validate_data_integrity(self) -> Dict[str, any]:
        """Validate the data returned by endpoints"""
        results = {}
        
        # Test signals endpoint
        try:
            response = requests.get(f"{self.base_url}/api/signals")
            data = response.json()
            
            results['signals'] = {
                'has_data': len(data.get('signals', [])) > 0,
                'signal_count': len(data.get('signals', [])),
                'has_summary': 'summary' in data,
                'accuracy_large_moves': data.get('summary', {}).get('accuracy_large_moves', 0)
            }
        except Exception as e:
            results['signals'] = {'error': str(e)}
        
        # Test predictions endpoint
        try:
            response = requests.get(f"{self.base_url}/api/predictions")
            data = response.json()
            
            results['predictions'] = {
                'has_dates': len(data.get('dates', [])) > 0,
                'data_points': len(data.get('dates', [])),
                'has_actuals': 'actual_returns' in data,
                'has_predictions': 'predicted_returns' in data
            }
        except Exception as e:
            results['predictions'] = {'error': str(e)}
            
        return results
    
    def run_comprehensive_test(self) -> Dict[str, any]:
        """Run all tests and return comprehensive results"""
        print("\n🔍 Dashboard Testing Agent - Comprehensive Test")
        print("=" * 60)
        
        # Start dashboard
        if not self.start_dashboard():
            return {
                'success': False,
                'error': 'Failed to start dashboard',
                'tests_passed': 0,
                'tests_failed': 1
            }
        
        # Define test cases
        test_cases = [
            ('/', None, "Main dashboard page"),
            ('/api/signals', ['signals', 'summary'], "Signals API"),
            ('/api/predictions', ['dates', 'actual_returns', 'predicted_returns'], "Predictions API"),
            ('/api/methodology', ['overview', 'models', 'techniques'], "Methodology API"),
            ('/api/data-sources', ['sources', 'integration'], "Data sources API"),
            ('/api/performance-metrics', ['overall', 'by_year', 'by_horizon'], "Performance metrics API")
        ]
        
        passed = 0
        failed = 0
        
        # Run tests
        for endpoint, expected_keys, description in test_cases:
            print(f"\n📍 Testing: {description}")
            print(f"   Endpoint: {endpoint}")
            
            success, message = self.test_endpoint(endpoint, expected_keys)
            
            if success:
                print(f"   ✅ PASSED: {message}")
                passed += 1
            else:
                print(f"   ❌ FAILED: {message}")
                failed += 1
            
            self.test_results.append({
                'endpoint': endpoint,
                'description': description,
                'success': success,
                'message': message
            })
        
        # Validate data integrity
        print("\n📊 Validating Data Integrity...")
        data_validation = self.validate_data_integrity()
        
        print("\nData Validation Results:")
        for endpoint, results in data_validation.items():
            print(f"\n{endpoint}:")
            for key, value in results.items():
                print(f"  {key}: {value}")
        
        # Stop dashboard
        self.stop_dashboard()
        
        # Summary
        total_tests = passed + failed
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            'success': failed == 0,
            'tests_passed': passed,
            'tests_failed': failed,
            'success_rate': success_rate,
            'test_results': self.test_results,
            'data_validation': data_validation
        }
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if failed == 0:
            print("\n✅ ALL TESTS PASSED - Dashboard is fully functional!")
        else:
            print("\n❌ Some tests failed - Dashboard needs fixes")
        
        return summary
    
    def stop_dashboard(self):
        """Stop the dashboard server"""
        if self.dashboard_process:
            print("\n🛑 Stopping dashboard server...")
            self.dashboard_process.terminate()
            self.dashboard_process.wait()
            print("✅ Dashboard server stopped")


if __name__ == "__main__":
    agent = DashboardTestingAgent()
    results = agent.run_comprehensive_test()
    
    # Save results
    with open('dashboard_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Full test results saved to: dashboard_test_results.json")
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)