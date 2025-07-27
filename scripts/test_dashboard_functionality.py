#!/usr/bin/env python3
"""
FUNCTIONAL TESTS FOR DASHBOARD
Actually tests if features WORK, not just if they exist
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
import json
import re
from sqlmodel import Session, select
from app.core.database import engine
from app.models.prediction import Prediction
from app.models.price_data import PriceData
from datetime import date, timedelta

class DashboardFunctionalTests:
    """
    Tests that actually verify functionality
    """
    
    def __init__(self, dashboard_url="http://localhost:8005"):
        self.url = dashboard_url
        self.failures = []
        self.passes = []
        
    def test_dashboard_loads(self):
        """Test 1: Dashboard actually loads"""
        print("\nTEST 1: Dashboard Loads")
        try:
            response = requests.get(self.url, timeout=5)
            if response.status_code == 200:
                self.passes.append("Dashboard loads successfully")
                print("  ✓ Dashboard returns 200 OK")
                return True
            else:
                self.failures.append(f"Dashboard returned {response.status_code}")
                print(f"  ❌ Dashboard returned {response.status_code}")
                return False
        except Exception as e:
            self.failures.append(f"Dashboard not accessible: {e}")
            print(f"  ❌ Dashboard not accessible: {e}")
            return False
    
    def test_kpis_populated(self):
        """Test 2: KPIs actually have data"""
        print("\nTEST 2: KPIs Populated")
        try:
            response = requests.get(self.url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find KPI values
            kpi_values = soup.find_all('div', class_='kpi-value')
            
            if not kpi_values:
                self.failures.append("No KPI values found on page")
                print("  ❌ No KPI values found")
                return False
            
            blank_kpis = []
            populated_kpis = []
            
            for kpi in kpi_values:
                text = kpi.text.strip()
                if text == "—" or text == "" or text == "None":
                    blank_kpis.append(text)
                else:
                    populated_kpis.append(text)
            
            print(f"  Found {len(kpi_values)} KPIs")
            print(f"  Populated: {len(populated_kpis)}")
            print(f"  Blank: {len(blank_kpis)}")
            
            if populated_kpis:
                print(f"  Values: {populated_kpis}")
            
            if len(blank_kpis) > len(populated_kpis):
                self.failures.append(f"Most KPIs are blank: {len(blank_kpis)}/{len(kpi_values)}")
                print(f"  ❌ Most KPIs are blank")
                return False
            else:
                self.passes.append(f"KPIs populated: {len(populated_kpis)}/{len(kpi_values)}")
                print(f"  ✓ KPIs are populated")
                return True
                
        except Exception as e:
            self.failures.append(f"Error checking KPIs: {e}")
            print(f"  ❌ Error: {e}")
            return False
    
    def test_actual_vs_predicted_data(self):
        """Test 3: Actual vs Predicted chart has BOTH lines with data"""
        print("\nTEST 3: Actual vs Predicted Chart Data")
        try:
            # Check July 2025 where we have data
            response = requests.get(f"{self.url}/?month=2025-07")
            
            # Extract JavaScript data
            actual_match = re.search(r'const actualPrices = .*?;', response.text, re.DOTALL)
            predicted_match = re.search(r'const predictedPrices = .*?;', response.text, re.DOTALL)
            
            if not actual_match or not predicted_match:
                self.failures.append("Chart data not found in page")
                print("  ❌ Chart data not found")
                return False
            
            # Check actualPrices
            actual_data = actual_match.group(0)
            actual_nulls = actual_data.count('null')
            # Look for numeric values in the array
            actual_array_match = re.search(r'\[(.*?)\]', actual_data, re.DOTALL)
            if actual_array_match:
                actual_array = actual_array_match.group(1)
                actual_values = len(re.findall(r'\d+(?:\.\d+)?(?=,|\])', actual_array))
            else:
                actual_values = 0
            
            print(f"  Actual prices: {actual_values} values, {actual_nulls} nulls")
            
            # Check predictedPrices
            predicted_data = predicted_match.group(0)
            predicted_nulls = predicted_data.count('null')
            # Look for numeric values in the array
            predicted_array_match = re.search(r'\[(.*?)\]', predicted_data, re.DOTALL)
            if predicted_array_match:
                predicted_array = predicted_array_match.group(1)
                predicted_values = len(re.findall(r'\d+(?:\.\d+)?(?=,|\])', predicted_array))
            else:
                predicted_values = 0
            
            print(f"  Predicted prices: {predicted_values} values, {predicted_nulls} nulls")
            
            if actual_values == 0:
                self.failures.append("No actual price data in chart")
                print("  ❌ No actual price data")
                return False
                
            if predicted_values == 0:
                self.failures.append("No predicted price data in chart")
                print("  ❌ No predicted price data")
                return False
            
            self.passes.append(f"Chart has both actual ({actual_values}) and predicted ({predicted_values}) data")
            print("  ✓ Both actual and predicted data present")
            return True
            
        except Exception as e:
            self.failures.append(f"Error checking chart data: {e}")
            print(f"  ❌ Error: {e}")
            return False
    
    def test_month_navigation(self):
        """Test 4: Month navigation actually works"""
        print("\nTEST 4: Month Navigation")
        try:
            # Test current month
            response = requests.get(self.url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            current_month = soup.find('div', class_='current-month')
            if not current_month:
                self.failures.append("Current month display not found")
                print("  ❌ Current month not displayed")
                return False
            
            print(f"  Current month: {current_month.text.strip()}")
            
            # Find navigation links
            nav_links = soup.find_all('a', class_='btn')
            prev_link = None
            next_link = None
            
            for link in nav_links:
                if 'Previous' in link.text:
                    prev_link = link.get('href')
                elif 'Next' in link.text:
                    next_link = link.get('href')
            
            if not prev_link or not next_link:
                self.failures.append("Navigation links not found")
                print("  ❌ Navigation links missing")
                return False
            
            print(f"  Previous link: {prev_link}")
            print(f"  Next link: {next_link}")
            
            # Test navigation
            prev_response = requests.get(f"{self.url.rstrip('/')}{prev_link}")
            if prev_response.status_code != 200:
                self.failures.append("Previous month navigation failed")
                print("  ❌ Previous month navigation failed")
                return False
            
            self.passes.append("Month navigation working")
            print("  ✓ Month navigation functional")
            return True
            
        except Exception as e:
            self.failures.append(f"Error testing navigation: {e}")
            print(f"  ❌ Error: {e}")
            return False
    
    def test_data_is_real(self):
        """Test 5: Verify data is from REAL database"""
        print("\nTEST 5: Real Data Verification")
        try:
            # Check database for real data
            with Session(engine) as session:
                # Check predictions
                predictions = session.exec(
                    select(Prediction)
                    .where(Prediction.model_name == "zen_consensus")
                    .limit(10)
                ).all()
                
                print(f"  Database predictions: {len(predictions)}")
                
                # Check prices
                prices = session.exec(
                    select(PriceData)
                    .where(PriceData.source == "Yahoo Finance")
                    .order_by(PriceData.date.desc())
                    .limit(10)
                ).all()
                
                print(f"  Database prices: {len(prices)}")
                
                if len(predictions) == 0:
                    self.failures.append("No predictions in database")
                    print("  ❌ No predictions found")
                    return False
                
                if len(prices) == 0:
                    self.failures.append("No price data in database")
                    print("  ❌ No price data found")
                    return False
                
                # Verify data sources
                for price in prices[:3]:
                    print(f"    Price: ${price.price:,.2f} on {price.date} from {price.source}")
                
                self.passes.append(f"Real data verified: {len(predictions)} predictions, {len(prices)} prices")
                print("  ✓ Data is REAL from database")
                return True
                
        except Exception as e:
            self.failures.append(f"Database error: {e}")
            print(f"  ❌ Database error: {e}")
            return False
    
    def test_prediction_table_populated(self):
        """Test 6: Prediction details table has data"""
        print("\nTEST 6: Prediction Table")
        try:
            response = requests.get(self.url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find table
            table = soup.find('table', class_='data-table')
            if not table:
                self.failures.append("Prediction table not found")
                print("  ❌ Table not found")
                return False
            
            # Count rows
            rows = table.find_all('tr')
            data_rows = [r for r in rows if r.find('td')]
            
            print(f"  Table rows: {len(data_rows)}")
            
            if len(data_rows) == 0:
                self.failures.append("No data in prediction table")
                print("  ❌ No data rows")
                return False
            
            # Check first row for actual data
            first_row = data_rows[0]
            cells = first_row.find_all('td')
            empty_cells = [c for c in cells if c.text.strip() in ['—', '', 'None']]
            
            print(f"  First row: {len(cells)} cells, {len(empty_cells)} empty")
            
            if len(empty_cells) == len(cells):
                self.failures.append("All cells empty in table")
                print("  ❌ Table has no real data")
                return False
            
            self.passes.append(f"Prediction table has {len(data_rows)} rows of data")
            print("  ✓ Table populated with data")
            return True
            
        except Exception as e:
            self.failures.append(f"Error checking table: {e}")
            print(f"  ❌ Error: {e}")
            return False
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*60)
        print("FUNCTIONALITY TEST REPORT")
        print("="*60)
        
        total_tests = len(self.passes) + len(self.failures)
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {len(self.passes)}")
        print(f"Failed: {len(self.failures)}")
        
        if self.failures:
            print("\n❌ FAILURES:")
            for i, failure in enumerate(self.failures, 1):
                print(f"  {i}. {failure}")
            
            print("\n⚠️  DASHBOARD IS NOT FUNCTIONAL!")
            return False
        else:
            print("\n✅ ALL FUNCTIONALITY TESTS PASSED")
            return True

def main():
    """Run functionality tests"""
    print("\n" + "#"*60)
    print("# DASHBOARD FUNCTIONALITY TESTS")
    print("# Testing if features ACTUALLY WORK")
    print("#"*60)
    
    # Allow URL parameter
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8005"
    tester = DashboardFunctionalTests(url)
    
    # Run all tests
    if tester.test_dashboard_loads():
        tester.test_kpis_populated()
        tester.test_actual_vs_predicted_data()
        tester.test_month_navigation()
        tester.test_data_is_real()
        tester.test_prediction_table_populated()
    
    # Generate report
    passed = tester.generate_report()
    
    if not passed:
        print("\n🚨 FIX THE DASHBOARD!")
        sys.exit(1)

if __name__ == "__main__":
    main()