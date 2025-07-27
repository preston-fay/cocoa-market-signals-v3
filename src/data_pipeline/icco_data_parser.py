#!/usr/bin/env python3
"""
ICCO Data Parser - Extracts production and trade statistics
Parses quarterly bulletins and production forecasts
REAL DATA ONLY - No synthetic generation
"""
import requests
import PyPDF2
import tabula
import pandas as pd
from datetime import datetime
import re
from typing import Dict, List, Optional
from sqlmodel import Session
from app.core.database import engine
from app.models.trade_data import TradeData
import json

class ICCODataParser:
    """Parse ICCO quarterly bulletins and statistics"""
    
    def __init__(self):
        self.base_url = "https://www.icco.org"
        self.statistics_url = f"{self.base_url}/statistics/"
        
        # Known report patterns
        self.report_patterns = {
            'quarterly_bulletin': r'QBCS.*\.pdf',
            'production_forecast': r'production.*forecast.*\.pdf',
            'grinding_data': r'grinding.*\.pdf',
            'stock_data': r'stock.*\.pdf'
        }
        
        # Key countries for cocoa
        self.key_producers = [
            'Ivory Coast', "Côte d'Ivoire", 'Ghana', 'Nigeria', 
            'Cameroon', 'Ecuador', 'Brazil', 'Indonesia'
        ]
        
        self.key_consumers = [
            'Netherlands', 'Germany', 'United States', 'Belgium',
            'Malaysia', 'France', 'United Kingdom', 'Switzerland'
        ]
    
    def fetch_report_urls(self) -> List[Dict[str, str]]:
        """Fetch available ICCO report URLs"""
        try:
            response = requests.get(self.statistics_url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML to find PDF links
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            reports = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.pdf'):
                    # Determine report type
                    report_type = 'other'
                    for pattern_name, pattern in self.report_patterns.items():
                        if re.search(pattern, href, re.IGNORECASE):
                            report_type = pattern_name
                            break
                    
                    # Make URL absolute
                    if not href.startswith('http'):
                        href = self.base_url + href if href.startswith('/') else self.base_url + '/' + href
                    
                    reports.append({
                        'url': href,
                        'type': report_type,
                        'title': link.get_text(strip=True)
                    })
            
            return reports
            
        except Exception as e:
            print(f"Error fetching ICCO reports: {str(e)}")
            return []
    
    def download_pdf(self, url: str) -> Optional[bytes]:
        """Download PDF file"""
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return None
    
    def extract_tables_from_pdf(self, pdf_content: bytes) -> List[pd.DataFrame]:
        """Extract tables from PDF using tabula"""
        try:
            # Save PDF temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(pdf_content)
                tmp_path = tmp.name
            
            # Extract tables
            tables = tabula.read_pdf(tmp_path, pages='all', multiple_tables=True)
            
            # Clean up
            import os
            os.unlink(tmp_path)
            
            return tables
            
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
            return []
    
    def parse_production_table(self, df: pd.DataFrame) -> List[Dict]:
        """Parse production data from table"""
        production_data = []
        
        # Look for country names in first column
        if len(df.columns) < 2:
            return []
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Find year columns (usually formatted as 'YYYY/YY')
        year_pattern = r'(\d{4})/\d{2}'
        year_columns = []
        
        for col in df.columns[1:]:  # Skip first column (countries)
            match = re.search(year_pattern, str(col))
            if match:
                year_columns.append((col, int(match.group(1))))
        
        # Extract data for each country
        for idx, row in df.iterrows():
            country = str(row.iloc[0]).strip()
            
            # Check if it's a producer country
            if not any(producer in country for producer in self.key_producers):
                continue
            
            # Extract production values
            for col_name, year in year_columns:
                try:
                    value = row[col_name]
                    if pd.notna(value):
                        # Clean value (remove commas, convert to float)
                        clean_value = str(value).replace(',', '').replace(' ', '')
                        production_tonnes = float(clean_value) * 1000  # Usually in thousand tonnes
                        
                        production_data.append({
                            'country': country,
                            'year': year,
                            'production_tonnes': production_tonnes,
                            'data_type': 'production'
                        })
                except:
                    continue
        
        return production_data
    
    def parse_grinding_table(self, df: pd.DataFrame) -> List[Dict]:
        """Parse grinding (processing) data from table"""
        grinding_data = []
        
        # Similar structure to production tables
        if len(df.columns) < 2:
            return []
        
        df.columns = [str(col).strip() for col in df.columns]
        
        # Find quarter columns (e.g., 'Q1 2023', '2023 Q1')
        quarter_pattern = r'Q(\d)\s*(\d{4})|(\d{4})\s*Q(\d)'
        
        for idx, row in df.iterrows():
            country = str(row.iloc[0]).strip()
            
            # Check if it's a consumer country
            if not any(consumer in country for consumer in self.key_consumers):
                continue
            
            for col in df.columns[1:]:
                match = re.search(quarter_pattern, str(col))
                if match:
                    if match.group(1):  # Q1 2023 format
                        quarter = int(match.group(1))
                        year = int(match.group(2))
                    else:  # 2023 Q1 format
                        year = int(match.group(3))
                        quarter = int(match.group(4))
                    
                    try:
                        value = row[col]
                        if pd.notna(value):
                            clean_value = str(value).replace(',', '').replace(' ', '')
                            grinding_tonnes = float(clean_value) * 1000
                            
                            grinding_data.append({
                                'country': country,
                                'year': year,
                                'quarter': quarter,
                                'grinding_tonnes': grinding_tonnes,
                                'data_type': 'grinding'
                            })
                    except:
                        continue
        
        return grinding_data
    
    def parse_stock_levels(self, df: pd.DataFrame) -> List[Dict]:
        """Parse stock level data"""
        stock_data = []
        
        # Look for stock-related keywords
        stock_keywords = ['stock', 'inventory', 'warehouse', 'certified']
        
        # Find relevant rows
        for idx, row in df.iterrows():
            row_text = ' '.join(str(val) for val in row.values).lower()
            
            if any(keyword in row_text for keyword in stock_keywords):
                # Extract numeric values
                for col_idx, value in enumerate(row):
                    try:
                        if pd.notna(value) and isinstance(value, (int, float)):
                            stock_data.append({
                                'date': datetime.now(),  # Will be updated if date found
                                'stock_tonnes': float(value) * 1000,
                                'location': 'global',  # Will be updated if location found
                                'data_type': 'stock'
                            })
                    except:
                        continue
        
        return stock_data
    
    def process_icco_reports(self, limit: int = 5):
        """Download and process ICCO reports"""
        print("Fetching ICCO report list...")
        reports = self.fetch_report_urls()
        
        if not reports:
            print("No reports found")
            return
        
        all_data = {
            'production': [],
            'grinding': [],
            'stocks': []
        }
        
        # Process reports
        for i, report in enumerate(reports[:limit]):
            print(f"\nProcessing {report['type']}: {report['title']}")
            
            # Download PDF
            pdf_content = self.download_pdf(report['url'])
            if not pdf_content:
                continue
            
            # Extract tables
            tables = self.extract_tables_from_pdf(pdf_content)
            print(f"Found {len(tables)} tables")
            
            # Parse each table based on report type
            for table_idx, table in enumerate(tables):
                if report['type'] == 'quarterly_bulletin' or report['type'] == 'production_forecast':
                    production_data = self.parse_production_table(table)
                    all_data['production'].extend(production_data)
                    
                    grinding_data = self.parse_grinding_table(table)
                    all_data['grinding'].extend(grinding_data)
                
                elif report['type'] == 'stock_data':
                    stock_data = self.parse_stock_levels(table)
                    all_data['stocks'].extend(stock_data)
        
        return all_data
    
    def save_to_database(self, data: Dict[str, List[Dict]]):
        """Save parsed data to database"""
        with Session(engine) as session:
            saved_count = 0
            
            # Save production data
            for item in data.get('production', []):
                trade_record = TradeData(
                    date=datetime(item['year'], 1, 1),  # Annual data
                    reporter_country=item['country'],
                    partner_country='World',
                    commodity_code='1801',  # Cocoa beans
                    commodity_desc='Cocoa beans, whole or broken, raw or roasted',
                    trade_flow='Production',
                    trade_value_usd=0,  # Not available
                    netweight_kg=item['production_tonnes'] * 1000,
                    quantity=item['production_tonnes'],
                    quantity_unit='tonnes'
                )
                session.add(trade_record)
                saved_count += 1
            
            # Save grinding data
            for item in data.get('grinding', []):
                # Convert quarter to month
                month = (item['quarter'] - 1) * 3 + 1
                
                trade_record = TradeData(
                    date=datetime(item['year'], month, 1),
                    reporter_country=item['country'],
                    partner_country='World',
                    commodity_code='1801',
                    commodity_desc='Cocoa grinding/processing',
                    trade_flow='Processing',
                    trade_value_usd=0,
                    netweight_kg=item['grinding_tonnes'] * 1000,
                    quantity=item['grinding_tonnes'],
                    quantity_unit='tonnes'
                )
                session.add(trade_record)
                saved_count += 1
            
            session.commit()
            print(f"Saved {saved_count} ICCO data records to database")

def main():
    """Run ICCO data collection"""
    parser = ICCODataParser()
    
    # Process reports
    data = parser.process_icco_reports(limit=5)
    
    # Report findings
    print("\n=== ICCO Data Summary ===")
    print(f"Production records: {len(data.get('production', []))}")
    print(f"Grinding records: {len(data.get('grinding', []))}")
    print(f"Stock records: {len(data.get('stocks', []))}")
    
    # Save to database
    if any(data.values()):
        parser.save_to_database(data)

if __name__ == "__main__":
    main()