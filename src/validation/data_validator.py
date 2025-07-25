"""
Data Validator - Core validation engine

Implements strict validation rules to ensure data integrity.
Every data point must be traceable to a verified source.
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from loguru import logger

class DataSource(BaseModel):
    """Model for data source metadata"""
    source_name: str
    source_type: str  # 'api', 'file', 'manual'
    source_url: Optional[str] = None
    retrieval_time: datetime
    data_hash: str
    verified: bool = False
    
class DataPoint(BaseModel):
    """Model for individual data points with full traceability"""
    timestamp: datetime
    value: float
    metric_name: str
    source: DataSource
    confidence_level: float = Field(ge=0.0, le=1.0)
    validation_status: str  # 'verified', 'estimated', 'rejected'
    notes: Optional[str] = None
    
    @validator('value')
    def value_must_be_finite(cls, v):
        if not np.isfinite(v):
            raise ValueError('Value must be finite')
        return v

class ValidationResult(BaseModel):
    """Result of data validation"""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    data_quality_score: float = Field(ge=0.0, le=1.0)
    
class DataValidator:
    """
    Core validation engine for ensuring data integrity
    """
    
    def __init__(self, sources_config_path: str = "data/sources.json"):
        self.sources_config = self._load_sources_config(sources_config_path)
        self.validation_log = []
        logger.add("logs/validation.log", rotation="1 day")
        
    def _load_sources_config(self, path: str) -> Dict:
        """Load approved data sources configuration"""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Sources config not found: {path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def validate_data_point(self, data_point: DataPoint) -> ValidationResult:
        """
        Validate a single data point against integrity rules
        """
        errors = []
        warnings = []
        
        # Check 1: Source verification
        if not data_point.source.verified:
            errors.append("Data source is not verified")
            
        # Check 2: Timestamp reasonableness
        if data_point.timestamp > datetime.now():
            errors.append("Timestamp is in the future")
            
        # Check 3: Value range checks based on metric
        if not self._check_value_range(data_point.metric_name, data_point.value):
            warnings.append(f"Value {data_point.value} outside expected range for {data_point.metric_name}")
            
        # Check 4: Source freshness
        age_hours = (datetime.now() - data_point.source.retrieval_time).total_seconds() / 3600
        if age_hours > 24:
            warnings.append(f"Data is {age_hours:.1f} hours old")
            
        # Calculate quality score
        quality_score = self._calculate_quality_score(data_point, errors, warnings)
        
        # Log validation
        self._log_validation(data_point, errors, warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            data_quality_score=quality_score
        )
    
    def validate_batch(self, data_points: List[DataPoint]) -> Dict[str, Any]:
        """
        Validate a batch of data points
        """
        results = []
        for dp in data_points:
            results.append(self.validate_data_point(dp))
            
        valid_count = sum(1 for r in results if r.is_valid)
        avg_quality = np.mean([r.data_quality_score for r in results])
        
        return {
            "total_points": len(data_points),
            "valid_points": valid_count,
            "invalid_points": len(data_points) - valid_count,
            "average_quality_score": avg_quality,
            "validation_results": results
        }
    
    def _check_value_range(self, metric_name: str, value: float) -> bool:
        """
        Check if value is within expected range for the metric
        """
        # Define reasonable ranges for different metrics
        ranges = {
            "price_usd_per_ton": (1000, 10000),  # $1k - $10k per ton
            "temperature_celsius": (-10, 50),     # -10°C to 50°C
            "rainfall_mm": (0, 500),              # 0 to 500mm
            "humidity_percent": (0, 100),         # 0% to 100%
            "export_volume_tons": (0, 1000000),   # Up to 1M tons
            "shipping_cost_usd": (0, 1000),       # Up to $1000 per ton
        }
        
        # Default range if metric not defined
        if metric_name not in ranges:
            return True
            
        min_val, max_val = ranges[metric_name]
        return min_val <= value <= max_val
    
    def _calculate_quality_score(self, data_point: DataPoint, 
                                errors: List[str], warnings: List[str]) -> float:
        """
        Calculate a quality score for the data point
        """
        if errors:
            return 0.0
            
        score = 1.0
        
        # Deduct for warnings
        score -= len(warnings) * 0.1
        
        # Boost for verified sources
        if data_point.source.verified:
            score += 0.1
            
        # Consider confidence level
        score *= data_point.confidence_level
        
        return max(0.0, min(1.0, score))
    
    def _log_validation(self, data_point: DataPoint, 
                       errors: List[str], warnings: List[str]) -> None:
        """
        Log validation results for audit trail
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "metric": data_point.metric_name,
            "value": data_point.value,
            "source": data_point.source.source_name,
            "errors": errors,
            "warnings": warnings,
            "validation_status": "rejected" if errors else "validated"
        }
        
        self.validation_log.append(log_entry)
        
        if errors:
            logger.error(f"Validation failed: {log_entry}")
        elif warnings:
            logger.warning(f"Validation warnings: {log_entry}")
        else:
            logger.info(f"Validation passed: {log_entry}")
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """
        Generate an audit report of all validations
        """
        if not self.validation_log:
            return {"message": "No validations performed yet"}
            
        df = pd.DataFrame(self.validation_log)
        
        return {
            "total_validations": len(self.validation_log),
            "rejected_count": len(df[df['validation_status'] == 'rejected']),
            "validated_count": len(df[df['validation_status'] == 'validated']),
            "metrics_validated": df['metric'].unique().tolist(),
            "sources_used": df['source'].unique().tolist(),
            "validation_period": {
                "start": df['timestamp'].min(),
                "end": df['timestamp'].max()
            }
        }
    
    def create_source_hash(self, data: Any) -> str:
        """
        Create a hash of data for verification
        """
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def verify_source_integrity(self, source_name: str, source_url: str) -> bool:
        """
        Verify that a data source is legitimate and accessible
        """
        # Check if source is in approved list
        for category in self.sources_config['data_sources'].values():
            for source_key, source_info in category.items():
                if source_info['name'] == source_name:
                    if 'url' in source_info and source_info['url'] == source_url:
                        return source_info.get('verified', False)
        
        logger.warning(f"Unknown source: {source_name} at {source_url}")
        return False


if __name__ == "__main__":
    # Example usage
    validator = DataValidator()
    
    # Create a sample data point
    source = DataSource(
        source_name="ICCO",
        source_type="api",
        source_url="https://www.icco.org/statistics/",
        retrieval_time=datetime.now(),
        data_hash=validator.create_source_hash({"price": 4800}),
        verified=True
    )
    
    data_point = DataPoint(
        timestamp=datetime(2024, 1, 15),
        value=4800.0,
        metric_name="price_usd_per_ton",
        source=source,
        confidence_level=0.95,
        validation_status="verified",
        notes="Peak price during January 2024 surge"
    )
    
    # Validate
    result = validator.validate_data_point(data_point)
    print(f"Validation result: {result}")
    
    # Generate audit report
    report = validator.generate_audit_report()
    print(f"Audit report: {report}")