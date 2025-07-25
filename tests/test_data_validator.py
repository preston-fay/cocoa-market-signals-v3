"""
Tests for Data Validator module
Ensures all data validation rules are properly enforced
"""

import pytest
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.validation.data_validator import (
    DataValidator, DataSource, DataPoint, ValidationResult
)


class TestDataValidator:
    """Test suite for data validation functionality"""
    
    @pytest.fixture
    def validator(self, tmp_path):
        """Create a validator with test configuration"""
        # Create test sources config
        test_config = {
            "data_sources": {
                "prices": {
                    "test_source": {
                        "name": "Test Source",
                        "type": "api",
                        "url": "https://test.com",
                        "verified": True
                    }
                }
            }
        }
        
        config_path = tmp_path / "sources.json"
        with open(config_path, 'w') as f:
            json.dump(test_config, f)
            
        return DataValidator(str(config_path))
    
    @pytest.fixture
    def valid_source(self):
        """Create a valid data source"""
        return DataSource(
            source_name="Test Source",
            source_type="api",
            source_url="https://test.com",
            retrieval_time=datetime.now(),
            data_hash="abc123",
            verified=True
        )
    
    @pytest.fixture
    def valid_data_point(self, valid_source):
        """Create a valid data point"""
        return DataPoint(
            timestamp=datetime.now() - timedelta(hours=1),
            value=3500.0,
            metric_name="price_usd_per_ton",
            source=valid_source,
            confidence_level=0.95,
            validation_status="verified"
        )
    
    def test_validate_valid_data_point(self, validator, valid_data_point):
        """Test validation of a completely valid data point"""
        result = validator.validate_data_point(valid_data_point)
        
        assert result.is_valid == True
        assert len(result.errors) == 0
        assert result.data_quality_score > 0.9
    
    def test_reject_unverified_source(self, validator, valid_data_point):
        """Test that unverified sources are rejected"""
        valid_data_point.source.verified = False
        result = validator.validate_data_point(valid_data_point)
        
        assert result.is_valid == False
        assert "Data source is not verified" in result.errors
        assert result.data_quality_score == 0.0
    
    def test_reject_future_timestamp(self, validator, valid_data_point):
        """Test that future timestamps are rejected"""
        valid_data_point.timestamp = datetime.now() + timedelta(days=1)
        result = validator.validate_data_point(valid_data_point)
        
        assert result.is_valid == False
        assert "Timestamp is in the future" in result.errors
    
    def test_warn_stale_data(self, validator, valid_data_point):
        """Test warning for stale data"""
        valid_data_point.source.retrieval_time = datetime.now() - timedelta(hours=30)
        result = validator.validate_data_point(valid_data_point)
        
        assert result.is_valid == True  # Still valid, just warning
        assert len(result.warnings) > 0
        assert any("hours old" in w for w in result.warnings)
    
    def test_value_range_validation(self, validator, valid_data_point):
        """Test that values outside expected ranges generate warnings"""
        # Test extremely high price
        valid_data_point.value = 15000.0  # $15k per ton (unusually high)
        result = validator.validate_data_point(valid_data_point)
        
        assert result.is_valid == True
        assert len(result.warnings) > 0
        assert any("outside expected range" in w for w in result.warnings)
    
    def test_batch_validation(self, validator, valid_data_point):
        """Test batch validation of multiple data points"""
        data_points = [
            valid_data_point,
            DataPoint(
                timestamp=datetime.now(),
                value=4000.0,
                metric_name="price_usd_per_ton",
                source=valid_data_point.source,
                confidence_level=0.8,
                validation_status="verified"
            )
        ]
        
        results = validator.validate_batch(data_points)
        
        assert results["total_points"] == 2
        assert results["valid_points"] == 2
        assert results["invalid_points"] == 0
        assert results["average_quality_score"] > 0.8
    
    def test_audit_report_generation(self, validator, valid_data_point):
        """Test audit report generation"""
        # Validate some data points first
        validator.validate_data_point(valid_data_point)
        
        report = validator.generate_audit_report()
        
        assert report["total_validations"] == 1
        assert report["validated_count"] == 1
        assert "price_usd_per_ton" in report["metrics_validated"]
    
    def test_source_hash_creation(self, validator):
        """Test deterministic hash creation"""
        data = {"price": 3500, "volume": 1000}
        
        hash1 = validator.create_source_hash(data)
        hash2 = validator.create_source_hash(data)
        
        assert hash1 == hash2  # Should be deterministic
        assert len(hash1) == 64  # SHA256 hash length
    
    def test_source_integrity_verification(self, validator):
        """Test source integrity checks"""
        # Test known source
        is_valid = validator.verify_source_integrity(
            "Test Source", 
            "https://test.com"
        )
        assert is_valid == True
        
        # Test unknown source
        is_valid = validator.verify_source_integrity(
            "Unknown Source",
            "https://fake.com"
        )
        assert is_valid == False


class TestDataModels:
    """Test Pydantic models for data validation"""
    
    def test_data_point_validation(self):
        """Test DataPoint model validation"""
        source = DataSource(
            source_name="Test",
            source_type="api",
            retrieval_time=datetime.now(),
            data_hash="test123",
            verified=True
        )
        
        # Test valid data point
        dp = DataPoint(
            timestamp=datetime.now(),
            value=3500.0,
            metric_name="test_metric",
            source=source,
            confidence_level=0.8,
            validation_status="verified"
        )
        assert dp.value == 3500.0
        
        # Test invalid confidence level
        with pytest.raises(ValueError):
            DataPoint(
                timestamp=datetime.now(),
                value=3500.0,
                metric_name="test_metric",
                source=source,
                confidence_level=1.5,  # >1.0
                validation_status="verified"
            )
        
        # Test infinite value rejection
        with pytest.raises(ValueError):
            DataPoint(
                timestamp=datetime.now(),
                value=float('inf'),
                metric_name="test_metric",
                source=source,
                confidence_level=0.8,
                validation_status="verified"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])