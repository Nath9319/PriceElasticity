"""
Enhanced Data Validation for B2B Price Elasticity Modeling
Implements comprehensive schema validation, data quality checks, and automated reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from pathlib import Path
import sys
import warnings
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config_loader import config_loader, logger

warnings.filterwarnings('ignore')

# Try to import validation libraries with graceful fallback
try:
    import jsonschema
    from jsonschema import validate, ValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    print("Warning: jsonschema not available. Using basic validation.")

try:
    from pydantic import BaseModel, ValidationError as PydanticValidationError, Field
    from pydantic import validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    print("Warning: pydantic not available. Using basic validation.")


class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result from a validation check"""
    check_name: str
    level: ValidationLevel
    passed: bool
    message: str
    details: Dict[str, Any]
    affected_rows: Optional[List[int]] = None
    suggested_action: Optional[str] = None


@dataclass
class DataQualityMetrics:
    """Data quality metrics summary"""
    total_records: int
    completeness_score: float
    consistency_score: float
    validity_score: float
    uniqueness_score: float
    overall_score: float
    quality_issues: List[ValidationResult]


class EnhancedDataValidator:
    """
    Enhanced Data Validation System
    Provides comprehensive data quality assessment and schema validation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Enhanced Data Validator
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config_loader if config is None else config
        self.logger = logger
        self.validation_results = []
        self.data_quality_metrics = {}
        self.schemas = {}
        
        # Get validation configuration
        self.validation_config = self.config.get('data_validation', {
            'max_missing_percentage': 20.0,
            'min_unique_values': 2,
            'outlier_threshold': 3.0,
            'date_format': '%Y-%m-%d',
            'required_columns': ['Customer_ID', 'Product_ID', 'Net_Price', 'Status'],
            'categorical_max_cardinality': 1000,
            'numerical_min_variance': 1e-8
        })
        
        self.logger.info("Enhanced Data Validator initialized")
        
        # Initialize default schemas
        self._initialize_default_schemas()
    
    def _initialize_default_schemas(self) -> None:
        """Initialize default data schemas"""
        
        # Quote history schema
        self.schemas['quote_history'] = {
            "type": "object",
            "required": ["Quote_ID", "Customer_ID", "Product_ID", "Status"],
            "properties": {
                "Quote_ID": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9_-]+$"
                },
                "Customer_ID": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9_-]+$"
                },
                "Product_ID": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9_-]+$"
                },
                "Status": {
                    "type": "string",
                    "enum": ["Won", "Lost"]
                },
                "Net_Price": {
                    "type": "number",
                    "minimum": 0
                },
                "List_Price": {
                    "type": "number",
                    "minimum": 0
                },
                "Quote_Date": {
                    "type": "string",
                    "pattern": "^\\d{4}-\\d{2}-\\d{2}.*$"
                },
                "Customer_Segment": {
                    "type": "string"
                },
                "Product_Category": {
                    "type": "string"
                }
            }
        }
        
        # Customer master schema
        self.schemas['customer_master'] = {
            "type": "object",
            "required": ["Customer_ID"],
            "properties": {
                "Customer_ID": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9_-]+$"
                },
                "Customer_Name": {
                    "type": "string"
                },
                "Industry": {
                    "type": "string"
                },
                "Region": {
                    "type": "string"
                },
                "Customer_Size": {
                    "type": "string",
                    "enum": ["Small", "Medium", "Large", "Enterprise"]
                }
            }
        }
        
        # Product master schema
        self.schemas['product_master'] = {
            "type": "object",
            "required": ["Product_ID"],
            "properties": {
                "Product_ID": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9_-]+$"
                },
                "Product_Name": {
                    "type": "string"
                },
                "Product_Category": {
                    "type": "string"
                },
                "Product_Line": {
                    "type": "string"
                },
                "Launch_Date": {
                    "type": "string",
                    "pattern": "^\\d{4}-\\d{2}-\\d{2}.*$"
                },
                "Lifecycle_Stage": {
                    "type": "string",
                    "enum": ["Introduction", "Growth", "Maturity", "Decline"]
                }
            }
        }
    
    def validate_dataset_schema(self, df: pd.DataFrame, schema_name: str) -> List[ValidationResult]:
        """
        Validate dataset against predefined schema
        
        Args:
            df: DataFrame to validate
            schema_name: Name of schema to use
            
        Returns:
            List of validation results
        """
        self.logger.info(f"Validating dataset schema: {schema_name}")
        
        validation_results = []
        
        if schema_name not in self.schemas:
            validation_results.append(ValidationResult(
                check_name="schema_exists",
                level=ValidationLevel.ERROR,
                passed=False,
                message=f"Schema '{schema_name}' not found",
                details={"available_schemas": list(self.schemas.keys())}
            ))
            return validation_results
        
        schema = self.schemas[schema_name]
        
        if not HAS_JSONSCHEMA:
            # Fallback validation without jsonschema
            return self._validate_schema_basic(df, schema, schema_name)
        
        # Check required columns
        required_columns = schema.get('required', [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results.append(ValidationResult(
                check_name="required_columns",
                level=ValidationLevel.ERROR,
                passed=False,
                message=f"Missing required columns: {missing_columns}",
                details={"missing_columns": missing_columns},
                suggested_action=f"Add missing columns: {', '.join(missing_columns)}"
            ))
        else:
            validation_results.append(ValidationResult(
                check_name="required_columns",
                level=ValidationLevel.INFO,
                passed=True,
                message="All required columns present",
                details={"required_columns": required_columns}
            ))
        
        # Validate each row against schema
        properties = schema.get('properties', {})
        row_validation_errors = []
        
        for idx, row in df.iterrows():
            row_errors = self._validate_row_against_schema(row, properties, idx)
            row_validation_errors.extend(row_errors)
            
            # Limit number of row-level errors reported
            if len(row_validation_errors) > 100:
                break
        
        if row_validation_errors:
            # Aggregate row errors by type
            error_types = {}
            for error in row_validation_errors:
                error_type = error['error_type']
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(error)
            
            for error_type, errors in error_types.items():
                affected_rows = [e['row_index'] for e in errors[:50]]  # Limit to first 50
                
                validation_results.append(ValidationResult(
                    check_name=f"schema_validation_{error_type}",
                    level=ValidationLevel.WARNING,
                    passed=False,
                    message=f"Schema validation errors: {error_type} ({len(errors)} occurrences)",
                    details={
                        "error_type": error_type,
                        "error_count": len(errors),
                        "sample_errors": errors[:5]
                    },
                    affected_rows=affected_rows,
                    suggested_action=f"Review and fix {error_type} validation errors"
                ))
        
        return validation_results
    
    def perform_data_quality_assessment(self, df: pd.DataFrame, 
                                      dataset_name: str = "unknown") -> DataQualityMetrics:
        """
        Perform comprehensive data quality assessment
        
        Args:
            df: DataFrame to assess
            dataset_name: Name of the dataset
            
        Returns:
            DataQualityMetrics with assessment results
        """
        self.logger.info(f"Performing data quality assessment for {dataset_name}")
        
        quality_issues = []
        
        # 1. Completeness Assessment
        completeness_results = self._assess_completeness(df)
        quality_issues.extend(completeness_results['issues'])
        completeness_score = completeness_results['score']
        
        # 2. Consistency Assessment
        consistency_results = self._assess_consistency(df)
        quality_issues.extend(consistency_results['issues'])
        consistency_score = consistency_results['score']
        
        # 3. Validity Assessment
        validity_results = self._assess_validity(df)
        quality_issues.extend(validity_results['issues'])
        validity_score = validity_results['score']
        
        # 4. Uniqueness Assessment
        uniqueness_results = self._assess_uniqueness(df)
        quality_issues.extend(uniqueness_results['issues'])
        uniqueness_score = uniqueness_results['score']
        
        # 5. Calculate overall quality score
        overall_score = (completeness_score + consistency_score + 
                        validity_score + uniqueness_score) / 4
        
        metrics = DataQualityMetrics(
            total_records=len(df),
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            validity_score=validity_score,
            uniqueness_score=uniqueness_score,
            overall_score=overall_score,
            quality_issues=quality_issues
        )
        
        self.data_quality_metrics[dataset_name] = metrics
        
        return metrics
    
    def check_data_drift(self, current_df: pd.DataFrame, reference_df: pd.DataFrame,
                        threshold: float = 0.1) -> List[ValidationResult]:
        """
        Check for data drift between current and reference datasets
        
        Args:
            current_df: Current dataset
            reference_df: Reference dataset
            threshold: Drift threshold for statistical tests
            
        Returns:
            List of validation results
        """
        self.logger.info("Checking for data drift...")
        
        drift_results = []
        
        # Check for column differences
        current_cols = set(current_df.columns)
        reference_cols = set(reference_df.columns)
        
        missing_cols = reference_cols - current_cols
        new_cols = current_cols - reference_cols
        
        if missing_cols:
            drift_results.append(ValidationResult(
                check_name="missing_columns_drift",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Columns missing from current dataset: {list(missing_cols)}",
                details={"missing_columns": list(missing_cols)},
                suggested_action="Investigate why columns are missing"
            ))
        
        if new_cols:
            drift_results.append(ValidationResult(
                check_name="new_columns_drift",
                level=ValidationLevel.INFO,
                passed=True,
                message=f"New columns in current dataset: {list(new_cols)}",
                details={"new_columns": list(new_cols)}
            ))
        
        # Check for distribution drift in common numerical columns
        common_cols = current_cols & reference_cols
        numerical_cols = [col for col in common_cols 
                         if current_df[col].dtype in ['int64', 'float64']]
        
        for col in numerical_cols:
            drift_result = self._check_numerical_drift(
                current_df[col], reference_df[col], col, threshold
            )
            drift_results.append(drift_result)
        
        # Check for categorical distribution changes
        categorical_cols = [col for col in common_cols 
                           if col not in numerical_cols and 
                           current_df[col].dtype == 'object']
        
        for col in categorical_cols:
            drift_result = self._check_categorical_drift(
                current_df[col], reference_df[col], col, threshold
            )
            drift_results.append(drift_result)
        
        return drift_results
    
    def validate_business_rules(self, df: pd.DataFrame) -> List[ValidationResult]:
        """
        Validate business-specific rules
        
        Args:
            df: DataFrame to validate
            
        Returns:
            List of validation results
        """
        self.logger.info("Validating business rules...")
        
        business_rule_results = []
        
        # Rule 1: Net Price should be <= List Price
        if all(col in df.columns for col in ['Net_Price', 'List_Price']):
            price_violations = df[df['Net_Price'] > df['List_Price']]
            
            if len(price_violations) > 0:
                business_rule_results.append(ValidationResult(
                    check_name="net_price_vs_list_price",
                    level=ValidationLevel.ERROR,
                    passed=False,
                    message=f"Net Price exceeds List Price in {len(price_violations)} records",
                    details={
                        "violation_count": len(price_violations),
                        "violation_percentage": len(price_violations) / len(df) * 100
                    },
                    affected_rows=price_violations.index.tolist()[:50],
                    suggested_action="Review pricing data for errors"
                ))
        
        # Rule 2: Quote Date should be within reasonable range
        if 'Quote_Date' in df.columns:
            try:
                quote_dates = pd.to_datetime(df['Quote_Date'], errors='coerce')
                current_date = datetime.now()
                future_dates = quote_dates > current_date
                old_dates = quote_dates < (current_date - timedelta(days=3650))  # 10 years ago
                
                if future_dates.sum() > 0:
                    business_rule_results.append(ValidationResult(
                        check_name="future_quote_dates",
                        level=ValidationLevel.WARNING,
                        passed=False,
                        message=f"Found {future_dates.sum()} quotes with future dates",
                        details={"future_date_count": int(future_dates.sum())},
                        affected_rows=df[future_dates].index.tolist()[:50]
                    ))
                
                if old_dates.sum() > 0:
                    business_rule_results.append(ValidationResult(
                        check_name="very_old_quote_dates",
                        level=ValidationLevel.WARNING,
                        passed=False,
                        message=f"Found {old_dates.sum()} quotes older than 10 years",
                        details={"old_date_count": int(old_dates.sum())},
                        affected_rows=df[old_dates].index.tolist()[:50]
                    ))
                    
            except Exception as e:
                business_rule_results.append(ValidationResult(
                    check_name="quote_date_parsing",
                    level=ValidationLevel.ERROR,
                    passed=False,
                    message=f"Error parsing Quote_Date: {str(e)}",
                    details={"error": str(e)}
                ))
        
        # Rule 3: Status should be Won or Lost
        if 'Status' in df.columns:
            valid_statuses = ['Won', 'Lost']
            invalid_statuses = df[~df['Status'].isin(valid_statuses)]
            
            if len(invalid_statuses) > 0:
                business_rule_results.append(ValidationResult(
                    check_name="invalid_status_values",
                    level=ValidationLevel.ERROR,
                    passed=False,
                    message=f"Found {len(invalid_statuses)} records with invalid Status values",
                    details={
                        "invalid_count": len(invalid_statuses),
                        "invalid_values": invalid_statuses['Status'].unique().tolist()
                    },
                    affected_rows=invalid_statuses.index.tolist()[:50],
                    suggested_action="Standardize Status values to 'Won' or 'Lost'"
                ))
        
        # Rule 4: Customer_ID and Product_ID should not be null/empty
        id_columns = ['Customer_ID', 'Product_ID', 'Quote_ID']
        for col in id_columns:
            if col in df.columns:
                null_ids = df[df[col].isnull() | (df[col] == '')]
                
                if len(null_ids) > 0:
                    business_rule_results.append(ValidationResult(
                        check_name=f"null_{col.lower()}",
                        level=ValidationLevel.ERROR,
                        passed=False,
                        message=f"Found {len(null_ids)} records with null/empty {col}",
                        details={"null_count": len(null_ids)},
                        affected_rows=null_ids.index.tolist()[:50],
                        suggested_action=f"Provide valid values for {col}"
                    ))
        
        # Rule 5: Prices should be positive
        price_columns = ['Net_Price', 'List_Price']
        for col in price_columns:
            if col in df.columns:
                negative_prices = df[df[col] <= 0]
                
                if len(negative_prices) > 0:
                    business_rule_results.append(ValidationResult(
                        check_name=f"negative_{col.lower()}",
                        level=ValidationLevel.ERROR,
                        passed=False,
                        message=f"Found {len(negative_prices)} records with non-positive {col}",
                        details={"negative_count": len(negative_prices)},
                        affected_rows=negative_prices.index.tolist()[:50],
                        suggested_action=f"Ensure {col} values are positive"
                    ))
        
        return business_rule_results
    
    def detect_outliers(self, df: pd.DataFrame) -> List[ValidationResult]:
        """
        Detect outliers in numerical columns
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of validation results
        """
        self.logger.info("Detecting outliers...")
        
        outlier_results = []
        
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        threshold = self.validation_config.get('outlier_threshold', 3.0)
        
        for col in numerical_cols:
            if df[col].std() == 0:  # Skip constant columns
                continue
            
            # Z-score method
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = df[z_scores > threshold]
            
            if len(outliers) > 0:
                outlier_percentage = len(outliers) / len(df) * 100
                
                level = ValidationLevel.WARNING if outlier_percentage < 5 else ValidationLevel.ERROR
                
                outlier_results.append(ValidationResult(
                    check_name=f"outliers_{col}",
                    level=level,
                    passed=outlier_percentage < 5,
                    message=f"Found {len(outliers)} outliers in {col} ({outlier_percentage:.1f}%)",
                    details={
                        "outlier_count": len(outliers),
                        "outlier_percentage": outlier_percentage,
                        "threshold": threshold,
                        "outlier_values": outliers[col].tolist()[:10]
                    },
                    affected_rows=outliers.index.tolist()[:50],
                    suggested_action=f"Review outlier values in {col} for data entry errors"
                ))
        
        return outlier_results
    
    def generate_validation_report(self, validation_results: List[ValidationResult],
                                 quality_metrics: Optional[DataQualityMetrics] = None) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        
        Args:
            validation_results: List of validation results
            quality_metrics: Data quality metrics (optional)
            
        Returns:
            Dictionary with validation report
        """
        self.logger.info("Generating validation report...")
        
        # Categorize results by level
        errors = [r for r in validation_results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in validation_results if r.level == ValidationLevel.WARNING]
        info = [r for r in validation_results if r.level == ValidationLevel.INFO]
        
        # Calculate pass rate
        passed_checks = [r for r in validation_results if r.passed]
        pass_rate = len(passed_checks) / len(validation_results) * 100 if validation_results else 0
        
        # Generate summary
        summary = {
            "total_checks": len(validation_results),
            "passed_checks": len(passed_checks),
            "failed_checks": len(validation_results) - len(passed_checks),
            "pass_rate_percentage": pass_rate,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "info_count": len(info)
        }
        
        # Create detailed results
        detailed_results = {
            "errors": [self._serialize_validation_result(r) for r in errors],
            "warnings": [self._serialize_validation_result(r) for r in warnings],
            "info": [self._serialize_validation_result(r) for r in info]
        }
        
        # Add quality metrics if provided
        quality_summary = None
        if quality_metrics:
            quality_summary = {
                "total_records": quality_metrics.total_records,
                "overall_score": quality_metrics.overall_score,
                "completeness_score": quality_metrics.completeness_score,
                "consistency_score": quality_metrics.consistency_score,
                "validity_score": quality_metrics.validity_score,
                "uniqueness_score": quality_metrics.uniqueness_score
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "quality_metrics": quality_summary,
            "detailed_results": detailed_results,
            "recommendations": recommendations,
            "next_steps": self._generate_next_steps(errors, warnings)
        }
        
        return report
    
    # Helper methods for internal operations
    
    def _validate_schema_basic(self, df: pd.DataFrame, schema: Dict, 
                             schema_name: str) -> List[ValidationResult]:
        """Basic schema validation without jsonschema library"""
        
        results = []
        
        # Check required columns
        required_columns = schema.get('required', [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            results.append(ValidationResult(
                check_name="required_columns",
                level=ValidationLevel.ERROR,
                passed=False,
                message=f"Missing required columns: {missing_columns}",
                details={"missing_columns": missing_columns}
            ))
        
        # Basic property validation
        properties = schema.get('properties', {})
        for col_name, col_schema in properties.items():
            if col_name not in df.columns:
                continue
                
            col_data = df[col_name]
            
            # Check data type
            expected_type = col_schema.get('type')
            if expected_type == 'string' and col_data.dtype not in ['object']:
                results.append(ValidationResult(
                    check_name=f"type_check_{col_name}",
                    level=ValidationLevel.WARNING,
                    passed=False,
                    message=f"Column {col_name} expected to be string but is {col_data.dtype}",
                    details={"column": col_name, "expected": "string", "actual": str(col_data.dtype)}
                ))
            elif expected_type == 'number' and col_data.dtype not in ['int64', 'float64']:
                results.append(ValidationResult(
                    check_name=f"type_check_{col_name}",
                    level=ValidationLevel.WARNING,
                    passed=False,
                    message=f"Column {col_name} expected to be number but is {col_data.dtype}",
                    details={"column": col_name, "expected": "number", "actual": str(col_data.dtype)}
                ))
        
        return results
    
    def _validate_row_against_schema(self, row: pd.Series, properties: Dict, 
                                   row_idx: int) -> List[Dict]:
        """Validate a single row against schema properties"""
        
        errors = []
        
        for col_name, col_schema in properties.items():
            if col_name not in row:
                continue
            
            value = row[col_name]
            
            # Skip null values unless explicitly required
            if pd.isna(value):
                continue
            
            # Check enum values
            if 'enum' in col_schema:
                allowed_values = col_schema['enum']
                if value not in allowed_values:
                    errors.append({
                        'row_index': row_idx,
                        'column': col_name,
                        'value': value,
                        'error_type': 'enum_violation',
                        'message': f"Value '{value}' not in allowed values {allowed_values}"
                    })
            
            # Check minimum values for numbers
            if col_schema.get('type') == 'number' and 'minimum' in col_schema:
                if pd.notna(value) and value < col_schema['minimum']:
                    errors.append({
                        'row_index': row_idx,
                        'column': col_name,
                        'value': value,
                        'error_type': 'minimum_violation',
                        'message': f"Value {value} is below minimum {col_schema['minimum']}"
                    })
            
            # Check pattern for strings
            if col_schema.get('type') == 'string' and 'pattern' in col_schema:
                pattern = col_schema['pattern']
                if not re.match(pattern, str(value)):
                    errors.append({
                        'row_index': row_idx,
                        'column': col_name,
                        'value': value,
                        'error_type': 'pattern_violation',
                        'message': f"Value '{value}' does not match pattern '{pattern}'"
                    })
        
        return errors
    
    def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness"""
        
        issues = []
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        completeness_percentage = (1 - missing_cells / total_cells) * 100
        
        # Check individual columns
        max_missing_pct = self.validation_config.get('max_missing_percentage', 20.0)
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = missing_count / len(df) * 100
            
            if missing_percentage > max_missing_pct:
                issues.append(ValidationResult(
                    check_name=f"completeness_{col}",
                    level=ValidationLevel.WARNING,
                    passed=False,
                    message=f"Column {col} has {missing_percentage:.1f}% missing values",
                    details={
                        "column": col,
                        "missing_count": missing_count,
                        "missing_percentage": missing_percentage
                    },
                    suggested_action=f"Investigate missing values in {col}"
                ))
        
        return {
            "score": completeness_percentage / 100,
            "issues": issues,
            "details": {
                "total_cells": total_cells,
                "missing_cells": missing_cells,
                "completeness_percentage": completeness_percentage
            }
        }
    
    def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency"""
        
        issues = []
        consistency_score = 1.0
        
        # Check for consistent data types within columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed data types in object columns
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    value_types = non_null_values.apply(type).unique()
                    if len(value_types) > 1:
                        consistency_score -= 0.1
                        issues.append(ValidationResult(
                            check_name=f"consistency_mixed_types_{col}",
                            level=ValidationLevel.WARNING,
                            passed=False,
                            message=f"Column {col} contains mixed data types: {[t.__name__ for t in value_types]}",
                            details={
                                "column": col,
                                "data_types": [t.__name__ for t in value_types]
                            }
                        ))
        
        # Check for consistent date formats
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            if col in df.columns:
                try:
                    # Try to parse dates
                    parsed_dates = pd.to_datetime(df[col], errors='coerce')
                    failed_parsing = parsed_dates.isnull() & df[col].notnull()
                    
                    if failed_parsing.sum() > 0:
                        consistency_score -= 0.1
                        issues.append(ValidationResult(
                            check_name=f"consistency_date_format_{col}",
                            level=ValidationLevel.WARNING,
                            passed=False,
                            message=f"Column {col} contains {failed_parsing.sum()} unparseable dates",
                            details={
                                "column": col,
                                "unparseable_count": int(failed_parsing.sum())
                            }
                        ))
                except Exception as e:
                    pass  # Skip if date parsing fails completely
        
        return {
            "score": max(0, consistency_score),
            "issues": issues
        }
    
    def _assess_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data validity"""
        
        issues = []
        validity_score = 1.0
        
        # Check for reasonable ranges in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                validity_score -= 0.1
                issues.append(ValidationResult(
                    check_name=f"validity_infinite_{col}",
                    level=ValidationLevel.ERROR,
                    passed=False,
                    message=f"Column {col} contains {inf_count} infinite values",
                    details={"column": col, "infinite_count": inf_count}
                ))
            
            # Check for extremely large values (potential data entry errors)
            if df[col].std() > 0:  # Skip constant columns
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                extreme_values = (z_scores > 5).sum()  # More than 5 standard deviations
                
                if extreme_values > 0:
                    validity_score -= 0.05
                    issues.append(ValidationResult(
                        check_name=f"validity_extreme_values_{col}",
                        level=ValidationLevel.INFO,
                        passed=True,
                        message=f"Column {col} contains {extreme_values} extreme values",
                        details={
                            "column": col,
                            "extreme_count": extreme_values,
                            "threshold": "5 standard deviations"
                        }
                    ))
        
        return {
            "score": max(0, validity_score),
            "issues": issues
        }
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data uniqueness"""
        
        issues = []
        uniqueness_score = 1.0
        
        # Check key columns for uniqueness
        key_columns = ['Quote_ID', 'Customer_ID', 'Product_ID']
        
        for col in key_columns:
            if col in df.columns:
                duplicate_count = df.duplicated(subset=[col]).sum()
                if duplicate_count > 0:
                    uniqueness_score -= 0.2
                    issues.append(ValidationResult(
                        check_name=f"uniqueness_{col}",
                        level=ValidationLevel.WARNING,
                        passed=False,
                        message=f"Column {col} has {duplicate_count} duplicate values",
                        details={
                            "column": col,
                            "duplicate_count": duplicate_count,
                            "unique_percentage": (1 - duplicate_count / len(df)) * 100
                        }
                    ))
        
        # Check overall record uniqueness
        total_duplicates = df.duplicated().sum()
        if total_duplicates > 0:
            uniqueness_score -= 0.3
            issues.append(ValidationResult(
                check_name="record_uniqueness",
                level=ValidationLevel.WARNING,
                passed=False,
                message=f"Dataset contains {total_duplicates} duplicate records",
                details={
                    "duplicate_records": total_duplicates,
                    "duplicate_percentage": total_duplicates / len(df) * 100
                }
            ))
        
        return {
            "score": max(0, uniqueness_score),
            "issues": issues
        }
    
    def _check_numerical_drift(self, current_series: pd.Series, reference_series: pd.Series,
                             column_name: str, threshold: float) -> ValidationResult:
        """Check for drift in numerical column"""
        
        # Calculate basic statistics
        current_mean = current_series.mean()
        reference_mean = reference_series.mean()
        current_std = current_series.std()
        reference_std = reference_series.std()
        
        # Calculate normalized differences
        mean_diff = abs(current_mean - reference_mean) / (reference_std + 1e-8)
        std_diff = abs(current_std - reference_std) / (reference_std + 1e-8)
        
        drift_detected = mean_diff > threshold or std_diff > threshold
        
        return ValidationResult(
            check_name=f"numerical_drift_{column_name}",
            level=ValidationLevel.WARNING if drift_detected else ValidationLevel.INFO,
            passed=not drift_detected,
            message=f"Numerical drift in {column_name}: {'detected' if drift_detected else 'not detected'}",
            details={
                "column": column_name,
                "mean_diff": float(mean_diff),
                "std_diff": float(std_diff),
                "threshold": threshold,
                "current_mean": float(current_mean),
                "reference_mean": float(reference_mean),
                "current_std": float(current_std),
                "reference_std": float(reference_std)
            }
        )
    
    def _check_categorical_drift(self, current_series: pd.Series, reference_series: pd.Series,
                               column_name: str, threshold: float) -> ValidationResult:
        """Check for drift in categorical column"""
        
        # Get value distributions
        current_dist = current_series.value_counts(normalize=True)
        reference_dist = reference_series.value_counts(normalize=True)
        
        # Calculate Jensen-Shannon divergence (simplified version)
        all_values = set(current_dist.index) | set(reference_dist.index)
        
        current_probs = np.array([current_dist.get(v, 0) for v in all_values])
        reference_probs = np.array([reference_dist.get(v, 0) for v in all_values])
        
        # Simple drift metric: sum of absolute differences
        drift_metric = np.sum(np.abs(current_probs - reference_probs)) / 2
        
        drift_detected = drift_metric > threshold
        
        return ValidationResult(
            check_name=f"categorical_drift_{column_name}",
            level=ValidationLevel.WARNING if drift_detected else ValidationLevel.INFO,
            passed=not drift_detected,
            message=f"Categorical drift in {column_name}: {'detected' if drift_detected else 'not detected'}",
            details={
                "column": column_name,
                "drift_metric": float(drift_metric),
                "threshold": threshold,
                "new_categories": list(set(current_dist.index) - set(reference_dist.index)),
                "missing_categories": list(set(reference_dist.index) - set(current_dist.index))
            }
        )
    
    def _serialize_validation_result(self, result: ValidationResult) -> Dict[str, Any]:
        """Serialize validation result to dictionary"""
        
        return {
            "check_name": result.check_name,
            "level": result.level.value,
            "passed": result.passed,
            "message": result.message,
            "details": result.details,
            "affected_rows_count": len(result.affected_rows) if result.affected_rows else 0,
            "suggested_action": result.suggested_action
        }
    
    def _generate_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Count issues by type
        errors = [r for r in validation_results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in validation_results if r.level == ValidationLevel.WARNING]
        
        if errors:
            recommendations.append(f"Address {len(errors)} critical errors before proceeding with model training")
            
            # Specific recommendations for common error types
            error_types = {}
            for error in errors:
                error_type = error.check_name.split('_')[0]
                if error_type not in error_types:
                    error_types[error_type] = 0
                error_types[error_type] += 1
            
            if 'schema' in error_types:
                recommendations.append("Review data schema compliance and fix validation errors")
            if 'business' in error_types or 'net' in str([e.check_name for e in errors]):
                recommendations.append("Validate business rule violations, especially pricing logic")
            if 'null' in str([e.check_name for e in errors]):
                recommendations.append("Handle missing values in key identifier columns")
        
        if warnings:
            recommendations.append(f"Review {len(warnings)} warnings that may impact model quality")
        
        # Quality-based recommendations
        completeness_issues = [r for r in validation_results if 'completeness' in r.check_name]
        if completeness_issues:
            recommendations.append("Consider imputation strategies for missing data")
        
        outlier_issues = [r for r in validation_results if 'outlier' in r.check_name]
        if outlier_issues:
            recommendations.append("Investigate outliers to determine if they are data errors or valid extreme values")
        
        drift_issues = [r for r in validation_results if 'drift' in r.check_name]
        if drift_issues:
            recommendations.append("Monitor data drift and consider retraining models if significant drift is detected")
        
        if not recommendations:
            recommendations.append("Data quality is good. Proceed with model training and monitoring.")
        
        return recommendations
    
    def _generate_next_steps(self, errors: List[ValidationResult], 
                           warnings: List[ValidationResult]) -> List[str]:
        """Generate next steps based on validation results"""
        
        next_steps = []
        
        if errors:
            next_steps.append("1. Fix all critical errors before proceeding")
            next_steps.append("2. Re-run validation after fixing errors")
        
        if warnings:
            next_steps.append("3. Evaluate warnings and determine acceptable risk levels")
            next_steps.append("4. Document any warnings that are accepted as business exceptions")
        
        next_steps.extend([
            "5. Set up automated validation for ongoing data monitoring",
            "6. Establish data quality metrics and thresholds",
            "7. Create data quality dashboard for continuous monitoring"
        ])
        
        return next_steps
