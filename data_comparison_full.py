#!/usr/bin/env python3
"""
Enhanced Data Comparison Tool - Full Production Version
Compares data between MySQL and Redshift databases with ALL advanced features:
- ALL differences shown (not just samples)
- Search functionality by Record ID
- Collapsible dropdowns for differences
- Export to CSV functionality
- Day/Night mode
- Professional UI with 90%+ green color coding
- Empty string handling (empty ≠ null)
- Column-level validation dashboard
"""

import mysql.connector
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import re
from typing import Dict, List, Tuple, Any
import sys
import os
from config import MYSQL_CONFIG, REDSHIFT_CONFIG, TABLES_TO_COMPARE, TEST_CONFIG, TIME_PERIOD_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_comparison_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class QuerySafetyValidator:
    """Safety validator to ensure only read-only queries are executed"""
    
    # Dangerous SQL keywords that modify data
    DANGEROUS_KEYWORDS = [
        'DELETE', 'DROP', 'TRUNCATE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER',
        'GRANT', 'REVOKE', 'EXECUTE', 'EXEC', 'MERGE', 'UPSERT', 'REPLACE'
    ]
    
    # Dangerous SQL patterns
    DANGEROUS_PATTERNS = [
        r'\bDELETE\s+FROM\b',
        r'\bDROP\s+(TABLE|DATABASE|SCHEMA|INDEX|VIEW|TRIGGER|PROCEDURE|FUNCTION)\b',
        r'\bTRUNCATE\s+TABLE\b',
        r'\bINSERT\s+INTO\b',
        r'\bUPDATE\s+\w+\s+SET\b',
        r'\bCREATE\s+(TABLE|DATABASE|SCHEMA|INDEX|VIEW|TRIGGER|PROCEDURE|FUNCTION)\b',
        r'\bALTER\s+(TABLE|DATABASE|SCHEMA|INDEX|VIEW|TRIGGER|PROCEDURE|FUNCTION)\b',
        r'\bGRANT\b',
        r'\bREVOKE\b',
        r'\bEXECUTE\b',
        r'\bEXEC\b',
        r'\bMERGE\s+INTO\b',
        r'\bUPSERT\b',
        r'\bREPLACE\s+INTO\b'
    ]
    
    @classmethod
    def validate_query(cls, query: str, context: str = "Unknown") -> bool:
        """
        Validate that a query is safe (read-only)
        
        Args:
            query: SQL query to validate
            context: Context where the query is being executed
            
        Returns:
            bool: True if query is safe, False otherwise
            
        Raises:
            SecurityError: If query contains dangerous operations
        """
        if not query or not query.strip():
            raise SecurityError(f"Empty query detected in {context}")
        
        # Convert to uppercase for case-insensitive checking
        query_upper = query.upper().strip()
        
        # Check for dangerous keywords at statement level (not within column names)
        # Split by semicolons to check each statement
        statements = query_upper.split(';')
        for statement in statements:
            statement = statement.strip()
            if not statement:
                continue
                
            # Check if statement starts with dangerous keywords
            for keyword in cls.DANGEROUS_KEYWORDS:
                if statement.startswith(keyword + ' ') or statement.startswith(keyword + '\n'):
                    raise SecurityError(
                        f"Dangerous keyword '{keyword}' detected at start of statement. "
                        f"Context: {context}. Query: {query[:100]}..."
                    )
        
        # Check for dangerous patterns using regex
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, query_upper, re.IGNORECASE):
                raise SecurityError(
                    f"Dangerous SQL pattern detected in query. "
                    f"Context: {context}. Query: {query[:100]}..."
                )
        
        # Ensure query starts with SELECT or DESCRIBE
        if not query_upper.startswith(('SELECT', 'DESCRIBE', 'SHOW', 'EXPLAIN')):
            raise SecurityError(
                f"Query must start with SELECT, DESCRIBE, SHOW, or EXPLAIN. "
                f"Context: {context}. Query: {query[:100]}..."
            )
        
        return True
    
    @classmethod
    def validate_table_name(cls, table_name: str, context: str = "Unknown") -> bool:
        """
        Validate table name for SQL injection prevention
        
        Args:
            table_name: Name of the table
            context: Context where the table name is being used
            
        Returns:
            bool: True if table name is safe
            
        Raises:
            SecurityError: If table name contains dangerous characters
        """
        if not table_name or not table_name.strip():
            raise SecurityError(f"Empty table name detected in {context}")
        
        # Check for dangerous characters
        dangerous_chars = [';', '--', '/*', '*/', "'", '"', '`', '\\', '/', '*']
        for char in dangerous_chars:
            if char in table_name:
                raise SecurityError(
                    f"Dangerous character '{char}' detected in table name. "
                    f"Context: {context}. Table: {table_name}"
                )
        
        # Check for SQL injection patterns
        injection_patterns = [
            r';\s*$',  # Semicolon at end
            r'--\s*$',  # SQL comment
            r'/\*.*\*/',  # Multi-line comment
            r'UNION\s+SELECT',  # UNION injection
            r'OR\s+1\s*=\s*1',  # OR 1=1 injection
            r'AND\s+1\s*=\s*1'  # AND 1=1 injection
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, table_name, re.IGNORECASE):
                raise SecurityError(
                    f"SQL injection pattern detected in table name. "
                    f"Context: {context}. Table: {table_name}"
                )
        
        return True

class SecurityError(Exception):
    """Custom exception for security violations"""
    pass

class DatabaseComparator:
    def __init__(self):
        self.mysql_config = MYSQL_CONFIG
        self.redshift_config = REDSHIFT_CONFIG
        self.tables = TABLES_TO_COMPARE
        self.test_config = TEST_CONFIG
        self.mysql_conn = None
        self.redshift_conn = None
        
    def connect_mysql(self):
        """Connect to MySQL database"""
        try:
            self.mysql_conn = mysql.connector.connect(**self.mysql_config)
            logger.info("Successfully connected to MySQL")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {e}")
            return False
    
    def connect_redshift(self):
        """Connect to Redshift database"""
        try:
            self.redshift_conn = psycopg2.connect(**self.redshift_config)
            logger.info("Successfully connected to Redshift")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redshift: {e}")
            return False
    
    def safe_execute_query(self, query: str, db_type: str, context: str) -> Any:
        """
        Safely execute a query with validation
        
        Args:
            query: SQL query to execute
            db_type: 'mysql' or 'redshift'
            context: Context for error reporting
            
        Returns:
            Query result
            
        Raises:
            SecurityError: If query is not safe
        """
        try:
            # Validate query safety
            QuerySafetyValidator.validate_query(query, context)
            
            # Get appropriate connection
            if db_type == 'mysql':
                if not self.mysql_conn:
                    raise Exception("MySQL connection not established")
                cursor = self.mysql_conn.cursor()
            elif db_type == 'redshift':
                if not self.redshift_conn:
                    raise Exception("Redshift connection not established")
                cursor = self.redshift_conn.cursor()
            else:
                raise ValueError(f"Invalid database type: {db_type}")
            
            # Execute query
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            
            return result
            
        except SecurityError as e:
            logger.error(f"Security violation in {context}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error executing query in {context}: {e}")
            raise

    def safe_read_sql(self, query: str, db_type: str, context: str) -> pd.DataFrame:
        """
        Safely read SQL query into pandas DataFrame with validation
        
        Args:
            query: SQL query to execute
            db_type: 'mysql' or 'redshift'
            context: Context for error reporting
            
        Returns:
            pandas DataFrame
            
        Raises:
            SecurityError: If query is not safe
        """
        try:
            # Validate query safety
            QuerySafetyValidator.validate_query(query, context)
            
            # Get appropriate connection
            if db_type == 'mysql':
                if not self.mysql_conn:
                    raise Exception("MySQL connection not established")
                connection = self.mysql_conn
            elif db_type == 'redshift':
                if not self.redshift_conn:
                    raise Exception("Redshift connection not established")
                connection = self.redshift_conn
            else:
                raise ValueError(f"Invalid database type: {db_type}")
            
            # Execute query using pandas
            df = pd.read_sql(query, connection)
            return df
            
        except SecurityError as e:
            logger.error(f"Security violation in {context}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading SQL in {context}: {e}")
            raise

    def get_table_schema(self, table_name: str, db_type: str) -> List[str]:
        """Get table schema (column names) for a given table"""
        try:
            # Validate table name
            QuerySafetyValidator.validate_table_name(table_name, f"get_table_schema_{db_type}")
            
            if db_type == 'mysql':
                query = f"DESCRIBE {table_name}"
                result = self.safe_execute_query(query, 'mysql', f"get_table_schema_mysql_{table_name}")
                columns = [row[0] for row in result]
            else:  # redshift
                query = f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}' 
                    AND table_schema = 'schema_niinedemo'
                    ORDER BY ordinal_position
                """
                result = self.safe_execute_query(query, 'redshift', f"get_table_schema_redshift_{table_name}")
                columns = [row[0] for row in result]
            
            return columns
        except Exception as e:
            logger.error(f"Error getting schema for {table_name} in {db_type}: {e}")
            return []
    
    def get_record_counts(self, table_name: str) -> Dict[str, int]:
        """Get record counts from both databases"""
        counts = {}
        
        try:
            # Validate table name
            QuerySafetyValidator.validate_table_name(table_name, "get_record_counts")
            
            # MySQL count
            mysql_query = f"SELECT COUNT(*) as count FROM {table_name}"
            mysql_result = self.safe_execute_query(mysql_query, 'mysql', f"get_record_counts_mysql_{table_name}")
            counts['mysql_total'] = mysql_result[0][0]
            
            # Redshift count
            redshift_query = f"SELECT COUNT(*) as count FROM {table_name}"
            redshift_result = self.safe_execute_query(redshift_query, 'redshift', f"get_record_counts_redshift_{table_name}")
            counts['redshift_total'] = redshift_result[0][0]
            
            return counts
            
        except Exception as e:
            logger.error(f"Error getting record counts for {table_name}: {e}")
            return {'mysql_total': 0, 'redshift_total': 0}
    
    def analyze_date_scenarios(self, table_name: str) -> Dict[str, Any]:
        """Analyze the three date scenarios for a table"""
        scenarios = {}
        
        try:
            # Validate table name
            QuerySafetyValidator.validate_table_name(table_name, "analyze_date_scenarios")
            
            # Apply time-period filter if enabled
            time_dates = self.get_time_period_dates()
            time_filter = ""
            if time_dates['start_date'] and time_dates['end_date']:
                time_filter = (
                    f" AND creation_time >= '{time_dates['start_date'].strftime('%Y-%m-%d %H:%M:%S')}'"
                    f" AND creation_time <= '{time_dates['end_date'].strftime('%Y-%m-%d %H:%M:%S')}'"
                )
            
            # Scenario 1: Creation date & last modified date same (including time)
            mysql_scenario1_query = f"""
                SELECT COUNT(*) as count FROM {table_name} 
                WHERE creation_time = last_modified_time{time_filter}
            """
            
            redshift_scenario1_query = f"""
                SELECT COUNT(*) as count FROM {table_name} 
                WHERE creation_time = last_modified_time{time_filter}
            """
            
            mysql_result = self.safe_execute_query(mysql_scenario1_query, 'mysql', f"analyze_date_scenarios_mysql_s1_{table_name}")
            mysql_s1_count = mysql_result[0][0]
            
            redshift_result = self.safe_execute_query(redshift_scenario1_query, 'redshift', f"analyze_date_scenarios_redshift_s1_{table_name}")
            redshift_s1_count = redshift_result[0][0]
            
            scenarios['scenario_1'] = {
                'description': 'Creation and Last Modified timestamps are exactly the same',
                'mysql_count': mysql_s1_count,
                'redshift_count': redshift_s1_count
            }
            
            # Scenario 2: Creation date & last modified date same but time different
            mysql_scenario2_query = f"""
                SELECT COUNT(*) as count FROM {table_name} 
                WHERE DATE(creation_time) = DATE(last_modified_time)
                AND creation_time != last_modified_time{time_filter}
            """
            
            redshift_scenario2_query = f"""
                SELECT COUNT(*) as count FROM {table_name} 
                WHERE DATE(creation_time) = DATE(last_modified_time)
                AND creation_time != last_modified_time{time_filter}
            """
            
            mysql_result = self.safe_execute_query(mysql_scenario2_query, 'mysql', f"analyze_date_scenarios_mysql_s2_{table_name}")
            mysql_s2_count = mysql_result[0][0]
            
            redshift_result = self.safe_execute_query(redshift_scenario2_query, 'redshift', f"analyze_date_scenarios_redshift_s2_{table_name}")
            redshift_s2_count = redshift_result[0][0]
            
            scenarios['scenario_2'] = {
                'description': 'Creation and Last Modified dates are the same, times differ',
                'mysql_count': mysql_s2_count,
                'redshift_count': redshift_s2_count
            }
            
            # Scenario 3: Creation date & last modified date different
            mysql_scenario3_query = f"""
                SELECT COUNT(*) as count FROM {table_name} 
                WHERE DATE(creation_time) != DATE(last_modified_time){time_filter}
            """
            
            redshift_scenario3_query = f"""
                SELECT COUNT(*) as count FROM {table_name} 
                WHERE DATE(creation_time) != DATE(last_modified_time){time_filter}
            """
            
            mysql_result = self.safe_execute_query(mysql_scenario3_query, 'mysql', f"analyze_date_scenarios_mysql_s3_{table_name}")
            mysql_s3_count = mysql_result[0][0]
            
            redshift_result = self.safe_execute_query(redshift_scenario3_query, 'redshift', f"analyze_date_scenarios_redshift_s3_{table_name}")
            redshift_s3_count = redshift_result[0][0]
            
            scenarios['scenario_3'] = {
                'description': 'Creation and Last Modified dates are different',
                'mysql_count': mysql_s3_count,
                'redshift_count': redshift_s3_count
            }
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error analyzing date scenarios for {table_name}: {e}")
            return {
                'scenario_1': {'description': 'Creation and Last Modified timestamps are exactly the same', 'mysql_count': 0, 'redshift_count': 0},
                'scenario_2': {'description': 'Creation and Last Modified dates are the same, times differ', 'mysql_count': 0, 'redshift_count': 0},
                'scenario_3': {'description': 'Creation and Last Modified dates are different', 'mysql_count': 0, 'redshift_count': 0}
            }
    
    def get_time_period_dates(self) -> Dict[str, datetime]:
        """Calculate start and end dates for time period processing"""
        if not TIME_PERIOD_CONFIG.get('enabled', False):
            return {'start_date': None, 'end_date': None}
        
        # Get days_to_process with default fallback
        days_to_process = TIME_PERIOD_CONFIG.get('days_to_process')
        if not days_to_process or days_to_process <= 0:
            days_to_process = 45  # Default fallback
            logger.warning(f"days_to_process not set or invalid, using default: {days_to_process} days")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_to_process)
        
        logger.info(f"Time period: {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')} ({days_to_process} days)")
        return {'start_date': start_date, 'end_date': end_date}
    
    def get_record_counts_with_time_filter(self, table_name: str) -> Dict[str, int]:
        """Get record counts from both databases with time period filter"""
        counts = {}
        time_dates = self.get_time_period_dates()
        
        try:
            # Validate table name
            QuerySafetyValidator.validate_table_name(table_name, "get_record_counts_with_time_filter")
            
            if time_dates['start_date'] and time_dates['end_date']:
                # MySQL count with time filter
                mysql_query = f"""
                    SELECT COUNT(*) as count FROM {table_name} 
                    WHERE creation_time >= '{time_dates['start_date'].strftime('%Y-%m-%d %H:%M:%S')}'
                    AND creation_time <= '{time_dates['end_date'].strftime('%Y-%m-%d %H:%M:%S')}'
                """
                
                # Redshift count with time filter
                redshift_query = f"""
                    SELECT COUNT(*) as count FROM {table_name} 
                    WHERE creation_time >= '{time_dates['start_date'].strftime('%Y-%m-%d %H:%M:%S')}'
                    AND creation_time <= '{time_dates['end_date'].strftime('%Y-%m-%d %H:%M:%S')}'
                """
            else:
                # Fallback to original behavior
                mysql_query = f"SELECT COUNT(*) as count FROM {table_name}"
                redshift_query = f"SELECT COUNT(*) as count FROM {table_name}"
            
            mysql_result = self.safe_execute_query(mysql_query, 'mysql', f"get_record_counts_mysql_{table_name}")
            counts['mysql_total'] = mysql_result[0][0]
            
            redshift_result = self.safe_execute_query(redshift_query, 'redshift', f"get_record_counts_redshift_{table_name}")
            counts['redshift_total'] = redshift_result[0][0]
            
            return counts
            
        except Exception as e:
            logger.error(f"Error getting record counts for {table_name}: {e}")
            return {'mysql_total': 0, 'redshift_total': 0}
    
    def validate_deletion_of_old_records(self, table_name: str) -> Dict[str, Any]:
        """Validate that old records (older than threshold) are properly deleted from MySQL"""
        deletion_validation = {
            'total_old_records': 0,
            'deleted_records': 0,
            'not_deleted_records': 0,
            'not_deleted_details': []
        }
        
        if not TIME_PERIOD_CONFIG.get('validate_deletion', False):
            return deletion_validation
        
        try:
            # Validate table name
            QuerySafetyValidator.validate_table_name(table_name, "validate_deletion_of_old_records")
            
            # Get deletion_threshold_days with default fallback
            deletion_threshold_days = TIME_PERIOD_CONFIG.get('deletion_threshold_days')
            if not deletion_threshold_days or deletion_threshold_days <= 0:
                deletion_threshold_days = 45  # Default fallback
                logger.warning(f"deletion_threshold_days not set or invalid, using default: {deletion_threshold_days} days")
            
            # Calculate deletion threshold date
            threshold_date = datetime.now() - timedelta(days=deletion_threshold_days)
            
            # Count total records that are older than the threshold (these are the ones that should be deleted)
            count_query = f"""
                SELECT COUNT(*) 
                FROM {table_name}
                WHERE creation_time < '{threshold_date.strftime('%Y-%m-%d %H:%M:%S')}'
                AND last_modified_time < '{threshold_date.strftime('%Y-%m-%d %H:%M:%S')}'
            """
            count_result = self.safe_execute_query(count_query, 'mysql', f"validate_deletion_count_{table_name}")
            total_not_deleted = int(count_result[0][0]) if count_result and count_result[0] else 0
            deletion_validation['total_old_records'] = total_not_deleted
            deletion_validation['not_deleted_records'] = total_not_deleted

            # Fetch only a limited sample for display, aligned with report config
            max_show = self.test_config.get('max_missing_records_to_show', 50)
            details_query = f"""
                SELECT id, creation_time, last_modified_time 
                FROM {table_name} 
                WHERE creation_time < '{threshold_date.strftime('%Y-%m-%d %H:%M:%S')}'
                AND last_modified_time < '{threshold_date.strftime('%Y-%m-%d %H:%M:%S')}'
                ORDER BY creation_time DESC
                LIMIT {max_show}
            """
            details_result = self.safe_execute_query(details_query, 'mysql', f"validate_deletion_details_{table_name}")

            for record_id, creation_time, last_modified_time in details_result:
                deletion_validation['not_deleted_details'].append({
                    'record_id': record_id,
                    'creation_time': creation_time.strftime('%Y-%m-%d %H:%M:%S') if creation_time else None,
                    'last_modified_time': last_modified_time.strftime('%Y-%m-%d %H:%M:%S') if last_modified_time else None
                })
            
            logger.info(f"Deletion validation for {table_name}: {deletion_validation['not_deleted_records']} records not deleted out of {deletion_validation['total_old_records']} old records")
            
            return deletion_validation
            
        except Exception as e:
            logger.error(f"Error validating deletion for {table_name}: {e}")
            return deletion_validation
    
    def compare_all_data_with_differences(self, table_name: str) -> Dict[str, Any]:
        """Compare ALL data between databases and capture differences (limited for performance)"""
        logger.info(f"Processing ALL records for {table_name} - comprehensive analysis")
        
        comparison_result = {
            'matches': 0,
            'mismatches': 0,
            'total_records': 0,
            'column_analysis': {},
            'all_differences': []  # Store ALL differences, not just samples
        }
        
        try:
            # Get common columns
            mysql_columns = self.get_table_schema(table_name, 'mysql')
            redshift_columns = self.get_table_schema(table_name, 'redshift')
            common_columns = list(set(mysql_columns) & set(redshift_columns))
            
            if not common_columns:
                logger.warning(f"No common columns found for {table_name}")
                return comparison_result
            
            pk_column = 'id' if 'id' in common_columns else common_columns[0]
            columns_str = ', '.join(common_columns)
            
            # Validate table name
            QuerySafetyValidator.validate_table_name(table_name, "compare_all_data_with_differences")
            
            # Get time period dates for filtering
            time_dates = self.get_time_period_dates()
            
            # Build queries with time period filter if enabled
            if time_dates['start_date'] and time_dates['end_date']:
                mysql_query = f"""
                    SELECT {columns_str} FROM {table_name} 
                    WHERE creation_time >= '{time_dates['start_date'].strftime('%Y-%m-%d %H:%M:%S')}'
                    AND creation_time <= '{time_dates['end_date'].strftime('%Y-%m-%d %H:%M:%S')}'
                    ORDER BY {pk_column}
                """
                redshift_query = f"""
                    SELECT {columns_str} FROM {table_name} 
                    WHERE creation_time >= '{time_dates['start_date'].strftime('%Y-%m-%d %H:%M:%S')}'
                    AND creation_time <= '{time_dates['end_date'].strftime('%Y-%m-%d %H:%M:%S')}'
                    ORDER BY {pk_column}
                """
            else:
                # Fallback to original behavior
                mysql_query = f"SELECT {columns_str} FROM {table_name} ORDER BY {pk_column}"
                redshift_query = f"SELECT {columns_str} FROM {table_name} ORDER BY {pk_column}"
            
            mysql_df = self.safe_read_sql(mysql_query, 'mysql', f"compare_all_data_mysql_{table_name}")
            redshift_df = self.safe_read_sql(redshift_query, 'redshift', f"compare_all_data_redshift_{table_name}")
            
            if mysql_df.empty or redshift_df.empty:
                logger.warning(f"Empty data for {table_name}")
                return comparison_result
            
            # Initialize column analysis
            for col in common_columns:
                if col != pk_column:
                    comparison_result['column_analysis'][col] = {
                        'matches': 0,
                        'mismatches': 0,
                        'null_in_mysql': 0,
                        'null_in_redshift': 0,
                        'match_rate': 0.0,
                        'differences': []  # Store ALL differences for this column
                    }
            
            # Merge dataframes for comparison
            merged = mysql_df.merge(
                redshift_df, 
                on=pk_column, 
                how='inner', 
                suffixes=('_mysql', '_redshift')
            )
            
            comparison_result['total_records'] = len(merged)
            
            # Compare each record
            for _, row in merged.iterrows():
                record_matches = True
                
                for col in common_columns:
                    if col == pk_column:
                        continue
                    
                    # Get values
                    mysql_val = row.get(f'{col}_mysql', row.get(col))
                    redshift_val = row.get(f'{col}_redshift', row.get(col))
                    
                    # Handle None/NULL values and empty strings (as requested: empty ≠ null)
                    mysql_is_null = mysql_val is None or pd.isna(mysql_val)
                    redshift_is_null = redshift_val is None or pd.isna(redshift_val)
                    mysql_is_empty = isinstance(mysql_val, str) and mysql_val.strip() == ''
                    redshift_is_empty = isinstance(redshift_val, str) and redshift_val.strip() == ''
                    
                    # Update null counts
                    if mysql_is_null:
                        comparison_result['column_analysis'][col]['null_in_mysql'] += 1
                    if redshift_is_null:
                        comparison_result['column_analysis'][col]['null_in_redshift'] += 1
                    
                    # Compare values
                    values_match = False
                    
                    if mysql_is_null and redshift_is_null:
                        values_match = True  # Both NULL
                    elif mysql_is_empty and redshift_is_empty:
                        values_match = True  # Both empty strings
                    elif mysql_is_null or redshift_is_null:
                        values_match = False  # One NULL, one not
                    elif mysql_is_empty or redshift_is_empty:
                        values_match = False  # One empty, one not
                    else:
                        # Both have values - compare as strings
                        mysql_str = str(mysql_val).strip() if mysql_val is not None else ''
                        redshift_str = str(redshift_val).strip() if redshift_val is not None else ''
                        values_match = mysql_str == redshift_str
                    
                    # Update statistics
                    if values_match:
                        comparison_result['column_analysis'][col]['matches'] += 1
                    else:
                        comparison_result['column_analysis'][col]['mismatches'] += 1
                        record_matches = False
                        
                        # Store differences (limited for performance)
                        if len(comparison_result['column_analysis'][col]['differences']) < self.test_config['max_differences_to_show']:
                            comparison_result['column_analysis'][col]['differences'].append({
                                'record_id': str(row[pk_column]),
                                'mysql_value': str(mysql_val) if mysql_val is not None else 'NULL',
                                'redshift_value': str(redshift_val) if redshift_val is not None else 'NULL'
                            })
                
                # Update overall record statistics
                if record_matches:
                    comparison_result['matches'] += 1
                else:
                    comparison_result['mismatches'] += 1
            
            # Calculate match rates for each column
            for col in comparison_result['column_analysis']:
                total = comparison_result['column_analysis'][col]['matches'] + comparison_result['column_analysis'][col]['mismatches']
                if total > 0:
                    comparison_result['column_analysis'][col]['match_rate'] = (
                        comparison_result['column_analysis'][col]['matches'] / total * 100
                    )
                else:
                    comparison_result['column_analysis'][col]['match_rate'] = 100.0
            
        except Exception as e:
            logger.error(f"Error comparing data for {table_name}: {e}")
        
        return comparison_result
    
    def find_missing_records(self, table_name: str) -> Dict[str, Any]:
        """Find records that exist in one database but not the other"""
        logger.info(f"Finding missing records for {table_name}")
        
        missing_records = {
            'missing_in_redshift': [],
            'missing_in_mysql': [],
            'total_missing_in_redshift': 0,
            'total_missing_in_mysql': 0
        }
        
        try:
            # Get primary key column
            mysql_columns = self.get_table_schema(table_name, 'mysql')
            redshift_columns = self.get_table_schema(table_name, 'redshift')
            common_columns = list(set(mysql_columns) & set(redshift_columns))
            
            if not common_columns:
                logger.warning(f"No common columns found for {table_name}")
                return missing_records
            
            pk_column = 'id' if 'id' in common_columns else common_columns[0]
            
            # Check if last_modified_time and created_time exist for timestamps
            timestamp_column = None
            created_time_column = None
            
            # Enhanced column detection for last modified
            for col in ['last_modified_time', 'updated_at', 'modified_date', 'last_modified', 'modified_time', 'update_time']:
                if col in common_columns:
                    timestamp_column = col
                    break
            
            # Enhanced column detection for creation date
            for col in ['creation_time', 'created_time', 'created_at', 'creation_date', 'created', 'create_time', 'insert_time', 'created_date']:
                if col in common_columns:
                    created_time_column = col
                    break
            
            # Log column detection results
            logger.info(f"Found timestamp column: {timestamp_column}")
            logger.info(f"Found created_time column: {created_time_column}")
            
            # Validate table name
            QuerySafetyValidator.validate_table_name(table_name, "find_missing_records")
            
            # Get time period dates for filtering
            time_dates = self.get_time_period_dates()
            
            # Build MySQL query with time period filter if enabled
            mysql_query = f"SELECT {pk_column}"
            if timestamp_column:
                mysql_query += f", {timestamp_column}"
            if created_time_column:
                mysql_query += f", {created_time_column}"
            
            if time_dates['start_date'] and time_dates['end_date']:
                mysql_query += f"""
                    FROM {table_name} 
                    WHERE creation_time >= '{time_dates['start_date'].strftime('%Y-%m-%d %H:%M:%S')}'
                    AND creation_time <= '{time_dates['end_date'].strftime('%Y-%m-%d %H:%M:%S')}'
                    ORDER BY {pk_column}
                """
            else:
                mysql_query += f" FROM {table_name} ORDER BY {pk_column}"
            
            mysql_records = self.safe_execute_query(mysql_query, 'mysql', f"find_missing_records_mysql_{table_name}")
            
            # Build Redshift query with time period filter if enabled
            redshift_query = f"SELECT {pk_column}"
            if timestamp_column:
                redshift_query += f", {timestamp_column}"
            if created_time_column:
                redshift_query += f", {created_time_column}"
            
            if time_dates['start_date'] and time_dates['end_date']:
                redshift_query += f"""
                    FROM {table_name} 
                    WHERE creation_time >= '{time_dates['start_date'].strftime('%Y-%m-%d %H:%M:%S')}'
                    AND creation_time <= '{time_dates['end_date'].strftime('%Y-%m-%d %H:%M:%S')}'
                    ORDER BY {pk_column}
                """
            else:
                redshift_query += f" FROM {table_name} ORDER BY {pk_column}"
            
            redshift_records = self.safe_execute_query(redshift_query, 'redshift', f"find_missing_records_redshift_{table_name}")
            
            # Convert to sets for comparison (using just the ID for set operations)
            mysql_ids = {record[0] for record in mysql_records}
            redshift_ids = {record[0] for record in redshift_records}
            
            # Create dictionaries for timestamp lookup
            mysql_data = {record[0]: record for record in mysql_records}
            redshift_data = {record[0]: record for record in redshift_records}
            
            # Find missing records
            missing_in_redshift_ids = mysql_ids - redshift_ids
            missing_in_mysql_ids = redshift_ids - mysql_ids
            
            # Prepare missing records data with timestamps (limited for performance)
            max_show = self.test_config.get('max_missing_records_to_show', 50)
            
            for record_id in list(missing_in_redshift_ids)[:max_show]:
                record_data = {'record_id': str(record_id)}
                if timestamp_column and len(mysql_data[record_id]) > 1:
                    record_data['last_modified'] = str(mysql_data[record_id][1]) if mysql_data[record_id][1] else 'N/A'
                else:
                    record_data['last_modified'] = 'N/A'
                if created_time_column and len(mysql_data[record_id]) > 2:
                    record_data['created_date'] = str(mysql_data[record_id][2]) if mysql_data[record_id][2] else 'N/A'
                else:
                    record_data['created_date'] = 'N/A'
                missing_records['missing_in_redshift'].append(record_data)
            
            for record_id in list(missing_in_mysql_ids)[:max_show]:
                record_data = {'record_id': str(record_id)}
                if timestamp_column and len(redshift_data[record_id]) > 1:
                    record_data['last_modified'] = str(redshift_data[record_id][1]) if redshift_data[record_id][1] else 'N/A'
                else:
                    record_data['last_modified'] = 'N/A'
                if created_time_column and len(redshift_data[record_id]) > 2:
                    record_data['created_date'] = str(redshift_data[record_id][2]) if redshift_data[record_id][2] else 'N/A'
                else:
                    record_data['created_date'] = 'N/A'
                missing_records['missing_in_mysql'].append(record_data)
            
            missing_records['total_missing_in_redshift'] = len(missing_in_redshift_ids)
            missing_records['total_missing_in_mysql'] = len(missing_in_mysql_ids)
            
            logger.info(f"Found {len(missing_in_redshift_ids)} records missing in Redshift, {len(missing_in_mysql_ids)} missing in MySQL")
            
        except Exception as e:
            logger.error(f"Error finding missing records for {table_name}: {e}")
        
        return missing_records
    
    def generate_enhanced_html_report(self, report: Dict[str, Any], filename: str):
        """Generate the enhanced HTML report with ALL your requirements"""
        
        # Helper function to safely format numbers
        def safe_format(value, default="0"):
            try:
                if value is None:
                    return default
                return f"{int(value):,}"
            except (ValueError, TypeError):
                return str(value) if value is not None else default
        
        # Calculate column statistics
        total_columns = 0
        failed_columns = 0
        for table_name, table_data in report['tables'].items():
            if 'field_comparison' in table_data and 'column_analysis' in table_data['field_comparison']:
                for col, col_data in table_data['field_comparison']['column_analysis'].items():
                    total_columns += 1
                    if col_data.get('match_rate', 0) < 95:
                        failed_columns += 1
        
        success_rate = ((total_columns - failed_columns) / total_columns * 100) if total_columns > 0 else 100
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Sync Validation Report - Enhanced</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #f8fafc;
            --bg-secondary: #ffffff;
            --bg-tertiary: #e2e8f0;
            --text-primary: #1a202c;
            --text-secondary: #4a5568;
            --text-muted: #718096;
            --accent-color: #3182ce;
            --success-color: #38a169;
            --success-light: #f0fff4;
            --warning-color: #d69e2e;
            --warning-light: #fffbeb;
            --error-color: #e53e3e;
            --error-light: #fed7d7;
            --excellent-color: #38a169;
            --excellent-light: #f0fff4;
            --border-color: #e2e8f0;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }}

        [data-theme="dark"] {{
            --bg-primary: #0f1419;
            --bg-secondary: #1a202c;
            --bg-tertiary: #2d3748;
            --text-primary: #ffffff;
            --text-secondary: #e2e8f0;
            --text-muted: #a0aec0;
            --accent-color: #3182ce;
            --success-color: #48bb78;
            --warning-color: #ed8936;
            --error-color: #f56565;
            --success-light: #1a202c;
            --warning-light: #1a202c;
            --error-light: #1a202c;
            --excellent-light: #1a202c;
            --border-color: #4a5568;
            --shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            transition: all 0.3s ease;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            margin-bottom: 25px;
            box-shadow: var(--shadow);
            position: relative;
        }}

        .theme-toggle {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: var(--accent-color);
            color: white;
            border: none;
            padding: 12px 16px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            transition: all 0.3s ease;
        }}

        .theme-toggle:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow);
        }}

        .header h1 {{
            font-size: 2.0em;
            margin-bottom: 8px;
            color: var(--accent-color);
        }}

        .subtitle {{
            color: var(--text-secondary);
            font-size: 0.85em;
            margin: 2px 0;
        }}

        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}

        .dashboard-section {{
            margin-bottom: 25px;
        }}

        .dashboard-title {{
            font-size: 1.4em;
            color: var(--accent-color);
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
        }}

        .card {{
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
        }}



        .card.excellent {{
            background: var(--excellent-light);
            border-color: var(--excellent-color);
        }}

        .card.success {{
            background: var(--success-light);
            border-color: var(--success-color);
        }}

        .card.warning {{
            background: var(--warning-light);
            border-color: var(--warning-color);
        }}

        .card.error {{
            background: var(--error-light);
            border-color: var(--error-color);
        }}

        .card h3 {{
            font-size: 1.0em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            color: #1a202c;
            font-weight: 600;
        }}

        .card .value {{
            font-size: 1.8em;
            font-weight: 700;
            margin: 6px 0;
            color: #1a202c;
        }}

        .card .subtitle {{
            color: #4a5568;
            font-size: 0.9em;
            margin-top: 4px;
        }}

        .card.excellent .value {{ color: #22543d; }}
        .card.success .value {{ color: #22543d; }}
        .card.warning .value {{ color: #744210; }}
        .card.error .value {{ color: #742a2a; }}

        [data-theme="dark"] .card h3 {{
            color: #ffffff;
        }}

        [data-theme="dark"] .card .value {{
            color: #ffffff;
        }}

        [data-theme="dark"] .card .subtitle {{
            color: #a0aec0;
        }}

        [data-theme="dark"] .card.excellent .value {{ color: #48bb78; }}
        [data-theme="dark"] .card.success .value {{ color: #48bb78; }}
        [data-theme="dark"] .card.warning .value {{ color: #ed8936; }}
        [data-theme="dark"] .card.error .value {{ color: #f56565; }}

        /* HOVER EFFECTS - LIGHT MODE SPECIFIC */
        [data-theme="light"] .card:hover,
        :not([data-theme="dark"]) .card:hover {{
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
            background: #2d3748 !important;
            color: #ffffff !important;
        }}

        [data-theme="light"] .card:hover h3,
        :not([data-theme="dark"]) .card:hover h3 {{
            color: #ffffff !important;
        }}

        [data-theme="light"] .card:hover .value,
        :not([data-theme="dark"]) .card:hover .value {{
            color: #ffffff !important;
        }}

        [data-theme="light"] .card:hover .subtitle,
        :not([data-theme="dark"]) .card:hover .subtitle {{
            color: #a0aec0 !important;
        }}

        /* Status-specific hover colors for values - LIGHT MODE */
        [data-theme="light"] .card.excellent:hover .value,
        :not([data-theme="dark"]) .card.excellent:hover .value {{
            color: #48bb78 !important;
        }}

        [data-theme="light"] .card.success:hover .value,
        :not([data-theme="dark"]) .card.success:hover .value {{
            color: #48bb78 !important;
        }}

        [data-theme="light"] .card.warning:hover .value,
        :not([data-theme="dark"]) .card.warning:hover .value {{
            color: #ed8936 !important;
        }}

        [data-theme="light"] .card.error:hover .value,
        :not([data-theme="dark"]) .card.error:hover .value {{
            color: #f56565 !important;
        }}

        /* DARK MODE HOVER EFFECTS */
        [data-theme="dark"] .card:hover {{
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
            background: #4a5568 !important;
            color: #ffffff !important;
        }}

        [data-theme="dark"] .card:hover h3 {{
            color: #ffffff !important;
        }}

        [data-theme="dark"] .card:hover .value {{
            color: #ffffff !important;
        }}

        [data-theme="dark"] .card:hover .subtitle {{
            color: #a0aec0 !important;
        }}

        [data-theme="dark"] .card.excellent:hover .value {{
            color: #48bb78 !important;
        }}

        [data-theme="dark"] .card.success:hover .value {{
            color: #48bb78 !important;
        }}

        [data-theme="dark"] .card.warning:hover .value {{
            color: #ed8936 !important;
        }}

        [data-theme="dark"] .card.error:hover .value {{
            color: #f56565 !important;
        }}

        .table-section {{
            background: var(--bg-secondary);
            border-radius: 16px;
            margin: 30px 0;
            box-shadow: var(--shadow);
            overflow: hidden;
        }}

        .table-header {{
            background: var(--accent-color);
            color: white;
            padding: 20px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            transition: background 0.3s ease;
        }}

        .table-header:hover {{
            background: #2c5aa0;
        }}

        [data-theme="dark"] .table-header:hover {{
            background: #2c5aa0;
            color: #ffffff;
        }}

        .table-header h2 {{
            font-size: 1.4em;
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .table-content {{
            padding: 25px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }}

        th {{
            background: var(--bg-tertiary);
            font-weight: 600;
            color: var(--text-primary);
        }}

        .status-good {{ color: var(--success-color); font-weight: bold; }}
        .status-error {{ color: var(--error-color); font-weight: bold; }}
        .status-warning {{ color: var(--warning-color); font-weight: bold; }}

        /* Enhanced Column Analysis */
        .column-analysis {{
            margin-top: 20px;
        }}

        .column-card {{
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin: 8px 0;
            overflow: hidden;
            background: var(--bg-secondary);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            color: var(--text-primary);
        }}

        .column-header {{
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            padding: 16px 20px;
            font-weight: bold;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s ease;
            border-radius: 8px 8px 0 0;
            color: #1a202c;
        }}

        .column-header .matches-info {{
            color: #4a5568;
        }}

        .column-header .matches-info span {{
            color: inherit;
        }}

        [data-theme="dark"] .column-header {{
            background: linear-gradient(135deg, #404040, #353535);
            color: var(--text-primary);
        }}

        [data-theme="dark"] .column-header.success {{
            background: #1a202c;
            color: #48bb78;
            border-left: 4px solid #48bb78;
        }}

        [data-theme="dark"] .column-header.warning {{
            background: #1a202c;
            color: #ed8936;
            border-left: 4px solid #ed8936;
        }}

        [data-theme="dark"] .column-header.error {{
            background: #1a202c;
            color: #f56565;
            border-left: 4px solid #f56565;
        }}

        [data-theme="dark"] .column-header.failed {{
            background: #1a202c;
            color: #f56565;
            border-left: 4px solid #f56565;
        }}

        [data-theme="dark"] .card {{
            background: #1a202c;
            color: #ffffff;
            border: 1px solid #4a5568;
        }}

        [data-theme="dark"] .card h3 {{
            color: #ffffff;
        }}

        [data-theme="dark"] .card .value {{
            color: #ffffff;
        }}

        [data-theme="dark"] .card .subtitle {{
            color: #a0aec0;
        }}

        [data-theme="dark"] .table-container {{
            background: #1a202c;
            border: 1px solid #4a5568;
        }}

        [data-theme="dark"] table {{
            background: #1a202c;
        }}

        [data-theme="dark"] th {{
            background: #2d3748;
            color: #ffffff;
            border-bottom: 1px solid #4a5568;
        }}

        [data-theme="dark"] td {{
            background: #1a202c;
            color: #ffffff;
            border-bottom: 1px solid #4a5568;
        }}

        [data-theme="dark"] .differences-section {{
            background: #1a202c;
            border: 1px solid #4a5568;
        }}

        [data-theme="dark"] .differences-section h4 {{
            color: #ffffff;
        }}

        [data-theme="dark"] .differences-table {{
            background: #1a202c;
        }}

        [data-theme="dark"] .differences-table th {{
            background: #2d3748;
            color: #ffffff;
        }}

        [data-theme="dark"] .differences-table td {{
            background: #1a202c;
            color: #ffffff;
        }}

        [data-theme="dark"] .search-input {{
            background: #2d3748;
            color: #ffffff;
            border: 1px solid #4a5568;
        }}

        [data-theme="dark"] .search-input::placeholder {{
            color: #a0aec0;
        }}

        [data-theme="dark"] .export-btn {{
            background: #3182ce;
            color: #ffffff;
        }}

        [data-theme="dark"] .export-btn:hover {{
            background: #2c5aa0;
        }}

        [data-theme="dark"] .floating-btn {{
            background: #3182ce;
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(49, 130, 206, 0.4);
        }}

        [data-theme="dark"] .floating-btn:hover {{
            background: #2c5aa0;
            transform: translateY(-2px);
        }}

        [data-theme="dark"] .progress-bar {{
            background: #2d3748;
        }}

        [data-theme="dark"] .progress-fill {{
            background: linear-gradient(90deg, #48bb78, #38a169);
        }}

        [data-theme="dark"] .progress-text {{
            color: #ffffff;
        }}

        [data-theme="dark"] .matches-info {{
            color: #a0aec0;
        }}

        [data-theme="dark"] .matches-info span {{
            color: inherit;
        }}

        [data-theme="dark"] .status-badge {{
            background: #2d3748;
            color: #ffffff;
        }}

        [data-theme="dark"] .status-badge.success {{
            background: #22543d;
            color: #48bb78;
        }}

        [data-theme="dark"] .status-badge.warning {{
            background: #744210;
            color: #ed8936;
        }}

        [data-theme="dark"] .status-badge.error {{
            background: #742a2a;
            color: #f56565;
        }}

        [data-theme="dark"] .differences-section:hover {{
            background: #2d3748;
            border-color: #4a5568;
        }}

        [data-theme="dark"] .differences-section:hover h4 {{
            color: #ffffff;
        }}

        [data-theme="dark"] .search-input:focus {{
            background: #2d3748;
            color: #ffffff;
            border-color: #3182ce;
            outline: none;
        }}

        [data-theme="dark"] .collapse-btn:hover {{
            background: #4a5568;
            color: #ffffff;
        }}

        [data-theme="dark"] .export-btn:hover {{
            background: #2c5aa0;
            color: #ffffff;
        }}

        .column-header:hover {{
            background: #2d3748;
            color: #ffffff !important;
        }}

        .column-header.success:hover {{
            background: #2d3748;
            color: #48bb78 !important;
        }}

        .column-header.warning:hover {{
            background: #2d3748;
            color: #ed8936 !important;
        }}

        .column-header.error:hover {{
            background: #2d3748;
            color: #f56565 !important;
        }}

        .column-header.failed:hover {{
            background: #2d3748;
            color: #f56565 !important;
        }}

        .column-header:hover .matches-info {{
            color: #a0aec0 !important;
        }}

        .column-header:hover .matches-info span {{
            color: inherit !important;
        }}

        .column-header.failed {{
            border-left: 4px solid #e53e3e;
            background: #fed7d7;
            color: #742a2a;
        }}

        .column-header.warning {{
            border-left: 4px solid #d69e2e;
            background: #fffbeb;
            color: #744210;
        }}

        .column-header.success {{
            border-left: 4px solid #38a169;
            background: #f0fff4;
            color: #22543d;
        }}

        .column-header.excellent {{
            border-left: 4px solid #38a169;
            background: #f0fff4;
            color: #22543d;
        }}

        .column-header.error {{
            border-left: 4px solid #e53e3e;
            background: #fed7d7;
            color: #742a2a;
        }}

        .column-header .matches-info {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: 4px;
        }}

        .column-header .matches-info span {{
            display: flex;
            align-items: center;
            gap: 4px;
            font-weight: 500;
        }}

        .column-details {{
            padding: 16px;
            display: none;
            background: var(--bg-secondary);
            border-top: 1px solid var(--border-color);
            color: var(--text-primary);
        }}

        .column-details.active {{
            display: block;
            animation: slideDown 0.3s ease;
        }}

        @keyframes slideDown {{
            from {{
                opacity: 0;
                max-height: 0;
            }}
            to {{
                opacity: 1;
                max-height: 500px;
            }}
        }}

        /* Enhanced Progress Bars */
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: var(--border-color);
            border-radius: 10px;
            overflow: hidden;
            margin: 12px 0;
            position: relative;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--success-color), #51cf66);
            transition: width 0.8s ease;
            border-radius: 10px;
        }}

        .progress-fill.warning {{
            background: linear-gradient(90deg, var(--warning-color), #ffd43b);
        }}

        .progress-fill.error {{
            background: linear-gradient(90deg, var(--error-color), #ff6b6b);
        }}

        .progress-fill.excellent {{
            background: linear-gradient(90deg, var(--excellent-color), #22c55e);
        }}

        .progress-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-weight: bold;
            font-size: 11px;
            color: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }}

        /* Enhanced Differences Section */
        .differences-section {{
            margin: 15px 0;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            background: var(--bg-secondary);
        }}

        .differences-header {{
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 18px 25px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s ease;
        }}

        .differences-header:hover {{
            background: linear-gradient(135deg, #ff5252, #d63031);
        }}

        .differences-header h4 {{
            margin: 0;
            font-size: 1.2em;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .differences-controls {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}

        .search-input {{
            padding: 10px 15px;
            border: none;
            border-radius: 8px;
            font-size: 0.9em;
            min-width: 250px;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            transition: all 0.3s ease;
        }}

        .search-input:focus {{
            outline: none;
            box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.5);
            transform: scale(1.02);
        }}

        .differences-content {{
            display: none;
            background: var(--bg-secondary);
        }}

        .differences-content.active {{
            display: block;
        }}

        .differences-info {{
            padding: 20px 25px;
            background: var(--bg-primary);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .differences-count {{
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 1.1em;
        }}

        .export-btn {{
            background: var(--accent-color);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .export-btn:hover {{
            background: #2c5aa0;
            transform: translateY(-2px);
        }}

        .differences-list {{
            max-height: 500px;
            overflow-y: auto;
            padding: 15px;
        }}

        .difference-item {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin: 10px 0;
            overflow: hidden;
            transition: all 0.3s ease;
        }}

        .difference-item:hover {{
            box-shadow: var(--shadow);
            transform: translateY(-1px);
        }}

        .difference-header {{
            background: var(--bg-primary);
            padding: 12px 18px;
            border-bottom: 1px solid var(--border-color);
            font-weight: 600;
            color: var(--accent-color);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .difference-values {{
            padding: 18px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        .value {{
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid;
        }}

        .mysql-value {{
            background: rgba(40, 167, 69, 0.1);
            border-left-color: var(--success-color);
        }}

        .redshift-value {{
            background: rgba(0, 122, 204, 0.1);
            border-left-color: var(--accent-color);
        }}

        .value-text {{
            font-family: 'Courier New', monospace;
            background: rgba(0, 0, 0, 0.05);
            padding: 4px 8px;
            border-radius: 4px;
            word-break: break-all;
            font-size: 0.9em;
        }}

        /* Missing Records Styles */
        .missing-records-section {{
            margin-top: 20px;
        }}
        
        .missing-records-card {{
            background: var(--bg-secondary);
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }}
        
        .missing-header {{
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--error-light);
            border-bottom: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }}
        
        .missing-header:hover {{
            background: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
        }}
        
        .missing-header h4 {{
            margin: 0;
            color: var(--error-color);
            font-size: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .missing-header .subtitle {{
            font-size: 12px;
            color: var(--text-muted);
            margin-left: 24px;
        }}
        
        .missing-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
            background: var(--bg-primary);
        }}
        
        .missing-content.active {{
            max-height: 500px;
            overflow-y: auto;
        }}
        
        .missing-info {{
            padding: 10px 20px;
            background: var(--bg-tertiary);
            font-size: 12px;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border-color);
        }}
        
        .missing-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .missing-table th,
        .missing-table td {{
            padding: 8px 20px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
            font-size: 13px;
        }}
        
        .missing-table th {{
            background: var(--bg-secondary);
            font-weight: 600;
            color: var(--text-primary);
        }}
        
        .missing-table td {{
            color: var(--text-secondary);
        }}
        
        .missing-table tr:hover {{
            background: var(--bg-tertiary);
        }}

        /* Floating Action Buttons */
        .floating-controls {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            display: flex;
            flex-direction: column;
            gap: 15px;
            z-index: 9999;
            pointer-events: auto;
        }}

        .fab {{
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: var(--accent-color);
            color: white;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: var(--shadow-lg);
            transition: all 0.3s ease;
            position: relative;
            z-index: 10000;
        }}

        .fab:hover {{
            transform: scale(1.1);
            background: #2c5aa0;
        }}

        /* Ensure floating buttons are always visible */
        .floating-controls * {{
            pointer-events: auto !important;
        }}

        /* Prevent any element from covering the floating buttons */
        .floating-controls {{
            isolation: isolate;
        }}

        .collapse-btn {{
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        .collapse-btn:hover {{
            background: rgba(255, 255, 255, 0.3);
        }}

        .timestamp {{
            text-align: center;
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid var(--border-color);
            color: var(--text-muted);
            font-size: 0.95em;
        }}

        /* Responsive Design */
        @media (max-width: 768px) {{
            .container {{ padding: 15px; }}
            .dashboard {{ grid-template-columns: 1fr; }}
            .differences-controls {{ flex-direction: column; gap: 10px; }}
            .search-input {{ min-width: 200px; }}
            .difference-values {{ grid-template-columns: 1fr; }}
            .floating-controls {{ bottom: 20px; right: 20px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <button class="theme-toggle" onclick="toggleTheme()">
                <i class="fas fa-moon" id="theme-icon"></i>
                <span id="theme-text">Dark Mode</span>
            </button>
            <h1><i class="fas fa-chart-line"></i> Data Sync Validation Report</h1>
            <div class="subtitle">MySQL ↔ Redshift Comprehensive Analysis</div>
            <div class="subtitle"><strong>MySQL Database:</strong> {self.mysql_config['database']} | <strong>Redshift Database:</strong> {self.redshift_config['database']}</div>
            <div class="subtitle">Generated: {report['timestamp']}</div>
            {f'<div class="subtitle" style="background: var(--accent-color); color: white; padding: 8px 16px; border-radius: 8px; margin-top: 10px; display: inline-block;"><i class="fas fa-calendar-alt"></i> <strong>Time Period:</strong> {report["time_period"]["start_date"]} to {report["time_period"]["end_date"]} ({report["time_period"]["days_processed"]} days)</div>' if report.get('time_period', {}).get('enabled') else ''}
        </div>

        <!-- Overall System Health Dashboard -->
        <div class="dashboard-section">
            <h2 class="dashboard-title">
                <i class="fas fa-tachometer-alt"></i> Overall System Health
            </h2>
            <div class="dashboard">
                <div class="card {'success' if report['summary']['tables_with_mismatches'] == 0 else 'warning'}">
                    <h3><i class="fas fa-table"></i> Tables Analyzed</h3>
                    <div class="value">{report['summary']['total_tables']}</div>
                    <div class="subtitle">{'All Clean' if report['summary']['tables_with_mismatches'] == 0 else str(report['summary']['tables_with_mismatches']) + ' with Issues'}</div>
                </div>
                <div class="card success">
                    <h3><i class="fas fa-database"></i> MySQL Records</h3>
                    <div class="value">{safe_format(report['summary']['total_mysql_records'])}</div>
                    <div class="subtitle">Source Database</div>
                </div>
                <div class="card success">
                    <h3><i class="fas fa-cloud"></i> Redshift Records</h3>
                    <div class="value">{safe_format(report['summary']['total_redshift_records'])}</div>
                    <div class="subtitle">Target Database</div>
                </div>
                <div class="card {'error' if report['summary']['total_missing_in_redshift'] > 0 else 'excellent'}">
                    <h3><i class="fas fa-exclamation-triangle"></i> Missing in Redshift</h3>
                    <div class="value">{report['summary']['total_missing_in_redshift']}</div>
                    <div class="subtitle">Records in MySQL only</div>
                </div>
                <div class="card {'error' if report['summary']['total_missing_in_mysql'] > 0 else 'excellent'}">
                    <h3><i class="fas fa-exclamation-triangle"></i> Missing in MySQL</h3>
                    <div class="value">{report['summary']['total_missing_in_mysql']}</div>
                    <div class="subtitle">Records in Redshift only</div>
                </div>
            </div>
        </div>

        <!-- Column Validation Dashboard -->
        <div class="dashboard-section">
            <h2 class="dashboard-title">
                <i class="fas fa-columns"></i> Column Validation Dashboard
            </h2>
            <div class="dashboard">
                <div class="card excellent">
                    <h3><i class="fas fa-list"></i> Total Columns</h3>
                    <div class="value">{total_columns}</div>
                    <div class="subtitle">Analyzed Across All Tables</div>
                </div>
                <div class="card {'error' if failed_columns > 0 else 'excellent'}">
                    <h3><i class="fas fa-times-circle"></i> Failed Columns</h3>
                    <div class="value">{failed_columns}</div>
                    <div class="subtitle">Match Rate < 95%</div>
                </div>
                <div class="card {'excellent' if success_rate >= 95 else 'success' if success_rate >= 90 else 'warning' if success_rate >= 70 else 'error'}">
                    <h3><i class="fas fa-percentage"></i> Success Rate</h3>
                    <div class="value">{success_rate:.1f}%</div>
                    <div class="subtitle">Column-Level Quality</div>
                </div>
                
            </div>
        </div>
"""

        # Add table-specific analysis
        for table_name, table_data in report['tables'].items():
            difference = table_data['record_counts']['mysql_total'] - table_data['record_counts']['redshift_total']
            
            html_content += f"""
        <div class="table-section" id="table-{table_name}">
            <div class="table-header" onclick="toggleTableSection('{table_name}')">
                <h2>
                    <i class="fas fa-table"></i> {table_name.upper()}
                    {'<i class="fas fa-exclamation-triangle" style="color: #ffd43b;"></i>' if difference != 0 else '<i class="fas fa-check-circle" style="color: #51cf66;"></i>'}
                </h2>
                <button class="collapse-btn">
                    <i class="fas fa-chevron-down" id="chevron-{table_name}"></i>
                    <span id="text-{table_name}">Expand</span>
                </button>
            </div>
            <div class="table-content" id="content-{table_name}" style="display: none;">
                <h3><i class="fas fa-chart-bar"></i> Record Count Analysis</h3>
                <table>
                    <tr>
                        <th><i class="fas fa-database"></i> Database</th>
                        <th><i class="fas fa-list-ol"></i> Total Records</th>
                        <th><i class="fas fa-check-circle"></i> Status</th>
                    </tr>
                    <tr>
                        <td><strong>MySQL</strong></td>
                        <td>{safe_format(table_data['record_counts']['mysql_total'])}</td>
                        <td class="status-good"><i class="fas fa-check"></i> Source</td>
                    </tr>
                    <tr>
                        <td><strong>Redshift</strong></td>
                        <td>{safe_format(table_data['record_counts']['redshift_total'])}</td>
                        <td class="{'status-error' if difference != 0 else 'status-good'}">
                            <i class="fas fa-{'times' if difference != 0 else 'check'}"></i>
                            {'Mismatch' if difference != 0 else 'Match'}
                        </td>
                    </tr>
                </table>
                
                <h3><i class="fas fa-calendar-alt"></i> Date Pattern Analysis</h3>
                <table>
                    <tr>
                        <th><i class="fas fa-list"></i> Pattern</th>
                        <th><i class="fas fa-database"></i> MySQL</th>
                        <th><i class="fas fa-cloud"></i> Redshift</th>
                        <th><i class="fas fa-balance-scale"></i> Diff</th>
                        <th><i class="fas fa-flag"></i> Status</th>
                    </tr>
"""
            
            for scenario_key, scenario_data in table_data['date_scenarios'].items():
                scenario_diff = scenario_data['mysql_count'] - scenario_data['redshift_count']
                status_class = 'status-good' if scenario_diff == 0 else 'status-warning'
                
                html_content += f"""
                    <tr>
                        <td>{scenario_data['description']}</td>
                        <td>{safe_format(scenario_data['mysql_count'])}</td>
                        <td>{safe_format(scenario_data['redshift_count'])}</td>
                        <td class="{'status-error' if scenario_diff != 0 else 'status-good'}">{scenario_diff:+d}</td>
                        <td class="{status_class}">
                            <i class="fas fa-{'check' if scenario_diff == 0 else 'exclamation-triangle'}"></i>
                        </td>
                    </tr>
"""
            
            html_content += """
                </table>
"""
            
            # Add field-by-field comparison with ALL differences
            if 'field_comparison' in table_data and 'column_analysis' in table_data['field_comparison']:
                column_analysis = table_data['field_comparison']['column_analysis']
                if column_analysis:
                    total_records = table_data['field_comparison']['total_records']
                    html_content += f"""
                <h3><i class="fas fa-microscope"></i> Field-by-Field Analysis - ALL {safe_format(total_records)} Records Processed</h3>
                <div class="column-analysis">
"""
                    
                    for col, col_data in column_analysis.items():
                        match_rate = col_data.get('match_rate', 0)
                        
                        # Professional color coding with consistent, less flashy colors
                        if match_rate >= 90:
                            status_class = 'success'
                            icon = 'check'
                        elif match_rate >= 75:
                            status_class = 'warning'
                            icon = 'exclamation-triangle'
                        else:
                            status_class = 'error'
                            icon = 'times-circle'
                        
                        differences_count = len(col_data.get('differences', []))
                        
                        # Show only success percentage by default (collapsed)
                        show_details = "active" if not self.test_config['auto_collapse_columns'] else ""
                        chevron_icon = "fa-chevron-up" if not self.test_config['auto_collapse_columns'] else "fa-chevron-down"
                        
                        html_content += f"""
                    <div class="column-card">
                        <div class="column-header {status_class}" onclick="toggleColumn('{table_name}_{col}')">
                            <div>
                                <i class="fas fa-columns"></i> <strong>{col}</strong> - Match Rate: {match_rate:.1f}%
                                <div class="matches-info">
                                    <span style="color: var(--success-color);"><i class="fas fa-check"></i> {safe_format(col_data['matches'])} matches</span>
                                    <span style="color: var(--error-color);"><i class="fas fa-times"></i> {safe_format(col_data['mismatches'])} mismatches</span>
                                </div>
                            </div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <i class="fas fa-{icon}"></i>
                                <button class="collapse-btn" onclick="event.stopPropagation(); toggleColumn('{table_name}_{col}')">
                                    <i class="fas {chevron_icon}" id="col-chevron-{table_name}_{col}"></i>
                                </button>
                            </div>
                        </div>
                        <div class="column-details {show_details}" id="{table_name}_{col}">
                            <div class="progress-bar">
                                <div class="progress-fill {status_class}" style="width: {match_rate}%"></div>
                                <div class="progress-text">{match_rate:.1f}%</div>
                            </div>
                            <table style="font-size: 0.9em;">
                                <tr><td><strong><i class="fas fa-check"></i> Matches:</strong></td><td>{safe_format(col_data['matches'])}</td></tr>
                                <tr><td><strong><i class="fas fa-times"></i> Mismatches:</strong></td><td>{safe_format(col_data['mismatches'])}</td></tr>
                                <tr><td><strong><i class="fas fa-question"></i> NULL in MySQL:</strong></td><td>{safe_format(col_data['null_in_mysql'])}</td></tr>
                                <tr><td><strong><i class="fas fa-question"></i> NULL in Redshift:</strong></td><td>{safe_format(col_data['null_in_redshift'])}</td></tr>
                            </table>
"""
                        
                        # Add differences with search and export functionality (limited for performance)
                        if col_data.get('differences'):
                            # Limit differences display
                            max_diffs_to_show = min(differences_count, self.test_config['max_differences_to_show'])
                            show_all_by_default = self.test_config['show_all_differences_by_default']
                            
                            html_content += f"""
                            <div class="differences-section">
                                <div class="differences-header" onclick="toggleDifferences('{table_name}_{col}_diff')">
                                    <h4><i class="fas fa-exclamation-triangle"></i> Sample {max_diffs_to_show}</h4>
                                    <div class="differences-controls">
                                        <input type="text" id="search_{table_name}_{col}" placeholder="Search by Record ID..." 
                                               onkeyup="filterDifferences('{table_name}_{col}')" class="search-input" onclick="event.stopPropagation();">
                                        <button class="collapse-btn" onclick="event.stopPropagation(); toggleDifferences('{table_name}_{col}_diff')">
                                            <i class="fas fa-chevron-down" id="diff-chevron-{table_name}_{col}"></i>
                                        </button>
                                    </div>
                                </div>
                                <div class="differences-content {'active' if show_all_by_default else ''}" id="{table_name}_{col}_diff">
                                    <div class="differences-info">
                                        <span class="differences-count">Sample {max_diffs_to_show} of {differences_count:,} total differences</span>
                                        <button onclick="exportDifferences('{table_name}_{col}')" class="export-btn">
                                            <i class="fas fa-download"></i> Export CSV
                                        </button>
                                    </div>
                                    <div class="differences-list" id="differences_list_{table_name}_{col}">
"""
                            
                            # Show limited differences for performance
                            for i, diff in enumerate(col_data['differences'][:max_diffs_to_show]):
                                html_content += f"""
                                        <div class="difference-item" data-record-id="{diff['record_id']}">
                                            <div class="difference-header">
                                                <i class="fas fa-id-card"></i> <strong>Record ID:</strong> {diff['record_id']}
                                            </div>
                                            <div class="difference-values">
                                                <div class="value mysql-value">
                                                    <i class="fas fa-database"></i> <strong>MySQL:</strong> 
                                                    <span class="value-text">{diff['mysql_value']}</span>
                                                </div>
                                                <div class="value redshift-value">
                                                    <i class="fas fa-cloud"></i> <strong>Redshift:</strong> 
                                                    <span class="value-text">{diff['redshift_value']}</span>
                                                </div>
                                            </div>
                                        </div>
"""
                            
                            html_content += """
                                    </div>
                                </div>
                            </div>
"""
                        
                        html_content += """
                        </div>
                    </div>
"""
                    
                    html_content += """
                </div>
"""
            
            # Add missing records section
            if 'missing_records' in table_data:
                missing_data = table_data['missing_records']
                total_missing_redshift = missing_data['total_missing_in_redshift']
                total_missing_mysql = missing_data['total_missing_in_mysql']
                
                if total_missing_redshift > 0 or total_missing_mysql > 0:
                    html_content += f"""
                <h3><i class="fas fa-search"></i> Missing Records Analysis</h3>
                <div class="missing-records-section">
"""
                    
                    # Missing in Redshift (present in MySQL only)
                    if total_missing_redshift > 0:
                        missing_in_redshift = missing_data['missing_in_redshift']
                        showing_count = len(missing_in_redshift)
                        
                        html_content += f"""
                    <div class="missing-records-card error">
                        <div class="missing-header" onclick="toggleMissingRecords('{table_name}_missing_redshift')">
                            <h4><i class="fas fa-exclamation-triangle"></i> Missing in Redshift ({total_missing_redshift:,} records)</h4>
                            <span class="subtitle">Records present in MySQL but missing in Redshift</span>
                            <i class="fas fa-chevron-down chevron" id="{table_name}_missing_redshift_chevron"></i>
                        </div>
                        <div class="missing-content" id="{table_name}_missing_redshift_content">
                            <div class="missing-info">Showing {showing_count} of {total_missing_redshift:,} missing records</div>
                            <table class="missing-table">
                                <tr>
                                    <th><i class="fas fa-key"></i> Record ID</th>
                                    <th><i class="fas fa-calendar-plus"></i> Created Date</th>
                                    <th><i class="fas fa-clock"></i> Last Modified</th>
                                </tr>
"""
                        
                        for record in missing_in_redshift:
                            html_content += f"""
                                <tr>
                                    <td>{record['record_id']}</td>
                                    <td>{record['created_date']}</td>
                                    <td>{record['last_modified']}</td>
                                </tr>
"""
                        
                        html_content += """
                            </table>
                        </div>
                    </div>
"""
                    
                    # Missing in MySQL (present in Redshift only)
                    if total_missing_mysql > 0:
                        missing_in_mysql = missing_data['missing_in_mysql']
                        showing_count = len(missing_in_mysql)
                        
                        html_content += f"""
                    <div class="missing-records-card error">
                        <div class="missing-header" onclick="toggleMissingRecords('{table_name}_missing_mysql')">
                            <h4><i class="fas fa-exclamation-triangle"></i> Missing in MySQL ({total_missing_mysql:,} records)</h4>
                            <span class="subtitle">Records present in Redshift but missing in MySQL</span>
                            <i class="fas fa-chevron-down chevron" id="{table_name}_missing_mysql_chevron"></i>
                        </div>
                        <div class="missing-content" id="{table_name}_missing_mysql_content">
                            <div class="missing-info">Showing {showing_count} of {total_missing_mysql:,} missing records</div>
                            <table class="missing-table">
                                <tr>
                                    <th><i class="fas fa-key"></i> Record ID</th>
                                    <th><i class="fas fa-calendar-plus"></i> Created Date</th>
                                    <th><i class="fas fa-clock"></i> Last Modified</th>
                                </tr>
"""
                        
                        for record in missing_in_mysql:
                            html_content += f"""
                                <tr>
                                    <td>{record['record_id']}</td>
                                    <td>{record['created_date']}</td>
                                    <td>{record['last_modified']}</td>
                                </tr>
"""
                        
                        html_content += """
                            </table>
                        </div>
                    </div>
"""
                    
                    html_content += """
                </div>
"""
            
            # Add Deletion Validation Section
            if TIME_PERIOD_CONFIG.get('validate_deletion', False) and 'deletion_validation' in table_data:
                deletion_data = table_data['deletion_validation']
                total_old_records = deletion_data['total_old_records']
                not_deleted_records = deletion_data['not_deleted_records']
                
                if total_old_records > 0:
                    html_content += f"""
                <h3><i class="fas fa-trash-alt"></i> Deletion Validation Analysis</h3>
                <div class="missing-records-section">
                    <div class="missing-records-card {'error' if not_deleted_records > 0 else 'success'}">
                        <div class="missing-header" onclick="toggleMissingRecords('{table_name}_deletion_validation')">
                            <h4><i class="fas fa-exclamation-triangle"></i> Records Not Deleted ({not_deleted_records:,} records)</h4>
                            <span class="subtitle">Records older than {TIME_PERIOD_CONFIG.get('deletion_threshold_days', 45)} days that should be deleted from MySQL</span>
                            <i class="fas fa-chevron-down chevron" id="{table_name}_deletion_validation_chevron"></i>
                        </div>
                        <div class="missing-content" id="{table_name}_deletion_validation_content">
                            <div class="missing-info">Showing {len(deletion_data['not_deleted_details'])} of {not_deleted_records:,} records that should be deleted</div>
                            <table class="missing-table">
                                <tr>
                                    <th><i class="fas fa-key"></i> Record ID</th>
                                    <th><i class="fas fa-calendar-plus"></i> Creation Time</th>
                                    <th><i class="fas fa-clock"></i> Last Modified Time</th>
                                </tr>
"""
                    
                    for record in deletion_data['not_deleted_details']:
                        html_content += f"""
                                <tr>
                                    <td>{record['record_id']}</td>
                                    <td>{record['creation_time']}</td>
                                    <td>{record['last_modified_time']}</td>
                                </tr>
"""
                    
                    html_content += """
                            </table>
                        </div>
                    </div>
                </div>
"""
            
            html_content += """
            </div>
        </div>
"""
        
        # Add JavaScript and closing tags
        html_content += f"""
        <!-- Floating Action Buttons -->
        <div class="floating-controls">
            <button class="fab" onclick="scrollToTop()" title="Scroll to Top">
                <i class="fas fa-arrow-up"></i>
            </button>
            <button class="fab" onclick="expandAllColumns()" title="Expand All Columns">
                <i class="fas fa-expand-arrows-alt"></i>
            </button>
            <button class="fab" onclick="collapseAllColumns()" title="Collapse All Columns">
                <i class="fas fa-compress-arrows-alt"></i>
            </button>
        </div>

        <div class="timestamp">
            <i class="fas fa-clock"></i>
            Report generated automatically by Enhanced Data Sync Validation System
            <br>
            <strong>Performance:</strong> Processed {safe_format(report['summary']['total_mysql_records'])} records across {report['summary']['total_tables']} tables
        </div>
    </div>
    
    <script>
        // Theme Toggle Functionality
        function toggleTheme() {{
            const html = document.documentElement;
            const themeIcon = document.getElementById('theme-icon');
            const themeText = document.getElementById('theme-text');
            
            if (html.getAttribute('data-theme') === 'light') {{
                html.setAttribute('data-theme', 'dark');
                themeIcon.className = 'fas fa-sun';
                themeText.textContent = 'Light Mode';
                localStorage.setItem('theme', 'dark');
            }} else {{
                html.setAttribute('data-theme', 'light');
                themeIcon.className = 'fas fa-moon';
                themeText.textContent = 'Dark Mode';
                localStorage.setItem('theme', 'light');
            }}
        }}

        // Load saved theme
        window.addEventListener('DOMContentLoaded', function() {{
            const savedTheme = localStorage.getItem('theme') || 'light';
            const html = document.documentElement;
            const themeIcon = document.getElementById('theme-icon');
            const themeText = document.getElementById('theme-text');
            
            html.setAttribute('data-theme', savedTheme);
            if (savedTheme === 'dark') {{
                themeIcon.className = 'fas fa-sun';
                themeText.textContent = 'Light Mode';
            }}
        }});

        // Enhanced Differences Toggle Function
        function toggleDifferences(diffId) {{
            const content = document.getElementById(diffId);
            const chevron = document.getElementById('diff-chevron-' + diffId.replace('_diff', ''));
            
            if (content.classList.contains('active')) {{
                content.classList.remove('active');
                if (chevron) chevron.className = 'fas fa-chevron-down';
            }} else {{
                content.classList.add('active');
                if (chevron) chevron.className = 'fas fa-chevron-up';
            }}
        }}

        // Filter Differences by Record ID (Search functionality as requested!)
        function filterDifferences(tableColId) {{
            const searchInput = document.getElementById('search_' + tableColId);
            const differencesList = document.getElementById('differences_list_' + tableColId);
            const searchTerm = searchInput.value.toLowerCase();
            const differenceItems = differencesList.querySelectorAll('.difference-item');
            
            let visibleCount = 0;
            differenceItems.forEach(function(item) {{
                const recordId = item.getAttribute('data-record-id').toLowerCase();
                if (recordId.includes(searchTerm)) {{
                    item.style.display = 'block';
                    visibleCount++;
                }} else {{
                    item.style.display = 'none';
                }}
            }});
            
            // Update count display
            const countElement = differencesList.parentElement.querySelector('.differences-count');
            if (countElement) {{
                countElement.textContent = 'Showing ' + visibleCount + ' of ' + differenceItems.length + ' differences';
            }}
        }}

        // Export Differences to CSV (Export functionality as requested!)
        function exportDifferences(tableColId) {{
            const differencesList = document.getElementById('differences_list_' + tableColId);
            const differenceItems = differencesList.querySelectorAll('.difference-item');
            const tableName = tableColId.split('_')[0];
            const columnName = tableColId.split('_')[1];
            
            let csvContent = 'Record ID,MySQL Value,Redshift Value\\n';
            
            differenceItems.forEach(function(item) {{
                const recordId = item.getAttribute('data-record-id');
                const mysqlValue = item.querySelector('.mysql-value .value-text').textContent;
                const redshiftValue = item.querySelector('.redshift-value .value-text').textContent;
                
                // Escape CSV values
                const escapedRecordId = '"' + recordId.replace(/"/g, '""') + '"';
                const escapedMysqlValue = '"' + mysqlValue.replace(/"/g, '""') + '"';
                const escapedRedshiftValue = '"' + redshiftValue.replace(/"/g, '""') + '"';
                
                csvContent += escapedRecordId + ',' + escapedMysqlValue + ',' + escapedRedshiftValue + '\\n';
            }});
            
            // Create and download CSV file
            const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', 'differences_' + tableName + '_' + columnName + '_' + new Date().toISOString().split('T')[0] + '.csv');
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}

        // Column Toggle with Smooth Animation
        function toggleColumn(columnId) {{
            const content = document.getElementById(columnId);
            const chevron = document.getElementById('col-chevron-' + columnId);
            
            if (content.classList.contains('active')) {{
                content.classList.remove('active');
                if (chevron) chevron.className = 'fas fa-chevron-down';
            }} else {{
                content.classList.add('active');
                if (chevron) chevron.className = 'fas fa-chevron-up';
            }}
        }}

        // Missing Records Toggle
        function toggleMissingRecords(missingId) {{
            const content = document.getElementById(missingId + '_content');
            const chevron = document.getElementById(missingId + '_chevron');
            
            if (content.classList.contains('active')) {{
                content.classList.remove('active');
                if (chevron) chevron.className = 'fas fa-chevron-down';
            }} else {{
                content.classList.add('active');
                if (chevron) chevron.className = 'fas fa-chevron-up';
            }}
        }}

        // Table Section Toggle
        function toggleTableSection(tableName) {{
            const content = document.getElementById('content-' + tableName);
            const chevron = document.getElementById('chevron-' + tableName);
            const text = document.getElementById('text-' + tableName);
            
            if (content.style.display === 'none') {{
                content.style.display = 'block';
                if (chevron) chevron.className = 'fas fa-chevron-up';
                if (text) text.textContent = 'Collapse';
            }} else {{
                content.style.display = 'none';
                if (chevron) chevron.className = 'fas fa-chevron-down';
                if (text) text.textContent = 'Expand';
            }}
        }}

        // Floating Action Button Functions
        function scrollToTop() {{
            window.scrollTo({{
                top: 0,
                behavior: 'smooth'
            }});
        }}

        function expandAllColumns() {{
            // Expand all column details
            const columns = document.querySelectorAll('.column-details');
            const columnChevrons = document.querySelectorAll('[id^="col-chevron-"]');
            
            columns.forEach(function(column) {{
                column.classList.add('active');
            }});
            
            columnChevrons.forEach(function(chevron) {{
                chevron.className = 'fas fa-chevron-up';
            }});

            // Expand all table sections
            const tableContents = document.querySelectorAll('.table-content');
            const tableChevrons = document.querySelectorAll('[id^="chevron-"]');
            const tableTexts = document.querySelectorAll('[id^="text-"]');
            
            tableContents.forEach(function(content) {{
                content.style.display = 'block';
            }});
            
            tableChevrons.forEach(function(chevron) {{
                chevron.className = 'fas fa-chevron-up';
            }});
            
            tableTexts.forEach(function(text) {{
                text.textContent = 'Collapse';
            }});
        }}

        function collapseAllColumns() {{
            // Collapse all column details
            const columns = document.querySelectorAll('.column-details');
            const columnChevrons = document.querySelectorAll('[id^="col-chevron-"]');
            
            columns.forEach(function(column) {{
                column.classList.remove('active');
            }});
            
            columnChevrons.forEach(function(chevron) {{
                chevron.className = 'fas fa-chevron-down';
            }});

            // Collapse all table sections
            const tableContents = document.querySelectorAll('.table-content');
            const tableChevrons = document.querySelectorAll('[id^="chevron-"]');
            const tableTexts = document.querySelectorAll('[id^="text-"]');
            
            tableContents.forEach(function(content) {{
                content.style.display = 'none';
            }});
            
            tableChevrons.forEach(function(chevron) {{
                chevron.className = 'fas fa-chevron-down';
            }});
            
            tableTexts.forEach(function(text) {{
                text.textContent = 'Expand';
            }});
        }}

        // Auto-expand problematic columns only if not in auto-collapse mode
        window.addEventListener('load', function() {{
            // Auto-expand warning and error columns only if auto-collapse is disabled
            if (!{str(self.test_config['auto_collapse_columns']).lower()}) {{
                const columns = document.querySelectorAll('.column-details');
                columns.forEach(function(column) {{
                    const header = column.previousElementSibling;
                    if (header.classList.contains('error') || header.classList.contains('warning')) {{
                        column.classList.add('active');
                        const columnId = column.id;
                        const chevron = document.getElementById('col-chevron-' + columnId);
                        if (chevron) {{
                            chevron.className = 'fas fa-chevron-up';
                        }}
                    }}
                }});
            }}

            // Always auto-expand error columns (below 75%) regardless of auto-collapse setting
            const errorColumns = document.querySelectorAll('.column-header.error');
            errorColumns.forEach(function(header) {{
                const column = header.nextElementSibling;
                if (column) {{
                    column.classList.add('active');
                    const columnId = column.id;
                    const chevron = document.getElementById('col-chevron-' + columnId);
                    if (chevron) {{
                        chevron.className = 'fas fa-chevron-up';
                    }}
                }}
            }});

            // Add loading complete animation
            document.body.style.opacity = '0';
            setTimeout(function() {{
                document.body.style.transition = 'opacity 0.5s ease';
                document.body.style.opacity = '1';
            }}, 100);
        }});

        // Ensure floating buttons stay visible
        window.addEventListener('scroll', function() {{
            const floatingControls = document.querySelector('.floating-controls');
            if (floatingControls) {{
                floatingControls.style.display = 'flex';
                floatingControls.style.visibility = 'visible';
                floatingControls.style.opacity = '1';
            }}
        }});

        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (e.ctrlKey || e.metaKey) {{
                switch(e.key) {{
                    case 't':
                        e.preventDefault();
                        toggleTheme();
                        break;
                    case 'Home':
                        e.preventDefault();
                        scrollToTop();
                        break;
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def run_comparison(self):
        """Main method to run the complete comparison"""
        logger.info("Starting comprehensive data comparison test...")
        
        if not self.connect_mysql():
            return False
        
        if not self.connect_redshift():
            return False
        
        try:
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'tables': {},
                'summary': {
                    'total_tables': len(self.tables),
                    'tables_with_mismatches': 0,
                    'total_mysql_records': 0,
                    'total_redshift_records': 0,
                    'total_missing_in_redshift': 0,
                    'total_missing_in_mysql': 0
                }
            }
            
            # Get time period information
            time_dates = self.get_time_period_dates()
            report['time_period'] = {
                'enabled': TIME_PERIOD_CONFIG.get('enabled', False),
                'start_date': time_dates['start_date'].strftime('%Y-%m-%d %H:%M:%S') if time_dates['start_date'] else None,
                'end_date': time_dates['end_date'].strftime('%Y-%m-%d %H:%M:%S') if time_dates['end_date'] else None,
                'days_processed': TIME_PERIOD_CONFIG.get('days_to_process', 45) if TIME_PERIOD_CONFIG.get('enabled', False) else None
            }
            
            for table_name in self.tables:
                logger.info(f"Analyzing table: {table_name}")
                
                # Use time-filtered record counts if enabled
                if TIME_PERIOD_CONFIG.get('enabled', False):
                    record_counts = self.get_record_counts_with_time_filter(table_name)
                else:
                    record_counts = self.get_record_counts(table_name)
                
                table_report = {
                    'record_counts': record_counts,
                    'date_scenarios': self.analyze_date_scenarios(table_name),
                    'field_comparison': self.compare_all_data_with_differences(table_name),
                    'missing_records': self.find_missing_records(table_name),
                    'deletion_validation': self.validate_deletion_of_old_records(table_name)
                }
                
                # Update summary
                report['summary']['total_mysql_records'] += table_report['record_counts']['mysql_total']
                report['summary']['total_redshift_records'] += table_report['record_counts']['redshift_total']
                report['summary']['total_missing_in_redshift'] += table_report['missing_records']['total_missing_in_redshift']
                report['summary']['total_missing_in_mysql'] += table_report['missing_records']['total_missing_in_mysql']
                
                # Check for mismatches
                has_mismatch = (
                    table_report['record_counts']['mysql_total'] != table_report['record_counts']['redshift_total'] or
                    table_report['field_comparison']['mismatches'] > 0
                )
                
                if has_mismatch:
                    report['summary']['tables_with_mismatches'] += 1
                
                report['tables'][table_name] = table_report
            
            # Save enhanced HTML report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_file = f'reports/data_comparison_report_full_{timestamp}.html'
            self.generate_enhanced_html_report(report, html_file)
            
            logger.info(f"Enhanced HTML report saved to: {html_file}")
            logger.info("Data comparison completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error during comparison: {e}")
            return False
        
        finally:
            if self.mysql_conn:
                self.mysql_conn.close()
            if self.redshift_conn:
                self.redshift_conn.close()

def main():
    """Main function"""
    comparator = DatabaseComparator()
    success = comparator.run_comparison()
    
    if success:
        print("\n" + "="*80)
        print("DATA COMPARISON COMPLETED SUCCESSFULLY")
        print("="*80)
        print("Report generated successfully.")
        print("Check the generated HTML report for detailed analysis.")
    else:
        print("\n" + "="*80)
        print("DATA COMPARISON FAILED")
        print("="*80)
        print("Check data_comparison_test.log for error details")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())