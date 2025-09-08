"""
Configuration file for Data Comparison Test
Modify these settings as needed for your environment
"""

# MySQL Database Configuration
MYSQL_CONFIG = {
    'host': '13.126.121.57',
    'user': 'demockart',
    'password': '09Nov2020#',
    'database': 'db_niinedemo',
    'port': 3342,
    'charset': 'utf8mb4'
}

# Redshift Database Configuration
REDSHIFT_CONFIG = {
    'host': 'redshift-channelkart-demo.c4ncoki1rp70.ap-south-1.redshift.amazonaws.com',
    'port': 5439,
    'database': 'channelkartdemo',
    'user': 'demoadmin',
    'password': 'SXDFIopzvk218.$',
    'options': '-c search_path=schema_niinedemo'
}

# Tables to compare
TABLES_TO_COMPARE = [
    'ck_order_details',
    'ck_orders', 
    'ck_sales',
    'ck_sales_details'
]

# Time Period Configuration
TIME_PERIOD_CONFIG = {
    'enabled': True,  # Set to False to process ALL records (original behavior)
    'days_to_process': 45,  # Number of days to look back from current date (default: 45)
    'validate_deletion': True,  # Check if old records are properly deleted from MySQL
    'deletion_threshold_days': 45  # Records older than this should be deleted from MySQL (default: 45)
}

# Test Configuration
TEST_CONFIG = {
    'sample_size': None,  # Process ALL records - full production mode
    'timeout_seconds': 600,  # Query timeout in seconds (increased for large datasets)
    'max_missing_records_to_show': 100,  # Maximum missing record IDs to display in report
    'max_differences_to_show': 100,  # Maximum differences to display per column (prevents UI lag)
    'show_all_differences_by_default': False,  # Whether to show all differences or just summary
    'compact_ui_mode': True,  # Use compact card sizes for better UX
    'auto_collapse_columns': True  # Auto-collapse columns by default
}

# Column mappings (if column names differ between databases)
# Format: {'table_name': {'mysql_column': 'redshift_column'}}
COLUMN_MAPPINGS = {
    # Example:
    # 'ck_orders': {
    #     'order_id': 'id',
    #     'created_at': 'created_time'
    # }
}

# Primary key columns for each table (if not 'id')
PRIMARY_KEY_COLUMNS = {
    # 'ck_orders': 'order_id',
    # 'ck_sales': 'sale_id'
}

# Email configuration (use environment variables for sensitive values)
# Create a .env file (not committed) with:
# EMAIL_SENDER=redshiftvalidationsystem@gmail.com
# EMAIL_PASSWORD=Salescode@321
EMAIL_CONFIG = {
    'enabled': True,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender': None,  # Loaded securely from env var EMAIL_SENDER
    'password': None,  # Loaded securely from env var EMAIL_PASSWORD
    'recipients': [
        'kunyalik.kanwar@salescode.ai'
    ]
}