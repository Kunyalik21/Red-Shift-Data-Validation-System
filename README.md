# Red Shift Data Validation System

This tool compares data between MySQL and Redshift databases to identify discrepancies and generate comprehensive interactive reports with professional UI/UX.

## 🚀 Features

- **Record Count Comparison**: Compare total records between MySQL and Redshift
- **Date Scenario Analysis (Time-Window Aware)**: Analyze three specific date scenarios within the configured time window:
  1. Creation and Last Modified timestamps are exactly the same
  2. Creation and Last Modified dates are the same, times differ
  3. Creation and Last Modified dates are different
- **✨ Enhanced Missing Records Detection**: Bidirectional detection of records that exist in one database but not the other
  - Records missing in Redshift (present in MySQL only)
  - Records missing in MySQL (present in Redshift only)
  - Detailed missing records with Record IDs and Last Modified timestamps
- **Field-by-Field Analysis**: Compare actual data values (respects time window when enabled) with difference tracking
- **Interactive HTML Reports**: Professional dashboard with:
  - 🌓 Dark/Light mode toggle
  - 📊 Real-time statistics dashboard
  - 🔍 Search functionality by Record ID
  - 📤 CSV export capability
  - 🎯 Collapsible sections for easy navigation
  - 📱 Responsive design
  - 🎨 Professional color coding and compact UI

## Prerequisites

- Python 3.7 or higher
- Network access to both MySQL and Redshift databases
- Required Python packages (automatically installed via requirements.txt)

## Quick Start

### For Linux/Mac:
```bash
chmod +x run_comparison.sh
./run_comparison.sh
```

### For Windows:
```cmd
run_comparison.bat
```

### Manual Setup:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate.bat  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the comparison
python3 data_comparison_full.py
```

## Configuration

Edit `config.py` to modify:
- Database connection parameters
- Tables to compare
- Sample size for data validation
- Primary key columns (if different from 'id')
- Column mappings (if column names differ between databases)
- **UI/UX Settings**:
  - `max_differences_to_show`: Maximum differences to display per column (default: 100)
  - `max_missing_records_to_show`: Maximum rows shown in expandable lists (default: 50)
  - `auto_collapse_columns`: Auto-collapse columns by default (default: True)
  - `compact_ui_mode`: Use compact card design (default: True)
- **Time Window & Deletion Validation** (`TIME_PERIOD_CONFIG`):
  ```python
  TIME_PERIOD_CONFIG = {
      'enabled': True,            # If True, compare only records within window
      'days_to_process': 45,      # Lookback window (defaults to 45 if missing/invalid)
      'validate_deletion': True,  # Validate old records are deleted from MySQL
      'deletion_threshold_days': 45  # Threshold for deletion validation (defaults to 45)
  }
  ```

Notes:
- Headers show exact totals; dropdowns list up to `max_missing_records_to_show` for performance.
- Deletion Validation shows the exact total count of records older than the threshold, while listing only up to the configured cap in the dropdown.

## Current Configuration

### MySQL Database:
- Host: 13.126.121.57
- Port: 3342
- Database: db_niinedemo
- Username: demockart

### Redshift Database:
- Host: redshift-channelkart-demo.c4ncoki1rp70.ap-south-1.redshift.amazonaws.com
- Port: 5439
- Database: channelkartdemo
- Schema: schema_niinedemo
- Username: demoadmin

### Tables Being Compared:
- ck_order_details
- ck_orders
- ck_sales
- ck_sales_details

## Output Files

The tool generates several output files:

1. **`reports/data_comparison_report_full_[timestamp].html`**: Interactive HTML report with professional UI
2. **`sample_report.html`**: Sample report demonstrating all features and functionality
3. **`logs/data_comparison_test.log`**: Execution logs with detailed information

### 📋 Sample Report
Check out `sample_report.html` to see a complete example of the generated report with:
- Missing records analysis with actual data (Record ID, Created Date, Last Modified)
- Field-by-field comparison results
- Interactive dashboard and collapsible sections
- Dark/light mode functionality
- Professional header with database names
- Bidirectional missing records detection

### 📁 Folder Structure
```
Red-Shift-Data-Validation-System/
├── data_comparison_full.py    # Main comparison script
├── config.py                  # Database configuration
├── requirements.txt           # Python dependencies
├── sample_report.html         # Sample report for demonstration
├── reports/                   # Generated reports folder
│   └── data_comparison_report_full_*.html
├── logs/                      # Execution logs folder
│   └── data_comparison_test.log
├── run_comparison.sh          # Linux/Mac execution script
├── run_comparison.bat         # Windows execution script
└── README.md                  # This documentation
```

## Report Structure

### Per-Table Analysis:
- **Record Counts**: Total counts (or filtered by configured time window)
- **Date Scenarios**: Analysis of the three patterns (time-window aware)
- **✨ Missing Records Analysis**: Bidirectional detection (header totals exact, dropdown capped)
- **Deletion Validation Analysis**: Exact total of old records + capped details list
- **Field-by-Field Comparison**: Complete value checks with differences

## Date Scenarios Explained (Time-Window Aware)

### Scenario 1: Creation and Last Modified timestamps are exactly the same
Records where `creation_time = last_modified_time` (exact timestamp match).

### Scenario 2: Creation and Last Modified dates are the same, times differ
Records where `DATE(creation_time) = DATE(last_modified_time)` but `creation_time != last_modified_time`.

### Scenario 3: Creation and Last Modified dates are different
Records where `DATE(creation_time) != DATE(last_modified_time)`.

## Understanding the Results

### Perfect Match:
- MySQL and Redshift record counts are identical
- All date scenarios have matching counts
- No missing records in either database
- Sample data comparison shows 100% match rate

### Common Issues:
- **Count Mismatch**: Different total record counts indicate missing or extra records
- **Missing Records**: Records exist in one database but not the other
- **Data Quality Issues**: Same records exist but with different values

## Troubleshooting

### Connection Issues:
- Verify database credentials in `config.py`
- Check network connectivity to database servers
- Ensure firewall rules allow connections

### Permission Issues:
- Verify database user has SELECT permissions on all tables
- Check if schema access is properly configured for Redshift

### Performance Issues:
- Reduce `sample_size` in `config.py` for faster execution
- Increase `timeout_seconds` for slow queries

## Customization

### Adding New Tables:
1. Add table names to `TABLES_TO_COMPARE` in `config.py`
2. If the table has a different primary key, add it to `PRIMARY_KEY_COLUMNS`

### Different Column Names:
If column names differ between MySQL and Redshift, add mappings to `COLUMN_MAPPINGS` in `config.py`

### Custom Date Filtering:
Modify `days_back` in `config.py` to change the date range for analysis

## Security Notes

- Database passwords are stored in plain text in `config.py`
- Consider using environment variables for production use
- Ensure proper network security for database connections

## Support

For issues or questions:
1. Check the log file (`data_comparison_test.log`) for detailed error information
2. Verify database connectivity manually
3. Review configuration settings in `config.py`