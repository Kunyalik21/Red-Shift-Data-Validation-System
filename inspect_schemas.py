#!/usr/bin/env python3
"""
Quick script to inspect table schemas and find the correct column names
"""

import mysql.connector
import psycopg2
from config import MYSQL_CONFIG, REDSHIFT_CONFIG, TABLES_TO_COMPARE

def inspect_mysql_schema():
    print("=== MYSQL TABLE SCHEMAS ===")
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        
        for table in TABLES_TO_COMPARE:
            print(f"\nTable: {table}")
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  {col[0]} - {col[1]}")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

def inspect_redshift_schema():
    print("\n=== REDSHIFT TABLE SCHEMAS ===")
    try:
        conn = psycopg2.connect(**REDSHIFT_CONFIG)
        cursor = conn.cursor()
        
        for table in TABLES_TO_COMPARE:
            print(f"\nTable: {table}")
            cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table}' 
                AND table_schema = 'schema_niinedemo'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            for col in columns:
                print(f"  {col[0]} - {col[1]}")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_mysql_schema()
    inspect_redshift_schema()