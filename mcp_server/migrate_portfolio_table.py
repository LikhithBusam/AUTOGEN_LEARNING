import sqlite3
import json

def migrate_portfolio_table():
    """Migrate the portfolios table to use portfolio_name instead of name"""
    try:
        conn = sqlite3.connect('./data/financial_analyst.db')
        cursor = conn.cursor()
        
        # Check if the table exists and what columns it has
        cursor.execute("PRAGMA table_info(portfolios)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        print("Current portfolios table columns:", column_names)
        
        if 'name' in column_names and 'portfolio_name' not in column_names:
            print("Migrating table from 'name' to 'portfolio_name' column...")
            
            # Create backup table
            cursor.execute('''
                CREATE TABLE portfolios_backup AS 
                SELECT * FROM portfolios
            ''')
            
            # Drop original table
            cursor.execute('DROP TABLE portfolios')
            
            # Create new table with correct schema
            cursor.execute('''
                CREATE TABLE portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    portfolio_name TEXT NOT NULL,
                    holdings TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Migrate data from backup
            cursor.execute('''
                INSERT INTO portfolios (id, user_id, portfolio_name, holdings, created_at, updated_at)
                SELECT id, user_id, name, holdings, created_at, updated_at FROM portfolios_backup
            ''')
            
            # Drop backup table
            cursor.execute('DROP TABLE portfolios_backup')
            
            conn.commit()
            print("Migration completed successfully!")
            
        elif 'portfolio_name' in column_names:
            print("Table already has portfolio_name column - no migration needed!")
            
        else:
            print("Creating new portfolios table...")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    portfolio_name TEXT NOT NULL,
                    holdings TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            print("New table created!")
        
        # Show final table structure
        cursor.execute("PRAGMA table_info(portfolios)")
        columns = cursor.fetchall()
        print("Final portfolios table structure:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Migration error: {e}")
        return False

if __name__ == "__main__":
    migrate_portfolio_table()
