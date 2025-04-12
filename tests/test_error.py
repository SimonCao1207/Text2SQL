import os
import sys
import unittest

# Add parent directory to path to import filter_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filter_data import is_error


class TestSQLError(unittest.TestCase):
    def test_valid_sql_query(self):
        """Test a valid SQL query that should not produce an error."""
        valid_sql = "SELECT admissions.dischtime FROM admissions WHERE admissions.subject_id = 10026406 AND strftime('%Y', admissions.admittime) >= '2100' ORDER BY admissions.admittime ASC LIMIT 1;"
        result = is_error(valid_sql)
        self.assertFalse(result, f"Valid SQL query produced an error: {valid_sql}")

    def test_invalid_sql_query(self):
        """Test an invalid SQL query that should produce an error."""
        invalid_sql = "SELECT * FROM nonexistent_table"
        result = is_error(invalid_sql)
        self.assertTrue(
            result, f"Invalid SQL query did not produce an error: {invalid_sql}"
        )

    def test_syntax_error_sql(self):
        """Test a SQL query with syntax error."""
        syntax_error_sql = "SELECT * FRO patients"
        result = is_error(syntax_error_sql)
        self.assertTrue(
            result,
            f"SQL query with syntax error did not produce an error: {syntax_error_sql}",
        )


if __name__ == "__main__":
    # unittest.main()
    TestSQLError().test_syntax_error_sql()
