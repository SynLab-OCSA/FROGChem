import pytest
from frogchem.printer import (
    print_title, print_info, print_success, print_warning, print_error,
    print_table, extract_table_data,
)


class TestExtractTableData:
    def test_basic_extraction(self):
        items = [
            {'name': 'A', 'value': 1.0},
            {'name': 'B', 'value': 2.0},
        ]
        result = extract_table_data(items, ['name', 'value'])
        assert result == [['A', 1.0], ['B', 2.0]]

    def test_missing_column(self):
        items = [{'name': 'A'}]
        result = extract_table_data(items, ['name', 'value'], missing_val='N/A')
        assert result == [['A', 'N/A']]

    def test_empty_items(self):
        result = extract_table_data([], ['a', 'b'])
        assert result == []

    def test_column_order(self):
        items = [{'x': 1, 'y': 2, 'z': 3}]
        result = extract_table_data(items, ['z', 'x'])
        assert result == [[3, 1]]


class TestPrintTable:
    def test_empty_data(self):
        print_table([], headers=['a', 'b'])

    def test_with_data(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        print_table(data, headers=['x', 'y'])

    def test_pastable_format(self):
        data = [[1.0, 2.0]]
        print_table(data, headers=['a', 'b'], pastable=True)

    def test_none_in_data(self):
        data = [[1.0, None]]
        print_table(data, headers=['a', 'b'])


class TestPrintFunctions:
    def test_print_info(self):
        print_info("test info")

    def test_print_success(self):
        print_success("test success")

    def test_print_warning(self):
        print_warning("test warning")

    def test_print_error_no_exit(self):
        print_error("test error")

    def test_print_title(self):
        print_title("Test Title")
