import pytest
from frogchem.printer import (
    print_title, print_info, print_success, print_warning, print_error,
    print_table,
)


class TestPrintTable:
    def test_empty_data(self):
        print_table([], headers=['a', 'b'])

    def test_with_data(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        print_table(data, headers=['x', 'y'])


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
