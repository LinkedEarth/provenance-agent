import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from notebook_parser import parse_notebook

SAMPLE = os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'sample.ipynb')


def test_returns_list():
    result = parse_notebook(SAMPLE)
    assert isinstance(result, list)


def test_correct_length():
    result = parse_notebook(SAMPLE)
    assert len(result) == 6


def test_correct_types():
    result = parse_notebook(SAMPLE)
    assert result == ['markdown', 'code', 'markdown', 'code', 'markdown', 'code']


def test_only_valid_values():
    result = parse_notebook(SAMPLE)
    assert all(t in ('code', 'markdown') for t in result)


if __name__ == '__main__':
    result = parse_notebook(SAMPLE)
    print('Cell types:', result)
    print('All tests passed.' if result == ['markdown', 'code', 'markdown', 'code', 'markdown', 'code'] else 'MISMATCH')
