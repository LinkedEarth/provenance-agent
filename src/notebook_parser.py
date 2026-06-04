import nbformat


def parse_notebook(path: str) -> list[str]:
    """
    Read a .ipynb file and return a list of cell types in order.

    Each element is either 'code' or 'markdown' corresponding to that cell.
    
    e.g. for an sample notebook: ['code', 'markdown', 'code', 'markdown', 'code', 'markdown']"""
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    return [cell.cell_type for cell in nb.cells]
