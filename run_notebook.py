"""Execute the portable notebook with the active Python environment's kernel."""
from pathlib import Path
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent


def clean_machine_paths(node):
    """Remove local absolute paths from the saved public-facing execution output."""
    if isinstance(node, str):
        replacements = [(str(ROOT), "."), (sys.executable, Path(sys.executable).name),
                        (str(Path(sys.prefix)), "<python-environment>")]
        for old, new in replacements:
            node = node.replace(old, new)
        return node
    if isinstance(node, list):
        return [clean_machine_paths(item) for item in node]
    if isinstance(node, dict):
        return {key: clean_machine_paths(value) for key, value in node.items()}
    return node


def main():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    path = ROOT / "three_finance_papers.ipynb"
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    print("Execute notebook using active Python:", Path(sys.executable).name, flush=True)
    client = NotebookClient(notebook, km=manager, timeout=1200,
                            resources={"metadata": {"path": str(ROOT)}})
    client.on_cell_execute = lambda cell, cell_index: print(f"Start cell {cell_index}: {cell.source.splitlines()[0]}", flush=True)
    client.on_cell_executed = lambda cell, cell_index, execute_reply: print(f"Completed cell {cell_index}", flush=True)
    try:
        client.execute()
    finally:
        sanitized = nbformat.from_dict(clean_machine_paths(dict(notebook)))
        nbformat.write(sanitized, path)
    print("Saved fully executed portable notebook", flush=True)


if __name__ == "__main__":
    main()
