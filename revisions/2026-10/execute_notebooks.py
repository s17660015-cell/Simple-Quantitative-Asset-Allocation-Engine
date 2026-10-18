"""Execute existing notebooks using the current environment and local data."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".jupyter" / "runtime"))
os.environ.setdefault("JUPYTER_DATA_DIR", str(ROOT / ".jupyter" / "data"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".jupyter" / "ipython"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)

import nbformat
from nbclient import NotebookClient

if __name__ == "__main__":
    import argparse
    from jupyter_client import KernelManager
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", default=["MVO.ipynb", "HRP.ipynb"])
    args = parser.parse_args()
    for name in args.notebooks:
        path = ROOT / name
        nb = nbformat.read(path, as_version=4)
        kernel = KernelManager(kernel_name="python3", transport="ipc")
        client = NotebookClient(nb, timeout=180, km=kernel,
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        nbformat.write(nb, path)
        assert not any(output.get("output_type") == "error"
                       for cell in nb.cells for output in cell.get("outputs", []))
        print("Executed successfully:", name, flush=True)
