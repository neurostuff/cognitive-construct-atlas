"""The dependency rule: the domain and the ports know nothing of adapters, use cases, files or the network."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "construct_atlas"
IO = {"json", "csv", "os", "sys", "pathlib", "sqlite3", "urllib", "subprocess", "requests", "shutil", "glob"}


def imports(path):
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add("." * node.level + (node.module or ""))
    return out


def test_domain_and_ports_do_no_io_and_do_not_look_outward():
    for layer in ("domain", "ports"):
        for f in (SRC / layer).glob("*.py"):
            for m in imports(f):
                assert m.split(".")[0] not in IO, f"{f.name} imports {m}"
                assert "adapters" not in m and "application" not in m and "cli" not in m, f"{f.name} imports {m}"


def test_application_does_not_import_adapters():
    for f in (SRC / "application").glob("*.py"):
        assert not any("adapters" in m for m in imports(f)), f.name
