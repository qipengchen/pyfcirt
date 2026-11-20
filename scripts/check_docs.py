"""Parse selected examples and check local link targets, without executing code."""

import ast
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from IPython.core.inputtransformer2 import TransformerManager

transform = TransformerManager()
for name in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["docs"]:
    path = Path(name)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        cells = json.loads(text)["cells"]
        blocks = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
        text = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    else:
        blocks = re.findall(r"(?ms)^```(?:python|py)\s*\n(.*?)^```\s*$", text)
    for code in blocks:
        ast.parse(transform.transform_cell(code), filename=name)
    for target in re.findall(r"\]\(<?([^\s)>]+)>?\)", text):
        url = urlsplit(target)
        if not url.scheme and url.path and not (path.parent / unquote(url.path)).exists():
            raise ValueError(f"{name}: missing local link target {target}")
    print(f"Checked syntax and local link targets: {name}")
