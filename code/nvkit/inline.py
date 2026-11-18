"""Splice nvkit.js and nvkit.css into pages between their nvkit markers (one source for both atlas pages).
usage: python3 inline.py PAGE.html [PAGE.html ...]   Each page holds, inside a <script> and a <style>, the marker pair
/* nvkit:start */ ... /* nvkit:end */; the JS block is the one in a <script>, the CSS block the one in a <style>."""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAT = re.compile(r"/\* nvkit:start \*/.*?/\* nvkit:end \*/", re.S)
for page in sys.argv[1:]:
    s = Path(page).read_text()
    blocks = list(PAT.finditer(s))
    assert len(blocks) == 2, f"{page}: expected 2 nvkit blocks, found {len(blocks)}"
    out, last = [], 0
    for m in blocks:
        before = s[:m.start()]
        kind = "css" if before.rfind("<style") > before.rfind("<script") else "js"
        out += [s[last:m.start()], (HERE / f"nvkit.{kind}").read_text().strip()]
        last = m.end()
    out.append(s[last:]); Path(page).write_text("".join(out)); print(page, "spliced")
