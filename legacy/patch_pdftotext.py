"""Add a text-only `pdftotext` PDF engine to an ns-pond checkout (construct-claims; James 2026-11-10: no tables).

usage: python -I patch_pdftotext.py NS_POND_CHECKOUT
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])

cfg = root / "ingestion_workflow/config/__init__.py"
s = cfg.read_text()
old = 'pdf_engines: List[Literal["mineru", "docling"]] = Field('
assert old in s, "config anchor"
cfg.write_text(s.replace(old, 'pdf_engines: List[Literal["mineru", "docling", "pdftotext"]] = Field('))

p = root / "ingestion_workflow/extractors/pdf_extractor.py"
s = p.read_text()

old = '''            if engine == "mineru":
                answers = self._extract_with_mineru(batch, extraction_root)
            else:'''
assert old in s, "dispatch anchor"
s = s.replace(old, '''            if engine == "mineru":
                answers = self._extract_with_mineru(batch, extraction_root)
            elif engine == "pdftotext":
                answers = self._extract_with_pdftotext(batch, extraction_root)
            else:''')

old = '''    def _worker_count(self, gpus: List[int]) -> int:'''
assert old in s, "worker_count anchor"
s = s.replace(old, '''    def _extract_with_pdftotext(
        self, download_results: List[DownloadResult], extraction_root: Path
    ) -> List[Any]:
        """Each PDF's text layer as poppler's pdftotext reads it, without tables, or why it read none.

        For uses that need a paper's sentences and not its tables: one short CPU process per
        PDF, where Docling on CPU takes minutes."""
        worker_count = self.settings.pdf_extract_workers or max(1, (os.cpu_count() or 2) // 4)
        logger.info("PDF extraction: %d pdftotext worker(s)", worker_count)
        contents = self._run_extraction_pipeline(
            download_results,
            extraction_root=extraction_root,
            worker=_extract_pdftotext_pdf,
            worker_count=worker_count,
            limits=Limits(memory_mb=1024, timeout_s=self.settings.pdf_extract_timeout_s),
            worker_initializer=None,
            worker_initargs=(),
            source_name="pdftotext",
            failure_message="pdftotext did not produce a result.",
            failure_builder=_build_failure_content,
        )
        return [content.error_message if _read_nothing(content) else content
                for content in contents]

    def _worker_count(self, gpus: List[int]) -> int:''')

old = '''def _pin_worker_to_gpu(slot: int, gpus: List[int]) -> None:'''
assert old in s, "pin anchor"
s = s.replace(old, '''def reflow_pdftotext(raw: str) -> str:
    """pdftotext's output as one paragraph per line.

    pdftotext breaks text where the PDF breaks lines; a paragraph ends at a blank line or a
    page break. Lines of a paragraph are joined with a space, and a word hyphenated across a
    line break ("inhib-" + "ition") is rejoined when the next line starts in lower case. A paragraph
    that stops without terminal punctuation is continued by the next one when that starts in lower
    case: a sentence cut by a column or page break. Margin line numbers of a manuscript are removed:
    numbers leading more than a third of the lines, and lines holding only a number when they are
    more than a tenth of the lines (page numbers and table cells go with them; no sentence is lost)."""
    raw = raw.replace("\\f", "\\n\\n")
    nonblank = [line for line in raw.split("\\n") if line.strip()]
    numbered = sum(bool(re.match(r"\\s*\\d{1,4}\\s+\\S", line)) for line in nonblank)
    if nonblank and numbered > len(nonblank) / 3:
        raw = re.sub(r"(?m)^\\s*\\d{1,4}\\s+(?=\\S)", "", raw)
    bare = sum(bool(re.fullmatch(r"\\s*\\d{1,4}\\s*", line)) for line in nonblank)
    if nonblank and bare > len(nonblank) / 10:
        raw = "\\n".join(line for line in raw.split("\\n") if not re.fullmatch(r"\\s*\\d{1,4}\\s*", line))
    paragraphs = []
    for block in re.split(r"\\n[ \\t]*\\n", raw):
        lines = [line.strip() for line in block.split("\\n") if line.strip()]
        if not lines:
            continue
        text = lines[0]
        for line in lines[1:]:
            if text.endswith("-") and not text.endswith(" -") and line[:1].islower():
                text = text[:-1] + line
            else:
                text = f"{text} {line}"
        if (paragraphs and text[:1].islower()
                and not re.search(r"[.!?:;)\\]]\\s*$", paragraphs[-1])):
            if paragraphs[-1].endswith("-") and not paragraphs[-1].endswith(" -"):
                paragraphs[-1] = paragraphs[-1][:-1] + text
            else:
                paragraphs[-1] = f"{paragraphs[-1]} {text}"
            continue
        paragraphs.append(text)
    return "\\n".join(paragraphs) + "\\n"


def _extract_pdftotext_pdf(download_result: DownloadResult, extraction_root: Path | str) -> Any:
    """One PDF's text layer read by poppler's pdftotext, reflowed into paragraphs; no tables."""
    import subprocess

    pdf_file = _pdf_file(download_result)
    if pdf_file is None:
        return _build_failure_content(download_result, "No PDF file in download result.")
    try:
        raw = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(pdf_file.file_path), "-"],
            capture_output=True, timeout=300, check=True,
        ).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError) as exc:
        return _build_failure_content(download_result, f"pdftotext failed: {exc}")
    full_text = reflow_pdftotext(raw)
    if not full_text.strip():
        return _build_failure_content(download_result, "pdftotext found no text layer.")
    misread = symbols_as_digits(full_text)
    if misread:
        return _set_aside(download_result, misread, 0)
    output_dir = Path(extraction_root) / safe_hash_stem(download_result.identifier.slug)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_text_path = output_dir / "article.txt"
    full_text_path.write_text(full_text, encoding="utf-8")
    return _pdf_content(download_result, full_text_path, [], [])


def _pin_worker_to_gpu(slot: int, gpus: List[int]) -> None:''')
p.write_text(s)
print("patched")
