"""reflow_pdftotext: pdftotext's line-broken output as one paragraph per line.

Inputs are shaped like real pdftotext output from the construct-claims sample (2026-11-10): a two-column paper
with words hyphenated across lines, a sentence cut by a page break, and an accepted manuscript whose margin line
numbers come out as their own lines (pltjajlxytj2) or as line prefixes.
"""
from ingestion_workflow.extractors.pdf_extractor import reflow_pdftotext


def test_lines_of_a_paragraph_join_and_hyphenated_words_rejoin():
    raw = "Response inhib-\nition was meas-\nured with the stop\nsignal task.\n\n2. Methods\n"
    assert reflow_pdftotext(raw) == "Response inhibition was measured with the stop signal task.\n2. Methods\n"


def test_a_sentence_cut_by_a_page_break_continues():
    raw = "A baseline signal was acquired during the rest\n\x0cinterval between tasks.\n"
    assert reflow_pdftotext(raw) == "A baseline signal was acquired during the rest interval between tasks.\n"


def test_a_finished_sentence_is_not_joined_to_the_next_paragraph():
    raw = "Each task lasted 30 s.\n\nparticipants then rested.\n"
    assert reflow_pdftotext(raw) == "Each task lasted 30 s.\nparticipants then rested.\n"


def test_margin_line_numbers_on_their_own_lines_are_dropped():
    raw = ("18\nNevertheless, the prevalence and factors influencing HT in acute\n19\n"
           "non-cardioembolic stroke remained less studied.\n20\nAdditionally, previous studies were CT-based.\n")
    assert reflow_pdftotext(raw) == (
        "Nevertheless, the prevalence and factors influencing HT in acute non-cardioembolic stroke remained "
        "less studied. Additionally, previous studies were CT-based.\n")


def test_margin_line_numbers_leading_the_lines_are_dropped():
    raw = "241 that cerebral amyloid\n242 angiopathy can predispose\n243 to HT.\n"
    assert reflow_pdftotext(raw) == "that cerebral amyloid angiopathy can predispose to HT.\n"


def test_a_line_that_starts_with_a_number_is_kept_in_an_ordinary_paper():
    raw = "2 groups were scanned.\nTable 1 shows ages.\nWe tested 30 people.\n"
    assert reflow_pdftotext(raw) == "2 groups were scanned. Table 1 shows ages. We tested 30 people.\n"
