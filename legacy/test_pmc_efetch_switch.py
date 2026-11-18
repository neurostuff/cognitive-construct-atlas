"""pmc_efetch: the pmc source asks NCBI efetch for what PMC's Cloud Service lacks, unless the setting is off.

The two network fetches are stubbed at the extractor's own seam (_fetch_cloud, _fetch_efetch): the test is about
which one runs, not about either service.
"""
from types import SimpleNamespace

from ingestion_workflow.extractors.pmc_extractor import PmcExtractor


def _extractor(efetch: bool, calls: list):
    ex = PmcExtractor.__new__(PmcExtractor)
    ex.settings = SimpleNamespace(pmc_efetch=efetch)
    ex._fetch_cloud = lambda ids: ({"1": "<article 1>"}, {})
    ex._fetch_efetch = lambda ids: (calls.append(list(ids)) or ({"2": "<article 2>"}, {}))
    return ex


def test_efetch_reads_what_the_bucket_lacks_by_default():
    calls = []
    articles, reasons = _extractor(True, calls)._fetch(["1", "2"])
    assert calls == [["2"]]
    assert set(articles) == {"1", "2"} and reasons == {}


def test_efetch_off_keeps_to_the_bucket():
    calls = []
    articles, reasons = _extractor(False, calls)._fetch(["1", "2"])
    assert calls == []
    assert set(articles) == {"1"}
    assert reasons == {"2": "not in PMC's Cloud Service (pmc_efetch off)"}
