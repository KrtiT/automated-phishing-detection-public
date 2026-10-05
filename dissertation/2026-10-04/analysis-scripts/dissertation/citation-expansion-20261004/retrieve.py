"""Retain public primary-source evidence for a bounded citation revision."""

import concurrent.futures
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import fitz
from lxml import html


HERE = Path(__file__).resolve().parent
SOURCES = HERE / "sources"
ITEMS = {
    "axelsson-crossref.json": "https://api.crossref.org/works/10.1145/357830.357849",
    "axelsson-author.pdf": "https://www.cse.chalmers.se/~sax/tisec.pdf",
    "dempster-crossref.json": "https://api.crossref.org/works/10.1111/j.2517-6161.1977.tb01600.x",
    "dempster-publisher.html": "https://academic.oup.com/jrsssb/article/39/1/1/7027539",
    "adamw-openreview.json": "https://api.openreview.net/notes?id=Bkg6RiCqY7",
    "adamw-paper.pdf": "https://openreview.net/pdf?id=Bkg6RiCqY7",
    "tranco-publisher.html": "https://www.ndss-symposium.org/ndss-paper/tranco-a-research-oriented-top-sites-ranking-hardened-against-manipulation/",
    "tail-publisher.html": "https://cacm.acm.org/research/the-tail-at-scale/",
    "tail-crossref.json": "https://api.crossref.org/works/10.1145/2408776.2408794",
    "schroeder-publisher.html": "https://www.usenix.org/legacy/events/nsdi06/tech/schroeder.html",
    "schroeder-paper.pdf": "https://www.usenix.org/legacy/events/nsdi06/tech/full_papers/schroeder/schroeder.pdf",
    "lipton-publisher.html": "https://proceedings.mlr.press/v80/lipton18a.html",
    "lipton-paper.pdf": "https://proceedings.mlr.press/v80/lipton18a/lipton18a.pdf",
    "brier-crossref.json": "https://api.crossref.org/works/" + quote("10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2", safe=""),
}


def retrieve(item):
    name, url = item
    path = SOURCES / name
    result = {"file": name, "url": url}
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 (scholarly citation verification)"})
        with urlopen(request, timeout=45) as response:
            content = response.read()
            result["resolved_url"] = response.url
        if content.startswith(b"\x1f\x8b"):
            content = gzip.decompress(content)
        path.write_bytes(content)
        if name.endswith(".pdf"):
            with fitz.open(path) as document:
                source_text = "\n".join(page.get_text() for page in document)
            path.with_suffix(".txt").write_text(source_text)
        elif name.endswith(".html"):
            document = html.fromstring(content)
            for element in document.xpath("//script|//style|//nav|//footer"):
                element.drop_tree()
            path.with_suffix(".txt").write_text(document.text_content())
        result.update(status="retrieved", bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    except Exception as error:
        result.update(status="not_retrieved", error=str(error))
    return result


if __name__ == "__main__":
    SOURCES.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(retrieve, ITEMS.items()))
    (HERE / "source-retrieval.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
