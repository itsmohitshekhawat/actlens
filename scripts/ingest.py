"""Turn the official EU AI Act (Regulation (EU) 2024/1689, XHTML from the EU Publications Office) into search chunks.

One chunk = one numbered paragraph of an article, one recital, or one part of an annex. Each chunk keeps a citation
label ("Art. 5(1)", "Recital 27", "Annex III") and its place in the law (chapter, section, article title), so the agent
can cite exactly where an answer comes from.

Usage: python scripts/ingest.py  (reads data/raw/ai_act.xhtml, writes data/corpus.json)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import warnings

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "ai_act.xhtml"
OUT = ROOT / "data" / "corpus.json"
SOURCE_URL = "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng"
ANNEX_PART_CHARS = 1800  # long annexes are cut into parts of about this size, at paragraph boundaries


def clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_of(node: Tag) -> str:
    """Text of a node, keeping list markers such as '(a)' next to their item."""
    return clean(node.get_text(" "))


def heading(container: Tag) -> str:
    """'CHAPTER III - HIGH-RISK AI SYSTEMS' from a chapter or section container."""
    label = container.find("p", recursive=False)
    title = container.find("div", class_="eli-title", recursive=False)
    return " - ".join(text_of(x) for x in (label, title) if x)


def chapter_of(article: Tag) -> tuple[str, str]:
    """Chapter and section titles an article sits in (from its parent containers)."""
    chapter = section = ""
    for parent in article.parents:
        pid = parent.get("id", "") if isinstance(parent, Tag) else ""
        if re.fullmatch(r"cpt_[IVXLC]+\.sct_\d+", pid) and not section:
            section = heading(parent)
        elif re.fullmatch(r"cpt_[IVXLC]+", pid) and not chapter:
            chapter = heading(parent)
    return chapter, section


def articles(soup: BeautifulSoup) -> list[dict]:
    chunks = []
    for art in soup.find_all("div", id=re.compile(r"^art_\d+$")):
        number = art["id"].split("_")[1]
        title_div = art.find("div", class_="eli-title")
        title = text_of(title_div) if title_div else ""
        chapter, section = chapter_of(art)
        paragraphs = art.find_all("div", id=re.compile(r"^\d{3}\.\d{3}$"), recursive=False)
        base = {"kind": "article", "article": int(number), "title": title, "chapter": chapter, "section": section}
        tables = art.find_all("table", recursive=False)
        numbered = [t for t in tables if re.match(r"\(\d+\)", text_of(t))]
        if not paragraphs and len(numbered) > 10:  # Article 3: a long list of numbered definitions, one chunk each
            for table in tables:
                point = text_of(table)
                m = re.match(r"\((\d+)\)\s*", point)
                if m:
                    chunks.append({**base, "id": f"art-{number}-{m.group(1)}", "label": f"Art. {number}({m.group(1)})",
                                   "paragraph": int(m.group(1)), "text": point})
            continue
        if not paragraphs:  # short articles have no numbered paragraphs
            body = [text_of(p) for p in art.find_all(["p", "table"], recursive=False)[1:]]
            body = [b for b in body if b and b != title]
            chunks.append({**base, "id": f"art-{number}", "label": f"Art. {number}", "text": " ".join(body)})
            continue
        for div in paragraphs:
            para = int(div["id"].split(".")[1])
            chunks.append({**base, "id": f"art-{number}-{para}", "label": f"Art. {number}({para})", "paragraph": para,
                           "text": text_of(div)})
    return chunks


def recitals(soup: BeautifulSoup) -> list[dict]:
    chunks = []
    for rct in soup.find_all("div", id=re.compile(r"^rct_\d+$")):
        number = int(rct["id"].split("_")[1])
        text = re.sub(r"^\(\d+\)\s*", "", text_of(rct))
        chunks.append({"kind": "recital", "id": f"rct-{number}", "label": f"Recital {number}", "recital": number,
                       "title": "", "text": text})
    return chunks


def annexes(soup: BeautifulSoup) -> list[dict]:
    chunks = []
    for anx in soup.find_all("div", id=re.compile(r"^anx_[IVXLC]+$")):
        roman = anx["id"].split("_")[1]
        heads = anx.find_all("p", class_="oj-doc-ti")
        title = clean(heads[1].get_text(" ")) if len(heads) > 1 else ""
        blocks = []
        for child in anx.children:
            if not isinstance(child, Tag) or "oj-doc-ti" in (child.get("class") or []):
                continue
            t = text_of(child)
            if t:
                blocks.append(t)
        parts, current = [], ""
        for block in blocks:
            if current and len(current) + len(block) > ANNEX_PART_CHARS:
                parts.append(current)
                current = ""
            current = f"{current} {block}".strip()
        if current:
            parts.append(current)
        for i, part in enumerate(parts, 1):
            label = f"Annex {roman}" if len(parts) == 1 else f"Annex {roman} (part {i})"
            chunks.append({"kind": "annex", "id": f"anx-{roman}-{i}", "label": label, "annex": roman, "title": title,
                           "text": part})
    return chunks


def main() -> None:
    soup = BeautifulSoup(RAW.read_text(encoding="utf-8"), "lxml")
    chunks = articles(soup) + annexes(soup) + recitals(soup)
    chunks = [c for c in chunks if len(c["text"]) > 20]
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"
    OUT.write_text(json.dumps({"source": SOURCE_URL, "law": "Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
                               "chunks": chunks}, ensure_ascii=False, indent=1), encoding="utf-8")
    kinds = {k: sum(c["kind"] == k for c in chunks) for k in ("article", "annex", "recital")}
    print(f"{len(chunks)} chunks -> {OUT.relative_to(ROOT)} {kinds}")


if __name__ == "__main__":
    main()
