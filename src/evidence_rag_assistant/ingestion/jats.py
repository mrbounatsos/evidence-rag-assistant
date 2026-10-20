"""Text extraction from JATS full-text XML.

Text within a block element (paragraph, title, caption) is joined without
separators, so inline formatting such as <bold>Dy</bold>ad cannot split
words. Numbered citation markers are dropped; author-year citations are kept
because they are part of the sentence.

Article extraction keeps the title, abstract, body paragraphs and figure
captions. Back matter is skipped, and every skipped element is recorded with
a reason so exclusions can be audited.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

# --- Text cleaning --------------------------------------------------------

# Elements nested inside a block that are not part of its prose.
# Figure and table captions are extracted separately.
_SKIP_INSIDE_TEXT = frozenset({"fig", "table-wrap", "tex-math"})

_NUMERIC_CITATION = re.compile(r"^[\s\d,;\-–—\[\]()]+$")
_EMPTY_BRACKETS = re.compile(r"[\[(][\s,;\-–—]*[\])]")
_REPEATED_SEPARATORS = re.compile(r"([,;])(?:\s*[,;])+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:)\]])")


def local_name(tag: str) -> str:
    """Strip an XML namespace: '{http://...}license' -> 'license'."""
    return tag.rsplit("}", 1)[-1]


def _is_numbered_citation(element: ET.Element) -> bool:
    if local_name(element.tag) != "xref" or element.get("ref-type") != "bibr":
        return False
    return bool(_NUMERIC_CITATION.match("".join(element.itertext())))


def _collect(element: ET.Element, parts: list[str]) -> None:
    if element.text:
        parts.append(element.text)
    for child in element:
        skip = local_name(child.tag) in _SKIP_INSIDE_TEXT or _is_numbered_citation(child)
        if not skip:
            _collect(child, parts)
        # Text after a child is stored as the child's .tail, so it is kept
        # even when the child itself is dropped.
        if child.tail:
            parts.append(child.tail)


def clean_text(text: str) -> str:
    """Collapse whitespace and tidy punctuation left behind by removed citations."""
    text = " ".join(text.split())
    text = _EMPTY_BRACKETS.sub("", text)
    text = _REPEATED_SEPARATORS.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", text)
    return " ".join(text.split())


def block_text(element: ET.Element) -> str:
    """Return the clean text of a block element such as <p>, <title> or <caption>."""
    parts: list[str] = []
    _collect(element, parts)
    return clean_text("".join(parts))


# --- Section selection ----------------------------------------------------

# sec-type values that mark non-content sections (from the corpus survey).
SKIP_SEC_TYPES = frozenset(
    {
        "ref-list",
        "ack",
        "associated-data",
        "supplementary-material",
        "contrib-info",
        "fn-group",
        "COI-statement",
        "glossary",
    }
)

# Normalized titles of non-content sections (from the corpus surveys).
SKIP_TITLES = frozenset(
    {
        "references",
        "acknowledgments",
        "acknowledgements",
        "funding",
        "funding statement",
        "author contributions",
        "authors' contributions",
        "credit authorship contribution statement",
        "contributors",
        "contributor information",
        "data availability",
        "data availability statement",
        "data accessibility",
        "data and materials availability",
        "code availability",
        "conflict of interest",
        "conflicts of interest",
        "conflict of interest declaration",
        "declaration of competing interest",
        "competing interests",
        "disclosure statement",
        "declarations",
        "declaration of ai use",
        "generative ai statement",
        "consent",
        "ethics",
        "ethics approval",
        "animal research statement",
        "publisher's note",
        "abbreviations",
        "footnotes",
        "supplementary material",
        "supplementary materials",
        "supplementary information",
        "supplementary data",
        "supplemental material",
        "supporting information",
        "associated data",
    }
)

# Abstract types that are kept, and the section path they are stored under.
# Untyped abstracts are kept as "Abstract"; all other types (graphical, toc,
# web-summary) are short blurbs or images and are skipped.
_KEPT_ABSTRACT_TYPES = {"primary": "Abstract", "summary": "Summary"}

# Containers whose children are walked as if they were part of the section.
_CONTAINERS = frozenset({"list", "list-item", "disp-quote", "boxed-text", "fig-group"})

# Elements consumed elsewhere (titles become the section path) and not logged.
_IGNORED = frozenset({"title", "label"})

_LEADING_NUMBERING = re.compile(r"^\d+(?:\.\d+)*\.?\s+")


@dataclass(frozen=True)
class Block:
    """One extracted unit of text: the title, a paragraph, or a caption."""

    kind: str  # "title" | "abstract" | "paragraph" | "caption"
    section_path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Skip:
    """One element that was not extracted, and why."""

    section_path: tuple[str, ...]
    element: str
    title: str
    reason: str


def _title_of(element: ET.Element) -> str:
    title = element.find("title")
    return block_text(title) if title is not None else ""


def skip_reason(sec: ET.Element) -> str | None:
    """Return why a <sec> is not content, or None if it should be extracted.

    Signals are checked from most to least reliable: contents, sec-type, title.
    """
    if any(local_name(e.tag) == "ref-list" for e in sec.iter()):
        return "contains_ref_list"
    content = [c for c in sec if local_name(c.tag) not in _IGNORED]
    if content and all(local_name(c.tag) == "supplementary-material" for c in content):
        return "only_supplementary_material"
    sec_type = sec.get("sec-type")
    if sec_type in SKIP_SEC_TYPES:
        return f"sec_type:{sec_type}"
    title = normalize_title(_title_of(sec))
    if title in SKIP_TITLES:
        return f"title:{title}"
    return None


def _caption_body(caption: ET.Element) -> str:
    parts = [block_text(c) for c in caption if local_name(c.tag) in {"title", "p"}]
    return " ".join(p for p in parts if p)


def caption_text(fig: ET.Element) -> str:
    """Return a figure's label and caption, e.g. 'Figure 1: Overview. Details.'"""
    label_element = fig.find("label")
    label = block_text(label_element) if label_element is not None else ""
    caption = fig.find("caption")
    body = _caption_body(caption) if caption is not None else ""
    if label and body:
        return f"{label}: {body}"
    return label or body


def _add(blocks: list[Block], kind: str, path: tuple[str, ...], text: str) -> None:
    if text:
        blocks.append(Block(kind, path, text))


def _walk(
    container: ET.Element,
    path: tuple[str, ...],
    blocks: list[Block],
    skips: list[Skip],
    paragraph_kind: str = "paragraph",
) -> None:
    for child in container:
        name = local_name(child.tag)
        if name == "p":
            _add(blocks, paragraph_kind, path, block_text(child))
            for nested in child.iter():
                if local_name(nested.tag) == "fig":
                    _add(blocks, "caption", path, caption_text(nested))
        elif name == "sec":
            title = _title_of(child)
            reason = skip_reason(child)
            if reason:
                skips.append(Skip(path, name, title, reason))
            else:
                child_path = (*path, title) if title else path
                _walk(child, child_path, blocks, skips, paragraph_kind)
        elif name == "fig":
            _add(blocks, "caption", path, caption_text(child))
        elif name in _CONTAINERS:
            _walk(child, path, blocks, skips, paragraph_kind)
        elif name == "caption":
            # A caption outside a figure, e.g. the heading of a <boxed-text>.
            _add(blocks, "caption", path, _caption_body(child))
        elif name == "disp-formula":
            skips.append(Skip(path, name, "", "formula_not_extracted"))
        elif name == "table-wrap":
            skips.append(Skip(path, name, _title_of(child), "table_not_extracted"))
        elif name not in _IGNORED:
            skips.append(Skip(path, name, "", "unhandled_element"))


def extract_article(root: ET.Element) -> tuple[list[Block], list[Skip]]:
    """Extract title, abstract, body paragraphs and figure captions from a JATS article."""
    blocks: list[Block] = []
    skips: list[Skip] = []

    meta = root.find("front/article-meta")
    if meta is not None:
        title = meta.find("title-group/article-title")
        if title is not None:
            _add(blocks, "title", (), block_text(title))
        for abstract in meta.findall("abstract"):
            abstract_type = abstract.get("abstract-type")
            if abstract_type is None:
                section = "Abstract"
            else:
                section = _KEPT_ABSTRACT_TYPES.get(abstract_type, "")
            if section:
                _walk(abstract, (section,), blocks, skips, paragraph_kind="abstract")
            else:
                skips.append(Skip((), "abstract", abstract_type or "", "typed_abstract"))

    body = root.find("body")
    if body is not None:
        _walk(body, (), blocks, skips)
    else:
        skips.append(Skip((), "body", "", "no_body"))

    back = root.find("back")
    if back is not None:
        for child in back:
            skips.append(Skip((), local_name(child.tag), _title_of(child), "in_back"))

    return blocks, skips


def normalize_title(title: str) -> str:
    """Normalize a section title for matching: '2.1 Funding:' -> 'funding'."""
    text = " ".join(title.lower().replace("’", "'").split())
    text = _LEADING_NUMBERING.sub("", text)
    return text.rstrip(" .:")
