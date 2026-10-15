from __future__ import annotations

from pathlib import Path


def parse_cell_metadata_file(path: str | Path) -> dict[str, str]:
    """
    解析 demo 中 key: value 形式的元数据文本。
    值可跨行（以 description 为例时通常单行）。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, _, rest = line.partition(":")
        key = k.strip().lower()
        val = rest.strip()
        if key:
            out[key] = val
    return out


def read_title_only(txt_path: str | Path) -> str:
    """仅从 txt 中读取 `title:` 行，与论文中文本模板对应的自然语言描述一致。"""
    meta = parse_cell_metadata_file(txt_path)
    return (meta.get("title") or "").strip()


_STATE_KEYWORDS = [
    ("mitosis", "mitosis state"),
    ("cytokinesis", "cytokinesis state"),
    ("apoptotic", "apoptotic state"),
    ("apoptosis", "apoptotic state"),
    ("interphase", "interphase state"),
    ("proliferat", "proliferating state"),
    ("senescence", "senescent state"),
]


def infer_state_from_text(blob: str) -> str | None:
    low = blob.lower()
    for kw, label in _STATE_KEYWORDS:
        if kw in low:
            return label
    return None


def title_to_mscdm_triplet(title: str) -> tuple[str, str, str]:
    """
    仅用一条 title 字符串映射为 MSCDM 三路条件，与模板
    T=[cell_type], [morphology], [state], [modality/channel], [attribute]
    的常见书写顺序一致：按英文逗号拆成若干子句，首子句偏「主体」，其余合并为「背景」，
    「状态」由全文关键词推断（无则默认 interphase）。
    """
    title = (title or "").strip()
    if not title:
        return "unknown cell", "interphase state", "microscopy"

    parts = [p.strip() for p in title.split(",") if p.strip()]
    if len(parts) == 1:
        subject = parts[0]
        state = infer_state_from_text(title) or "interphase state"
        return subject, state, ""

    subject = parts[0]
    state = infer_state_from_text(title) or "interphase state"
    background = ", ".join(parts[1:])
    return subject, state, background


def triplet_from_txt_path(txt_path: str | Path) -> tuple[str, str, str]:
    """训练/推理入口；优先读取显式三路字段，兼容旧版仅含 title 的文本。"""
    meta = parse_cell_metadata_file(txt_path)
    explicit = (
        (meta.get("subject") or "").strip(),
        (meta.get("state") or "").strip(),
        (meta.get("background") or "").strip(),
    )
    if all(explicit):
        return explicit
    t = (meta.get("title") or "").strip()
    if not t:
        return "unknown cell", "interphase state", "microscopy"
    return title_to_mscdm_triplet(t)
