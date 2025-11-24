#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from datetime import datetime


TOKEN_HEADER_RE = re.compile(r"^\*\*([IVXLCDM]+)\.\s+([^*]+)\*\*\s+—\s+\*([^*]+)\*\s*$")
PATHWAY_HEADER_RE = re.compile(r"^###\s+Pathway\s+(\d+):\s*(.*)$", re.IGNORECASE)
ROMAN_LINE_RE = re.compile(r"^([IVXLCDM]+)\.\s+([^()]+?)(?:\s*\(([^)]+)\))?\s*$")
BULLET_ITEM_RE = re.compile(r"^\-\s*\*\*([IVXLCDM]+)\.\s*([^*]+)\*\*\s*\(([^)]+)\)")


def normalize_name(s: str) -> str:
    return (
        re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", re.sub(r"\(.*?\)", "", s.lower()))).strip()
    )


def parse_token_index(md_text: str):
    tokens = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        m = TOKEN_HEADER_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        roman, name, epithet = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        rec = {
            "id": roman,
            "name": name,
            "epithet": epithet,
            "phase": None,
            "function": None,
            "modulation": None,
            "wumbo_roles": [],
        }
        i += 1
        # consume until blank line
        while i < len(lines) and lines[i].strip() != "":
            L = lines[i].strip()
            if L.lower().startswith("phase:"):
                rec["phase"] = L.split(":", 1)[1].strip()
            elif L.lower().startswith("function:"):
                rec["function"] = L.split(":", 1)[1].strip()
            elif L.lower().startswith("modulation:"):
                rec["modulation"] = L.split(":", 1)[1].strip()
            elif re.match(r"^wumbo\s*role(s)?\s*:\s*", L, re.IGNORECASE):
                raw = L.split(":", 1)[1].strip()
                parts = [p.strip() for p in re.split(r"[;,/|•]+", raw) if p.strip()]
                rec["wumbo_roles"] = sorted(set(parts))
            i += 1
        tokens.append(rec)
        # skip blank line
        while i < len(lines) and lines[i].strip() == "":
            i += 1
    return tokens


def parse_matrix(md_text: str):
    pathways = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        mh = PATHWAY_HEADER_RE.match(lines[i].strip())
        if not mh:
            i += 1
            continue
        number = int(mh.group(1))
        title = mh.group(2).strip()
        i += 1
        start = i
        # collect until next pathway header or EOF
        section_lines = []
        while i < len(lines) and not PATHWAY_HEADER_RE.match(lines[i]):
            section_lines.append(lines[i])
            i += 1
        text = "\n".join(section_lines)
        nodes = []
        modulators = []
        bottlenecks = []
        # Parse roman lines (often inside code blocks)
        for L in section_lines:
            m = ROMAN_LINE_RE.match(L.strip())
            if m:
                nodes.append({
                    "roman": m.group(1),
                    "name": m.group(2).strip(),
                    "note": (m.group(3) or "").strip(),
                })
        # Track context for bullets
        current = None
        for L in section_lines:
            s = L.strip()
            if re.search(r"\*\*Modulation Nodes\*\*", s, re.IGNORECASE):
                current = "mod"
                continue
            if re.search(r"\*\*Critical Bottlenecks\*\*", s, re.IGNORECASE):
                current = "bot"
                continue
            bm = BULLET_ITEM_RE.match(s)
            if bm and current:
                item = {"roman": bm.group(1), "name": bm.group(2).strip(), "note": bm.group(3).strip()}
                if current == "mod":
                    modulators.append(item)
                elif current == "bot":
                    bottlenecks.append(item)
        pathways.append({
            "number": number,
            "title": title,
            "nodes": nodes,
            "modulators": modulators,
            "bottlenecks": bottlenecks,
            "raw": text,
        })
    pathways.sort(key=lambda p: p["number"])
    return pathways


def cross_reference(tokens, pathways):
    def escape_re(s: str) -> str:
        return re.escape(s)

    token_to_paths_scored = {}
    path_to_tokens_scored = {p["number"]: [] for p in pathways}

    # Build name index for pathways
    path_name_index = []
    for p in pathways:
        names = []
        for n in (p.get("nodes") or []):
            names.append({"kind": "node", "n": normalize_name(n.get("name", "")), "raw": n.get("name", ""), "roman": n.get("roman", "")})
        for n in (p.get("modulators") or []):
            names.append({"kind": "mod", "n": normalize_name(n.get("name", "")), "raw": n.get("name", ""), "roman": n.get("roman", "")})
        for n in (p.get("bottlenecks") or []):
            names.append({"kind": "bot", "n": normalize_name(n.get("name", "")), "raw": n.get("name", ""), "roman": n.get("roman", "")})
        path_name_index.append({"p": p, "names": names})

    for t in tokens:
        key = normalize_name(t.get("name", ""))
        epi = normalize_name(t.get("epithet", ""))
        results = []
        for entry in path_name_index:
            p = entry["p"]
            names = entry["names"]
            conf = 0.0
            reasons = []
            # Roman match
            if any((nn.get("roman", "").strip() == t["id"]) for nn in (p.get("nodes") or [])):
                conf = max(conf, 0.95)
                reasons.append(f"roman-match:{t['id']}")
            # Exact name
            if any(obj["n"] == key for obj in names):
                conf = max(conf, 1.0)
                reasons.append("exact-name")
            # Word-boundary partial
            if conf < 1.0 and key:
                pattern = re.compile(rf"\b{escape_re(key)}\b")
                if any(pattern.search(obj["n"]) for obj in names):
                    conf = max(conf, 0.85)
                    reasons.append("name-boundary")
            # Epithet hint
            if conf < 1.0 and epi and any(epi in obj["n"] for obj in names):
                conf = max(conf, 0.8)
                reasons.append("epithet-hint")
            # Raw text mention fallback
            if conf == 0.0 and key and key in (p.get("raw", "").lower()):
                conf = 0.6
                reasons.append("raw-mention")
            if conf > 0:
                results.append({"number": p["number"], "confidence": round(conf, 2), "reasons": reasons})
        results.sort(key=lambda r: (-r["confidence"], r["number"]))
        token_to_paths_scored[t["id"]] = results
        for r in results:
            path_to_tokens_scored.setdefault(r["number"], []).append({"token_id": t["id"], "confidence": r["confidence"]})

    for num, arr in path_to_tokens_scored.items():
        arr.sort(key=lambda a: (-a["confidence"], a["token_id"]))

    # Convenience non-scored
    token_to_paths = {tid: [x["number"] for x in arr] for tid, arr in token_to_paths_scored.items()}
    path_to_tokens = {num: [x["token_id"] for x in arr] for num, arr in path_to_tokens_scored.items()}

    return {
        "token_to_paths": token_to_paths,
        "path_to_tokens": path_to_tokens,
        "token_to_paths_scored": token_to_paths_scored,
        "path_to_tokens_scored": path_to_tokens_scored,
    }


def build_pack(token_md_path: str, matrix_md_path: str, version: str):
    with open(token_md_path, "r", encoding="utf-8") as f:
        tok_text = f.read()
    with open(matrix_md_path, "r", encoding="utf-8") as f:
        mat_text = f.read()

    tokens = parse_token_index(tok_text)
    pathways = parse_matrix(mat_text)
    links = cross_reference(tokens, pathways)
    pack = {
        "meta": {
            "name": "Ace Neural Codex Language Pack",
            "version": version,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "sources": [
                os.path.basename(token_md_path),
                os.path.basename(matrix_md_path),
            ],
        },
        "tokens": tokens,
        "pathways": pathways,
        "links": links,
    }
    # Markdown summary
    lines = []
    lines.append(f"# Language Pack Summary ({version})")
    lines.append("")
    lines.append(f"Generated: {pack['meta']['generated_at']}")
    lines.append("")
    for t in tokens:
        scored = (links["token_to_paths_scored"].get(t["id"], []) or [])[:5]
        label = (
            " (Pathways: "
            + ", ".join(f"{x['number']}({x['confidence']:.2f})" for x in scored)
            + ")"
            if scored
            else ""
        )
        lines.append(f"- {t['id']}. {t['name']} — Phase: {t.get('phase') or ''}{label}")

    return pack, "\n".join(lines) + "\n"


def next_version_dir(base_dir: str, requested: str | None) -> str:
    if requested:
        return os.path.join(base_dir, requested)
    # auto-increment vN
    existing = []
    if os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            if re.match(r"^v\d+$", name):
                existing.append(int(name[1:]))
    v = max(existing) + 1 if existing else 1
    return os.path.join(base_dir, f"v{v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", help="version label (e.g., v1). If omitted, auto-increments.")
    ap.add_argument("--outdir", default="language", help="output base directory under repo root")
    ap.add_argument("--token", default="TOKEN_INDEX.md")
    ap.add_argument("--matrix", default="NEURAL_PATHING_MATRIX.md")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    token_md = os.path.join(repo_root, args.token)
    matrix_md = os.path.join(repo_root, args.matrix)
    out_base = os.path.join(repo_root, args.outdir)
    out_version_dir = next_version_dir(out_base, args.version)
    os.makedirs(out_version_dir, exist_ok=True)

    pack, summary_md = build_pack(token_md, matrix_md, os.path.basename(out_version_dir))

    # write
    pack_path = os.path.join(out_version_dir, "pack.json")
    with open(pack_path, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)
    sum_path = os.path.join(out_version_dir, "summary.md")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    # latest pointers
    latest_json = os.path.join(out_base, "latest.json")
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)
    latest_md = os.path.join(out_base, "latest.md")
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(summary_md)

    # version index
    index_path = os.path.join(out_base, "index.json")
    index = {"versions": []}
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            index = {"versions": []}
    # update index
    versions = {v.get("version"): v for v in index.get("versions", []) if isinstance(v, dict) and v.get("version")}
    versions[pack["meta"]["version"]] = {
        "version": pack["meta"]["version"],
        "generated_at": pack["meta"]["generated_at"],
        "paths": {
            "json": os.path.relpath(pack_path, out_base),
            "summary": os.path.relpath(sum_path, out_base),
        },
    }
    index["versions"] = [versions[k] for k in sorted(versions.keys(), key=lambda x: (x[0] == 'v' and int(x[1:]) or x))]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("Wrote:")
    print("  ", os.path.relpath(pack_path, repo_root))
    print("  ", os.path.relpath(sum_path, repo_root))
    print("  ", os.path.relpath(latest_json, repo_root))
    print("  ", os.path.relpath(index_path, repo_root))


if __name__ == "__main__":
    sys.exit(main())

