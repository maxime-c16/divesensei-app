from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from divesensei.workflows.export_rough_clips import (
    CANDIDATE_SOURCES,
    DEFAULT_POST_ROLL,
    DEFAULT_PRE_ROLL,
    MEDIA_MODES,
    build_plan,
    ensure_run_dir,
    ensure_safe_output_root,
    optional_context,
    sanitize_slug,
    timestamp_for_row,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/session_indexes/snmt")
ALLOWED_OUTPUT_ROOT = Path("outputs/session_indexes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei export-session-index",
        description="Export a metadata-only full-session marker/index package for an evaluation session.",
    )
    parser.add_argument("evaluation_root", help="Evaluation session output directory")
    parser.add_argument("--media", choices=sorted(MEDIA_MODES), default="auto")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--pre-roll", type=float, default=DEFAULT_PRE_ROLL)
    parser.add_argument("--post-roll", type=float, default=DEFAULT_POST_ROLL)
    parser.add_argument("--candidate-source", choices=sorted(CANDIDATE_SOURCES), default="auto")
    parser.add_argument("--rough-clip-manifest", default="")
    parser.add_argument("--include-debug-fields", action="store_true")
    parser.add_argument("--html", action="store_true", help="Also write a static local index.html next to marker files")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true", help="Metadata-only; this is the default behavior")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing output run directory")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hhmmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def safe_session_output_root(output_root: Path) -> Path:
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_root = output_root.resolve()
    allowed = (Path.cwd() / ALLOWED_OUTPUT_ROOT).resolve()
    try:
        output_root.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Output root must be under {allowed}") from exc
    return output_root


def load_rough_clip_links(path_text: str) -> tuple[dict[str, str], dict[str, Any]]:
    if not path_text:
        return {}, {"provided": False, "matched_count": 0, "unmatched_count": 0, "manifest_path": None}
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise ValueError(f"Rough clip manifest not found: {path}")
    data = read_json(path)
    rows = data.get("manifest_rows") or data.get("planned_clips") or []
    links: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or row.get("source_candidate_id") or "")
        output_path = row.get("output_path")
        if candidate_id and output_path:
            links.setdefault(candidate_id, str(output_path))
    return links, {"provided": True, "manifest_path": str(path), "manifest_rows": len(rows), "linkable_rows": len(links)}


def marker_optional_fields(context: dict[str, Any], include_debug: bool) -> dict[str, Any]:
    out = {
        "event_label": context.get("event_label") or context.get("suggested_event_label"),
        "anchor_quality": context.get("anchor_quality") if context.get("anchor_quality") is not None else context.get("anchorQuality"),
        "candidate_usefulness": context.get("candidate_usefulness") or context.get("usefulness"),
        "clip_quality": context.get("clip_quality"),
        "timing_group": None,
        "thumbnail_path": None,
        "notes": None,
    }
    if include_debug:
        out["candidate_quality_shadow_debug"] = {
            key: value for key, value in context.items() if key.startswith("candidate_quality_shadow") or key == "review_relevance_score"
        } or None
        out["timing_semantics_shadow_debug"] = context.get("timing_semantics_shadow")
    return out


def build_markers(args: argparse.Namespace) -> dict[str, Any]:
    rough_args = argparse.Namespace(
        evaluation_root=args.evaluation_root,
        media=args.media,
        output_root="outputs/clip_exports/snmt_rough",
        pre_roll=args.pre_roll,
        post_roll=args.post_roll,
        min_duration=1.0,
        max_duration=12.0,
        candidate_source=args.candidate_source,
        limit=None,
        run_id=args.run_id or "session-index-planning",
        dry_run=True,
        execute=False,
        force=True,
        fail_fast=False,
        json=False,
    )
    plan = build_plan(rough_args)
    output_root = safe_session_output_root(Path(args.output_root))
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / plan["session_slug"] / run_id
    rough_links, link_summary = load_rough_clip_links(args.rough_clip_manifest)
    markers: list[dict[str, Any]] = []
    failed = list(plan["failed_clips"])
    for row in plan["planned_clips"]:
        candidate_id = str(row["candidate_id"])
        rough_clip_path = rough_links.get(candidate_id)
        context = row.get("optional_context") or {}
        marker = {
            "marker_index": len(markers) + 1,
            "session_id": Path(plan["evaluation_root"]).name,
            "session_slug": plan["session_slug"],
            "candidate_id": candidate_id,
            "timestamp_seconds": row.get("timestamp_seconds"),
            "timestamp_hhmmss": hhmmss(float(row.get("timestamp_seconds") or row["clip_start_seconds"])),
            "marker_start_seconds": row["clip_start_seconds"],
            "marker_end_seconds": row["clip_end_seconds"],
            "source_candidate_path": row.get("candidate_source_artifact"),
            "source_media_path": plan["media_path"],
            "media_mode": plan["media_mode_selected"],
            "rough_status": "rough_unreviewed",
            "rough_clip_path": rough_clip_path,
            **marker_optional_fields(context, bool(args.include_debug_fields)),
        }
        markers.append(marker)
    markers.sort(key=lambda item: (float(item["timestamp_seconds"] or 0.0), int(item["marker_index"])))
    for index, marker in enumerate(markers, start=1):
        marker["marker_index"] = index
    link_summary["matched_count"] = sum(1 for marker in markers if marker.get("rough_clip_path"))
    link_summary["unmatched_count"] = len(markers) - int(link_summary.get("matched_count") or 0)
    return {
        "schema_version": "s005_session_index_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dry_run": True,
        "evaluation_root": plan["evaluation_root"],
        "session_id": Path(plan["evaluation_root"]).name,
        "session_slug": plan["session_slug"],
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "media_path": plan["media_path"],
        "media_mode_requested": args.media,
        "media_mode_selected": plan["media_mode_selected"],
        "candidate_source": plan["candidate_source"],
        "candidate_source_artifact": plan["candidate_source_artifact"],
        "candidate_rows_loaded": plan["candidate_rows_loaded"],
        "marker_count": len(markers),
        "failed_marker_count": len(failed),
        "markers": markers,
        "failed_markers": failed,
        "rough_clip_linking": link_summary,
        "chronological_order_valid": all(
            float(markers[i]["timestamp_seconds"] or 0.0) <= float(markers[i + 1]["timestamp_seconds"] or 0.0)
            for i in range(max(0, len(markers) - 1))
        ),
    }


CSV_FIELDS = [
    "marker_index",
    "session_id",
    "session_slug",
    "candidate_id",
    "timestamp_seconds",
    "timestamp_hhmmss",
    "marker_start_seconds",
    "marker_end_seconds",
    "source_candidate_path",
    "source_media_path",
    "media_mode",
    "rough_status",
    "rough_clip_path",
    "event_label",
    "anchor_quality",
    "candidate_usefulness",
    "clip_quality",
    "timing_group",
    "thumbnail_path",
    "notes",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})


def write_session_index(path: Path, package: dict[str, Any]) -> None:
    lines = [
        "# Session Index",
        "",
        f"Session: `{package['session_id']}`",
        f"Media: `{package['media_path']}`",
        f"Candidate source: `{package['candidate_source']}`",
        f"Markers: {package['marker_count']}",
        f"Rough status: `rough_unreviewed`",
        "",
        "| # | Time | Candidate | Rough Clip |",
        "|---:|---|---|---|",
    ]
    for marker in package["markers"]:
        clip = marker.get("rough_clip_path") or ""
        lines.append(f"| {marker['marker_index']} | {marker['timestamp_hhmmss']} | `{marker['candidate_id']}` | `{clip}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, package: dict[str, Any]) -> None:
    lines = [
        "# Session Index Export Summary",
        "",
        f"Run id: `{package['run_id']}`",
        f"Session: `{package['session_id']}`",
        f"Markers: {package['marker_count']}",
        f"Failed/skipped markers: {package['failed_marker_count']}",
        f"Chronological order valid: {package['chronological_order_valid']}",
        f"Rough clip manifest provided: {package['rough_clip_linking'].get('provided')}",
        f"Rough clip matches: {package['rough_clip_linking'].get('matched_count', 0)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def html_attr(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def html_text(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=False)


def friendly_time(seconds: Any) -> str:
    try:
        total = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        total = 0
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"{hours} h {minutes:02d} min {secs:02d} s"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def friendly_status(marker: dict[str, Any]) -> str:
    if marker.get("rough_clip_path"):
        return "Clip rapide - À vérifier par l’entraîneur"
    return "Repère non vérifié"


def link_for_local_path(path_text: str | None, run_dir: Path) -> dict[str, Any]:
    if not path_text:
        return {"href": None, "display": "", "exists": False, "relative": False}
    path = Path(path_text)
    exists = path.exists()
    if not path.is_absolute():
        return {"href": path.as_posix(), "display": path.as_posix(), "exists": exists, "relative": True}
    cwd = Path.cwd().resolve()
    resolved_run_dir = run_dir.resolve()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    try:
        resolved.relative_to(cwd)
        return {
            "href": Path(os.path.relpath(resolved, resolved_run_dir)).as_posix(),
            "display": str(resolved),
            "exists": exists,
            "relative": True,
        }
    except ValueError:
        return {"href": None, "display": str(resolved), "exists": exists, "relative": False}


def marker_search_text(marker: dict[str, Any]) -> str:
    parts = [
        marker.get("candidate_id"),
        marker.get("event_label"),
        marker.get("anchor_quality"),
        marker.get("candidate_usefulness"),
        marker.get("clip_quality"),
        friendly_status(marker),
    ]
    return " ".join(str(part) for part in parts if part is not None).lower()


def write_html_index(path: Path, package: dict[str, Any]) -> None:
    run_dir = path.parent
    source_link = link_for_local_path(package.get("media_path"), run_dir)
    linked_count = int(package["rough_clip_linking"].get("matched_count") or 0)
    source_href = source_link.get("href")
    embed_video = bool(source_href and source_link.get("exists"))
    video_markup = ""
    if embed_video:
        video_markup = (
            '<video id="session-video" class="session-video" controls preload="metadata" '
            f'src="{html_attr(source_href)}"></video>'
        )
    else:
        video_markup = (
            '<div class="video-fallback">La Vidéo complète est indiquée comme chemin local. '
            "Si le lien ne marche pas, copiez la Vidéo complète manuellement et utilisez les boutons Copier le temps.</div>"
        )

    rows: list[str] = []
    for marker in package["markers"]:
        clip_info = link_for_local_path(marker.get("rough_clip_path"), run_dir)
        clip_link = ""
        if clip_info.get("href"):
            clip_link = f'<a class="row-link" href="{html_attr(clip_info["href"])}" target="_blank" rel="noreferrer">Ouvrir le clip</a>'
        else:
            clip_link = '<span class="muted">Pas de clip rapide</span>'
        detail_bits = [
            marker.get("event_label"),
            marker.get("anchor_quality"),
            marker.get("candidate_usefulness"),
            marker.get("clip_quality"),
        ]
        details = " / ".join(str(bit) for bit in detail_bits if bit)
        if details:
            details = f'<span class="details">{html_text(details)}</span>'
        jump_time = friendly_time(marker["timestamp_seconds"])
        window_text = f"{friendly_time(marker['marker_start_seconds'])} -> {friendly_time(marker['marker_end_seconds'])}"
        display_id = f"Repère {int(marker['marker_index']):03d}"
        rows.append(
            '<tr class="marker-row" '
            f'data-search="{html_attr(marker_search_text(marker))}" '
            f'data-linked="{str(bool(marker.get("rough_clip_path"))).lower()}">'
            f'<td class="index">{int(marker["marker_index"])}</td>'
            f'<td><button class="time-button" type="button" data-seconds="{html_attr(marker["timestamp_seconds"])}" '
            f'data-time="{html_attr(jump_time)}"><strong>Temps de saut : {html_text(jump_time)}</strong></button></td>'
            f'<td><strong>{html_text(display_id)}</strong><span class="details">Référence technique : {html_text(marker["candidate_id"])}</span>{details}</td>'
            f'<td><span class="status">{html_text(friendly_status(marker))}</span></td>'
            f'<td class="window"><strong>Fenêtre du clip :</strong> {html_text(window_text)}</td>'
            '<td class="actions">'
            f'<button type="button" data-action="jump" data-seconds="{html_attr(marker["timestamp_seconds"])}" '
            f'data-time="{html_attr(jump_time)}">Aller au temps</button>'
            f'<button type="button" data-action="copy" data-seconds="{html_attr(marker["timestamp_seconds"])}" '
            f'data-time="{html_attr(jump_time)}">Copier le temps</button>'
            f"{clip_link}</td>"
            "</tr>"
        )

    source_reference = html_text(source_link.get("display") or package.get("media_path") or "")
    source_anchor = (
        f'<a href="{html_attr(source_href)}">{source_reference}</a>'
        if source_href
        else f"<span>{source_reference}</span>"
    )
    html_doc = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_text(package["session_slug"])} - Index de la séance</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --surface: #ffffff;
      --text: #1f2328;
      --muted: #666d76;
      --border: #d8d7d2;
      --accent: #116329;
      --accent-bg: #dafbe1;
      --button: #24292f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
      font-size: 15px;
      line-height: 1.45;
    }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 24px; }}
    header {{ border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 20px; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; font-weight: 650; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px 18px; color: var(--muted); }}
    .media {{ background: var(--surface); border: 1px solid var(--border); padding: 16px; margin-bottom: 16px; }}
    .media-path {{ word-break: break-all; margin: 8px 0 0; }}
    .session-video {{ width: 100%; max-height: 520px; background: #111; margin-top: 12px; }}
    .video-fallback {{ margin-top: 12px; color: var(--muted); }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 16px 0; }}
    input[type="search"] {{
      width: min(420px, 100%);
      padding: 8px 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font: inherit;
      background: var(--surface);
    }}
    label {{ color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--surface); border: 1px solid var(--border); }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }}
    th {{ font-size: 13px; color: var(--muted); font-weight: 650; background: #fbfbfa; }}
    tr[hidden] {{ display: none; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }}
    button, .row-link {{
      border: 1px solid var(--border);
      background: #fff;
      color: var(--button);
      border-radius: 6px;
      padding: 5px 8px;
      font: inherit;
      font-size: 13px;
      text-decoration: none;
      cursor: pointer;
      display: inline-block;
    }}
    button:hover, .row-link:hover {{ border-color: #8c959f; }}
    .time-button {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .actions {{ white-space: nowrap; display: flex; gap: 6px; flex-wrap: wrap; }}
    .index, .window {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
    .status {{ background: var(--accent-bg); color: var(--accent); border-radius: 6px; padding: 2px 6px; font-size: 13px; }}
    .details {{ display: block; color: var(--muted); font-size: 13px; margin-top: 2px; }}
    .muted {{ color: var(--muted); }}
    #copy-status {{ color: var(--muted); min-height: 20px; }}
    @media (max-width: 760px) {{
      main {{ padding: 14px; }}
      table, thead, tbody, tr, th, td {{ display: block; }}
      thead {{ display: none; }}
      tr {{ border-bottom: 1px solid var(--border); padding: 8px; }}
      td {{ border: 0; padding: 5px 0; }}
      .actions {{ white-space: normal; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{html_text(package["session_slug"])}</h1>
      <div class="meta">
        <span>{int(package["marker_count"])} repères dans l'Index de la séance</span>
        <span>{linked_count} Clips rapides liés</span>
        <span>Clip non vérifié</span>
      </div>
    </header>
    <section class="media" aria-labelledby="media-heading">
      <h2 id="media-heading">Vidéo complète</h2>
      <div class="media-path">{source_anchor}</div>
      {video_markup}
    </section>
    <section class="toolbar" aria-label="Marker controls">
      <input id="marker-search" type="search" placeholder="Rechercher un repère, un temps ou un libellé" autocomplete="off">
      <label><input id="linked-only" type="checkbox" {"disabled" if linked_count == 0 else ""}> Afficher seulement les clips rapides</label>
      <span id="copy-status" aria-live="polite"></span>
    </section>
    <table>
      <thead>
        <tr><th>#</th><th>Temps de saut</th><th>Repère</th><th>Statut</th><th>Fenêtre du clip</th><th>Actions</th></tr>
      </thead>
      <tbody id="marker-body">
        {''.join(rows)}
      </tbody>
    </table>
  </main>
  <script>
    const video = document.getElementById('session-video');
    const rows = Array.from(document.querySelectorAll('.marker-row'));
    const search = document.getElementById('marker-search');
    const linkedOnly = document.getElementById('linked-only');
    const status = document.getElementById('copy-status');

    function copyTimestamp(time, seconds) {{
      const text = `Temps de saut : ${{time}} (${{seconds}} s)`;
      if (navigator.clipboard && navigator.clipboard.writeText) {{
        navigator.clipboard.writeText(text).then(() => {{
          status.textContent = `Temps copié : ${{text}}`;
        }}).catch(() => {{
          status.textContent = text;
        }});
      }} else {{
        status.textContent = text;
      }}
    }}

    function jumpTo(seconds, time) {{
      if (video) {{
        video.currentTime = Number(seconds);
        video.focus();
        video.play().catch(() => undefined);
        status.textContent = `Position video : ${{time}}`;
      }} else {{
        copyTimestamp(time, seconds);
      }}
    }}

    function applyFilters() {{
      const query = search.value.trim().toLowerCase();
      const onlyLinked = linkedOnly && linkedOnly.checked;
      for (const row of rows) {{
        const matchesQuery = !query || row.dataset.search.includes(query);
        const matchesLink = !onlyLinked || row.dataset.linked === 'true';
        row.hidden = !(matchesQuery && matchesLink);
      }}
    }}

    document.addEventListener('click', (event) => {{
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      if (target.matches('.time-button')) {{
        jumpTo(target.dataset.seconds, target.dataset.time);
      }}
      if (target.dataset.action === 'jump') {{
        jumpTo(target.dataset.seconds, target.dataset.time);
      }}
      if (target.dataset.action === 'copy') {{
        copyTimestamp(target.dataset.time, target.dataset.seconds);
      }}
    }});
    search.addEventListener('input', applyFilters);
    if (linkedOnly) linkedOnly.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")


def write_sharing_readme(path: Path, package: dict[str, Any]) -> None:
    linked_count = int(package["rough_clip_linking"].get("matched_count") or 0)
    marker_count = int(package["marker_count"])
    media_path = package.get("media_path") or ""
    lines = [
        "# Guide de partage - séance SNMT",
        "",
        "Ce fichier fonctionne en local. Il sert à expliquer quoi ouvrir et quoi envoyer aux plongeurs.",
        "",
        "## Quel fichier ouvrir en premier ?",
        "",
        "Ouvrez `index.html`. C'est l'Index de la séance : il affiche les repères dans l'ordre de la Vidéo complète.",
        "",
        "## À partager",
        "",
        "- `index.html` : Index de la séance.",
        "- `README_PARTAGE.md` ou `LISEZ_MOI.html` : ce guide.",
        "- `markers.csv` : liste simple des temps, utile si quelqu'un préfère un tableur.",
        "- Le dossier `clips/` si vous voulez envoyer les Clips rapides.",
        "- La Vidéo complète si vous voulez permettre aux plongeurs de tout revoir.",
        "",
        "Important : la Vidéo complète n'est pas copiée automatiquement. Le chemin affiché dans l'index peut être local à l'ordinateur du club.",
        "",
        "## Vidéo complète",
        "",
        f"Chemin indiqué : `{media_path}`",
        "",
        "Si le lien vidéo ne marche pas, copiez la Vidéo complète manuellement avec le dossier partagé, puis utilisez les Temps de saut dans `index.html`.",
        "",
        "## Clips rapides",
        "",
        f"Clips rapides liés dans l'index : {linked_count}.",
        f"Repères dans l'Index de la séance : {marker_count}.",
        "",
        "Tous les repères n'ont pas forcément un clip rapide. C'est normal : l'index peut aussi servir à naviguer dans la Vidéo complète.",
        "",
        "## Que signifie Clip non vérifié ?",
        "",
        "Clip non vérifié signifie : clip rapide créé automatiquement pour gagner du temps. Il est À vérifier par l’entraîneur.",
        "",
        "Cela ne veut pas dire que le plongeon est confirmé, que le cadrage est parfait, ni que le système connaît le plongeur.",
        "",
        "## Comment utiliser l'index",
        "",
        "- Utilisez Temps de saut pour aller au bon moment dans la Vidéo complète.",
        "- Utilisez Fenêtre du clip pour comprendre le début et la fin approximatifs du clip.",
        "- Utilisez Copier le temps pour envoyer un temps à quelqu'un.",
        "- Utilisez Ouvrir le clip quand un Clip rapide existe.",
        "- Utilisez la recherche pour trouver un repère ou un libellé.",
        "",
        "## Fichiers techniques à ignorer",
        "",
        "Les fichiers `markers.json`, `EXPORT_SUMMARY.md` et certains identifiants techniques servent au suivi interne. Ils ne sont pas nécessaires pour les plongeurs.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme_html(path: Path, package: dict[str, Any]) -> None:
    linked_count = int(package["rough_clip_linking"].get("matched_count") or 0)
    marker_count = int(package["marker_count"])
    media_path = html_text(package.get("media_path") or "")
    html_doc = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lisez-moi - partage séance SNMT</title>
  <style>
    body {{ margin: 0; background: #f7f7f4; color: #1f2328; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; line-height: 1.5; }}
    main {{ max-width: 860px; margin: 0 auto; padding: 24px; }}
    section {{ background: #fff; border: 1px solid #d8d7d2; padding: 16px; margin: 14px 0; }}
    h1 {{ font-size: 24px; margin: 0 0 12px; }}
    h2 {{ font-size: 17px; margin: 0 0 8px; }}
    a {{ color: #116329; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; }}
    li {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <main>
    <h1>Guide de partage - séance SNMT</h1>
    <p>Ce fichier fonctionne en local. Il explique quoi ouvrir et quoi envoyer aux plongeurs.</p>
    <section>
      <h2>Quel fichier ouvrir en premier ?</h2>
      <p>Ouvrez <a href="index.html">index.html</a>. C'est l'Index de la séance.</p>
    </section>
    <section>
      <h2>À partager</h2>
      <ul>
        <li><code>index.html</code> : Index de la séance.</li>
        <li><code>README_PARTAGE.md</code> ou <code>LISEZ_MOI.html</code> : guide de partage.</li>
        <li><code>markers.csv</code> : liste des Temps de saut.</li>
        <li>Le dossier <code>clips/</code> si vous voulez envoyer les Clips rapides.</li>
        <li>La Vidéo complète si vous voulez permettre aux plongeurs de tout revoir.</li>
      </ul>
    </section>
    <section>
      <h2>Vidéo complète</h2>
      <p>Chemin indiqué : <code>{media_path}</code></p>
      <p>Si le lien ne marche pas, copiez la Vidéo complète manuellement et utilisez les Temps de saut.</p>
      <p>La Fenêtre du clip indique le début et la fin approximatifs autour du repère.</p>
    </section>
    <section>
      <h2>Clips rapides</h2>
      <p>{linked_count} Clips rapides sont liés dans l'index. L'Index de la séance contient {marker_count} repères.</p>
      <p>Tous les repères n'ont pas forcément un clip rapide.</p>
    </section>
    <section>
      <h2>Clip non vérifié</h2>
      <p>Un Clip non vérifié est un clip rapide créé automatiquement. Il est À vérifier par l’entraîneur.</p>
      <p>Il ne confirme pas la qualité du plongeon et ne remplace pas l'avis de l'entraîneur.</p>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")


def write_outputs(package: dict[str, Any], *, force: bool, include_html: bool = False) -> None:
    run_dir = Path(package["run_dir"])
    allowed = (Path.cwd() / ALLOWED_OUTPUT_ROOT).resolve()
    try:
        run_dir.resolve().relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Run directory must be under {allowed}") from exc
    if run_dir.exists():
        if not force:
            raise ValueError(f"Output run directory already exists; use --force to replace: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "markers.json", package)
    write_csv(run_dir / "markers.csv", package["markers"])
    write_session_index(run_dir / "SESSION_INDEX.md", package)
    write_summary(run_dir / "EXPORT_SUMMARY.md", package)
    if package["failed_markers"]:
        write_json(run_dir / "failed_markers.json", package["failed_markers"])
    if include_html:
        write_html_index(run_dir / "index.html", package)
        write_sharing_readme(run_dir / "README_PARTAGE.md", package)
        write_readme_html(run_dir / "LISEZ_MOI.html", package)


def export_session_index(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.pre_roll < 0 or args.post_roll <= 0:
        raise ValueError("--pre-roll must be >= 0 and --post-roll must be > 0.")
    package = build_markers(args)
    if not package["chronological_order_valid"]:
        raise ValueError("Chronological marker order validation failed.")
    write_outputs(package, force=bool(args.force), include_html=bool(args.html))
    return package


def main(argv: Sequence[str] | None = None) -> int:
    try:
        package = export_session_index(argv)
    except Exception as exc:
        print(f"export-session-index failed: {exc}", file=sys.stderr)
        return 2
    summary = {
        "run_dir": package["run_dir"],
        "marker_count": package["marker_count"],
        "failed_marker_count": package["failed_marker_count"],
        "chronological_order_valid": package["chronological_order_valid"],
        "rough_clip_matches": package["rough_clip_linking"].get("matched_count", 0),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
