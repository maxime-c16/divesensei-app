from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from divesensei.workflows.export_rough_clips import sanitize_slug


DEFAULT_OUTPUT_ROOT = Path("outputs/mobile_share/snmt")
ALLOWED_OUTPUT_ROOT = Path("outputs/mobile_share")

MESSAGE_TEXT = (
    "Voici les clips rapides de la séance SNMT.\n"
    "Ils sont générés automatiquement : si un clip est trop court, mal cadré, "
    "ou s’il manque un passage, demande-moi la vidéo complète.\n"
    "Tu peux ouvrir directement les fichiers MP4 sur ton téléphone.\n"
)

VIDEO_COMPLETE_TEXT = (
    "La vidéo complète n’est pas incluse automatiquement dans ce dossier.\n"
    "L’entraîneur peut l’envoyer séparément si besoin.\n"
    "Les temps indiqués dans la liste des clips peuvent aider à retrouver les passages.\n"
    "Un chemin local de vidéo, s’il existe, sert seulement au coach sur l’ordinateur du club.\n"
)

TECHNICAL_TERMS_FOR_MESSAGE = {
    "candidate",
    "manifest",
    "index",
    "markers",
    "debug",
    "rank",
    "model",
    "json",
    "jsonl",
    "rough_unreviewed",
}

TRACEABILITY_FIELDS = [
    "clip_number",
    "mobile_filename",
    "source_clip_path",
    "original_filename",
    "candidate_id",
    "timestamp_seconds",
    "timestamp_friendly",
    "clip_start_seconds",
    "clip_end_seconds",
    "clip_window_friendly",
    "duration_seconds",
    "file_size_mb",
    "container",
    "video_codec",
    "audio_codec",
    "width",
    "height",
    "fps",
    "mobile_status",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei export-mobile-share",
        description="Create a phone-friendly French share package from an explicit rough clip manifest.",
    )
    parser.add_argument("--rough-manifest", required=True, help="Explicit rough clip manifest.json from export-rough-clips")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing mobile share run directory")
    parser.add_argument("--json", action="store_true", help="Print summary as JSON")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def safe_output_root(raw: str) -> Path:
    output_root = resolve_path(raw)
    allowed = (Path.cwd() / ALLOWED_OUTPUT_ROOT).resolve()
    try:
        output_root.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Output root must be under {allowed}") from exc
    return output_root


def ensure_run_dir(run_dir: Path, *, force: bool) -> None:
    allowed = (Path.cwd() / ALLOWED_OUTPUT_ROOT).resolve()
    try:
        run_dir.resolve().relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Run directory must be under {allowed}") from exc
    if run_dir.exists():
        if not force:
            raise ValueError(f"Output run directory already exists; use --force to replace: {run_dir}")
        shutil.rmtree(run_dir)
    (run_dir / "clips").mkdir(parents=True, exist_ok=False)
    (run_dir / "_technique").mkdir(parents=True, exist_ok=False)


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def friendly_time(seconds: Any) -> str:
    value = finite_float(seconds)
    if value is None:
        return "temps inconnu"
    total = max(0, int(round(value)))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def filename_time(seconds: Any) -> str:
    value = finite_float(seconds)
    if value is None:
        return "temps-inconnu"
    total = max(0, int(round(value)))
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}min{secs:02d}"
    return f"{secs}s"


def fps_from_stream(stream: dict[str, Any]) -> float | None:
    value = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not value or value == "0/0":
        return None
    if isinstance(value, str) and "/" in value:
        num, den = value.split("/", 1)
        try:
            denominator = float(den)
            if denominator == 0:
                return None
            return round(float(num) / denominator, 3)
        except ValueError:
            return None
    return finite_float(value)


def probe_clip(path: Path) -> tuple[dict[str, Any], bool]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}, False
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}, True
    try:
        return json.loads(completed.stdout), True
    except json.JSONDecodeError:
        return {}, True


def file_size_status(size_mb: float) -> str:
    if size_mb < 20:
        return "excellent"
    if size_mb <= 80:
        return "acceptable"
    return "warning"


def codec_status(container: str | None, video_codec: str | None, audio_codec: str | None) -> str:
    container_ok = bool(container and "mp4" in container.lower())
    video_ok = video_codec in {"h264", "avc1"}
    audio_ok = audio_codec in {"aac", None, ""}
    if container_ok and video_ok and audio_ok:
        return "compatible_prefere"
    if container_ok and video_ok:
        return "probablement_compatible"
    return "a_verifier"


def audit_for_file(path: Path) -> dict[str, Any]:
    size_bytes = path.stat().st_size
    size_mb = round(size_bytes / (1024 * 1024), 3)
    probed, ffprobe_available = probe_clip(path)
    fmt = probed.get("format") if isinstance(probed, dict) else {}
    streams = probed.get("streams") if isinstance(probed, dict) else []
    streams = streams if isinstance(streams, list) else []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    container = fmt.get("format_name") if isinstance(fmt, dict) else None
    duration = finite_float(fmt.get("duration") if isinstance(fmt, dict) else None)
    video_codec = video_stream.get("codec_name") if isinstance(video_stream, dict) else None
    audio_codec = audio_stream.get("codec_name") if isinstance(audio_stream, dict) else None
    width = video_stream.get("width") if isinstance(video_stream, dict) else None
    height = video_stream.get("height") if isinstance(video_stream, dict) else None
    fps = fps_from_stream(video_stream) if isinstance(video_stream, dict) else None
    return {
        "ffprobe_available": ffprobe_available,
        "file_size_bytes": size_bytes,
        "file_size_mb": size_mb,
        "duration_seconds_probe": round(duration, 3) if duration is not None else None,
        "container": container,
        "video_codec": video_codec,
        "audio_codec": audio_codec,
        "width": width,
        "height": height,
        "fps": fps,
        "size_status": file_size_status(size_mb),
        "codec_status": codec_status(container, video_codec, audio_codec),
    }


def valid_clip_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("manifest_rows") or manifest.get("planned_clips") or []
    usable: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_path = row.get("output_path")
        if not raw_path:
            continue
        path = resolve_path(str(raw_path))
        if path.suffix.lower() != ".mp4":
            continue
        usable.append({**row, "_resolved_output_path": str(path)})
    usable.sort(
        key=lambda item: (
            finite_float(item.get("timestamp_seconds")) if finite_float(item.get("timestamp_seconds")) is not None else 1e18,
            int(item.get("index") or len(usable) + 1),
        )
    )
    return usable


def write_traceability_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRACEABILITY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in TRACEABILITY_FIELDS})


def write_mobile_readme(path: Path, package: dict[str, Any]) -> None:
    clip_count = int(package["summary"]["clip_count"])
    total_mb = float(package["summary"]["total_size_mb"])
    doc = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Clips rapides SNMT</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #f7f7f4;
      color: #1f2328;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
      line-height: 1.5;
    }}
    main {{
      max-width: 720px;
      margin: 0 auto;
      padding: 20px;
    }}
    h1 {{ font-size: 24px; margin: 0 0 12px; }}
    h2 {{ font-size: 17px; margin: 24px 0 8px; }}
    p {{ margin: 8px 0; }}
    ul {{ padding-left: 20px; }}
    li {{ margin: 7px 0; }}
    .note {{
      border: 1px solid #d8d7d2;
      background: #fff;
      padding: 12px;
      border-radius: 8px;
      margin: 14px 0;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Clips rapides SNMT</h1>
    <p>Ouvrez directement les fichiers MP4 dans le dossier <code>clips</code> sur votre téléphone.</p>
    <div class="note">
      <p><strong>Clip non vérifié.</strong> Ces clips sont générés automatiquement. Ils permettent de revoir vite les passages, mais ils ne remplacent pas la vérification de l’entraîneur.</p>
    </div>
    <h2>Que contient ce dossier ?</h2>
    <ul>
      <li>{clip_count} clips rapides au format MP4.</li>
      <li>Un message court à copier dans WhatsApp, SMS ou email.</li>
      <li>Une note sur la vidéo complète si vous voulez revoir toute la séance.</li>
    </ul>
    <h2>Si un clip ne convient pas</h2>
    <p>Si un clip est trop court, mal cadré, ou s’il manque un passage, demandez la vidéo complète à l’entraîneur.</p>
    <h2>Vidéo complète</h2>
    <p>La vidéo complète n’est pas incluse automatiquement dans ce dossier. Elle peut être envoyée séparément si besoin.</p>
    <h2>Fichiers techniques</h2>
    <p>Le dossier <code>_technique</code> sert seulement au coach pour retrouver l’origine des clips. Vous pouvez l’ignorer.</p>
    <p>Taille totale des clips : {total_mb:.1f} Mo.</p>
  </main>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def write_manifest_source(path: Path, manifest_path: Path, manifest: dict[str, Any]) -> None:
    provenance = {
        "schema_version": "s011_manifest_source_v1",
        "copied_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_path": str(manifest_path),
        "source_manifest": manifest,
    }
    write_json(path, provenance)


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = resolve_path(args.rough_manifest)
    if not manifest_path.exists():
        raise ValueError(f"Rough clip manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    session_slug = sanitize_slug(str(manifest.get("session_slug") or manifest_path.parents[1].name))
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = safe_output_root(args.output_root)
    run_dir = output_root / session_slug / run_id
    ensure_run_dir(run_dir, force=bool(args.force))

    rows = valid_clip_rows(manifest)
    if not rows:
        raise ValueError("No rough MP4 clip output paths found in the explicit manifest.")

    copied_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for number, row in enumerate(rows, start=1):
        source_path = Path(row["_resolved_output_path"])
        if not source_path.exists() or not source_path.is_file():
            failures.append(
                {
                    "clip_number": number,
                    "candidate_id": row.get("candidate_id"),
                    "source_clip_path": str(source_path),
                    "error": "source_clip_missing",
                }
            )
            continue
        timestamp = finite_float(row.get("timestamp_seconds"))
        filename = f"{number:02d}_{filename_time(timestamp)}.mp4"
        destination = (run_dir / "clips" / filename).resolve()
        try:
            destination.relative_to((run_dir / "clips").resolve())
        except ValueError as exc:
            raise ValueError(f"Unsafe destination path for clip {number}: {destination}") from exc
        shutil.copy2(source_path, destination)
        audit = audit_for_file(destination)
        start = row.get("clip_start_seconds")
        end = row.get("clip_end_seconds")
        trace = {
            "clip_number": number,
            "mobile_filename": filename,
            "source_clip_path": str(source_path),
            "original_filename": source_path.name,
            "candidate_id": row.get("candidate_id") or row.get("source_candidate_id"),
            "timestamp_seconds": timestamp,
            "timestamp_friendly": friendly_time(timestamp),
            "clip_start_seconds": finite_float(start),
            "clip_end_seconds": finite_float(end),
            "clip_window_friendly": f"{friendly_time(start)} -> {friendly_time(end)}",
            "duration_seconds": finite_float(row.get("duration_seconds")) or audit.get("duration_seconds_probe"),
            "file_size_mb": audit["file_size_mb"],
            "container": audit.get("container"),
            "video_codec": audit.get("video_codec"),
            "audio_codec": audit.get("audio_codec"),
            "width": audit.get("width"),
            "height": audit.get("height"),
            "fps": audit.get("fps"),
            "mobile_status": audit["size_status"],
        }
        copied_rows.append(
            {
                **trace,
                "mobile_clip_path": str(destination),
                "ffprobe_available": audit["ffprobe_available"],
                "codec_status": audit["codec_status"],
                "file_size_bytes": audit["file_size_bytes"],
            }
        )
        trace_rows.append(trace)

    if failures:
        raise ValueError(f"{len(failures)} source clips are missing; mobile package not complete: {failures[:3]}")

    total_size_mb = round(sum(float(row["file_size_mb"]) for row in copied_rows), 3)
    package_status = "warning" if total_size_mb > 500 else "ok"
    ffprobe_available = all(bool(row["ffprobe_available"]) for row in copied_rows)
    codec_warnings = [row for row in copied_rows if row.get("codec_status") != "compatible_prefere"]
    size_warnings = [row for row in copied_rows if row.get("mobile_status") == "warning"]
    audit_package = {
        "schema_version": "s011_mobile_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ffprobe_available": ffprobe_available,
        "clip_count": len(copied_rows),
        "total_size_mb": total_size_mb,
        "package_status": package_status,
        "package_warning_threshold_mb": 500,
        "per_clip_thresholds_mb": {"excellent_lt": 20, "acceptable_max": 80, "warning_gt": 80},
        "codec_preference": "mp4 container + h264 video + aac audio",
        "codec_warning_count": len(codec_warnings),
        "size_warning_count": len(size_warnings),
        "clips": copied_rows,
    }

    message_path = run_dir / "MESSAGE_A_COPIER.txt"
    video_info_path = run_dir / "VIDEO_COMPLETE_INFO.txt"
    readme_path = run_dir / "README_MOBILE.html"
    technique_dir = run_dir / "_technique"
    message_path.write_text(MESSAGE_TEXT, encoding="utf-8")
    video_info_path.write_text(VIDEO_COMPLETE_TEXT, encoding="utf-8")
    package_summary = {
        "summary": {
            "clip_count": len(copied_rows),
            "total_size_mb": total_size_mb,
        }
    }
    write_mobile_readme(readme_path, package_summary)
    write_traceability_csv(technique_dir / "LISTE_DES_CLIPS.csv", trace_rows)
    write_manifest_source(technique_dir / "manifest_source.json", manifest_path, manifest)
    write_json(technique_dir / "mobile_audit.json", audit_package)

    package = {
        "schema_version": "s011_mobile_share_package_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rough_manifest_path": str(manifest_path),
        "session_slug": session_slug,
        "run_id": run_id,
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "visible_files": {
            "clips_dir": str(run_dir / "clips"),
            "MESSAGE_A_COPIER.txt": str(message_path),
            "VIDEO_COMPLETE_INFO.txt": str(video_info_path),
            "README_MOBILE.html": str(readme_path),
        },
        "technical_files": {
            "LISTE_DES_CLIPS.csv": str(technique_dir / "LISTE_DES_CLIPS.csv"),
            "manifest_source.json": str(technique_dir / "manifest_source.json"),
            "mobile_audit.json": str(technique_dir / "mobile_audit.json"),
        },
        "summary": {
            "source_manifest_rows": len(rows),
            "copied_clip_count": len(copied_rows),
            "total_size_mb": total_size_mb,
            "package_status": package_status,
            "ffprobe_available": ffprobe_available,
            "codec_warning_count": len(codec_warnings),
            "size_warning_count": len(size_warnings),
            "zip_created": False,
            "transcoding_performed": False,
            "new_clips_created": False,
        },
        "clips": copied_rows,
    }
    write_json(technique_dir / "package_summary.json", package)
    return package


def validate_message_text(text: str) -> dict[str, Any]:
    lower = text.lower()
    found = sorted(term for term in TECHNICAL_TERMS_FOR_MESSAGE if term in lower)
    return {"technical_terms_found": found, "technical_terms_absent": not found}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        package = build_package(args)
    except Exception as exc:
        print(f"export-mobile-share failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(package, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(f"Mobile share package: {package['run_dir']}")
        print(f"Copied clips: {package['summary']['copied_clip_count']}")
        print(f"Total size: {package['summary']['total_size_mb']} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
