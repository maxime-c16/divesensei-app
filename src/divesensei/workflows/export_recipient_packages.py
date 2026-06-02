from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from divesensei.workflows.export_rough_clips import sanitize_slug


DEFAULT_OUTPUT_ROOT = Path("outputs/mobile_share_by_recipient/snmt")
ALLOWED_RECIPIENT_ROOT = Path("outputs/mobile_share_by_recipient")
ALLOWED_GROUP_ROOT = Path("outputs/mobile_share_by_group")
FORBIDDEN_VISIBLE_TERMS = {"candidate", "manifest", "rank", "model", "debug", "shadow", "json", "rough_unreviewed"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="divesensei export-recipient-packages",
        description="Export per-recipient mobile packages from validated clip assignments.",
    )
    parser.add_argument("--assignment-validation", required=True, type=Path)
    parser.add_argument("--source-mobile-package", required=True, type=Path)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true", help="Plan recipient packages without copying MP4 files")
    parser.add_argument("--execute", action="store_true", help="Copy existing MP4 files into recipient folders")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing recipient package run")
    parser.add_argument("--package-mode", choices=("recipient", "group", "both"), default="recipient")
    parser.add_argument("--json", action="store_true")
    return parser


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def repo_root() -> Path:
    return Path.cwd().resolve()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def ensure_under(path: Path, allowed: Path, label: str) -> None:
    try:
        path.resolve().relative_to(allowed.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay under {allowed.resolve()}") from exc


def load_validated_assignments(assignment_validation: Path) -> list[dict[str, Any]]:
    path = assignment_validation / "validated_assignments.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing validated assignments: {path}")
    data = read_json(path)
    return list(data.get("assignments", []))


def load_recipient_package_plan(assignment_validation: Path) -> dict[str, Any]:
    path = assignment_validation / "recipient_package_plan.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing recipient package plan: {path}")
    return read_json(path)


def load_assignment_artifacts(assignment_validation: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assignment_validation = resolve(assignment_validation)
    return load_validated_assignments(assignment_validation), load_recipient_package_plan(assignment_validation)


def validate_source_clips(plan: dict[str, Any], source_mobile_package: Path) -> list[dict[str, Any]]:
    source_mobile_package = resolve(source_mobile_package)
    issues: list[dict[str, Any]] = []
    allowed_clips_dir = (source_mobile_package / "clips").resolve()
    for recipient in plan.get("recipients", []):
        for item in recipient.get("planned_files", []):
            source = resolve(Path(item.get("source_mobile_clip_path") or ""))
            try:
                source.relative_to(allowed_clips_dir)
            except ValueError:
                issues.append({"type": "source_outside_mobile_package", "source": str(source), "recipient": recipient.get("recipient_key")})
                continue
            if not source.exists():
                issues.append({"type": "source_missing", "source": str(source), "recipient": recipient.get("recipient_key")})
            elif source.stat().st_size <= 0:
                issues.append({"type": "source_empty", "source": str(source), "recipient": recipient.get("recipient_key")})
    return issues


def _filename_time(value: str) -> str:
    cleaned = value.replace(" ", "").replace("s", "")
    if "min" in cleaned:
        return cleaned.replace("min", "m")
    return cleaned or "t"


def _descriptive_filename(index: int, item: dict[str, Any]) -> str:
    time_part = str(item.get("timestamp_friendly") or item.get("mobile_filename") or f"clip{index}")
    time_part = time_part.replace(" ", "").replace("s", "")
    dive_code = str(item.get("dive_code") or "").strip()
    series = item.get("dive_code_series_index")
    attempt = item.get("attempt_index_within_dive_code_series") or item.get("attempt_index_for_dive_code") or index
    if dive_code:
        if series and int(series) > 1:
            suffix = f"{dive_code}_serie{int(series)}_essai{int(attempt)}"
        else:
            suffix = f"{dive_code}_essai{int(attempt)}"
    elif item.get("recipient_key") == "a_identifier":
        suffix = "a_identifier"
    else:
        suffix = f"essai{index}"
    return f"{index:02d}_{time_part}_{suffix}.mp4"


def format_recipient_filename(index: int, item: dict[str, Any]) -> str:
    time_part = _filename_time(str(item.get("timestamp_friendly") or item.get("mobile_filename") or f"clip{index}"))
    dive_code = str(item.get("dive_code") or "").strip()
    series = item.get("dive_code_series_index")
    attempt = item.get("attempt_index_within_dive_code_series") or item.get("attempt_index_for_dive_code") or index
    if dive_code:
        if series and int(series) > 1:
            suffix = f"{dive_code}_s{int(series)}e{int(attempt)}"
        else:
            suffix = f"{dive_code}_e{int(attempt)}"
    elif item.get("recipient_key") == "a_identifier":
        suffix = "a_identifier"
    else:
        suffix = f"essai{index}"
    return f"{index:02d}_{time_part}_{suffix}.mp4"


def _assignment_by_mobile(assignments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("mobile_filename")): row for row in assignments}


def _package_message(recipient_key: str) -> str:
    abbreviation_note = (
        "Dans les noms de fichiers : e = essai, s = série. "
        "Exemple : 201C_s2e1 = 201C, série 2, essai 1.\n"
    )
    if recipient_key == "a_identifier":
        return (
            "Voici les clips à identifier de la séance SNMT.\n"
            "Ces clips ne sont pas encore attribués à un plongeur.\n"
            "Ils sont à vérifier par l’entraîneur avant partage.\n"
            f"{abbreviation_note}"
        )
    return (
        "Voici tes clips rapides de la séance SNMT.\n"
        "Les clips sont générés automatiquement et vérifiés par l’entraîneur avant partage.\n"
        "Si un clip est trop court, mal cadré, ou s’il manque un passage, demande-moi la vidéo complète.\n"
        "Tu peux ouvrir directement les fichiers MP4 sur ton téléphone.\n"
        f"{abbreviation_note}"
    )


def _video_complete_text() -> str:
    return (
        "La vidéo complète n’est pas incluse automatiquement dans ce dossier.\n"
        "L’entraîneur peut l’envoyer séparément si besoin.\n"
        "Les noms des clips et les informations techniques peuvent aider le coach à retrouver les passages.\n"
    )


def _readme_html(recipient_label: str, clip_count: int) -> str:
    title = html.escape(f"Clips SNMT - {recipient_label}")
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; line-height: 1.45; color: #171717; }}
    main {{ max-width: 680px; }}
    h1 {{ font-size: 1.35rem; }}
    li {{ margin: 0.35rem 0; }}
  </style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p>Ce dossier contient {clip_count} clip(s) MP4 à ouvrir directement sur téléphone.</p>
  <ul>
    <li>Les clips sont automatiques et gérés par l’entraîneur.</li>
    <li>Si un clip manque ou semble trop court, demande la vidéo complète.</li>
    <li>Dans les noms de fichiers : <strong>e</strong> = essai, <strong>s</strong> = série. Exemple : <code>201C_s2e1</code> = 201C, série 2, essai 1.</li>
    <li>Le dossier <code>_technique</code> sert seulement au coach et peut être ignoré.</li>
  </ul>
</main>
</body>
</html>
"""


def build_recipient_export_plan(
    assignments: list[dict[str, Any]],
    recipient_plan: dict[str, Any],
    source_mobile_package: Path,
    output_root: Path,
    run_id: str,
    package_mode: str,
) -> dict[str, Any]:
    source_mobile_package = resolve(source_mobile_package)
    session_slug = source_mobile_package.parent.name
    assignments_by_mobile = _assignment_by_mobile(assignments)
    packages: list[dict[str, Any]] = []

    def add_package(kind: str, key: str, label: str, files: list[dict[str, Any]]) -> None:
        root = resolve(output_root) if kind == "recipient" else (repo_root() / "outputs/mobile_share_by_group/snmt").resolve()
        package_dir = root / session_slug / run_id / sanitize_slug(key or label)
        planned_files = []
        seen: dict[str, int] = defaultdict(int)
        for index, item in enumerate(files, start=1):
            source = resolve(Path(item["source_mobile_clip_path"]))
            assignment = assignments_by_mobile.get(str(item.get("mobile_filename")), {})
            merged = {**assignment, **item}
            filename = format_recipient_filename(index, merged)
            full_descriptive_filename = _descriptive_filename(index, merged)
            seen[filename] += 1
            if seen[filename] > 1:
                stem = Path(filename).stem
                filename = f"{stem}__dup{seen[filename]:02d}.mp4"
            planned_files.append(
                {
                    "source_mobile_clip_path": str(source),
                    "output_clip_path": str(package_dir / "clips" / filename),
                    "output_filename": filename,
                    "full_descriptive_filename": full_descriptive_filename,
                    "recipient_key": key,
                    "recipient_label": label,
                    "package_kind": kind,
                    "mobile_filename": item.get("mobile_filename"),
                    "candidate_id": item.get("candidate_id"),
                    "dive_code": item.get("dive_code") or "",
                    "dive_code_series_index": item.get("dive_code_series_index"),
                    "attempt_index_within_dive_code_series": item.get("attempt_index_within_dive_code_series"),
                    "assignment_status": item.get("assignment_status"),
                    "share_status": item.get("share_status"),
                    "timestamp_seconds": assignment.get("timestamp_seconds"),
                    "timestamp_friendly": assignment.get("timestamp_friendly"),
                    "training_group": assignment.get("training_group") or "",
                }
            )
        packages.append({"package_kind": kind, "package_key": key, "package_label": label, "package_dir": str(package_dir), "clip_count": len(planned_files), "planned_files": planned_files})

    if package_mode in {"recipient", "both"}:
        for recipient in recipient_plan.get("recipients", []):
            add_package("recipient", recipient["recipient_key"], recipient["recipient_label"], recipient.get("planned_files", []))

    if package_mode in {"group", "both"}:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        labels: dict[str, str] = {}
        for recipient in recipient_plan.get("recipients", []):
            for item in recipient.get("planned_files", []):
                assignment = assignments_by_mobile.get(str(item.get("mobile_filename")), {})
                group = str(assignment.get("training_group") or "groupe_inconnu")
                grouped[group].append({**item, "recipient_key": recipient["recipient_key"], "recipient_label": recipient["recipient_label"]})
                labels[group] = group
        for group, files in sorted(grouped.items()):
            add_package("group", group, labels[group], files)

    return {"session_slug": session_slug, "run_id": run_id, "package_mode": package_mode, "packages": packages}


def _prepare_run_root(output_root: Path, session_slug: str, run_id: str, force: bool, execute: bool) -> Path:
    root = resolve(output_root)
    ensure_under(root, (repo_root() / ALLOWED_RECIPIENT_ROOT).resolve(), "Output root")
    run_root = root / session_slug / run_id
    ensure_under(run_root, (repo_root() / ALLOWED_RECIPIENT_ROOT).resolve(), "Run root")
    if run_root.exists():
        if not force:
            raise FileExistsError(f"Output run directory exists; pass --force to reuse: {run_root}")
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _write_package_files(package: dict[str, Any], execute: bool) -> dict[str, Any]:
    package_dir = Path(package["package_dir"])
    clips_dir = package_dir / "clips"
    tech_dir = package_dir / "_technique"
    copied = 0
    total_size = 0
    if execute:
        clips_dir.mkdir(parents=True, exist_ok=True)
        tech_dir.mkdir(parents=True, exist_ok=True)
        for item in package["planned_files"]:
            source = Path(item["source_mobile_clip_path"])
            target = Path(item["output_clip_path"])
            shutil.copy2(source, target)
            copied += 1
            total_size += target.stat().st_size
        write_recipient_readmes(package)
        write_recipient_technical_traceability(package)
    return {"package_dir": str(package_dir), "copied_clip_count": copied, "package_size_mb": round(total_size / (1024 * 1024), 3)}


def copy_recipient_clips(export_plan: dict[str, Any], execute: bool) -> list[dict[str, Any]]:
    return [_write_package_files(package, execute=execute) for package in export_plan["packages"]]


def write_recipient_readmes(package: dict[str, Any]) -> None:
    package_dir = Path(package["package_dir"])
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "MESSAGE_A_COPIER.txt").write_text(_package_message(package["package_key"]), encoding="utf-8")
    (package_dir / "VIDEO_COMPLETE_INFO.txt").write_text(_video_complete_text(), encoding="utf-8")
    (package_dir / "README_MOBILE.html").write_text(_readme_html(package["package_label"], package["clip_count"]), encoding="utf-8")


def write_recipient_technical_traceability(package: dict[str, Any]) -> None:
    tech_dir = Path(package["package_dir"]) / "_technique"
    tech_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in enumerate(package["planned_files"], start=1):
        rows.append(
            {
                "clip_number": index,
                "recipient_key": package["package_key"],
                "recipient_label": package["package_label"],
                "recipient_filename": item["output_filename"],
                "full_descriptive_filename": item.get("full_descriptive_filename") or "",
                "source_mobile_clip_path": item["source_mobile_clip_path"],
                "source_mobile_filename": item["mobile_filename"],
                "candidate_id": item["candidate_id"],
                "timestamp_seconds": item.get("timestamp_seconds") or "",
                "timestamp_friendly": item.get("timestamp_friendly") or "",
                "dive_code": item.get("dive_code") or "",
                "dive_code_series_index": item.get("dive_code_series_index") or "",
                "attempt_index_within_dive_code_series": item.get("attempt_index_within_dive_code_series") or "",
                "assignment_status": item.get("assignment_status") or "",
                "share_status": item.get("share_status") or "",
            }
        )
    fields = list(rows[0].keys()) if rows else ["clip_number", "recipient_filename", "source_mobile_clip_path", "candidate_id"]
    write_csv(tech_dir / "LISTE_DES_CLIPS_RECIPIENT.csv", rows, fields)
    write_json(tech_dir / "assignment_source.json", {"package": package})
    write_json(
        tech_dir / "package_summary.json",
        {
            "package_kind": package["package_kind"],
            "package_key": package["package_key"],
            "package_label": package["package_label"],
            "clip_count": package["clip_count"],
        },
    )


def write_export_summary(run_root: Path, export_plan: dict[str, Any], copy_results: list[dict[str, Any]], dry_run: bool) -> None:
    total_clips = sum(package["clip_count"] for package in export_plan["packages"])
    total_copied = sum(item["copied_clip_count"] for item in copy_results)
    write_json(
        run_root / ("dry_run_report.json" if dry_run else "export_report.json"),
        {
            "dry_run": dry_run,
            "session_slug": export_plan["session_slug"],
            "run_id": export_plan["run_id"],
            "package_mode": export_plan["package_mode"],
            "package_count": len(export_plan["packages"]),
            "planned_clip_count": total_clips,
            "copied_clip_count": total_copied,
            "packages": export_plan["packages"],
            "copy_results": copy_results,
        },
    )
    (run_root / "EXPORT_SUMMARY.md").write_text(
        "\n".join(
            [
                "# Export destinataires SNMT",
                "",
                f"- Mode: {'dry-run' if dry_run else 'execute'}",
                f"- Dossiers: {len(export_plan['packages'])}",
                f"- Clips planifiés: {total_clips}",
                f"- Clips copiés: {total_copied}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def export_recipient_packages(
    assignment_validation: Path,
    source_mobile_package: Path,
    output_root: Path,
    run_id: str,
    dry_run: bool,
    execute: bool,
    force: bool,
    package_mode: str,
) -> dict[str, Any]:
    if dry_run and execute:
        raise ValueError("--dry-run and --execute cannot be used together")
    if not dry_run and not execute:
        dry_run = True
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    assignments, recipient_plan = load_assignment_artifacts(assignment_validation)
    source_mobile_package = resolve(source_mobile_package)
    if not source_mobile_package.exists():
        raise FileNotFoundError(f"Source mobile package not found: {source_mobile_package}")
    source_issues = validate_source_clips(recipient_plan, source_mobile_package)
    if source_issues:
        raise FileNotFoundError(f"Source clip validation failed: {source_issues[:3]}")
    export_plan = build_recipient_export_plan(assignments, recipient_plan, source_mobile_package, output_root, run_id, package_mode)
    run_root = _prepare_run_root(output_root, export_plan["session_slug"], run_id, force, execute)
    if package_mode in {"group", "both"}:
        group_root = (repo_root() / "outputs/mobile_share_by_group/snmt" / export_plan["session_slug"] / run_id).resolve()
        ensure_under(group_root, (repo_root() / ALLOWED_GROUP_ROOT).resolve(), "Group run root")
        if group_root.exists() and force:
            shutil.rmtree(group_root)
    copy_results = copy_recipient_clips(export_plan, execute=execute)
    write_export_summary(run_root, export_plan, copy_results, dry_run=dry_run)
    return {
        "run_root": str(run_root),
        "dry_run": dry_run,
        "execute": execute,
        "package_mode": package_mode,
        "package_count": len(export_plan["packages"]),
        "planned_clip_count": sum(package["clip_count"] for package in export_plan["packages"]),
        "copied_clip_count": sum(item["copied_clip_count"] for item in copy_results),
        "packages": export_plan["packages"],
        "copy_results": copy_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = export_recipient_packages(
            assignment_validation=args.assignment_validation,
            source_mobile_package=args.source_mobile_package,
            output_root=args.output_root,
            run_id=args.run_id,
            dry_run=args.dry_run,
            execute=args.execute,
            force=args.force,
            package_mode=args.package_mode,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"Recipient package export written to {summary['run_root']}")
        print(f"Mode: {'execute' if summary['execute'] else 'dry-run'}")
        print(f"Packages: {summary['package_count']}")
        print(f"Planned clips: {summary['planned_clip_count']}")
        print(f"Copied clips: {summary['copied_clip_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
