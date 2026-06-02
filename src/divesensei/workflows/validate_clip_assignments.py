from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence


ALLOWED_ASSIGNMENT_STATUS = {"a_verifier", "a_identifier", "garder", "supprimer", "ignore"}
ALLOWED_SHARE_STATUS = {"a_verifier", "a_partager", "ne_pas_partager", "coach_seulement"}
UNKNOWN_DIVER_LABELS = {"", "a identifier", "a_identifier", "inconnu", "inconnue", "unknown"}
OUTPUT_ROOT = Path("outputs/assignment_validation")

REQUIRED_ASSIGNMENT_COLUMNS = {
    "mobile_filename",
    "diver_query",
    "training_group",
    "apparatus_type",
    "apparatus_height_m",
    "apparatus_id",
    "dive_code",
    "assignment_status",
    "share_status",
    "notes",
}


@dataclass(frozen=True)
class NormalizedText:
    original: str
    normalized: str
    compact: str
    initials: str


def normalize_text(value: Any) -> NormalizedText:
    original = "" if value is None else str(value).strip()
    decomposed = unicodedata.normalize("NFKD", original)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    lower = re.sub(r"\s+", " ", without_accents.lower()).strip()
    compact = re.sub(r"[^a-z0-9]+", "", lower)
    words = re.findall(r"[a-z0-9]+", lower)
    initials = "".join(word[0] for word in words if word)
    return NormalizedText(original=original, normalized=lower, compact=compact, initials=initials)


def _repo_root() -> Path:
    return Path.cwd().resolve()


def _safe_output_dir(output_root: Path, session_slug: str, run_id: str, force: bool) -> Path:
    root = output_root.resolve()
    allowed = (_repo_root() / OUTPUT_ROOT).resolve()
    try:
        root.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Output root must stay under {allowed}") from exc

    out_dir = root / session_slug / run_id
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory exists; pass --force to reuse: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bool_value(value: Any, default: bool = True) -> bool:
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "oui", "actif", "active"}


def load_mobile_clip_list(mobile_package: Path) -> list[dict[str, Any]]:
    clip_list_path = mobile_package / "_technique" / "LISTE_DES_CLIPS.csv"
    if not clip_list_path.exists():
        raise FileNotFoundError(f"Missing mobile clip list: {clip_list_path}")
    rows = _read_csv(clip_list_path)
    normalized: list[dict[str, Any]] = []
    for row in rows:
        mobile_filename = (row.get("mobile_filename") or "").strip()
        if not mobile_filename:
            continue
        enriched = dict(row)
        enriched["mobile_filename"] = mobile_filename
        enriched["timestamp_seconds"] = _float_or_none(row.get("timestamp_seconds"))
        enriched["clip_start_seconds"] = _float_or_none(row.get("clip_start_seconds"))
        enriched["clip_end_seconds"] = _float_or_none(row.get("clip_end_seconds"))
        enriched["duration_seconds"] = _float_or_none(row.get("duration_seconds"))
        enriched["mobile_clip_path"] = str((mobile_package / "clips" / mobile_filename).resolve())
        normalized.append(enriched)
    if not normalized:
        raise ValueError(f"No clips found in {clip_list_path}")
    return normalized


def load_assignment_csv(assignments_path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(assignments_path)
    if not rows:
        raise ValueError(f"Assignment CSV is empty: {assignments_path}")
    columns = set(rows[0].keys())
    missing = sorted(REQUIRED_ASSIGNMENT_COLUMNS - columns)
    if missing:
        raise ValueError(f"Assignment CSV missing required columns: {', '.join(missing)}")
    return rows


def load_diver_roster(divers_path: Path | None) -> list[dict[str, Any]]:
    if not divers_path:
        return []
    if not divers_path.exists():
        raise FileNotFoundError(f"Diver roster not found: {divers_path}")
    if divers_path.suffix.lower() == ".json":
        data = json.loads(divers_path.read_text(encoding="utf-8"))
        rows = data.get("divers", data) if isinstance(data, dict) else data
    else:
        rows = _read_csv(divers_path)
    roster: list[dict[str, Any]] = []
    for row in rows:
        if not _bool_value(row.get("active"), default=True):
            continue
        display_name = str(row.get("display_name") or "").strip()
        if not display_name:
            continue
        aliases_raw = row.get("aliases") or []
        if isinstance(aliases_raw, str):
            aliases = [alias.strip() for alias in re.split(r"[|,;]", aliases_raw) if alias.strip()]
        else:
            aliases = [str(alias).strip() for alias in aliases_raw if str(alias).strip()]
        norm = normalize_text(display_name)
        roster.append(
            {
                "diver_id": str(row.get("diver_id") or norm.compact),
                "display_name": display_name,
                "aliases": aliases,
                "group": row.get("group") or "",
                "active": True,
                "normalized_name": norm.normalized,
                "compact_name": norm.compact,
                "initials": norm.initials,
            }
        )
    return roster


def build_alias_index(roster: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for diver in roster:
        keys = {
            diver["normalized_name"],
            diver["compact_name"],
            diver["initials"],
        }
        keys.update(normalize_text(alias).normalized for alias in diver.get("aliases", []))
        keys.update(normalize_text(alias).compact for alias in diver.get("aliases", []))
        for key in keys:
            if key:
                index[key].append(diver)
    return dict(index)


def _resolution(status: str, query: str, **kwargs: Any) -> dict[str, Any]:
    payload = {
        "typed_value": query,
        "status": status,
        "resolved_diver_id": None,
        "resolved_diver_label": None,
        "match_type": None,
        "match_score": 0.0,
        "warning": None,
    }
    payload.update(kwargs)
    return payload


def _single_match(matches: list[dict[str, Any]], query: str, match_type: str, score: float) -> dict[str, Any]:
    if len(matches) == 1:
        diver = matches[0]
        return _resolution(
            "resolved",
            query,
            resolved_diver_id=diver["diver_id"],
            resolved_diver_label=diver["display_name"],
            match_type=match_type,
            match_score=score,
        )
    return _resolution(
        "ambiguous",
        query,
        match_type=match_type,
        match_score=score,
        candidates=[{"diver_id": item["diver_id"], "display_name": item["display_name"]} for item in matches],
        warning="ambiguous diver query; manual resolution required",
    )


def resolve_diver_query(query: Any, roster: list[dict[str, Any]], alias_index: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    text = normalize_text(query)
    alias_index = alias_index or build_alias_index(roster)
    if text.normalized in UNKNOWN_DIVER_LABELS:
        return _resolution(
            "unknown",
            text.original,
            resolved_diver_id="a_identifier",
            resolved_diver_label="À identifier",
            match_type="unknown",
            match_score=1.0,
        )
    if not roster:
        return _resolution("free_text", text.original, resolved_diver_label=text.original, match_type="manual_free_text", match_score=0.5)

    alias_matches = alias_index.get(text.normalized, [])
    if alias_matches:
        return _single_match(alias_matches, text.original, "exact_alias_or_name", 1.0)

    compact_matches = alias_index.get(text.compact, [])
    if compact_matches:
        return _single_match(compact_matches, text.original, "compact_alias", 0.98)

    initials_matches = [diver for diver in roster if diver["initials"] == text.compact]
    if initials_matches:
        return _single_match(initials_matches, text.original, "initials", 0.92)

    prefix_matches = [
        diver
        for diver in roster
        if diver["normalized_name"].startswith(text.normalized)
        or diver["compact_name"].startswith(text.compact)
        or any(normalize_text(alias).compact.startswith(text.compact) for alias in diver.get("aliases", []))
    ]
    if prefix_matches:
        return _single_match(prefix_matches, text.original, "prefix", 0.85)

    substring_matches = [
        diver
        for diver in roster
        if text.normalized in diver["normalized_name"]
        or text.compact in diver["compact_name"]
        or any(text.compact in normalize_text(alias).compact for alias in diver.get("aliases", []))
    ]
    if substring_matches:
        return _single_match(substring_matches, text.original, "substring", 0.78)

    near = detect_diver_duplicates(text.original, roster)
    if near:
        return _resolution(
            "near_duplicate_unconfirmed",
            text.original,
            match_type="near_duplicate",
            match_score=near[0]["score"],
            near_duplicates=near,
            warning="possible duplicate diver; coach confirmation required",
        )

    return _resolution("unresolved", text.original, resolved_diver_label=text.original, match_type="manual_free_text", match_score=0.0, warning="no roster match")


def detect_diver_duplicates(query: Any, roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = normalize_text(query)
    matches: list[dict[str, Any]] = []
    for diver in roster:
        ratio = SequenceMatcher(None, text.compact, diver["compact_name"]).ratio()
        tokens = set(text.normalized.split())
        diver_tokens = set(diver["normalized_name"].split())
        token_overlap = len(tokens & diver_tokens) / max(1, len(tokens | diver_tokens))
        score = max(ratio, token_overlap)
        if score >= 0.72:
            matches.append({"diver_id": diver["diver_id"], "display_name": diver["display_name"], "score": round(score, 3)})
    return sorted(matches, key=lambda item: item["score"], reverse=True)


def _dive_code(value: Any) -> str:
    raw = "" if value is None else str(value).strip()
    norm = normalize_text(raw).normalized
    if norm in {"", "inconnu", "unknown", "a identifier", "a_identifier"}:
        return ""
    return raw.upper()


def _is_packageable(row: dict[str, Any]) -> bool:
    return row.get("share_status") == "a_partager" and row.get("assignment_status") not in {"supprimer", "ignore"}


def compute_attempt_indexes(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row.get("timestamp_seconds") is None, row.get("timestamp_seconds") or 0.0, row.get("mobile_filename") or ""))
    session_count = 0
    by_diver: dict[str, int] = defaultdict(int)
    by_apparatus: dict[str, int] = defaultdict(int)
    by_diver_apparatus: dict[tuple[str, str], int] = defaultdict(int)
    by_diver_code_total: dict[tuple[str, str], int] = defaultdict(int)
    series_counter_by_diver_code: dict[tuple[str, str], int] = defaultdict(int)
    last_code_by_diver: dict[str, str] = {}
    active_series_by_diver_code: dict[tuple[str, str], int] = {}
    attempt_in_series: dict[tuple[str, str, int], int] = defaultdict(int)
    audit_rows: list[dict[str, Any]] = []

    for row in ordered:
        row["packageable"] = _is_packageable(row)
        row["attempt_index_in_session"] = None
        row["attempt_index_for_diver"] = None
        row["attempt_index_for_apparatus"] = None
        row["attempt_index_for_diver_on_apparatus"] = None
        row["attempt_index_for_dive_code"] = None
        row["dive_code_series_index"] = None
        row["attempt_index_within_dive_code_series"] = None
        if not row["packageable"]:
            continue

        diver_key = row.get("recipient_key") or "a_identifier"
        apparatus_key = "|".join(str(row.get(key) or "") for key in ("apparatus_type", "apparatus_height_m", "apparatus_id"))
        code = row.get("dive_code_normalized") or ""

        session_count += 1
        by_diver[diver_key] += 1
        by_apparatus[apparatus_key] += 1
        by_diver_apparatus[(diver_key, apparatus_key)] += 1
        row["attempt_index_in_session"] = session_count
        row["attempt_index_for_diver"] = by_diver[diver_key]
        row["attempt_index_for_apparatus"] = by_apparatus[apparatus_key]
        row["attempt_index_for_diver_on_apparatus"] = by_diver_apparatus[(diver_key, apparatus_key)]

        if code:
            by_diver_code_total[(diver_key, code)] += 1
            row["attempt_index_for_dive_code"] = by_diver_code_total[(diver_key, code)]
            manual_series = _int_or_none(row.get("manual_series_override"))
            if manual_series:
                series_index = manual_series
                series_counter_by_diver_code[(diver_key, code)] = max(series_counter_by_diver_code[(diver_key, code)], series_index)
                active_series_by_diver_code[(diver_key, code)] = series_index
            elif last_code_by_diver.get(diver_key) == code and (diver_key, code) in active_series_by_diver_code:
                series_index = active_series_by_diver_code[(diver_key, code)]
            else:
                series_counter_by_diver_code[(diver_key, code)] += 1
                series_index = series_counter_by_diver_code[(diver_key, code)]
                active_series_by_diver_code[(diver_key, code)] = series_index
            manual_attempt = _int_or_none(row.get("manual_attempt_override"))
            if manual_attempt:
                attempt_value = manual_attempt
            else:
                attempt_in_series[(diver_key, code, series_index)] += 1
                attempt_value = attempt_in_series[(diver_key, code, series_index)]
            row["dive_code_series_index"] = series_index
            row["attempt_index_within_dive_code_series"] = attempt_value
            last_code_by_diver[diver_key] = code

        audit_rows.append(
            {
                "mobile_filename": row.get("mobile_filename"),
                "recipient_key": diver_key,
                "dive_code": code,
                "attempt_index_in_session": row.get("attempt_index_in_session"),
                "attempt_index_for_diver": row.get("attempt_index_for_diver"),
                "attempt_index_for_dive_code": row.get("attempt_index_for_dive_code"),
                "dive_code_series_index": row.get("dive_code_series_index"),
                "attempt_index_within_dive_code_series": row.get("attempt_index_within_dive_code_series"),
                "manual_series_override": row.get("manual_series_override") or "",
                "manual_attempt_override": row.get("manual_attempt_override") or "",
            }
        )

    seen_labels: dict[tuple[str, str, int, int], str] = {}
    duplicates: list[dict[str, Any]] = []
    for row in ordered:
        if not row.get("packageable") or not row.get("dive_code_normalized"):
            continue
        key = (
            row.get("recipient_key") or "a_identifier",
            row.get("dive_code_normalized") or "",
            int(row.get("dive_code_series_index") or 0),
            int(row.get("attempt_index_within_dive_code_series") or 0),
        )
        previous = seen_labels.get(key)
        if previous:
            duplicates.append({"first": previous, "second": row.get("mobile_filename"), "label_key": list(key)})
        else:
            seen_labels[key] = str(row.get("mobile_filename"))

    return ordered, {"attempt_rows": audit_rows, "duplicate_attempt_labels": duplicates}


def _future_filename(recipient_index: int, row: dict[str, Any]) -> str:
    time_label = (row.get("timestamp_friendly") or "clip").replace(" ", "")
    code = row.get("dive_code_normalized") or ""
    if code:
        series = row.get("dive_code_series_index")
        attempt = row.get("attempt_index_within_dive_code_series") or row.get("attempt_index_for_dive_code") or recipient_index
        if series and int(series) > 1:
            suffix = f"{code}_serie{int(series)}_essai{int(attempt)}"
        else:
            suffix = f"{code}_essai{int(attempt)}"
    else:
        suffix = f"essai{row.get('attempt_index_for_diver') or recipient_index}"
    return f"{recipient_index:02d}_{time_label}_{suffix}.mp4"


def build_recipient_package_plan(rows: list[dict[str, Any]], mobile_package: Path) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        if not row.get("packageable"):
            continue
        key = row.get("recipient_key") or "a_identifier"
        grouped[key].append(row)
        labels[key] = row.get("recipient_label") or ("À identifier" if key == "a_identifier" else key)

    recipients: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item == "a_identifier", labels.get(item, item))):
        clips = sorted(grouped[key], key=lambda row: (row.get("timestamp_seconds") or 0.0, row.get("mobile_filename") or ""))
        planned_files = []
        for index, row in enumerate(clips, start=1):
            planned_files.append(
                {
                    "source_mobile_clip_path": str((mobile_package / "clips" / str(row.get("mobile_filename"))).resolve()),
                    "future_output_filename": _future_filename(index, row),
                    "mobile_filename": row.get("mobile_filename"),
                    "candidate_id": row.get("candidate_id"),
                    "assignment_status": row.get("assignment_status"),
                    "share_status": row.get("share_status"),
                    "dive_code": row.get("dive_code_normalized") or "",
                    "dive_code_series_index": row.get("dive_code_series_index"),
                    "attempt_index_within_dive_code_series": row.get("attempt_index_within_dive_code_series"),
                }
            )
        recipients.append({"recipient_key": key, "recipient_label": labels[key], "clip_count": len(planned_files), "planned_files": planned_files})
    return {"recipient_count": len(recipients), "recipients": recipients}


def validate_assignments(
    mobile_clips: list[dict[str, Any]],
    assignment_rows: list[dict[str, Any]],
    roster: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    roster = roster or []
    alias_index = build_alias_index(roster)
    clip_by_name = {row["mobile_filename"]: row for row in mobile_clips}
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    diver_resolution_audit: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []

    for raw_index, assignment in enumerate(assignment_rows, start=1):
        filename = str(assignment.get("mobile_filename") or "").strip()
        clip = clip_by_name.get(filename)
        if not clip:
            errors.append({"row": raw_index, "mobile_filename": filename, "error": "mobile_filename not found in mobile package"})
            continue

        assignment_status = str(assignment.get("assignment_status") or "").strip() or "a_verifier"
        share_status = str(assignment.get("share_status") or "").strip() or "a_partager"
        if assignment_status not in ALLOWED_ASSIGNMENT_STATUS:
            errors.append({"row": raw_index, "mobile_filename": filename, "error": f"invalid assignment_status: {assignment_status}"})
        if share_status not in ALLOWED_SHARE_STATUS:
            errors.append({"row": raw_index, "mobile_filename": filename, "error": f"invalid share_status: {share_status}"})

        diver_query = assignment.get("diver_query") or assignment.get("typed_diver_query") or ""
        resolution = resolve_diver_query(diver_query, roster, alias_index)
        resolution["mobile_filename"] = filename
        diver_resolution_audit.append(resolution)
        if resolution.get("warning"):
            warnings.append({"row": raw_index, "mobile_filename": filename, "type": "diver_resolution", "message": resolution["warning"], "details": resolution})

        code = _dive_code(assignment.get("dive_code"))
        if not code:
            warnings.append({"row": raw_index, "mobile_filename": filename, "type": "missing_dive_code", "message": "dive code is blank/inconnu; sharing remains allowed"})

        autofill_source = str(assignment.get("autofill_source") or "").strip()
        autofill_confirmed = str(assignment.get("autofill_confirmed") or "").strip().lower()
        if autofill_source and autofill_confirmed not in {"1", "true", "yes", "oui"}:
            warnings.append({"row": raw_index, "mobile_filename": filename, "type": "autofill_unconfirmed", "message": "autofill value is suggestion-only until coach confirms"})

        recipient_key = resolution.get("resolved_diver_id")
        recipient_label = resolution.get("resolved_diver_label")
        if resolution.get("status") in {"near_duplicate_unconfirmed", "ambiguous", "unknown"} or not recipient_key:
            recipient_key = "a_identifier" if resolution.get("status") != "free_text" else normalize_text(recipient_label).compact
            recipient_label = "À identifier" if recipient_key == "a_identifier" else recipient_label

        row = {**clip, **assignment}
        row.update(
            {
                "row_index": raw_index,
                "assignment_status": assignment_status,
                "share_status": share_status,
                "dive_code_original": assignment.get("dive_code") or "",
                "dive_code_normalized": code,
                "typed_diver_query": diver_query,
                "resolved_diver_id": resolution.get("resolved_diver_id"),
                "resolved_diver_label": resolution.get("resolved_diver_label"),
                "diver_match_type": resolution.get("match_type"),
                "diver_match_score": resolution.get("match_score"),
                "diver_resolution_status": resolution.get("status"),
                "recipient_key": recipient_key,
                "recipient_label": recipient_label,
                "manual_series_override": assignment.get("manual_series_override") or "",
                "manual_attempt_override": assignment.get("manual_attempt_override") or "",
                "bulk_operation_id": assignment.get("bulk_operation_id") or "",
                "bulk_operation_label": assignment.get("bulk_operation_label") or "",
                "autofill_source": autofill_source,
                "autofill_confirmed": assignment.get("autofill_confirmed") or "",
            }
        )
        validated.append(row)

    validated, attempt_audit = compute_attempt_indexes(validated)
    for duplicate in attempt_audit.get("duplicate_attempt_labels", []):
        warnings.append({"type": "duplicate_attempt_label", "message": "manual override created duplicate attempt label", "details": duplicate})

    return {
        "validated_assignments": validated,
        "warnings": warnings,
        "errors": errors,
        "diver_resolution_audit": diver_resolution_audit,
        "attempt_index_audit": attempt_audit,
    }


def write_assignment_outputs(result: dict[str, Any], output_dir: Path, mobile_package: Path) -> dict[str, Path]:
    rows = result["validated_assignments"]
    plan = build_recipient_package_plan(rows, mobile_package)
    series_rows = [
        {
            "mobile_filename": row.get("mobile_filename"),
            "recipient_key": row.get("recipient_key"),
            "dive_code": row.get("dive_code_normalized"),
            "dive_code_series_index": row.get("dive_code_series_index"),
            "attempt_index_within_dive_code_series": row.get("attempt_index_within_dive_code_series"),
            "manual_series_override": row.get("manual_series_override") or "",
        }
        for row in rows
        if row.get("packageable") and row.get("dive_code_normalized")
    ]

    csv_fields = [
        "mobile_filename",
        "candidate_id",
        "timestamp_seconds",
        "timestamp_friendly",
        "diver_query",
        "resolved_diver_id",
        "resolved_diver_label",
        "diver_resolution_status",
        "training_group",
        "apparatus_type",
        "apparatus_height_m",
        "apparatus_id",
        "dive_code_normalized",
        "assignment_status",
        "share_status",
        "attempt_index_in_session",
        "attempt_index_for_diver",
        "attempt_index_for_apparatus",
        "attempt_index_for_diver_on_apparatus",
        "attempt_index_for_dive_code",
        "dive_code_series_index",
        "attempt_index_within_dive_code_series",
        "bulk_operation_id",
        "bulk_operation_label",
        "autofill_source",
        "autofill_confirmed",
        "notes",
    ]
    paths = {
        "validated_assignments_json": output_dir / "validated_assignments.json",
        "validated_assignments_csv": output_dir / "validated_assignments.csv",
        "recipient_package_plan": output_dir / "recipient_package_plan.json",
        "assignment_warnings": output_dir / "assignment_warnings.json",
        "assignment_errors": output_dir / "assignment_errors.json",
        "diver_resolution_audit": output_dir / "diver_resolution_audit.json",
        "attempt_index_audit": output_dir / "attempt_index_audit.json",
        "series_logic_audit": output_dir / "series_logic_audit.json",
        "summary": output_dir / "VALIDATION_SUMMARY.md",
    }
    _write_json(paths["validated_assignments_json"], {"assignments": rows})
    _write_csv(paths["validated_assignments_csv"], rows, csv_fields)
    _write_json(paths["recipient_package_plan"], plan)
    _write_json(paths["assignment_warnings"], {"warnings": result["warnings"]})
    _write_json(paths["assignment_errors"], {"errors": result["errors"]})
    _write_json(paths["diver_resolution_audit"], {"resolutions": result["diver_resolution_audit"]})
    _write_json(paths["attempt_index_audit"], result["attempt_index_audit"])
    _write_json(paths["series_logic_audit"], {"series_rows": series_rows})
    paths["summary"].write_text(
        "\n".join(
            [
                "# Validation des affectations",
                "",
                f"- Clips valides: {len(rows)}",
                f"- Avertissements: {len(result['warnings'])}",
                f"- Erreurs: {len(result['errors'])}",
                f"- Destinataires planifiés: {plan['recipient_count']}",
                "",
                "Aucun clip n'a été copié, généré ou transcodé.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return paths


def validate_clip_assignments(
    mobile_package: Path,
    assignments: Path,
    divers: Path | None,
    output_root: Path,
    run_id: str,
    force: bool,
) -> dict[str, Any]:
    mobile_package = mobile_package.resolve()
    if not mobile_package.exists():
        raise FileNotFoundError(f"Mobile package not found: {mobile_package}")
    session_slug = mobile_package.parent.name
    output_dir = _safe_output_dir(output_root, session_slug, run_id, force)
    mobile_clips = load_mobile_clip_list(mobile_package)
    assignment_rows = load_assignment_csv(assignments)
    roster = load_diver_roster(divers)
    result = validate_assignments(mobile_clips, assignment_rows, roster)
    paths = write_assignment_outputs(result, output_dir, mobile_package)
    return {
        "output_dir": str(output_dir),
        "paths": {key: str(path) for key, path in paths.items()},
        "assignment_count": len(result["validated_assignments"]),
        "warning_count": len(result["warnings"]),
        "error_count": len(result["errors"]),
        "recipient_count": json.loads(paths["recipient_package_plan"].read_text(encoding="utf-8"))["recipient_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate manual clip assignments and build recipient package plans.")
    parser.add_argument("--mobile-package", required=True, type=Path)
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--divers", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/assignment_validation/snmt"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = validate_clip_assignments(
            mobile_package=args.mobile_package,
            assignments=args.assignments,
            divers=args.divers,
            output_root=args.output_root,
            run_id=args.run_id,
            force=args.force,
        )
    except Exception as exc:  # CLI boundary: print a clear error and fail safely.
        print(f"ERROR: {exc}")
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"Assignment validation written to {summary['output_dir']}")
        print(f"Assignments: {summary['assignment_count']}")
        print(f"Warnings: {summary['warning_count']}")
        print(f"Errors: {summary['error_count']}")
        print(f"Recipient plans: {summary['recipient_count']}")
    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
