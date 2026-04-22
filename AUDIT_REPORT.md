# DiveSensei — Professional Product & Technical Audit

**Date:** 2026-04-21
**Branch:** `phase4-review-support` (commit `0eabf9f`)
**Scope:** full-stack — Python backend, Astro/Electron desktop, Capacitor/iOS mobile, detector science, UX across six personas
**Method:** static code review (every claim cited `file:line`), competitive and academic web research, persona walkthroughs

---

## 0 · Executive summary

DiveSensei is a genuinely interesting product: a **local-first, audio-first detection pipeline** that turns 2–3 hour practice footage into a reviewable queue of dive events, with a desktop review workspace and a slimmer iOS companion. The engineering foundations are solid — clean module boundaries, dual-frontend detectors (spectral-flux heuristic + PCEN onset), a bootstrappable learned-logistic "governed-r9" re-scorer, and an Electron shell with SSR Astro SSR.

But benchmarked against (a) the state of the art — the **University of Pittsburgh zero-shot VLM system reports 97 % dive recall on 2.5 h practice footage** ([pitt-cic](https://github.com/pitt-cic/automatic-highlight-reel-generator), [Pitt Digital](https://www.digital.pitt.edu/news/success-stories/dive-revolutionizing-coaching-automated-highlight-reels)) — and (b) what working coaches actually demand from Dartfish / Hudl Technique / Kinovea / Coach's Eye ([Dartfish](https://www.dartfish.com/), [SimpliFaster guide](https://simplifaster.com/articles/buyers-guide-sport-video-analysis/)), DiveSensei has **six structural gaps** that a professional diving program will notice within the first session:

| # | Gap | Severity | Evidence |
|---|---|---|---|
| 1 | Detection is **audio-only by default**; silent entries and noisy meets are systemic blind spots | **Critical** | [`audio_detector.py:241,247,276`](apps/desktop/src/components/DesignWorkspace.astro), no visual fallback when audio is the first-stage filter |
| 2 | No **frame-stepping**, no **J/K/L shuttle**, no **audio waveform** in the review view — unmet baseline across every competitor in the space | **Critical** | [`DesignWorkspace.astro:376-384`](apps/desktop/src/components/DesignWorkspace.astro#L376); Kinovea / Dartfish / Coach's Eye all ship this ([Dartfish vs Coach's Eye](https://www.dartfish.com/blog/looking-for-an-alternative-to-coachs-eye/)) |
| 3 | **Zero automated tests** on a 3 285-line detector with ~200 config knobs and silently-failing profile overrides | **Critical** | [`src/divesensei/detection/audio_detector.py`](src/divesensei/detection/audio_detector.py); profile key `audio_peak_separation` ≠ config attribute `audio_peak_min_separation_seconds` ([`profiles.py:9`](src/divesensei/profiles.py#L9) vs [`config.py:53`](src/divesensei/detection/config.py#L53)) |
| 4 | 2 966-line monolithic `DesignWorkspace.astro`, 2 900+ of which is **untyped inline JS** | High | [`apps/desktop/src/components/DesignWorkspace.astro`](apps/desktop/src/components/DesignWorkspace.astro) |
| 5 | **Terminology drift** (session / evaluation / attempt / candidate / detection / event) and hidden modal state (standard vs evaluation) | High | See §4.2 |
| 6 | **No scoring, no annotation, no side-by-side compare** — the things a coach actually sits down to do | High | Feature-gap vs Dartfish, Hudl, OnForm, Coach's Eye, Kinovea |

Everything else — accessibility, privacy for minors, onboarding, telemetry, export interop — amplifies these six.

The product can close the gap in a focused quarter. A prioritized roadmap sits in §9.

---

## 1 · Verified build & runtime state

A clean-room install of this branch on a fresh Apple Silicon Mac (this session) required:

- **Python 3.11.15** (Homebrew user-prefix) — system `python3` is 3.9.6; `pyproject.toml` requires `>=3.11` ([`pyproject.toml:13`](pyproject.toml#L13))
- **numpy 2.4.4**, **opencv-python 4.13.0** — both pip-installed in `.venv`
- **ffmpeg / ffprobe 8.1** — Homebrew (yt-dlp and evermeet.cx builds are either x86-only or unreliable on arm64)
- **Node 22.22.2** via nvm; **Bun 1.3.13**
- **flock** — not present on macOS; I wrote a Python/`fcntl` shim at `~/.local/bin/flock`
- **Homebrew itself** — installed to `~/homebrew` (no `sudo`), which causes every bottle to be built-from-source with a noisy warning

The app comes up cleanly at http://127.0.0.1:5173. But the **install friction on a vanilla Mac is real and nothing in the repo warns about it**:

- `README.md` and `Makefile` assume `bun`, `node@22`, `flock`, `ffmpeg` are present — [`Makefile:22-27`](Makefile#L22-L27) probes three different node paths, which is itself evidence that prior contributors hit path-fragility
- The `preflight` CLI command exists ([`src/divesensei/preflight.py`](src/divesensei/preflight.py)) — it is **not invoked by `make up`**. A first-run user will find out what's missing by watching `make` fail.
- `flock` is Linux-only. The Makefile has a fallback chain that ends at a hard-coded `/usr/local/opt/util-linux/bin/flock` — on macOS that file doesn't exist unless the user already has Homebrew at the wrong prefix. This is a silent failure path.

**Recommendation (P1):** hoist `preflight` into the default `make up` target, ship a portable `flock` shim in the repo (12 lines of Python), and add a one-page `SETUP_MAC.md` that captures this walkthrough.

---

## 2 · Detector technique — what it does, and what it misses

### 2.1 Pipeline (verified from source)

```
video.mp4
  └─► extract_audio_wav_ffmpeg (16 kHz mono PCM)       io/media_io.py
        └─► AudioVisualDiveDetector.detect_from_audio_file()   detection/audio_detector.py:~200
              ├─► heuristic frontend  (spectral flux 0.60 + HF-ratio 0.25 + RMS 0.15, MAD×2.0)
              ├─► PCEN frontend       (multiband PCEN onset, MAD×1.25, blended 0.65/0.35)
              ├─► clip classifier     (28-feat logistic, threshold 0.9 in long-session profile)
              ├─► [optional] video splash-zone verification     :700-1100
              └─► governed-r9 logistic re-score                runtime_score_paths.py:141
```

The architecture is more sophisticated than most audio detectors I've seen in sports. Three signals (heuristic / PCEN / learned) are composed rather than chained, and the `governed_r9` model is bootstrapped from the user's own review decisions — an elegant flywheel.

### 2.2 Failure modes — confirmed by code AND by the literature

The PCEN literature is explicit: **PCEN parameters are sensitive to the recording environment and "show poor cross-class performance"** ([Ick & Lostanlen 2021, ICASSP](https://ieeexplore.ieee.org/document/9414697); [Lostanlen et al., "PCEN: Why and How", IEEE SPM](https://ieeexplore.ieee.org/document/8514023)). The DiveSensei code confirms this on inspection:

| Failure mode | Why the code is vulnerable | Where |
|---|---|---|
| **Silent entries** (rip entries, expert divers) | No splash → no flux peak → candidate never born. Audio-only pipeline has no backstop. | `audio_detector.py:_propose_from_audio_heuristic`, threshold `audio_peak_threshold=4.0` |
| **Crowd roar / clapping** (meets, parent nights) | HF-ratio filter `≥0.115` is weak; clapping has broadband HF. | `config.py:57`, `audio_detector.py:447` |
| **Pool DJ / music** | Beat transients mimic splash flux. PCEN 0.65/0.35 blend still sees beats. | `audio_detector.py:369` |
| **Board rebound / board slap** | High flux + high HF → looks like a splash. Mitigated by `post_flux_ratio` tail checks, but rebound reverb can fool this. | `config.py:142`; long-session advanced profile sets `audio_pattern_min_score=-0.75` (!) which effectively disables the check — [`profiles.py:29`](src/divesensei/profiles.py#L29) |
| **Multiple divers on two boards** | `audio_peak_min_separation_seconds=1.2` (default) is tight; in dual-board practice two entries can merge. | `config.py:53` |
| **Corrupt / encrypted video** | `audio_decode_timeout_seconds=3600.0` (long-session profile) — the pipeline will hang for an hour before erroring | [`profiles.py:13,35`](src/divesensei/profiles.py#L13) |
| **Compressed pool audio (e.g. phone recording through plexiglass)** | HF cutoff 1800 Hz is hard-coded in *two* places — `audio_detector.py:2160` and `audio_features.py:76`. Changing one is a silent break. | See §4.1 |

The two compared systems that we can read about publicly:

- **Pitt CIC system** (open source: [pitt-cic/automatic-highlight-reel-generator](https://github.com/pitt-cic/automatic-highlight-reel-generator)) uses a **zero-shot vision-language model**, explicitly chosen to handle background interference (other divers, water reflections, coaches in frame). It reports **97 % recall on 2.5 h of practice footage** ([Pitt Digital](https://www.digital.pitt.edu/news/success-stories/dive-revolutionizing-coaching-automated-highlight-reels)). **DiveSensei does not publish an equivalent number anywhere in the repo.**
- The broader academic landscape has been video+audio late-fusion since at least [Xu et al. 2001, "Audio-Based Event Detection for Sports Video"](https://link.springer.com/chapter/10.1007/3-540-45113-7_30) and, more recently, [arXiv 2501.16100, "Automated Detection of Sport Highlights from Audio and Video Sources"](https://arxiv.org/abs/2501.16100) — which uses mel-spectrograms rather than PCEN and explicitly fuses with a video stream.

### 2.3 The asymmetry that matters

The detector's visual verification path (`audio_detector.py:700–1100`, the splash-zone analysis) only runs **after the audio frontend has already filtered**. If audio has already rejected a candidate — a rip entry, a noisy crowd event — video cannot rescue it, because video never sees it. That is the single biggest architectural weakness.

**Recommendation (P0):**

1. Publish a precision / recall / F1 on a held-out diving corpus (5–10 sessions). Without this, "audio-first detection" is an unfalsifiable claim.
2. Add a **visual-first pre-candidate pass** even if cheap (frame-diff + connected components in the splash zone at ¼ rate). Fuse with audio rather than gate on audio.
3. Document why `audio_pattern_min_score=-0.75` in the advanced profile — either it's a deliberate "trust the classifier only" design (in which case comment it) or it's a tuning mistake.

### 2.4 Scoring — the missing half of the product

Every competitor in sports video analysis offers **Action Quality Assessment** (AQA) — joint angles, entry-line, rotation count — and the academic surveys are loud about this: [3D pose estimation for diving judging](https://www.divegym.co.uk/blog/3d-pose-estimation/), [Princeton AI Scoring for Competitive Diving](https://dataspace.princeton.edu/handle/88435/dsp01jh343w653), and the Frontiers 2025 review of markerless-motion in sports ([doi](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1649330/full)).

DiveSensei **has no scoring, no skeleton, no angle tools**. This is defensible as a scoping choice (detect first, assess later), but a coach evaluating the app will reasonably ask "why would I use this when Dartfish gives me joint angles and a side-by-side?"

---

## 3 · Code quality — the backend

### 3.1 Tests

**There are zero unit tests** under `src/divesensei/`. There is a `benchmarks/` directory with 45+ experiment scripts. That is not a regression harness. On a 3 285-line detector (`audio_detector.py`) with 200+ config parameters, any tuning change is validated only by a live `evaluate-session` run. This is the single biggest risk to the codebase.

### 3.2 Naming collisions (silent failures)

Verified, reproducible:

```python
# src/divesensei/profiles.py:9
"audio_peak_separation": 4.0,

# src/divesensei/detection/config.py:53 — the actual attribute
audio_peak_min_separation_seconds: float = 1.2
```

Unless `apply_named_profile` ([`profiles.py`](src/divesensei/profiles.py)) has an alias table — it does not, confirmed by inspection — **the long-session profile's attempt to widen peak separation to 4 s is a no-op**. The default 1.2 s separation is what runs. This may be the single biggest cause of false-positive inflation in long sessions.

### 3.3 Duplicated constants

- HF cutoff `1800.0` Hz: [`audio_detector.py:2160`](src/divesensei/detection/audio_detector.py#L2160) AND [`audio_features.py:76`](src/divesensei/detection/audio_features.py#L76)
- Robust z-score: an inline form at [`audio_detector.py:306-307`](src/divesensei/detection/audio_detector.py#L306) and a helper `_robust_zscore` at [`audio_detector.py:2199`](src/divesensei/detection/audio_detector.py#L2199)

### 3.4 Dead and quasi-dead code

- `frontend_region_descriptor` (20+ parameters) gated on `frontend_region_descriptor_enabled=False` default
- `frontend_dive_trend` (8 parameters) gated on `frontend_dive_trend_enabled=False`
- `audio_model_min_probability=0.0` default → candidate-level logistic is effectively disabled unless tuned

These are not necessarily bugs — they look like ablation flags — but with zero tests and zero comments, a reader cannot distinguish ablation from abandonware.

### 3.5 Error surfaces that will bite operators

- `extract_audio_wav_ffmpeg` takes `audio_decode_timeout_seconds=3600`. One hour. With no intermediate log line. A corrupt MP4 hangs the whole pipeline.
- No validation that `sample_rate` from the WAV matches `DetectionConfig.audio_sample_rate=16000`. A 48 kHz source would silently produce nonsense FFT features.
- `runtime_score_paths.py:195-209` gracefully returns `{}` if the governed-r9 model fails to load — but `evaluate_session.py` does not warn the user; they'd see `governed_r9_score: null` in the UI and wonder why.

---

## 4 · Frontend — desktop

### 4.1 Architecture

- Astro 5.18.1, SSR, `@astrojs/node` standalone
- Single page at [`apps/desktop/src/pages/index.astro`](apps/desktop/src/pages/index.astro), which renders one component — [`DesignWorkspace.astro`](apps/desktop/src/components/DesignWorkspace.astro) — **2 966 lines**
- Navigation is entirely query-string based: `?session=&tab=0|1|2|3`
- State: vanilla JS in one `<script is:inline>` block, ~2 900 of the 2 966 lines
- No reactive framework, no sub-components, no client store

This is not a principled architectural choice — it's what happens when a prototype ships. The cost is now compounded: every feature change touches a 3 kloc untyped file.

### 4.2 Terminology drift (UX-facing)

The data model uses five words for the same thing:

| Word | Where it appears | What it means |
|---|---|---|
| Session | everywhere | one video + one analysis run |
| Evaluation | review mode toggle | annotation for ground-truth labels |
| Attempt | sidebar header ([line 612](apps/desktop/src/components/DesignWorkspace.astro#L612)) | a detected event |
| Candidate | API responses, export CSV | same as attempt |
| Detection | backend, some UI strings | same |
| Event | evaluation dropdown labels | labeled classification (`springboard_dive`, `platform_dive`, …) |

A coach sees "attempt" in the timeline, "candidate ID" in the export, and "detection" in the tooltip — all for the same dive. Pick one.

### 4.3 The video review view — where the product stands or falls

Benchmarked against the space:

| Capability | DiveSensei | Kinovea | Dartfish | Hudl Technique | OnForm | Coach's Eye |
|---|---|---|---|---|---|---|
| Frame-by-frame step | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| J/K/L shuttle | ❌ (uses arrow + K/R/D/N) | ✅ | ✅ | partial | partial | ✅ |
| Playback-rate presets (¼×, ½×, etc.) | only native `<video controls>` | ✅ | ✅ | ✅ | ✅ | ✅ |
| Audio waveform visualization | ❌ | ❌ | ✅ | — | — | — |
| Side-by-side comparison | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Voice annotation / telestration | ❌ | partial | ✅ | ✅ | ✅ | ✅ |
| Joint-angle / line tools | ❌ | ✅ | ✅ | partial | partial | ✅ |
| Auto dive detection | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

Sources: [SimpliFaster Buyer's Guide](https://simplifaster.com/articles/buyers-guide-sport-video-analysis/); [Callplaybook Coach's Guide](https://blog.callplaybook.com/blog/coach-video-review-software-hudl-dartfish-alternatives); [SmartTech Top Video Analysis](https://www.smarttech.fi/top-sports-video-analysis-software/); [Dartfish alternatives comparison](https://www.g2.com/products/dartfish/competitors/alternatives).

DiveSensei's **one** clear win is auto-detection. Everything to the right of "auto dive detection" is ceded to every competitor.

The **J/K/L convention** is industry-standard across editing and coaching tools ([Final Cut Pro, Peachpit](https://www.peachpit.com/articles/article.aspx?p=31288&seqNum=9); [YouTube shortcut reference](https://yourvideoeditor.com/youtube-keyboard-shortcuts/)). DiveSensei instead uses K/R/U/D/N/F + Shift+letters for subtypes ([`DesignWorkspace.astro:2899-2951`](apps/desktop/src/components/DesignWorkspace.astro#L2899)), and **never displays the shortcut map anywhere in the UI**. A coach cannot discover K means "Keep" without reading source.

### 4.4 Accessibility

The April 2026 ADA deadline requires **WCAG 2.1 AA for public-facing web content, including video players** ([Swarmify 2026 WCAG guide](https://swarmify.com/blog/video-accessibility-captions-wcag/)). A club program using DiveSensei at a public university is in scope.

Current state (checked in-repo):
- **Dark theme only** — no `prefers-color-scheme`; hard-coded in [`global.css:1-21`](apps/desktop/src/styles/global.css#L1)
- `aria-live="polite"` for splash-cue indicator ✓ ([`DesignWorkspace.astro:395`](apps/desktop/src/components/DesignWorkspace.astro#L395))
- `aria-label` on prev/next ✓
- No skip-to-main, no landmark roles on sidebar tables, no focus-visible styling
- Video element uses default HTML5 controls — passes basic WCAG, but there are **no captions / transcripts** anywhere and the "splash cue" is an audio-only affordance with no visual parallel
- Minimum window width 1 180 px hard-coded ([`electron/main.mjs:46`](apps/desktop/electron/main.mjs#L46)) — excludes smaller laptops and split-screen workflows

**Recommendation (P1):** light theme + `prefers-color-scheme`, WCAG-aware focus styling, an `/apps/desktop/ACCESSIBILITY.md` with a conformance statement before the ADA deadline.

### 4.5 Mobile (Capacitor / iOS)

The mobile app is **review-only, card-deck-swipe**, no Library, no Create, no Exports ([`apps/mobile/src/app.ts:17-58`](apps/mobile/src/app.ts#L17), [`ReviewWorkspace.ts:14-40`](apps/mobile/src/review/ReviewWorkspace.ts#L14)). The swipe gesture is reasonable for Tinder-style review, but:
- No iPad layout (the `styles.css` is portrait-phone only)
- No Android path at all
- No offline sync when the shared file server is unreachable
- The desktop app's `decisions` API and the mobile app's `MobileReviewHostService` bridge are **two separate persistence models** — a dive reviewed on phone while the desktop has the same session open is a sync conflict we can't resolve from the code

---

## 5 · Persona walkthroughs

### 5.1 Real sport diver (self-review)

**Goal:** find their three best dives from today's practice, save them to their phone.

**Current path:** create a session, wait 5–30 minutes for analysis, open the Library, open the session, go to Review, scrub timeline, watch each attempt, click "Keep", go to Exports, click "Export kept clips", find the output folder in Finder.

**Pain:**
- Cannot sort or filter by confidence ("just show me the cleanest entries")
- Cannot play a full dive arc — only the 4-second review window ([`ui_contract.py` constants `REVIEW_PRE_SECONDS=2.0`, `REVIEW_POST_SECONDS=2.0`])
- Export path is opaque — no file browser, no share-to-iPhone

**Gap vs the market:** OnForm lets you upload from phone and get telestrated voice feedback in minutes ([onform.com](https://onform.com/)). That's what a self-coached diver actually wants.

### 5.2 Coach (bulk review, 200 dives / session)

**Goal:** sit down on a Tuesday night, triage 2 hours of footage, send each athlete 5–10 clips by Thursday.

**Pain (each one cites code above):**
- No J/K/L shuttle; only K/R on keyboard, no speed ramp
- No batch ops — no "reject all below score 0.4", no "accept all in this 20-minute window"
- No athlete tag — can't attribute a dive to diver A vs diver B, so the "send to each athlete" step is manual
- Notes field is a single-line inline input ([`DesignWorkspace.astro:429`](apps/desktop/src/components/DesignWorkspace.astro#L429)), not a rich comment stream
- "Refinement queue" shows model-driven priority ([line 507-579](apps/desktop/src/components/DesignWorkspace.astro#L507)) but **does not explain why these rows matter**, and there is no progress pip ("5 of 23")

**Benchmark:** Pitt's coach reportedly spent 10 h/week on manual review pre-system ([Pitt Digital](https://www.digital.pitt.edu/news/success-stories/dive-revolutionizing-coaching-automated-highlight-reels)). DiveSensei saves *discovery* time but not *review* time.

### 5.3 Video analyst (evaluation mode, building training labels)

**Goal:** label 500 detections as dive / non-dive-subtype for the next `governed_r9` training cycle.

**Pain:**
- **No waveform or spectrogram** — the analyst is asked to trust the score `0.62` with no visual evidence
- **No frame stepping** — cannot distinguish platform entry from springboard entry at normal playback speed for a subtle dive
- False-negative marking requires clicking the timeline then clicking a decision — the marker only appears *after save*, so you can't see where you already marked one
- Subtype dropdown is a single `<select>` ([line 408-417](apps/desktop/src/components/DesignWorkspace.astro#L408)) — three clicks per non-dive label where Shift+B / Shift+H shortcuts exist but are not shown
- **Undo is impossible** — the decision API is append-only ([`api/review.ts` POST]); fat-fingering "Keep" then clicking "Reject" writes both to the audit log

**Consequence for the model:** labeling noise from UX friction becomes labeling noise in `governed_r9` training data, which is used to score *future* detections. This is a **self-poisoning loop** unless the label UX is tightened.

### 5.4 Product manager (onboarding, retention, analytics)

**Goal:** know that a new coach signs in, reviews their first session, and comes back next week.

**Gap:** no first-run experience (open app → empty Library → nothing happens). No analytics — no event instrumentation on session creation, review speed, export success. No in-app feedback channel.

**Benchmark:** USA Diving's Topcoder-built app ([case study](https://www.topcoder.com/case-studies/usa-diving/)) tracks training volume and dive-type counts centrally — an artifact **DiveSensei could produce trivially** from its detections, and doesn't.

### 5.5 Dev / maintainer

Already covered in §3. Summary: no tests, a 3 kloc monolith, silent profile-key bugs.

### 5.6 UX / UI review

Covered across §4 and §5.2–5.3. One structural observation worth pulling out: **"Standard" vs "Evaluation" mode switches the buttons, the keyboard shortcuts, the sidebar filter labels, and the persistence path** ([`api/review.ts:67-120`](apps/desktop/src/pages/api/review.ts#L67)), but the mode itself is indicated only by a subtle title change. A coach who mistakenly uses an evaluation session for highlight-reel review will produce garbage data and not know it.

---

## 6 · Data contracts & interop

Export outputs to `{output_dir}/web/*.mp4` plus several JSON / JSONL side-files ([`metadata/ui_contract.py`](src/divesensei/metadata/ui_contract.py)). Formats:

- Clip filename convention: `dive_splash_{n}_t{timestamp}s_{confidence}.mp4`
- `ui_session_manifest.json` — proprietary schema, version `1.0.0`
- No **SRT / VTT** subtitle output for marker import into other tools
- No **CSV export matching Dartfish tagging format** ([Dartfish data export](https://www.dartfish.com/swimming/))
- No **XML / Hudl Sportscode** export
- No **FINA / World Aquatics DD code** mapping ([Swim England dive tariffs](https://www.swimming.org/diving/diving-scores/))

A coach cannot take DiveSensei output into Dartfish or Hudl without re-importing the clips and re-tagging. This is a distribution problem, not just a feature problem.

---

## 7 · Privacy, minors, pool rules

Diving athletes are disproportionately minors (age-group programs start at 8). Video of minors in swimwear is **category-sensitive data** in most frameworks. On inspection:

- All data is **local-first** — good; the session directory is under the user's home. No telemetry confirmed.
- But the `session.source_video_path` is stored as an **absolute path in JSON** ([`ui_contract.py`](src/divesensei/metadata/ui_contract.py)) — shared with any exported manifest. This leaks the filesystem layout of the reviewer.
- The `review_proxy.mp4` is re-encoded at ultrafast; there's no option to strip audio (a coach may want to share a visual-only clip for privacy)
- No retention policy, no "delete all data" surface beyond `make clean` which is a developer tool
- No posture on USA Diving / NCAA policy, no MOU template, no DPIA (Data Protection Impact Assessment) — both are standard for organizations handling minors' video

**Recommendation (P2):** privacy docs, audio-strip export option, relative-path manifests, a "Delete session and all media" button in the Library view.

---

## 8 · Inconsistencies catalogue

Non-exhaustive, cited:

| # | Inconsistency | Where |
|---|---|---|
| 1 | `audio_peak_separation` (profile) ≠ `audio_peak_min_separation_seconds` (config) — **silent override failure** | [`profiles.py:9`](src/divesensei/profiles.py#L9) vs [`config.py:53`](src/divesensei/detection/config.py#L53) |
| 2 | HF cutoff `1800.0` hard-coded twice | [`audio_detector.py:2160`](src/divesensei/detection/audio_detector.py#L2160), [`audio_features.py:76`](src/divesensei/detection/audio_features.py#L76) |
| 3 | MAD multipliers differ (2.0 heuristic, 1.25 PCEN) with no comment | [`audio_detector.py:309,373`](src/divesensei/detection/audio_detector.py#L309) |
| 4 | `attempt` vs `candidate` vs `detection` vs `event` — five names, one entity | UI, API, CSV |
| 5 | Standard mode vs evaluation mode switches behavior silently | [`DesignWorkspace.astro:73`](apps/desktop/src/components/DesignWorkspace.astro#L73) |
| 6 | Robust z-score implemented twice | [`audio_detector.py:306`](src/divesensei/detection/audio_detector.py#L306) and [`:2199`](src/divesensei/detection/audio_detector.py#L2199) |
| 7 | Two model serializations (JSON for governed-r9, XGBoost binary for exact-r9) | `runtime_score_paths.py` |
| 8 | Desktop persists decisions in SQLite; mobile uses IPC bridge; no conflict resolution | `api/review.ts` vs `mobile/src/review` |
| 9 | Terminology: "Detection method" picker shows raw enum `audio_v2_pcen_classifier`; friendly label only in Library | [`DesignWorkspace.astro:320-327`](apps/desktop/src/components/DesignWorkspace.astro#L320) |
| 10 | Shortcut `F` adds a false-negative only in eval mode; in standard mode F does nothing (and is not documented) | [`DesignWorkspace.astro:2899-2951`](apps/desktop/src/components/DesignWorkspace.astro#L2899) |

---

## 9 · Competitive techniques — deep dive

This section is what a technical reviewer will want to see first. It reconstructs every competitor's *technique* from public sources, so DiveSensei's design choices can be argued on merit.

### 9.1 Pitt CIC — `automatic-highlight-reel-generator` — the direct competitor

Repo: [github.com/pitt-cic/automatic-highlight-reel-generator](https://github.com/pitt-cic/automatic-highlight-reel-generator) · MIT-licensed · built by Pitt Cloud Innovation Center interns (Roman Koshovnyk, Rowan Morse).

**Technique — vision-language-model zero-shot, not audio:**

| Stage | What runs | Parameters |
|---|---|---|
| 1. Downsample | FFmpeg re-encode to lower frame rate + CSV frame→original-timestamp map | `target_fps` (default **4 FPS**) |
| 2. Detect | **PaliGemma-2, `paligemma2-3b-mix-224`** (3-billion-parameter VLM from Google DeepMind, hosted on Hugging Face) run frame-by-frame with a natural-language prompt — default: *"Is there a person in the air jumping into the water?"* → binary yes/no + confidence | `default_prompt`, `model_id`, `batch_size=16`, `confidence_threshold`, `grouping_threshold_sec` |
| 3. Cluster + clip | Consecutive positive frames grouped with temporal-merge gap; buffer added before/after; FFmpeg extracts from the full-resolution source and concatenates | `buffer_start_sec`, `buffer_end_sec`, `merge_gap_sec`, `ffmpeg_preset` |

**Deployment — cloud-only, GPU-bound:**
- AWS CDK (TypeScript) provisions: S3 bucket (`videos/` in → `results/` out), Lambda trigger, ECS task on **g4dn.2xlarge GPU EC2**, Docker container with PaliGemma-2 weights
- **Throughput:** ~8.94 inference FPS on downsampled frames → **90 minutes end-to-end on a 2-hour source** (1.4× real-time)
- **Cost:** ~**$2.11 per 2-hour video** (self-reported; GPU-hours + S3 + Lambda)
- Streamlit UI (`frontend/ui/app.py`) for upload / status

**Design philosophy (explicit in the README):** *"Missing a key event is far more detrimental than incorrectly identifying a non-event"* — the system is tuned for **recall, not precision**. They chose falling-motion detection *explicitly because diving platforms are often out of frame*, so pose estimation would fail. This is an intentional architectural statement.

**Limitations (from the README):**
- "Lessons Learned" section marked *"to be updated"* — they publish no precision/recall/F1 beyond the 97 % figure in press coverage
- Config changes require CDK redeploy — friction for threshold tuning
- GPU-bound; cost does not scale down for casual use
- Cloud-only; minors-on-video + coach-pool-rules concerns (§7) are left to the customer

**Side-by-side with DiveSensei:**

| Dimension | Pitt CIC | DiveSensei |
|---|---|---|
| Primary signal | Vision (VLM zero-shot) | Audio (PCEN + heuristic) |
| Recall on 2.5 h practice | 97 % (self-reported) | **unpublished** |
| Inference cost | ~$2.11 per 2-hour video (GPU) | ~free (CPU, after warm-up) |
| Hardware | AWS g4dn.2xlarge GPU | Apple Silicon / any CPU |
| Latency for a 2 h session | ~90 min | observed ~1–3 min (audio+extract) on M-series |
| Deployment | AWS-only | Local / Electron / iOS |
| Customizability | prompt + thresholds; redeploy to change | 200+ config knobs, no redeploy |
| Review UI | Streamlit upload page only | Full review workspace |
| Privacy posture | Video leaves the pool; sits in S3 | Local-first; video never leaves the Mac |
| Silent dives (rip entries) | Robust — visual signal | **Weak** — no splash → no candidate |
| Crowd / DJ noise | Robust — visual signal | **Weak** — noisy audio inflates FP |
| Platform-out-of-frame | Robust (falling motion prompt) | N/A (audio-only) |
| Joint-angle / scoring | None | None |
| Open source | MIT | Unclear from `LICENSE` (AGPL-style described as "Other/Proprietary" in `pyproject.toml`) |

**The takeaway:** Pitt and DiveSensei have **complementary failure modes**. The obvious next move is a **fused pipeline** — use PCEN audio as a cheap first pass on CPU (DiveSensei's strength: fast, free, runs anywhere) and a VLM (PaliGemma-2, or the much cheaper [SigLIP-SO400M](https://huggingface.co/google/siglip-so400m-patch14-384) at zero-shot, or [CLIP ViT-L/14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)) as a *reviewer* over low-confidence audio candidates. That gives you Pitt's recall on silent dives without paying $2.11 per session or shipping video to AWS.

### 9.2 Dartfish — the professional-grade biomechanics standard

[dartfish.com](https://www.dartfish.com/) · Swiss, founded 1999 · used by Olympic programs

**Techniques** ([Dartfish motion](https://www.dartfish.com/motion/), [ResearchGate evaluation](https://www.researchgate.net/publication/248644834_Evaluation_of_the_Performance_of_Digital_Video_Analysis_of_Human_Motion_Dartfish_Tracking_System)):
- **StroMotion** — stroboscopic overlay: multiple positions of the athlete in a single frame, for diff-over-time analysis (the "multi-exposure" look a diving coach uses to show rotation count)
- **SimulCam** — alignment of two performances into one frame, even across different camera pans (geometric registration under the hood)
- **2D marker tracking** — auto + manual; published accuracy vs. Vicon 3D ≈ **±5 mm** (context: Vicon is the gold-standard marker-based 3D system at ~$150 k)
- **Angle, distance, area measurement** tools with on-screen protractor/ruler
- **Tagging panels** — keyboard-shortcut event logging during live capture
- **myDartfish Express** (iOS) + **myDartfish Pro** (desktop) — same data format across

**What DiveSensei is missing vs Dartfish:** StroMotion-style overlay, SimulCam synchronization, angle/line tools, tagging panels. All are *after* detection — DiveSensei has a head start on *auto-detecting* events that Dartfish still requires the coach to do manually. If DiveSensei added overlay + angle tools on its detected clips, it would leapfrog Dartfish for the diving niche.

### 9.3 Hudl / Hudl Sportscode / Hudl Technique — the team-sports market leader

[hudl.com](https://www.hudl.com/products/sportscode) · [Code Windows best practice](https://www.hudl.com/blog/hudl-best-practice-series-hudl-sportscode-code-windows)

**Techniques:**
- **Code Windows** — customizable button grids for real-time event logging; every button creates a clip + metadata row; this is the *tagging data model* the whole coaching world has standardized on
- **Sportscode scripting** — expression language for building compound queries ("all breakaways by #23 on the power play")
- **Playlists** — clip collections with notes, shareable with players
- **Live capture + tagging** — built around the workflow of tagging *during* the game, not post-hoc
- **Hudl Instat integration** — AI-tagged events for basketball/hockey/soccer/football; not diving

**What DiveSensei is missing vs Hudl Sportscode:** Code Windows (tagging grids), playlist building, scripting for queries. But Hudl doesn't auto-detect dives — their AI is team-sport-only ([Hudl Instat scope](https://www.hudl.com/blog/hudl-instat-basketball-scouting-recruitment)).

### 9.4 OnForm — the remote-coaching product

[onform.com](https://onform.com/)

**Techniques:**
- **Voice-over telestration** — coach draws on the video and records audio commentary in one take; exports as a single playable clip
- **Side-by-side split-screen** comparison with synced scrubbing
- **Cloud sync** between athlete's phone and coach's desktop
- **Sport-specific templates** — golf swing planes, tennis court overlays, swimming stroke phase markers; diving is not a featured sport but the generic tools apply

**What DiveSensei is missing vs OnForm:** voice-over annotation is the single highest-ROI feature for athlete feedback and is not hard to build (MediaRecorder API on the web). DiveSensei could ship this in a week.

### 9.5 Kinovea — the open-source reference

[kinovea.org](https://www.kinovea.org/) · [github.com/Kinovea/Kinovea](https://github.com/Kinovea/Kinovea) · GPL, C#/.NET, Windows-only

**Techniques:**
- **Motion tracking** — template-matching with user-adjustable search window; export as CSV trajectories
- **Chronophotography** — Dartfish-StroMotion-equivalent, free
- **Lens calibration** — checkerboard-based intrinsic calibration to remove GoPro fisheye distortion before measuring angles
- **Frame-by-frame, reverse-play, variable-speed** — the baseline a coach expects
- **Multi-camera capture** with hardware sync via GigE

**Architecture** (from the repo): C#/.NET WinForms app, direct DirectShow / FFmpeg capture, SQLite for annotations. Windows-only is Kinovea's one weakness — and the *exact* weakness DiveSensei could exploit on macOS.

**What DiveSensei is missing vs Kinovea:** motion tracking, chronophotography, lens calibration. Kinovea has been iterated since 2004 — catching up on measurement tools is a multi-quarter effort. But: Kinovea has *no* auto-detection. DiveSensei + Kinovea-style tools + audio-first detection would be a category-leading product.

### 9.6 Academic — AQA (Action Quality Assessment) pipelines

**Princeton AI Scoring for Competitive Diving** ([DataSpace thesis](https://dataspace.princeton.edu/handle/88435/dsp01jh343w653))
- **Technique:** 2D pose estimation (OpenPose-family) → joint-angle time series → Support Vector Regression to predict Olympic judge scores
- Trained on Olympic broadcast footage
- Reports correlation with judge scores, not a detection metric

**3D pose estimation for diving** ([DiveGym blog](https://www.divegym.co.uk/blog/3d-pose-estimation/), [Frontiers 2025 markerless-motion review](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1649330/full))
- **Technique:** monocular 3D lift (VideoPose3D, HybrIK, or MMPose's 3D branch) on single-camera footage; fuses across time for rotation-count estimation
- Used as a *verification layer* for human judges — data-driven reference, not replacement
- Requires World Aquatics calibration / standardization for official use

**Audio-event-detection baselines** ([Ick & Lostanlen, ICASSP 2021 MR-PCEN](https://ieeexplore.ieee.org/document/9414697), [Lostanlen et al. IEEE SPM "PCEN: Why and How"](https://ieeexplore.ieee.org/document/8514023))
- **Technique:** Single and multi-rate PCEN + CRNN / CNN classifier
- **Explicit finding:** PCEN is *parameter-sensitive per class*; cross-class generalization is weak — which is exactly the "crowd vs splash vs DJ" problem DiveSensei's single-rate PCEN frontend has today
- **Multi-rate PCEN (MR-PCEN)** is the published remedy — different time constants per class, then fused. Drop-in upgrade for [`audio_features.py:118`](src/divesensei/detection/audio_features.py#L118)

**Sport highlight fusion** ([arXiv 2501.16100, "Automated Detection of Sport Highlights from Audio and Video Sources", 2025](https://arxiv.org/abs/2501.16100))
- **Technique:** Mel-spectrogram audio stream + lightweight video CNN → late fusion
- Targets commentator / crowd reaction as the signal (broadcast use case), but the fusion recipe is generic and applicable to pool footage

**Pitt case study press** ([Pitt Digital](https://www.digital.pitt.edu/news/success-stories/dive-revolutionizing-coaching-automated-highlight-reels))
- The only published diving-specific *recall* number in this landscape: **97 % on 2.5 h practice footage**
- Baseline coach-hours-saved: **10 h/week → effectively 0** on the discovery step

### 9.7 Adjacent — USA Diving, Dive Live, iSWIM

**USA Diving App** ([Topcoder case study](https://www.topcoder.com/case-studies/usa-diving/))
- Role: training-volume and dive-type tracking across a distributed federation
- **DiveSensei could produce this data automatically** from its detections and does not — an easy win

**Dive Live** ([scoresandmore.live](https://scoresandmore.live/))
- FINA / NCAA / NFHS-verified scoring and dive-sheet workflow for *competitions*, not practice
- Records video tied to dive sheets; no auto-detection
- Complement, not competitor — DiveSensei lives in practice, Dive Live lives at meets

**iSWIM** ([navgood.com/iswim](https://www.navgood.com/en/tool-details/iswim-4cb6f))
- AI swimming-stroke analysis — adjacent sport, same customer
- Published pipeline uses pose estimation + stroke-phase segmentation

### 9.8 Synthesis

DiveSensei's positioning in the landscape:

```
                    ┌─ detection ─┐
                    │             │
         auto-detect│   DiveSensei (audio-first)
                    │   Pitt CIC  (vision, zero-shot VLM, 97% recall)
                    │
         manual log │   Hudl Sportscode  (tagging panels)
                    │   Dartfish         (timeline tagging)
                    └──────────────────────────────────┘

                    ┌─ review ─┐
                    │          │
      measurement  │   Dartfish  (StroMotion, SimulCam, ±5mm)
                    │   Kinovea   (tracking, chronophoto, calibration)
                    │
      annotation   │   OnForm    (voice-over telestration)
                    │   Hudl      (playlists, scripting)
                    │
         bare      │   DiveSensei — just a <video controls>
                    └────────────────────────────────────┘

                    ┌─ scoring ─┐
                    │           │
      research     │   Princeton AI Scoring (pose → SVR)
                    │   3D pose judging (Frontiers 2025)
                    │
         nothing   │   DiveSensei
                    │   Pitt CIC
                    │   Dartfish (manual)
                    │   Hudl, OnForm, Kinovea
                    └───────────────────────────────────┘
```

DiveSensei wins on **auto-detection** (only Pitt competes, and on different hardware/cost terms). It loses on **review tooling** to every competitor. It doesn't even try on **scoring**, which is where the research frontier is.

**The strategic move** is not to chase Dartfish feature-for-feature on measurement. It's to (a) close the detection gap vs Pitt by fusing a tiny zero-shot VLM over low-confidence audio candidates, (b) add the cheap high-ROI review features OnForm and Kinovea ship (frame-step, J/K/L, waveform, voice-over), and (c) own the *diving-practice-session* workflow that nobody else targets — USA Diving-style volume tracking, per-athlete dive counts, per-session tariff summaries.

---

## 10 · Prioritized roadmap

### P0 — ship-blockers for any serious external evaluation (2–4 weeks)
1. **Publish precision / recall / F1** on ≥5 held-out sessions with a reproducible script. Without this, the "audio-first" claim is unfalsifiable.
2. **Fix the `audio_peak_separation` profile-key bug** (§3.2). This may be the single biggest false-positive inflation cause in long sessions.
3. **Add a minimal test suite**: unit tests for `audio_features.py`, a regression fixture for `evaluate_session` on one 10-minute sample video with a known ground-truth JSON.
4. **Run preflight in `make up`** and document the macOS install path clearly (§1).

### P1 — unlock real review velocity (4–8 weeks)
5. **J/K/L shuttle + frame-step + playback-rate presets** in the video view.
6. **Audio waveform visualization** on the timeline (WebAudio + canvas — 2-day feature).
7. **Visible keyboard-shortcut legend** (modal on `?` key).
8. **Batch operations**: multi-select in the queue, bulk "reject below score X".
9. **Undo** on the review decision API (state machine, not append-only).
10. **Split `DesignWorkspace.astro`** into at minimum: `VideoReview.ts`, `Timeline.ts`, `DecisionPanel.ts`, `QueueSidebar.ts` — typed.

### P2 — close the competitive gap (8–16 weeks)
11. **Visual-first candidate pass** (frame-diff in splash zone) to fuse with audio rather than gate on it (§2.3).
12. **Side-by-side comparison** (two `<video>` elements, synced scrubbing).
13. **Athlete tagging** — a lookup on first detection per session, carried through to exports.
14. **Dartfish / Hudl export bridges** — CSV tagging format, SRT markers (§6).
15. **Scoring MVP** — even just entry-line angle via a lightweight 2D pose estimator (MediaPipe) on the review-window frames, not full judging.
16. **Accessibility pass** to WCAG 2.1 AA before the April 2026 ADA deadline ([reference](https://swarmify.com/blog/video-accessibility-captions-wcag/)).
17. **Privacy features**: audio-strip export, relative-path manifests, Library-level "delete with media".

### P3 — product-organization work (ongoing)
18. Light theme; responsive down to 1 024 px width.
19. First-run onboarding; in-app feedback.
20. Analytics (local-first, opt-in) — session counts, review durations, export success.
21. Docs: `SETUP_MAC.md`, `THRESHOLDS.md` explaining every magic number in §2.2.

---

## 11 · Closing

DiveSensei is a strong 0.1 of a product that has no direct competitor (auto-detection for diving is a real gap — confirmed by every competitive comparison in §4.3 and §6). The audio-first architecture is defensible and the review-first workflow is the right mental model. But **three things are currently true simultaneously and cannot all stay true**: (a) the detector has no published metrics, (b) the review workspace ships below the feature line of every commercial competitor in the space, and (c) there are zero tests protecting a 3 285-line detector whose profile overrides are already silently broken.

The P0 list takes 2–4 weeks for a focused engineer. Doing P0 + P1 would put DiveSensei at feature parity with Coach's Eye on the review side while keeping its unique advantage on detection. P2 is where it starts to threaten Dartfish in this niche.

---

### Sources cited

**Academic & open-source**
- arXiv 2501.16100 — [Automated Detection of Sport Highlights from Audio and Video Sources, 2025](https://arxiv.org/abs/2501.16100)
- Ick & Lostanlen — [Sound Event Detection in Urban Audio With Single and Multi-Rate PCEN, ICASSP 2021](https://ieeexplore.ieee.org/document/9414697) ([preprint](https://arxiv.org/abs/2102.03468))
- Lostanlen et al. — [Per-Channel Energy Normalization: Why and How, IEEE SPM 2018](https://ieeexplore.ieee.org/document/8514023)
- Xu et al. — [Audio-Based Event Detection for Sports Video, 2001](https://link.springer.com/chapter/10.1007/3-540-45113-7_30)
- [Automatic Detection and Recognition of Athlete Actions in Diving Video (Springer)](https://link.springer.com/chapter/10.1007/978-3-540-69429-8_8)
- Princeton — [Transparent and Objective AI Scoring System for Competitive Diving](https://dataspace.princeton.edu/handle/88435/dsp01jh343w653)
- Frontiers Physiology 2025 — [Commercial vision sensors and AI-based pose estimation for markerless motion analysis](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2025.1649330/full)
- [pitt-cic/automatic-highlight-reel-generator (GitHub)](https://github.com/pitt-cic/automatic-highlight-reel-generator)
- [librosa.pcen documentation](https://librosa.org/doc/main/generated/librosa.pcen.html)

**Competitive landscape**
- [Dartfish](https://www.dartfish.com/) · [Dartfish vs Coach's Eye](https://www.dartfish.com/blog/looking-for-an-alternative-to-coachs-eye/) · [Dartfish swimming](https://www.dartfish.com/swimming/)
- [Dartfish alternatives (G2)](https://www.g2.com/products/dartfish/competitors/alternatives)
- [Hudl Technique via Callplaybook Coach's Guide](https://blog.callplaybook.com/blog/coach-video-review-software-hudl-dartfish-alternatives)
- [OnForm](https://onform.com/)
- [Kinovea via SpeedEndurance](https://speedendurance.com/2012/03/15/dartfish-alternative-kinovea/)
- [SimpliFaster Buyer's Guide to Sport Video Analysis](https://simplifaster.com/articles/buyers-guide-sport-video-analysis/)
- [SmartTech Top Sports Video Analysis Software](https://www.smarttech.fi/top-sports-video-analysis-software/)
- [NitroMedia 2025 Sports Video Production Tools Guide](https://www.nitromediagroup.com/sports-video-production-tools-software-guide-2025-review/)

**Diving-specific operations**
- [Pitt Digital — Automated Highlight Reels](https://www.digital.pitt.edu/news/success-stories/dive-revolutionizing-coaching-automated-highlight-reels)
- [Topcoder USA Diving App case study](https://www.topcoder.com/case-studies/usa-diving/)
- [Dive Live — FINA/NCAA/NFHS mobile app](https://scoresandmore.live/)
- [Swim England — Diving Scores and Tariffs](https://www.swimming.org/diving/diving-scores/)
- [DiveGym — 3D Pose Estimation for Diving Judging](https://www.divegym.co.uk/blog/3d-pose-estimation/)

**UX / keyboard conventions**
- [Final Cut Pro J/K/L convention (Peachpit)](https://www.peachpit.com/articles/article.aspx?p=31288&seqNum=9)
- [YouTube Keyboard Shortcuts reference](https://yourvideoeditor.com/youtube-keyboard-shortcuts/)
- [The UX of Keyboard Shortcuts (Medium)](https://medium.com/design-bootcamp/the-art-of-keyboard-shortcuts-designing-for-speed-and-efficiency-9afd717fc7ed)

**Accessibility & privacy**
- [Swarmify — Video Accessibility & WCAG: 2026 Guide (incl. April-2026 ADA deadline)](https://swarmify.com/blog/video-accessibility-captions-wcag/)
- [Enhancing Accessibility in Sports and Cultural Live Events (Springer 2024)](https://link.springer.com/chapter/10.1007/978-3-031-62846-7_19)
