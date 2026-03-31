# Mobile Scaffold Handoff

Branch:

- `feat/mobile-capacitor-scaffold`

Goal of this handoff:

- continue the mobile scaffold work on a Mac with full Xcode platform support
- validate the generated Capacitor iOS project on simulator or device
- continue from the first real native persistence and review-proxy spike

## What Exists On This Branch

Scaffolded mobile app:

- `apps/mobile/`

Key files:

- [apps/mobile/package.json](/home/mcauchy/divesensei-app/apps/mobile/package.json)
- [apps/mobile/capacitor.config.ts](/home/mcauchy/divesensei-app/apps/mobile/capacitor.config.ts)
- [apps/mobile/src/native/types.ts](/home/mcauchy/divesensei-app/apps/mobile/src/native/types.ts)
- [apps/mobile/src/native/client.ts](/home/mcauchy/divesensei-app/apps/mobile/src/native/client.ts)
- [apps/mobile/src/host/types.ts](/home/mcauchy/divesensei-app/apps/mobile/src/host/types.ts)
- [apps/mobile/src/host/mobileHost.ts](/home/mcauchy/divesensei-app/apps/mobile/src/host/mobileHost.ts)
- [apps/mobile/src/review/ReviewWorkspace.ts](/home/mcauchy/divesensei-app/apps/mobile/src/review/ReviewWorkspace.ts)
- [apps/mobile/src/app.ts](/home/mcauchy/divesensei-app/apps/mobile/src/app.ts)

Native plugin source:

- [apps/mobile/ios/App/App/Plugins/DiveSenseiMedia/DiveSenseiMediaPlugin.swift](/home/mcauchy/divesensei-app/apps/mobile/ios/App/App/Plugins/DiveSenseiMedia/DiveSenseiMediaPlugin.swift)

Generated iOS project:

- [apps/mobile/ios/App/App.xcodeproj/project.pbxproj](/home/mcauchy/divesensei-app/apps/mobile/ios/App/App.xcodeproj/project.pbxproj)
- [apps/mobile/ios/App/App.xcworkspace/contents.xcworkspacedata](/home/mcauchy/divesensei-app/apps/mobile/ios/App/App.xcworkspace/contents.xcworkspacedata)
- [apps/mobile/ios/App/Podfile](/home/mcauchy/divesensei-app/apps/mobile/ios/App/Podfile)

Docs:

- [docs/mobile/v1-shape.md](/home/mcauchy/divesensei-app/docs/mobile/v1-shape.md)

## Current State

Already done:

- Capacitor/Vite app scaffold exists under `apps/mobile`
- bridge contract is the current source of truth
- browser shim exists for smoke testing without native iOS
- host abstraction exists so the mobile UI does not depend on desktop `/api/*`
- real Capacitor iOS project has been generated under `apps/mobile/ios`
- Swift plugin file is included in the generated app target
- `npm install` and `npm run build` both succeed
- `npx cap sync ios` succeeds when run with UTF-8 locale and the external Xcode `DEVELOPER_DIR`
- first native persistence layer is in place for:
  - sources
  - sessions
  - manifests
  - decisions
- first real native media spike is in place for:
  - `pickSourceVideo`
  - `getSourceAvailability`
  - `repairSource`
  - `getReviewProxy`

Not yet verified:

- Xcode compile from this machine
- Photos picker flow on simulator/device
- end-to-end relaunch and repair flow on hardware

Current blocker:

- the full Xcode app is on `/Volumes/DiveRecorderGPT/Applications/Xcode.app`
- that Xcode 16.3 bundle opens the workspace, but `xcodebuild` currently reports no available iOS destinations
- `xcrun simctl list devices available` returns no devices
- `xcodebuild` reports `iOS 18.4 is not installed` for the scheme destination

## Exact Next Commands On Mac

From repo root:

```sh
git fetch origin
git checkout feat/mobile-capacitor-scaffold
```

Install mobile dependencies:

```sh
cd apps/mobile
npm install
```

If CocoaPods is not available yet:

```sh
brew install cocoapods
```

Build the web bundle:

```sh
npm run build
```

Generate the iOS project:

```sh
npx cap add ios
```

If `ios/` already exists but is only a partial scaffold, back up the plugin file, remove `apps/mobile/ios`, rerun `npx cap add ios`, then restore the plugin file into:

```sh
apps/mobile/ios/App/App/Plugins/DiveSenseiMedia/DiveSenseiMediaPlugin.swift
```

Install native platform dependency if missing:

```sh
npm install @capacitor/ios
```

Sync after generation:

```sh
RUBYOPT='-EUTF-8:UTF-8' LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
DEVELOPER_DIR=/Volumes/DiveRecorderGPT/Applications/Xcode.app/Contents/Developer \
npx cap sync ios
```

Open in Xcode:

```sh
RUBYOPT='-EUTF-8:UTF-8' LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
DEVELOPER_DIR=/Volumes/DiveRecorderGPT/Applications/Xcode.app/Contents/Developer \
npx cap open ios
```

## Important iOS Integration Check

After regeneration, confirm that this file is included in the generated app target:

- `apps/mobile/ios/App/App/Plugins/DiveSenseiMedia/DiveSenseiMediaPlugin.swift`

If Capacitor generation changes folder contents, keep this file in the generated project under:

- `ios/App/App/Plugins/DiveSenseiMedia/DiveSenseiMediaPlugin.swift`

inside the `apps/mobile` iOS project tree.

The current intent is:

- plugin source lives with the mobile app
- not as a detached repo-root artifact

## First Validation Checklist In Xcode

1. Install the missing iOS platform/runtime in the external Xcode bundle so the `App` scheme has a valid destination.
2. Build succeeds with Capacitor and the plugin file included.
3. App launches to the smoke page.
4. `Pick Source` triggers the native Photos picker.
5. Picking a video returns a payload to JS.
6. `getSourceAvailability` resolves the returned `sourceRef`.
7. Create a session and confirm it appears again after relaunch.
8. Open review UI, save decisions, relaunch, and confirm decision persistence.
9. Call `getReviewProxy` and confirm the returned URL plays in the web review surface.
10. Delete or de-scope the original Photos asset and validate `missing` / repair behavior.

## Known Gaps To Address Next

1. The native project still needs a successful Xcode compile on this machine once the iOS runtime/platform is installed.
2. Placeholder manifest generation is persisted, but it is still synthetic review data rather than detector output.
3. `startAnalysis` is still a placeholder state transition, not real analysis.
4. Review proxy generation exists only for Photos-backed sources.
5. Export is still unimplemented.

## Scope Guard

Do next:

- install the missing iOS runtime/platform in Xcode
- compile the plugin
- validate the Photos/source/session/review-proxy flow end to end
- verify relaunch, missing-asset, permission-change, and limited-library behavior on simulator or device

Do not do next:

- detector parity work
- export pipeline work
- full desktop UI port
- architecture redesign
