# Mobile Scaffold Handoff

Branch:

- `feat/mobile-capacitor-scaffold`

Goal of this handoff:

- continue the mobile scaffold work on a Mac with Xcode installed
- generate the real Capacitor iOS project
- compile and validate the first native source-pick flow

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

Docs:

- [docs/mobile/v1-shape.md](/home/mcauchy/divesensei-app/docs/mobile/v1-shape.md)

## Current State

Already done:

- Capacitor/Vite app scaffold exists under `apps/mobile`
- bridge contract is the current source of truth
- browser shim exists for smoke testing without native iOS
- host abstraction exists so the mobile UI does not depend on desktop `/api/*`
- Swift plugin file exists in the mobile-owned iOS path
- first native spike started for:
  - `pickSourceVideo`
  - `getSourceAvailability`
  - `repairSource`

Not yet verified:

- `npm install` / Vite build in `apps/mobile`
- generated Capacitor iOS project
- Xcode compile
- Photos picker flow on simulator/device

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

Run the browser scaffold:

```sh
npm run dev
```

Build the web bundle:

```sh
npm run build
```

Generate the iOS project:

```sh
npx cap add ios
```

Sync after generation:

```sh
npx cap sync ios
```

Open in Xcode:

```sh
npx cap open ios
```

## Important iOS Integration Check

After `cap add ios`, confirm that this file is included in the generated app target:

- `apps/mobile/ios/App/App/Plugins/DiveSenseiMedia/DiveSenseiMediaPlugin.swift`

If Capacitor generation changes folder contents, keep this file in the generated project under:

- `ios/App/App/Plugins/DiveSenseiMedia/DiveSenseiMediaPlugin.swift`

inside the `apps/mobile` iOS project tree.

The current intent is:

- plugin source lives with the mobile app
- not as a detached repo-root artifact

## First Validation Checklist In Xcode

1. Build succeeds with Capacitor and the plugin file included.
2. App launches to the smoke page.
3. `Pick Source` triggers the native Photos picker.
4. Picking a video returns a payload to JS.
5. `getSourceAvailability` resolves the returned `sourceRef`.
6. Relaunch app and confirm `sourceRef` persistence still works.

## Known Gaps To Address Next

1. Real session persistence is still stubbed.
2. `repairSource` currently returns a new source payload but does not yet rewrite a durable session record.
3. `createSession`, `listSessions`, `getSessionManifest`, and export flows are still placeholder-native or browser-shim backed.
4. No actual generated Xcode project has been committed yet.

## Scope Guard

Do next:

- get the iOS project generated
- compile the plugin
- validate the Photos/sourceRef spike end to end

Do not do next:

- detector parity work
- export pipeline work
- full desktop UI port
- architecture redesign
