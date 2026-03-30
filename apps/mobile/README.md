# DiveSensei Mobile

This is the Capacitor mobile app scaffold for the iOS-first mobile product.

## Install

```sh
cd apps/mobile
npm install
```

## Run The Web Bundle

```sh
cd apps/mobile
npm run dev
```

Default dev URL:

- `http://127.0.0.1:4173`

## Build The Web Bundle

```sh
cd apps/mobile
npm run build
```

## Generate / Sync iOS

From the mobile app root:

```sh
cd apps/mobile
npm install
npx cap add ios
npx cap sync ios
npx cap open ios
```

If the iOS project already exists:

```sh
cd apps/mobile
npx cap sync ios
npx cap open ios
```

## Native Plugin Location

The app-owned native plugin source lives at:

- `apps/mobile/ios/App/App/Plugins/DiveSenseiMedia/DiveSenseiMediaPlugin.swift`

If `npx cap add ios` regenerates the project, keep that file mirrored into the generated app target at the same relative path:

- `ios/App/App/Plugins/DiveSenseiMedia/DiveSenseiMediaPlugin.swift` inside the generated Capacitor iOS project under `apps/mobile/ios`

## Current Status

- typed bridge contract exists
- web-side bridge client exists
- typed host abstraction exists
- browser smoke page exists
- Swift plugin scaffold exists in the mobile-owned iOS path
- native media methods are still mostly stubbed except for the source-pick spike path being the next target
