import type { CapacitorConfig } from "@capacitor/cli";

const config: CapacitorConfig = {
  appId: "com.divesensei.mobile",
  appName: "DiveSensei Mobile",
  webDir: "dist",
  bundledWebRuntime: false,
  ios: {
    path: "ios",
  },
};

export default config;
