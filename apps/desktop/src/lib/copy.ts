export type WorkspaceLocale = "en";

export interface WorkspaceCopy {
  tabs: [string, string, string, string];
  sessions: {
    heading: string;
    subtitle: string;
    newRunHeading: string;
    newRunSubtitle: string;
    currentSessionHeading: string;
    noSessionSelected: string;
    idleStatus: string;
    chooseVideoStep: string;
    startRunStep: string;
    reviewExportStep: string;
    emptySessionNote: string;
    noAttemptsFound: string;
    nextActionReady: string;
    nextActionPending: string;
    defaultAction: string;
  };
  fields: {
    sessionName: string;
    videoFile: string;
    sessionType: string;
    detector: string;
  };
  actions: {
    chooseFile: string;
    startAnalysis: string;
    open: string;
    rename: string;
    delete: string;
    openReview: string;
    openDiagnostics: string;
    keep: string;
    reject: string;
    unsure: string;
    exportKeptClips: string;
  };
  review: {
    attemptsHeading: string;
    allFilter: string;
    pendingFilter: string;
    keptFilter: string;
    rejectedFilter: string;
    unsureFilter: string;
    notePlaceholder: string;
    noDecisionYet: string;
  };
  exports: {
    heading: string;
    subtitle: string;
    idleStatus: string;
    readyHeading: string;
    exportedHeading: string;
    progressHeading: string;
    progressSubtitle: string;
    recentSessionsHeading: string;
    recentSessionsSubtitle: string;
  };
  diagnostics: {
    tab: string;
    heading: string;
    subtitle: string;
    previewHeading: string;
    previewSubtitle: string;
  };
  messages: {
    filePickerIdle: string;
    fileSelectedSuffix: string;
    reviewListHint: string;
    noDecisionSaved: string;
    noDecisionDetail: string;
    noDecisionsYet: string;
    noNoteAdded: string;
    noPendingExports: string;
    noExportedClips: string;
    noRecentSessions: string;
  };
}

const en: WorkspaceCopy = {
  tabs: ["Sessions", "Review", "Exports", "Diagnostics"],
  sessions: {
    heading: "Sessions",
    subtitle: "Start an analysis or reopen an existing session.",
    newRunHeading: "Start analysis",
    newRunSubtitle: "Choose a video, select a detector, and start a new analysis.",
    currentSessionHeading: "Selected session",
    noSessionSelected: "No session selected",
    idleStatus: "Ready to start.",
    chooseVideoStep: "Choose a video",
    startRunStep: "Start the analysis",
    reviewExportStep: "Review and export",
    emptySessionNote: "Start an analysis from the left. This panel will show session details after the run starts.",
    noAttemptsFound: "Analysis completed, but no dive was detected in this session.",
    nextActionReady: "Open Review to confirm detections, then export the clips you keep.",
    nextActionPending: "Wait for the analysis to finish, then open Review.",
    defaultAction: "Choose a video, keep the default detector, and start the analysis.",
  },
  fields: {
    sessionName: "Session name",
    videoFile: "Video file",
    sessionType: "Session type",
    detector: "Detector",
  },
  actions: {
    chooseFile: "Choose file",
    startAnalysis: "Start analysis",
    open: "Open",
    rename: "Rename",
    delete: "Delete",
    openReview: "Open review",
    openDiagnostics: "Open diagnostics",
    keep: "Keep",
    reject: "Reject",
    unsure: "Unsure",
    exportKeptClips: "Export kept clips",
  },
  review: {
    attemptsHeading: "Attempts",
    allFilter: "All",
    pendingFilter: "Pending",
    keptFilter: "Kept",
    rejectedFilter: "Rejected",
    unsureFilter: "Unsure",
    notePlaceholder: "Add a note (optional)",
    noDecisionYet: "No decision saved for this attempt.",
  },
  exports: {
    heading: "Exports",
    subtitle: "Only clips marked Keep appear here.",
    idleStatus: "Keep the clips you want, then export them.",
    readyHeading: "Marked Keep",
    exportedHeading: "Exported clips",
    progressHeading: "Review status",
    progressSubtitle: "Track what is kept, rejected, exported, or still pending.",
    recentSessionsHeading: "Recent sessions",
    recentSessionsSubtitle: "Sessions you opened recently.",
  },
  diagnostics: {
    tab: "Diagnostics",
    heading: "Diagnostics",
    subtitle: "Saved files and technical output for the selected session.",
    previewHeading: "Preview",
    previewSubtitle: "Preview the key output files without leaving the app.",
  },
  messages: {
    filePickerIdle: "No file selected. You can still paste a path manually.",
    fileSelectedSuffix: "selected. It will be copied into the local runtime when you start the analysis.",
    reviewListHint: "Choose an attempt, review it, and save your decision.",
    noDecisionSaved: "No decision saved for this attempt.",
    noDecisionDetail: "No decision yet",
    noDecisionsYet: "No decisions yet",
    noNoteAdded: "No note added.",
    noPendingExports: "No clips are waiting for export. Mark clips as Keep in Review first.",
    noExportedClips: "No exported clips yet.",
    noRecentSessions: "No recent sessions.",
  },
};

export function getWorkspaceCopy(locale: WorkspaceLocale = "en"): WorkspaceCopy {
  if (locale === "en") return en;
  return en;
}
