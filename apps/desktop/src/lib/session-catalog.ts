import { sqliteCatalogBackend } from "@/lib/session-catalog-sqlite";

export type { CatalogSessionRecord } from "@/lib/session-catalog-core";

export const registerSessionManifest = sqliteCatalogBackend.registerSessionManifest.bind(sqliteCatalogBackend);
export const listCatalogSessions = sqliteCatalogBackend.listCatalogSessions.bind(sqliteCatalogBackend);
export const saveReviewDecision = sqliteCatalogBackend.saveReviewDecision.bind(sqliteCatalogBackend);
export const listReviewDecisions = sqliteCatalogBackend.listReviewDecisions.bind(sqliteCatalogBackend);
export const markSessionOpened = sqliteCatalogBackend.markSessionOpened.bind(sqliteCatalogBackend);
export const renameSessionRun = sqliteCatalogBackend.renameSessionRun.bind(sqliteCatalogBackend);
export const getManifestPathForAnalysisRun = sqliteCatalogBackend.getManifestPathForAnalysisRun.bind(sqliteCatalogBackend);
export const deleteSessionRun = sqliteCatalogBackend.deleteSessionRun.bind(sqliteCatalogBackend);
export const refreshCatalogAvailability = sqliteCatalogBackend.refreshCatalogAvailability.bind(sqliteCatalogBackend);
export const relinkCatalogSource = sqliteCatalogBackend.relinkCatalogSource.bind(sqliteCatalogBackend);
export const resolveCatalogManifestPaths = sqliteCatalogBackend.resolveCatalogManifestPaths.bind(sqliteCatalogBackend);
