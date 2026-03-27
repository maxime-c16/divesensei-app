import fs from "node:fs";
import path from "node:path";
import type { APIRoute } from "astro";
import { ensureRuntimeDirs, importsRoot } from "@/lib/runtime-config";

function sanitizeName(filename: string): string {
  const trimmed = filename.trim();
  const ext = path.extname(trimmed);
  const stem = path.basename(trimmed, ext).replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");
  return `${stem || "video"}${ext}`;
}

export const POST: APIRoute = async ({ request }) => {
  const contentType = String(request.headers.get("content-type") ?? "").toLowerCase();
  let bytes = Buffer.alloc(0);
  let originalName = "";

  if (contentType.includes("multipart/form-data")) {
    const form = await request.formData().catch(() => null);
    const file = form?.get("video");
    if (!(file instanceof File)) {
      return Response.json({ error: "Video file is required." }, { status: 400 });
    }
    bytes = Buffer.from(await file.arrayBuffer());
    originalName = file.name || "";
  } else {
    bytes = Buffer.from(await request.arrayBuffer().catch(() => new ArrayBuffer(0)));
    originalName = String(request.headers.get("x-file-name") ?? "").trim();
    if (!originalName) {
      return Response.json({ error: "Video file name is required." }, { status: 400 });
    }
  }

  if (!bytes.length) {
    return Response.json({ error: "The selected file is empty." }, { status: 400 });
  }

  ensureRuntimeDirs();
  const timestamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
  const filename = sanitizeName(originalName || "video");
  const targetDir = path.join(importsRoot, timestamp);
  const targetPath = path.join(targetDir, filename);
  fs.mkdirSync(targetDir, { recursive: true });
  fs.writeFileSync(targetPath, bytes);

  return Response.json({
    path: targetPath,
    originalName: originalName || filename,
    size: bytes.length,
  });
};
