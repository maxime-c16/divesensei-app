import fs from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
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
  let originalName = "";
  let size = 0;

  ensureRuntimeDirs();
  const timestamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);

  if (contentType.includes("multipart/form-data")) {
    const form = await request.formData().catch(() => null);
    const file = form?.get("video");
    if (!(file instanceof File)) {
      return Response.json({ error: "Video file is required." }, { status: 400 });
    }
    originalName = file.name || "";
    const filename = sanitizeName(originalName || "video");
    const targetDir = path.join(importsRoot, timestamp);
    const targetPath = path.join(targetDir, filename);
    fs.mkdirSync(targetDir, { recursive: true });

    await pipeline(
      Readable.fromWeb(file.stream() as ReadableStream<Uint8Array>),
      fs.createWriteStream(targetPath),
    );

    const written = fs.statSync(targetPath, { throwIfNoEntry: false });
    size = written?.size ?? 0;
    if (!size) {
      fs.rmSync(targetPath, { force: true });
      return Response.json({ error: "The selected file is empty." }, { status: 400 });
    }
    return Response.json({
      path: targetPath,
      originalName: originalName || filename,
      size,
    });
  }

  originalName = String(request.headers.get("x-file-name") ?? "").trim();
  if (!originalName) {
    return Response.json({ error: "Video file name is required." }, { status: 400 });
  }
  if (!request.body) {
    return Response.json({ error: "The selected file is empty." }, { status: 400 });
  }

  const filename = sanitizeName(originalName || "video");
  const targetDir = path.join(importsRoot, timestamp);
  const targetPath = path.join(targetDir, filename);
  fs.mkdirSync(targetDir, { recursive: true });

  const sizeCounter = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      size += chunk.byteLength;
      controller.enqueue(chunk);
    },
  });

  await pipeline(
    Readable.fromWeb(request.body.pipeThrough(sizeCounter)),
    fs.createWriteStream(targetPath),
  );

  if (!size) {
    fs.rmSync(targetPath, { force: true });
    return Response.json({ error: "The selected file is empty." }, { status: 400 });
  }

  return Response.json({
    path: targetPath,
    originalName: originalName || filename,
    size,
  });
};
