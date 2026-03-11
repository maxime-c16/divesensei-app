import fs from "node:fs";
import path from "node:path";
import type { APIRoute } from "astro";
import { isAllowedLocalPath } from "@/lib/runtime-config";

function contentTypeFor(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".mp4") return "video/mp4";
  if (ext === ".mov") return "video/quicktime";
  if (ext === ".webm") return "video/webm";
  if (ext === ".json") return "application/json";
  if (ext === ".csv") return "text/csv";
  return "application/octet-stream";
}

export const GET: APIRoute = async ({ request, url }) => {
  const filePath = url.searchParams.get("path");
  if (!filePath) return new Response("Missing path", { status: 400 });
  if (!isAllowedLocalPath(filePath)) return new Response("Forbidden", { status: 403 });
  if (!fs.existsSync(filePath)) return new Response("Not found", { status: 404 });

  const stat = fs.statSync(filePath);
  const range = request.headers.get("range");
  const mime = contentTypeFor(filePath);

  if (!range) {
    const stream = fs.createReadStream(filePath);
    return new Response(stream as unknown as BodyInit, {
      status: 200,
      headers: {
        "Content-Type": mime,
        "Content-Length": String(stat.size),
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
      },
    });
  }

  const [startValue, endValue] = range.replace("bytes=", "").split("-");
  const start = Number.parseInt(startValue, 10);
  const end = endValue ? Number.parseInt(endValue, 10) : stat.size - 1;
  if (Number.isNaN(start) || Number.isNaN(end) || start > end || end >= stat.size) {
    return new Response("Invalid range", { status: 416 });
  }

  const stream = fs.createReadStream(filePath, { start, end });
  return new Response(stream as unknown as BodyInit, {
    status: 206,
    headers: {
      "Content-Type": mime,
      "Content-Length": String(end - start + 1),
      "Content-Range": `bytes ${start}-${end}/${stat.size}`,
      "Accept-Ranges": "bytes",
      "Cache-Control": "no-store",
    },
  });
};
