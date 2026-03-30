import type { Detection } from "@/native/types";

export function formatSeconds(value: number): string {
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}:${seconds.toFixed(3).padStart(6, "0")}`;
}

export function reviewStartFor(item: Detection): number {
  return Number(item.review_start_seconds ?? item.start_time_seconds ?? 0);
}

export function reviewEndFor(item: Detection): number {
  const fallbackStart = reviewStartFor(item);
  return Number(item.review_end_seconds ?? item.end_time_seconds ?? fallbackStart + 0.5);
}
