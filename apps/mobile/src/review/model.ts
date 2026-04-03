import type { Detection } from "@/native/types";

export function formatSeconds(value: number): string {
  const total = Math.max(0, value);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total - hours * 3600 - minutes * 60;

  if (hours > 0) {
    return `${hours}h${minutes}m${seconds.toFixed(1)}s`;
  }

  if (minutes > 0) {
    return `${minutes}m${seconds.toFixed(1)}s`;
  }

  return `${seconds.toFixed(1)}s`;
}

export function reviewStartFor(item: Detection): number {
  return Number(item.review_start_seconds ?? item.start_time_seconds ?? 0);
}

export function reviewEndFor(item: Detection): number {
  const fallbackStart = reviewStartFor(item);
  return Number(item.review_end_seconds ?? item.end_time_seconds ?? fallbackStart + 0.5);
}
