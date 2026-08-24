// Validated categorical/sequential palette (light mode), fixed ordering.
export const SERIES = {
  blue: "#2a78d6",
  orange: "#eb6834",
  aqua: "#1baf7a",
  yellow: "#eda100",
};

export const SEQUENTIAL_BLUE = [
  "#eaf1fc",
  "#cde2fb",
  "#9ec5f4",
  "#6da7ec",
  "#3987e5",
  "#256abf",
  "#184f95",
  "#0d366b",
];

export function sequentialStep(value: number, max: number): string {
  if (max <= 0) return SEQUENTIAL_BLUE[0];
  const ratio = Math.min(1, value / max);
  const idx = Math.round(ratio * (SEQUENTIAL_BLUE.length - 1));
  return SEQUENTIAL_BLUE[idx];
}
