#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exe=""
model=""
output="$script_dir/results"
confidence="0.4"
min_total_contours="0"
images=()

usage() {
  cat <<'USAGE'
Usage:
  bash ci_smoke/run_unix_smoke.sh [options]

Options:
  --exe PATH                 ShapeStarSmoke executable path.
  --model PATH               Model checkpoint path.
  -i, --images PATH...        Input image paths.
  -o, --output PATH           Output directory.
  --confidence VALUE          Confidence threshold, default 0.4.
  --min-total-contours VALUE  Minimum total contour count, default 0.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exe)
      exe="$2"
      shift 2
      ;;
    --model|-m)
      model="$2"
      shift 2
      ;;
    --output|-o)
      output="$2"
      shift 2
      ;;
    --confidence)
      confidence="$2"
      shift 2
      ;;
    --min-total-contours)
      min_total_contours="$2"
      shift 2
      ;;
    --images|-i)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        images+=("$1")
        shift
      done
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$exe" ]]; then
  while IFS= read -r candidate; do
    exe="$candidate"
    break
  done < <(find "$script_dir/bin" -type f -name "ShapeStarSmoke" 2>/dev/null | sort)
fi

if [[ -z "$exe" || ! -f "$exe" ]]; then
  echo "ShapeStarSmoke executable was not found under $script_dir/bin" >&2
  exit 1
fi

chmod +x "$exe" || true

if [[ -z "$model" ]]; then
  while IFS= read -r candidate; do
    model="$candidate"
    break
  done < <(find "$script_dir/models" -maxdepth 1 -type f -name "*.pth" 2>/dev/null | sort)
fi

if [[ -z "$model" || ! -f "$model" ]]; then
  echo "Model checkpoint (.pth) was not found under $script_dir/models" >&2
  exit 1
fi

if [[ ${#images[@]} -eq 0 ]]; then
  while IFS= read -r image; do
    images+=("$image")
  done < <(find "$script_dir/images" -maxdepth 1 -type f \( \
      -iname "*.tif" -o -iname "*.tiff" -o -iname "*.png" -o \
      -iname "*.jpg" -o -iname "*.jpeg" \
    \) 2>/dev/null | sort)
fi

if [[ ${#images[@]} -eq 0 ]]; then
  echo "No test images were found under $script_dir/images" >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-}"
if [[ -z "$python_bin" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  elif command -v python >/dev/null 2>&1; then
    python_bin="python"
  else
    echo "Python was not found; it is needed only to read JSON smoke outputs." >&2
    exit 1
  fi
fi

rm -rf "$output"
mkdir -p "$output"

echo "ShapeStarSmoke executable: $exe"
echo "ShapeStar model: $model"
echo "Input images:"
for image in "${images[@]}"; do
  echo " - $image"
done
echo "Output directory: $output"
echo "Confidence threshold: $confidence"

"$exe" -i "${images[@]}" -m "$model" -o "$output" \
  --device cpu \
  --confidence "$confidence" \
  --keep-boundary \
  --min-contours 0

total_contours=0
for image in "${images[@]}"; do
  file_name="$(basename "$image")"
  stem="${file_name%.*}"
  json_path="$output/$stem.json"
  mask_path="$output/$stem.png"
  overlay_path="$output/${stem}_overlay.png"

  for path in "$json_path" "$mask_path" "$overlay_path"; do
    if [[ ! -f "$path" ]]; then
      echo "Missing expected output: $path" >&2
      exit 1
    fi
  done

  count="$("$python_bin" -c 'import json, sys; print(int(json.load(open(sys.argv[1], "r", encoding="utf-8")).get("contour_count", 0)))' "$json_path")"
  total_contours=$((total_contours + count))
  echo "Verified $stem: contour_count=$count"
  echo "Generated: $json_path"
  echo "Generated: $mask_path"
  echo "Generated: $overlay_path"
done

if (( total_contours < min_total_contours )); then
  echo "Smoke test produced $total_contours total contours, expected at least $min_total_contours." >&2
  exit 1
fi

echo "ShapeStar smoke test completed: image_count=${#images[@]}, total_contours=$total_contours"

