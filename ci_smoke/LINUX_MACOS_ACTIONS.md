# Linux and macOS ShapeStar Smoke-Test Handoff

This note is for continuing the smoke-test work on Linux and Apple Silicon
macOS machines. It intentionally records the assumptions that are easy to lose
when moving to another system or another coding agent.

## Goal

Build a small CPU-only `ShapeStarSmoke` command-line package on each target
platform, upload the package to the existing `smoke-assets-v1` GitHub Release,
and let GitHub Actions run the same three repository test images with the same
model checkpoint.

The smoke test is not the full ApuShape GUI package. It only verifies that the
ShapeStar inference path can start, load the model, process sample images, and
write `*.json`, `*.png`, and `*_overlay.png` outputs.

## Release Assets

Upload these assets to the same release tag used by the Windows workflow:

```text
smoke-assets-v1
```

Expected asset names:

```text
ShapeStarSmoke-win-x64.zip
ShapeStarSmoke-linux-x64.tar.gz
ShapeStarSmoke-macos-arm64.tar.gz
*.pth
```

The Linux/macOS archives should contain a PyInstaller onedir folder similar to:

```text
ShapeStarSmoke/
  ShapeStarSmoke
  _internal/
```

The model can be shared across operating systems. If several `*.pth` files are
uploaded, the smoke script uses the first one after alphabetical sorting, so a
single model asset is recommended.

## Build on Linux

Copy the ShapeStarSmoke source folder to a Linux x64 machine. Use a CPU-only
environment that can run `shapestar_smoke.py` successfully, then build:

```bash
cd /path/to/claude-code-ApuShape_win_action
python -m PyInstaller --clean --noconfirm ShapeStarSmoke.spec
tar -C dist -czf ShapeStarSmoke-linux-x64.tar.gz ShapeStarSmoke
```

Upload:

```bash
gh release upload smoke-assets-v1 ShapeStarSmoke-linux-x64.tar.gz \
  --repo BUAA-LiuLab/ApuShape \
  --clobber
```

## Build on macOS arm64

Use an Apple Silicon Mac with an arm64 Python/conda environment. After
`shapestar_smoke.py` works from source:

```bash
cd /path/to/claude-code-ApuShape_win_action
python -m PyInstaller --clean --noconfirm ShapeStarSmoke.spec
tar -C dist -czf ShapeStarSmoke-macos-arm64.tar.gz ShapeStarSmoke
```

Upload:

```bash
gh release upload smoke-assets-v1 ShapeStarSmoke-macos-arm64.tar.gz \
  --repo BUAA-LiuLab/ApuShape \
  --clobber
```

Unsigned macOS command-line executables can usually run in GitHub Actions when
they are extracted by the runner. If local macOS blocks the executable, remove
quarantine locally or run it from Terminal during testing.

## Enable the Workflows

Commit these files to the public repository:

```text
.github/workflows/linux-shapestar-smoke.yml
.github/workflows/macos-shapestar-smoke.yml
ci_smoke/run_unix_smoke.sh
ci_smoke/LINUX_MACOS_ACTIONS.md
```

Recommended commit:

```bash
git add .github/workflows/linux-shapestar-smoke.yml \
  .github/workflows/macos-shapestar-smoke.yml \
  ci_smoke/run_unix_smoke.sh \
  ci_smoke/LINUX_MACOS_ACTIONS.md \
  .gitattributes \
  ci_smoke/README.md
git commit -m "Add Linux and macOS ShapeStar smoke test templates"
git push
```

Then open GitHub:

```text
BUAA-LiuLab/ApuShape -> Actions
```

Select `Linux ShapeStar smoke test` or `macOS ShapeStar smoke test`, click
`Run workflow`, and choose the branch containing the workflow.

## Expected CI Outputs

Each workflow downloads the platform archive and model from `smoke-assets-v1`,
runs:

```bash
bash ci_smoke/run_unix_smoke.sh
```

and uploads `ci_smoke/results/` as an Actions artifact. The artifact should
contain, for each input image:

```text
<image-name>.json
<image-name>.png
<image-name>_overlay.png
```

## Runner Labels

The Linux workflow uses:

```text
ubuntu-22.04
ubuntu-24.04
```

The macOS workflow uses one stable Apple Silicon runner by default:

```text
macos-15
```

If a second macOS configuration is needed, add another currently supported
Apple Silicon label from GitHub's hosted runner table. Avoid deprecated labels
and public-preview labels for reviewer-facing runs.

Avoid `*-latest` labels for this reviewer-facing smoke test because they move
over time.
