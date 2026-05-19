# ShapeStar CI Smoke Test

This folder is intended to be copied into the public ApuShape GitHub
repository together with the ShapeStar smoke-test workflows.

Expected repository layout:

```text
.github/
  workflows/
    windows-shapestar-smoke.yml
    linux-shapestar-smoke.yml
    macos-shapestar-smoke.yml

ci_smoke/
  images/
    sample1.tif
    sample2.tif
    sample3.tif
  run_windows_smoke.ps1
  run_unix_smoke.sh
```

Large runtime assets should not be committed to Git. Upload them to the GitHub
Release tag configured in the workflow, by default:

```text
smoke-assets-v1
```

Release assets:

```text
ShapeStarSmoke-win-x64.zip
ShapeStarSmoke-linux-x64.tar.gz
ShapeStarSmoke-macos-arm64.tar.gz
*.pth
```

The Windows zip should contain `ShapeStarSmoke.exe` and its `_internal` folder.
The Linux/macOS tar archives should contain a `ShapeStarSmoke` executable and
its `_internal` folder. During GitHub Actions, each workflow downloads the
matching runtime package and model, extracts the executable under
`ci_smoke/bin`, runs the three sample images, and uploads `ci_smoke/results` as
workflow artifacts.

See `LINUX_MACOS_ACTIONS.md` before continuing this work on Linux or macOS.
