# ShapeStar CI Smoke Test

This folder is intended to be copied into the public ApuShape GitHub
repository together with `.github/workflows/windows-shapestar-smoke.yml`.

Expected repository layout:

```text
.github/
  workflows/
    windows-shapestar-smoke.yml

ci_smoke/
  images/
    sample1.tif
    sample2.tif
    sample3.tif
  run_windows_smoke.ps1
```

Large runtime assets should not be committed to Git. Upload them to the GitHub
Release tag configured in the workflow, by default:

```text
smoke-assets-v1
```

Release assets:

```text
ShapeStarSmoke-win-x64.zip
*.pth
```

The zip should contain `ShapeStarSmoke.exe` and its `_internal` folder. During
GitHub Actions, the workflow downloads the zip and model, extracts the
executable under `ci_smoke/bin`, runs the three sample images, and uploads
`ci_smoke/results` as workflow artifacts.
