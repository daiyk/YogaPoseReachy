# Changelog

## 2026-11-11

- Prepare version 1.2.0 for private distribution through the eth-sbs Hugging Face Space and the official Reachy app installer/updater.
- Apply changed Wireless pose-runtime dependency locks instead of silently adopting the previous environment; retry failed updates.
- Track the bundled pose model with Git LFS for private Space publication.
- Reject unresolved Git LFS assets during wheel builds and verify bundled model, reference, runtime lock and profile assets.
- Document private access, installation, official updates and maintainer publishing.

## 2026-11-05

- Bundle the pose runtime and reference assets for standalone pose coaching.
- Allow observation rates up to 30 FPS and refresh the setup documentation.
