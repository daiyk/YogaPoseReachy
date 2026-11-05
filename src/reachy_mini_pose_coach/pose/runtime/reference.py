"""Reference photo validation (copied from `reachy_mini_pose_compare.runner.validate_reference`)."""

import io


def validate_reference(data: bytes) -> bytes:
    """Return an EXIF-upright RGB JPEG of at most 1280 px, rejecting unsafe or unreadable images."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not data or len(data) > 5 * 1024 * 1024:
        raise ValueError("Choose a JPEG/PNG reference of at most 5 MB.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if (
                image.format not in ("JPEG", "PNG")
                or max(image.size) > 4096
                or image.width * image.height > 12_000_000
            ):
                raise ValueError("Use JPEG/PNG up to 4096 pixels per side and 12 megapixels.")
            upright = ImageOps.exif_transpose(image).convert("RGB")
            upright.thumbnail((1280, 1280))
            result = io.BytesIO()
            upright.save(result, format="JPEG", quality=92)
            return result.getvalue()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("Cannot read this reference image.") from exc
