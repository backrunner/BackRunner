#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "assets/hero/backrunner-workbench.png"
DEFAULT_OUTPUT = ROOT / "assets/hero/backrunner-workbench-animated.gif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the animated BackRunner hero.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=324)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--radius", type=int, default=24)
    return parser.parse_args()


def crop_and_resize(source: Image.Image, size: tuple[int, int]) -> Image.Image:
    crop_height = round(source.width * size[1] / size[0])
    top = max(0, round((source.height - crop_height) * 0.47))
    cropped = source.crop((0, top, source.width, top + crop_height))
    return cropped.resize(size, Image.Resampling.LANCZOS).convert("RGB")


def signal_masks(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    red_values = []
    cyan_values = []

    for red, green, blue in image.get_flattened_data():
        red_score = max(0, red - max(green, blue) * 0.62)
        red_values.append(
            min(255, round(red_score * 2.2))
            if red > 55 and red > green * 1.16 and red > blue * 1.04
            else 0
        )

        cyan_score = max(0, (green + blue) / 2 - red * 0.72)
        cyan_values.append(
            min(255, round(cyan_score * 2.05))
            if blue > 82 and green > 70 and (green + blue) / 2 > red * 1.08
            else 0
        )

    red_mask = Image.new("L", image.size)
    cyan_mask = Image.new("L", image.size)
    red_mask.putdata(red_values)
    cyan_mask.putdata(cyan_values)
    return red_mask, cyan_mask


def scaled_mask(mask: Image.Image, strength: float) -> Image.Image:
    return mask.point(lambda value: min(255, round(value * strength)))


def add_screen_glow(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int],
    strength: float,
) -> Image.Image:
    screen = ImageChops.screen(image, Image.new("RGB", image.size, color))
    return Image.composite(screen, image, scaled_mask(mask, strength))


def moving_band(
    size: tuple[int, int],
    progress: float,
    start: float,
    span: float,
    sigma: float,
) -> Image.Image:
    width, height = size
    center = start + progress * span
    values = []

    for x in range(width):
        value = 0.0
        for wrapped_center in (center - span, center, center + span):
            distance = (x - wrapped_center) / sigma
            value = max(value, math.exp(-0.5 * distance * distance))
        values.append(round(255 * value))

    band = Image.new("L", (width, 1))
    band.putdata(values)
    return band.resize((width, height))


def build_frames(base: Image.Image, frame_count: int) -> list[Image.Image]:
    red_mask, cyan_mask = signal_masks(base)
    frames = []

    for index in range(frame_count):
        progress = index / frame_count
        frame = base.copy()

        red_band = moving_band(base.size, progress, -100, 820, 54)
        cyan_band = moving_band(base.size, progress, 290, 920, 60)
        red_pulse = ImageChops.multiply(red_mask, red_band).filter(
            ImageFilter.GaussianBlur(3)
        )
        cyan_pulse = ImageChops.multiply(cyan_mask, cyan_band).filter(
            ImageFilter.GaussianBlur(3)
        )
        frame = add_screen_glow(frame, red_pulse, (255, 48, 24), 0.92)
        frame = add_screen_glow(frame, cyan_pulse, (26, 218, 255), 0.90)
        frames.append(frame)

    return frames


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1),
        radius=radius,
        fill=255,
    )
    return mask


def shared_palette(frames: list[Image.Image], colors: int = 112) -> Image.Image:
    sample_width = max(1, frames[0].width // 4)
    sample_height = max(1, frames[0].height // 4)
    samples = Image.new("RGB", (sample_width, sample_height * len(frames)))

    for index, frame in enumerate(frames):
        sample = frame.resize((sample_width, sample_height), Image.Resampling.BILINEAR)
        samples.paste(sample, (0, index * sample_height))

    return samples.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)


def main() -> None:
    args = parse_args()
    if (
        args.width <= 0
        or args.height <= 0
        or args.frames <= 1
        or args.fps <= 0
        or args.radius < 0
    ):
        raise SystemExit("width, height, frames, fps, and radius must be valid")

    with Image.open(args.source) as source:
        base = crop_and_resize(source, (args.width, args.height))

    frames = build_frames(base, args.frames)
    palette = shared_palette(frames)
    quantized = [
        frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for frame in frames
    ]
    transparent_corners = ImageChops.invert(
        rounded_mask(base.size, min(args.radius, min(base.size) // 2))
    )
    for frame in quantized:
        frame.paste(255, mask=transparent_corners)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        args.output,
        save_all=True,
        append_images=quantized[1:],
        duration=round(1000 / args.fps),
        loop=0,
        optimize=True,
        disposal=2,
        transparency=255,
    )


if __name__ == "__main__":
    main()
