#!/usr/bin/env python3
"""Build paper-ready structure figures from DrawIO PNG exports.

The DrawIO XML files remain the editable source.  This script applies only
lightweight composition/annotation so the paper can use stable figure files
without hand-editing coordinates in DrawIO.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "paper_assets" / "05_figures" / "fig_structure" / "png"
OUT = ROOT / "paper_assets" / "05_figures" / "final"

FONT = Path("/usr/share/fonts/dejavu/DejaVuSans.ttf")
BOLD = Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf")


BLUE = (22, 89, 199)
DARK = (20, 28, 38)
PURPLE = (105, 45, 215)
WHITE = (255, 255, 255)
LIGHT_BLUE = (232, 241, 255)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BOLD if bold else FONT), size)


def fit_width(im: Image.Image, width: int) -> Image.Image:
    ratio = width / im.width
    height = round(im.height * ratio)
    return im.resize((width, height), Image.Resampling.LANCZOS)


def fit_box(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    ratio = min(max_w / im.width, max_h / im.height)
    return im.resize((round(im.width * ratio), round(im.height * ratio)), Image.Resampling.LANCZOS)


def centered_text(draw: ImageDraw.ImageDraw, xy, text: str, fnt, fill=DARK, spacing=4):
    x, y = xy
    lines = text.split("\n")
    widths = [draw.textbbox((0, 0), line, font=fnt)[2] for line in lines]
    heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total_h = sum(heights) + spacing * (len(lines) - 1)
    cy = y - total_h / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((x - w / 2, cy), line, font=fnt, fill=fill)
        cy += h + spacing


def save_all(im: Image.Image, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    im.save(png, dpi=(300, 300))
    im.convert("RGB").save(pdf, resolution=300.0)
    print(png)
    print(pdf)


def build_method_overview():
    im = Image.open(SRC / "fig1.drawio.png").convert("RGBA")
    draw = ImageDraw.Draw(im)

    # Replace the ambiguous bottom caption with the real round-to-round state.
    draw.rounded_rectangle((905, 843, 1285, 923), radius=10, fill=WHITE)
    centered_text(
        draw,
        (1095, 881),
        "global adapter θᵏ⁺¹\nfor the next round",
        font(27, bold=True),
        fill=PURPLE,
        spacing=6,
    )

    # Add a lightweight backward cue: candidate set is model-discovered before
    # client scoring; client configuration then produces feasible sets.
    draw.line((822, 154, 780, 154, 780, 210), fill=BLUE, width=4)
    draw.polygon([(780, 222), (772, 206), (788, 206)], fill=BLUE)
    draw.rounded_rectangle((658, 118, 858, 153), radius=7, fill=WHITE, outline=BLUE, width=2)
    centered_text(draw, (758, 136), "candidate set A", font(18, bold=True), fill=BLUE)

    draw.rounded_rectangle((1230, 332, 1450, 372), radius=7, fill=WHITE, outline=BLUE, width=2)
    centered_text(draw, (1340, 351), "feasible configs Kᵏᵢ", font(18, bold=True), fill=BLUE)

    save_all(im, "fig1_method_overview")


def add_panel_header(draw, x, y, w, label, title):
    draw.rounded_rectangle((x, y, x + w, y + 46), radius=8, fill=LIGHT_BLUE, outline=BLUE, width=2)
    draw.text((x + 16, y + 10), label, font=font(24, True), fill=BLUE)
    draw.text((x + 78, y + 10), title, font=font(24, True), fill=DARK)


def build_scheduler_pipeline():
    fig2 = Image.open(SRC / "fig2.drawio.png").convert("RGBA")
    fig4 = Image.open(SRC / "fig4.drawio.png").convert("RGBA")
    fig5 = Image.open(SRC / "fig5.drawio.png").convert("RGBA")

    # Trim white margins slightly while keeping all semantic content.
    fig2 = fig2.crop((65, 0, 1535, 825))
    fig4 = fig4.crop((0, 0, 1706, 905))
    fig5 = fig5.crop((0, 82, 1642, 857))

    top_h = 520
    a = fit_box(fig2, 785, top_h)
    b = fit_box(fig4, 830, top_h)
    c = fit_width(fig5, 1600)

    margin = 50
    gap = 35
    header_h = 56
    top_y = margin + header_h
    bottom_y = top_y + top_h + 82
    canvas_w = 1700
    canvas_h = bottom_y + header_h + c.height + 45
    canvas = Image.new("RGBA", (canvas_w, canvas_h), WHITE)
    draw = ImageDraw.Draw(canvas)

    add_panel_header(draw, margin, margin, 785, "(a)", "Matrix-level candidate selection")
    add_panel_header(draw, margin + 785 + gap, margin, 830, "(b)", "Client feasible configurations")

    canvas.alpha_composite(a, (margin + (785 - a.width) // 2, top_y + (top_h - a.height) // 2))
    canvas.alpha_composite(
        b,
        (margin + 785 + gap + (830 - b.width) // 2, top_y + (top_h - b.height) // 2),
    )

    add_panel_header(draw, margin, bottom_y, 1600, "(c)", "Server-side budget/latency-constrained MCKP scheduling")
    canvas.alpha_composite(c, (margin + (1600 - c.width) // 2, bottom_y + header_h + 12))

    save_all(canvas, "fig2_matrix_scheduler_pipeline")


def main():
    build_method_overview()
    build_scheduler_pipeline()


if __name__ == "__main__":
    main()
