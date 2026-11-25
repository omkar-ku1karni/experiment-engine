"""Record a short demo GIF (registry -> audit -> decision memo) from a running app (`make demo`)."""

import io
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "reports" / "demo.gif"
frames: list[tuple[Image.Image, int]] = []


def settle(page, extra=2.0):
    page.wait_for_selector("[data-testid='stApp']", timeout=60000)
    for _ in range(120):
        if page.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        time.sleep(0.4)
    time.sleep(extra)


def snap(page, ms):
    img = Image.open(io.BytesIO(page.screenshot())).convert("RGB").resize((1100, 786), Image.LANCZOS)
    frames.append((img, ms))


def choose(page, label_text, option):
    box = page.get_by_role("combobox", name=label_text)
    box.click()
    page.get_by_role("option", name=option, exact=True).click()
    settle(page)


def scroll(page, y):
    page.evaluate(f"document.querySelector('[data-testid=\"stMain\"]')?.scrollTo(0, {y}) || window.scrollTo(0, {y})")
    time.sleep(1.2)


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(BASE)
    settle(page)
    snap(page, 2500)
    scroll(page, 900)
    snap(page, 2000)
    page.goto(f"{BASE}/Experiment_Audit")
    settle(page)
    snap(page, 2500)
    scroll(page, 900)
    snap(page, 2500)
    scroll(page, 1500)
    snap(page, 2500)
    scroll(page, 0)
    choose(page, "Experiment", "recs-carousel")
    scroll(page, 1500)
    snap(page, 3000)
    page.goto(f"{BASE}/Readout")
    settle(page, 3)
    snap(page, 3500)
    b.close()

imgs = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f, _ in frames]
imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=[d for _, d in frames], loop=0, optimize=True)
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(imgs)} frames)")
