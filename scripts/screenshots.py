"""Capture the five app screens for the README (needs a running app: `make demo`)."""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "reports" / "screenshots"
PAGES = [("1_registry", ""), ("2_audit", "Experiment_Audit"), ("3_planner", "Planner"),
         ("4_simulation_lab", "Simulation_Lab"), ("5_readout", "Readout")]


def wait_idle(page):
    page.wait_for_selector("[data-testid='stApp']", timeout=60000)
    for _ in range(120):  # until Streamlit stops running scripts
        if page.locator("[data-testid='stStatusWidget']").count() == 0:
            break
        time.sleep(0.5)
    time.sleep(2.5)


OUT.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1400, "height": 1000})
    for name, path in PAGES:
        page.goto(f"{BASE}/{path}")
        wait_idle(page)
        page.screenshot(path=str(OUT / f"{name}.png"), full_page=False)
        print("saved", name)
    b.close()
