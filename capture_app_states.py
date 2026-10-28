#!/usr/bin/env python3
"""
capture_app_states.py
Captures interactive states of the app (search, citations, model ladder, trend)
and renders nvidia-smi as an image for paper_facts/images/.
"""

import os
import subprocess
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    base_dir = Path("/Users/durveshnarkhede/Desktop/COPILOTE")
    out_dir = base_dir / "paper_facts" / "images"
    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

    # 1. Render nvidia-smi as a nice monospaced graphic
    smi_file = base_dir / "results" / "nvidia_smi.txt"
    if smi_file.exists():
        smi_text = smi_file.read_text()
        plt.figure(figsize=(10, 4.5), dpi=300, facecolor="#0f172a")
        ax = plt.gca()
        ax.set_facecolor("#0f172a")
        plt.text(0.02, 0.95, smi_text, fontfamily="monospace", fontsize=7.5, color="#38bdf8", va="top")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "nvidia_smi.png", facecolor="#0f172a")
        plt.close()
        print("Generated nvidia_smi.png")

    # 2. Prepare HTML drivers to capture UI states
    # State A: Model Comparison Panel Expanded
    html_model_panel = base_dir / "temp_ui_model_panel.html"
    index_html = (base_dir / "static" / "index.html").read_text()
    
    # Inject auto-open for details
    mod_html_1 = index_html.replace("<details>", "<details open>")
    # Point fetch calls to absolute http://127.0.0.1:8000
    mod_html_1 = mod_html_1.replace('fetch("/api/', 'fetch("http://127.0.0.1:8000/api/')
    html_model_panel.write_text(mod_html_1)

    # State B: Search Query Run
    html_search = base_dir / "temp_ui_search.html"
    search_inject = """
    window.addEventListener('load', () => {
        setTimeout(async () => {
            document.querySelector('#q').value = "Contrastive visual representation learning";
            document.querySelectorAll('.seg button')[1].click(); // Find mode
            document.querySelector('#go').click();
        }, 1000);
    });
    """
    mod_html_2 = mod_html_1.replace("</script>", search_inject + "\n</script>")
    html_search.write_text(mod_html_2)

    # State C: Ask Query Run with Citations
    html_ask = base_dir / "temp_ui_ask.html"
    ask_inject = """
    window.addEventListener('load', () => {
        setTimeout(async () => {
            document.querySelector('#q').value = "What is LoRA parameter efficient fine tuning?";
            document.querySelector('#go').click();
        }, 1000);
    });
    """
    mod_html_3 = mod_html_1.replace("</script>", ask_inject + "\n</script>")
    html_ask.write_text(mod_html_3)

    # State D: Topic Trend
    html_trend = base_dir / "temp_ui_trend.html"
    trend_inject = """
    window.addEventListener('load', () => {
        setTimeout(async () => {
            if (window.trend) { window.trend(0); }
            else {
                fetch('http://127.0.0.1:8000/api/trend/0').then(r=>r.json()).then(d=>{
                    const s = d.years.map((y,i)=>`<span>${y}: ${d.counts[i]}</span>`).join(' ');
                    document.querySelector('#res').innerHTML = `<h2>Topic Trend: ${d.label}</h2><div class='muted'>${s}</div>`;
                });
            }
        }, 1200);
    });
    """
    mod_html_4 = mod_html_1.replace("</script>", trend_inject + "\n</script>")
    html_trend.write_text(mod_html_4)

    # Capture screenshots
    print("Capturing app_model_comparison_panel.png...")
    subprocess.run([
        chrome_bin,
        "--headless",
        "--disable-gpu",
        "--window-size=1440,900",
        "--virtual-time-budget=4000",
        f"--screenshot={out_dir / 'app_model_comparison_panel.png'}",
        f"file://{html_model_panel}"
    ], capture_output=True)

    print("Capturing app_search_results.png...")
    subprocess.run([
        chrome_bin,
        "--headless",
        "--disable-gpu",
        "--window-size=1440,900",
        "--virtual-time-budget=5000",
        f"--screenshot={out_dir / 'app_search_results.png'}",
        f"file://{html_search}"
    ], capture_output=True)

    print("Capturing app_answer_with_citations.png...")
    subprocess.run([
        chrome_bin,
        "--headless",
        "--disable-gpu",
        "--window-size=1440,900",
        "--virtual-time-budget=5000",
        f"--screenshot={out_dir / 'app_answer_with_citations.png'}",
        f"file://{html_ask}"
    ], capture_output=True)

    print("Capturing app_topic_trend.png...")
    subprocess.run([
        chrome_bin,
        "--headless",
        "--disable-gpu",
        "--window-size=1440,900",
        "--virtual-time-budget=4000",
        f"--screenshot={out_dir / 'app_topic_trend.png'}",
        f"file://{html_trend}"
    ], capture_output=True)

    # Cleanup temp files
    for p in [html_model_panel, html_search, html_ask, html_trend]:
        if p.exists():
            p.unlink()

    print("All UI state screenshots successfully captured!")

if __name__ == "__main__":
    main()
