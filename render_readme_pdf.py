import base64
import json
import os
import re
import subprocess
import tempfile

import mistune

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
README = os.path.join(BASE_DIR, "README.md")
OUT_PDF = os.path.join(BASE_DIR, "README.pdf")
CHROMIUM = "chromium"


def to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    return f"data:image/{ext};base64,{b64}"


def render_mermaid(mermaid_src: str, tmpdir: str) -> str:
    """Render a mermaid diagram to PNG via mermaid-cli (mmdc) and return its data URI."""
    mmd_path = os.path.join(tmpdir, "diagram.mmd")
    png_path = os.path.join(tmpdir, "diagram.png")
    puppeteer_config = os.path.join(tmpdir, "puppeteer-config.json")
    with open(mmd_path, "w") as f:
        f.write(mermaid_src)
    with open(puppeteer_config, "w") as f:
        json.dump({"args": ["--no-sandbox"], "executablePath": f"/usr/bin/{CHROMIUM}"}, f)
    subprocess.run(
        ["mmdc", "-i", mmd_path, "-o", png_path, "-b", "transparent", "-w", "1200", "-p", puppeteer_config],
        check=True,
    )
    return to_data_uri(png_path)


def main():
    with open(README, "r", encoding="utf-8") as f:
        md_text = f.read()

    with tempfile.TemporaryDirectory() as tmpdir:
        def replace_mermaid(match):
            src = match.group(0)[len("```mermaid"):-len("```")].strip()
            uri = render_mermaid(src, tmpdir)
            return f"![pipeline diagram]({uri})"

        md_text = re.sub(r"```mermaid.*?```", replace_mermaid, md_text, flags=re.DOTALL)

        def inline_image(match):
            alt, src = match.group(1), match.group(2)
            if src.startswith("http") or src.startswith("data:"):
                return match.group(0)
            return f"![{alt}]({to_data_uri(os.path.join(BASE_DIR, src))})"

        md_text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", inline_image, md_text)

        body_html = mistune.html(md_text)

        page_html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  max-width: 900px;
  margin: 40px auto;
  padding: 0 20px;
  color: #1f2328;
  line-height: 1.6;
}}
h1, h2, h3 {{ border-bottom: 1px solid #d0d7de; padding-bottom: 0.3em; }}
code {{ background: #f6f8fa; padding: 0.15em 0.4em; border-radius: 4px; font-size: 0.9em; }}
pre {{ background: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto; }}
pre code {{ background: none; padding: 0; }}
img {{ max-width: 100%; display: block; margin: 12px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #d0d7de; padding: 6px 12px; text-align: left; }}
th {{ background: #f6f8fa; }}
blockquote {{ border-left: 4px solid #d0d7de; margin: 0; padding-left: 1em; color: #57606a; }}
a {{ color: #0969da; }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""
        html_path = os.path.join(tmpdir, "README_render.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(page_html)

        subprocess.run(
            [
                CHROMIUM,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                f"--print-to-pdf={OUT_PDF}",
                "--no-pdf-header-footer",
                f"file://{html_path}",
            ],
            check=True,
        )
    print(f"PDF written to {OUT_PDF}")


if __name__ == "__main__":
    main()
