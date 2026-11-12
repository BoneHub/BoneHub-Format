"""The database's webpage, a static site for GitHub Pages, in <database>/github-webpage/.

build_webpage() copies the page from the package's template, fills in the database's name and links, and writes its
data (data.js) from the index. github-webpage/README.md says how to publish the page, and
github-webpage/update_data.py rewrites data.js after the index changes, without this package.

Requires pyarrow: pip install "bonehub-format[converter]".
"""

from __future__ import annotations

import html
import runpy
import shutil
from importlib import resources
from pathlib import Path

from . import layout

TEMPLATE_FOLDER = "webpage_template"  # in the package
TEMPLATE_FILES = ("index.html", "app.js", "overview.js", "styles.css", "logo.png", ".nojekyll", "update_data.py", "README.md")

_GITHUB_ICON = (
    '<svg width="48" height="48" viewBox="0 0 16 16" fill="white" xmlns="http://www.w3.org/2000/svg"><path d="M8 0C3.58 0 '
    "0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48"
    "-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64"
    "-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 "
    ".27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 "
    '1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" /></svg>'
)


def _link(url: str, css_class: str, title: str, content: str) -> str:
    return (
        f'<a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer" class="{css_class}" '
        f'title="{html.escape(title)}">{content}</a>'
    )


def build_webpage(
    root: Path,
    title: str,
    subtitle: str | None = None,
    about_html: str | None = None,
    note_html: str | None = None,
    contact_html: str | None = None,
    hf_repo_id: str | None = None,
    github_repo: str | None = None,
    logo_link: str | None = None,
    analytics_html: str | None = None,
) -> Path:
    """Write <database>/github-webpage/ and return its path.

    title, subtitle: shown at the top of the page (plain text).
    about_html, note_html, contact_html: the About section, a note under the title and a Contact section (HTML);
        the note and the Contact section are left out when not given.
    hf_repo_id: the database's Hugging Face repository, linked from the page, e.g. 'BoneHub/<database>'.
    github_repo: a GitHub repository linked from the page's GitHub icon, e.g. 'BoneHub/<database>'.
    logo_link: where the logo links to; the Hugging Face repository by default.
    analytics_html: HTML added at the end of the page, e.g. the script tag of a visitor counter.

    The template files are replaced, so edits made to them in github-webpage/ are lost; data.js is rewritten.
    """
    root = Path(root)
    index_dir = root / layout.INDEX_DIR
    if not (index_dir / "scans.parquet").is_file():
        raise FileNotFoundError(f"The webpage shows the index, which is missing in '{index_dir}'; run build_index first.")
    target = root / layout.WEBPAGE_DIR
    target.mkdir(exist_ok=True)
    template = resources.files("bonehub_format") / TEMPLATE_FOLDER
    for name in TEMPLATE_FILES:
        with resources.as_file(template / name) as source:
            shutil.copyfile(source, target / name)

    hf_url = f"https://huggingface.co/datasets/{hf_repo_id}" if hf_repo_id else None
    links = []
    if hf_url:
        links.append(_link(hf_url, "btn-link", "The dataset on Hugging Face", "Hugging Face"))
    if github_repo:
        links.append(_link(f"https://github.com/{github_repo}", "github-link", "View on GitHub", _GITHUB_ICON))
    about = about_html or (
        "<p>This dataset holds medical images, bone segmentations and the 3D bone shapes derived from them, converted "
        "from public datasets into the <a href=\"https://github.com/BoneHub/BoneHub-Format\" target=\"_blank\" "
        "rel=\"noopener noreferrer\">BoneHub format</a>. It is intended for biomedical research.</p>"
    )
    contact = f'<section class="contact">\n            <h2>📬 Contact</h2>\n            {contact_html}\n        </section>' if contact_html else ""
    values = {
        "{{TITLE}}": html.escape(title),
        "{{SUBTITLE}}": html.escape(subtitle or "Medical images, bone segmentations and 3D bone shapes."),
        "{{LOGO_LINK}}": html.escape(logo_link or hf_url or "https://github.com/BoneHub/BoneHub-Format"),
        "{{LINKS}}": "\n                ".join(links),
        "{{NOTE}}": f'<p class="note">{note_html}</p>' if note_html else "",
        "{{ABOUT}}": about,
        "{{CONTACT}}": contact,
        "{{ANALYTICS}}": analytics_html or "",
        "{{HF_URL}}": hf_url or "(not published yet)",
    }
    for name in ("index.html", "README.md"):
        text = (target / name).read_text(encoding="utf-8")
        for placeholder, value in values.items():
            text = text.replace(placeholder, value)
        (target / name).write_text(text, encoding="utf-8", newline="\n")

    update = runpy.run_path(str(target / "update_data.py"), run_name="bonehub_update_data")
    update["write_data_js"](index_dir, target / "data.js")
    return target
