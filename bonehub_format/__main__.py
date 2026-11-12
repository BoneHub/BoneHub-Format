"""Command line of BoneHub-Format.

    python -m bonehub_format validate DATABASE [--publishing] [--registry SOURCE]
    python -m bonehub_format pack DATABASE
    python -m bonehub_format unpack DATABASE [--overwrite]
    python -m bonehub_format index DATABASE
    python -m bonehub_format prepare DATABASE --repo-id ORG/NAME --title TITLE [--description-file FILE]
                                              [--homepage URL] [--github-repo ORG/NAME] [--webpage-options FILE]
                                              [--registry SOURCE]
    python -m bonehub_format upload DATABASE --repo-id ORG/NAME [--commit-message TEXT] [--registry SOURCE]

DATABASE is the root folder of a database. prepare gets it ready to upload: it packs the zips, rebuilds the index,
validates it, and writes the Hugging Face files (README.md, LICENSE.md, .gitattributes, .gitignore, upload.md) and
github-webpage/. The webpage options file is a JSON object of build_webpage arguments, e.g. {"subtitle": ...,
"about_html": ..., "contact_html": ..., "note_html": ..., "logo_link": ..., "analytics_html": ...}.

validate, prepare and upload check every dataset ID against the BoneHub registry, read from GitHub unless --registry
gives another URL or a local copy of the CSV; `validate --registry none` skips the check.
"""

import argparse
import json
import sys
from pathlib import Path


def _print_problems(problems: list[str]) -> int:
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problem(s) found." if problems else "No problems found.")
    return 1 if problems else 0


def _registry(source: str, optional: bool):
    """The registry from a URL or path, None for 'none' where the check may be skipped; raises SystemExit if it cannot
    be read."""
    from .registry import load_registry

    if source.lower() == "none":
        if optional:
            return None
        raise SystemExit("A database is checked against the registry before it is published; --registry none is not allowed here.")
    try:
        return load_registry(source)
    except (OSError, ValueError) as e:
        raise SystemExit(f"Could not read the registry from {source}: {e}\nPass --registry with a local copy of the CSV.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bonehub_format", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="check a database")
    validate.add_argument("--publishing", action="store_true", help="also check that it is ready to upload")
    commands.add_parser("pack", help="bring the mesh.zip and nurbs.zip of every scan up to date")
    unpack = commands.add_parser("unpack", help="extract the zips of every scan into mesh/ and nurbs/ folders")
    unpack.add_argument("--overwrite", action="store_true", help="also replace folders that exist")
    commands.add_parser("index", help="rebuild the index")
    prepare = commands.add_parser("prepare", help="get a database ready to upload to Hugging Face")
    prepare.add_argument("--repo-id", required=True, help="Hugging Face repository, e.g. BoneHub/<database>")
    prepare.add_argument("--title", required=True, help="name of the database")
    prepare.add_argument("--description-file", type=Path, help="Markdown file with the opening paragraph of the card")
    prepare.add_argument("--homepage", help="URL of the database's webpage")
    prepare.add_argument("--github-repo", help="GitHub repository the webpage links to, e.g. BoneHub/<database>")
    prepare.add_argument("--webpage-options", type=Path, help="JSON file of further build_webpage arguments")
    upload = commands.add_parser("upload", help="upload a prepared database to its Hugging Face repository")
    upload.add_argument("--repo-id", required=True, help="Hugging Face repository, e.g. BoneHub/<database>")
    upload.add_argument("--commit-message", help="message of the upload commit")
    for command in commands.choices.values():
        command.add_argument("root", type=Path, metavar="DATABASE", help="the root folder of the database")
    from .registry import REGISTRY_URL

    for command in (validate, prepare, upload):
        command.add_argument(
            "--registry", default=REGISTRY_URL, metavar="SOURCE", help="URL or path of the registry CSV (default: on GitHub)"
        )
    args = parser.parse_args(argv)

    from .archive import pack, unpack
    from .validate import validate as validate_root

    if args.command == "validate":
        registry = _registry(args.registry, optional=True)
        return _print_problems(validate_root(args.root, publishing=args.publishing, registry=registry))
    if args.command == "pack":
        print(f"{len(pack(args.root))} zip(s) written.")
        return 0
    if args.command == "unpack":
        print(f"{len(unpack(args.root, args.overwrite))} folder(s) written.")
        return 0
    if args.command == "index":
        from .index import build_index

        for path in build_index(args.root).values():
            print(f"Wrote {path}")
        return 0
    if args.command == "upload":
        from .publish import upload as upload_root

        problems = validate_root(args.root, publishing=True, registry=_registry(args.registry, optional=False))
        if problems:
            print("The database is not ready to upload; run prepare first:")
            return _print_problems(problems)
        upload_root(args.root, args.repo_id, args.commit_message)
        print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")
        return 0

    # prepare
    from .index import build_index
    from .publish import write_hub_files
    from .webpage import build_webpage

    registry = _registry(args.registry, optional=False)
    print(f"{len(pack(args.root))} zip(s) written.")
    build_index(args.root)
    print("Index rebuilt.")
    problems = validate_root(args.root, publishing=True, registry=registry)
    if problems:
        print("The database is not ready to upload:")
        return _print_problems(problems)
    description = args.description_file.read_text(encoding="utf-8") if args.description_file else None
    for path in write_hub_files(args.root, args.repo_id, args.title, description, args.homepage):
        print(f"Wrote {path}")
    options = json.loads(args.webpage_options.read_text(encoding="utf-8")) if args.webpage_options else {}
    webpage = build_webpage(args.root, args.title, hf_repo_id=args.repo_id, github_repo=args.github_repo, **options)
    print(f"Wrote {webpage}")
    print("Ready to upload: see upload.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
