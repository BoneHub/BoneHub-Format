"""Command line of BoneHub-Format.

    python -m bonehub_format validate ROOT [--publishing]
    python -m bonehub_format pack ROOT
    python -m bonehub_format unpack ROOT [--overwrite]
    python -m bonehub_format index ROOT
    python -m bonehub_format match ROOT [--min-slices N] [--workers N]
    python -m bonehub_format prepare ROOT --repo-id ORG/NAME --title TITLE [--description-file FILE]
                                          [--homepage URL] [--github-repo ORG/NAME] [--webpage-options FILE]
    python -m bonehub_format upload ROOT --repo-id ORG/NAME [--commit-message TEXT]

prepare gets a root ready to upload: it packs the zips, rebuilds the index, validates the root, and writes the
Hugging Face files (README.md, LICENSE.md, .gitattributes, .gitignore, upload.md) and github-webpage/. The webpage
options file is a JSON object of build_webpage arguments, e.g. {"subtitle": ..., "about_html": ..., "contact_html":
..., "note_html": ..., "logo_link": ..., "analytics_html": ...}.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bonehub_format", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="check a root folder")
    validate.add_argument("--publishing", action="store_true", help="also check that it is ready to upload")
    commands.add_parser("pack", help="bring the mesh.zip and nurbs.zip of every scan up to date")
    unpack = commands.add_parser("unpack", help="extract the zips of every scan into mesh/ and nurbs/ folders")
    unpack.add_argument("--overwrite", action="store_true", help="also replace folders that exist")
    commands.add_parser("index", help="rebuild the index")
    match = commands.add_parser("match", help="find scans that are the same image and add them to links.json")
    match.add_argument("--min-slices", type=int, default=5, help="identical slices needed for a match (default 5)")
    match.add_argument("--workers", type=int, default=None, help="images read at once (default: up to 4)")
    prepare = commands.add_parser("prepare", help="get a root ready to upload to Hugging Face")
    prepare.add_argument("--repo-id", required=True, help="Hugging Face repository, e.g. BoneHub/BoneHub-Dataset")
    prepare.add_argument("--title", required=True, help="name of the dataset, e.g. 'BoneHub Dataset'")
    prepare.add_argument("--description-file", type=Path, help="Markdown file with the opening paragraph of the card")
    prepare.add_argument("--homepage", help="URL of the dataset's webpage")
    prepare.add_argument("--github-repo", help="GitHub repository the webpage links to, e.g. BoneHub/BoneHub-Dataset")
    prepare.add_argument("--webpage-options", type=Path, help="JSON file of further build_webpage arguments")
    upload = commands.add_parser("upload", help="upload a prepared root to its Hugging Face repository")
    upload.add_argument("--repo-id", required=True, help="Hugging Face repository, e.g. BoneHub/BoneHub-Dataset")
    upload.add_argument("--commit-message", help="message of the upload commit")
    for command in commands.choices.values():
        command.add_argument("root", type=Path, help="the root folder of the datasets")
    args = parser.parse_args(argv)

    from .archive import pack, unpack
    from .validate import validate as validate_root

    if args.command == "validate":
        return _print_problems(validate_root(args.root, publishing=args.publishing))
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
    if args.command == "match":
        from .matching import find_identical_images

        found, errors = find_identical_images(args.root, min_slices=args.min_slices, workers=args.workers)
        for key, error in errors.items():
            print(f"Could not read the image of {key}: {error}")
        print(f"{len(found)} pair(s) of scans are the same image; links.json is updated.")
        return 0
    if args.command == "upload":
        from .publish import upload as upload_root

        problems = validate_root(args.root, publishing=True)
        if problems:
            print("The root is not ready to upload; run prepare first:")
            return _print_problems(problems)
        upload_root(args.root, args.repo_id, args.commit_message)
        print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")
        return 0

    # prepare
    from .index import build_index
    from .publish import write_hub_files
    from .webpage import build_webpage

    print(f"{len(pack(args.root))} zip(s) written.")
    build_index(args.root)
    print("Index rebuilt.")
    problems = validate_root(args.root, publishing=True)
    if problems:
        print("The root is not ready to upload:")
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
