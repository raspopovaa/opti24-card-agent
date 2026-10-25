"""Загрузка документов в базу знаний агента (регламенты, тарифы, FAQ, инструкции).

    python scripts/ingest_knowledge.py add regulation.docx --title "Регламент блокировки" \\
        --kind regulation --version 2026-10 --valid-from 2026-10-01
    python scripts/ingest_knowledge.py list
    python scripts/ingest_knowledge.py archive <id>

Хранилище лежит в knowledge_store/ (в git не попадает). Новая версия документа с тем же
названием переводит прежнюю в архив. Типы файлов: txt, md, html, csv, tsv, json, docx,
xlsx, pptx, eml, pdf с текстом. Сканы, фото, аудио и видео отклоняются с подсказкой.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.ingest import CLASSIFICATIONS, KINDS, IngestError, Store  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--store", type=Path, help="папка хранилища (по умолчанию knowledge_store/)")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="загрузить файл")
    add.add_argument("file", type=Path)
    add.add_argument("--title", required=True)
    add.add_argument("--kind", choices=KINDS, default="other")
    add.add_argument("--version", default="")
    add.add_argument("--valid-from", default="")
    add.add_argument("--classification", choices=CLASSIFICATIONS, default="internal")
    add.add_argument("--source", default="", help="откуда документ (URL или название)")
    commands.add_parser("list", help="показать документы")
    archive = commands.add_parser("archive", help="убрать документ из поиска")
    archive.add_argument("id")
    args = parser.parse_args(argv)

    store = Store(args.store)
    try:
        if args.command == "add":
            doc = store.add(
                args.file, title=args.title, kind=args.kind, version=args.version,
                valid_from=args.valid_from, classification=args.classification, source=args.source,
            )
            print(f"Загружено: {doc.title} ({doc.id}), фрагментов: {doc.chunks}")
        elif args.command == "list":
            for doc in store.documents():
                print(f"{doc.id}  {doc.status:8}  {doc.kind:11}  {doc.version or '-':10}  {doc.title}")
        else:
            store.archive(args.id)
            print(f"В архиве: {args.id}")
    except IngestError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
