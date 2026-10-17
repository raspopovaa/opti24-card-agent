"""Проверка актуальности базы знаний по документам оператора топливных карт.

Сайт opti-24.ru отдаёт документы только браузеру человека, поэтому скрипт ничего
не скачивает. Он ищет скачанные вручную файлы в папках-«входящих», считает их
хеш и сравнивает с версией, учтённой в docs/knowledge/sources.json.

    python scripts/sync_knowledge.py                    # что изменилось (по умолчанию ~/Downloads)
    python scripts/sync_knowledge.py --inbox ~/Obsidian\\ Vault/Inbox
    python scripts/sync_knowledge.py --accept api-tariffs   # выжимки обновлены, зафиксировать хеш

Код выхода 1, если есть новые или изменившиеся документы: выжимки, перечисленные
в topics, нужно пересмотреть и провести через PR.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "docs" / "knowledge" / "sources.json"
DEFAULT_INBOX = Path("~/Downloads").expanduser()


@dataclass(frozen=True, slots=True)
class SourceStatus:
    source_id: str
    state: str  # new | changed | unchanged | missing
    path: Path | None
    sha256: str | None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def newest_match(inboxes: list[Path], patterns: list[str]) -> Path | None:
    """Самый свежий файл во входящих, подходящий под один из шаблонов имени."""
    found = [
        path
        for inbox in inboxes
        if inbox.is_dir()
        for path in inbox.rglob("*")
        if path.is_file() and any(fnmatch.fnmatch(path.name.casefold(), p.casefold()) for p in patterns)
    ]
    return max(found, key=lambda path: path.stat().st_mtime, default=None)


def check_sources(registry: dict[str, list[dict[str, object]]], inboxes: list[Path]) -> list[SourceStatus]:
    statuses: list[SourceStatus] = []
    for source in registry["sources"]:
        patterns = [str(p) for p in source["files"]]  # type: ignore[union-attr]
        path = newest_match(inboxes, patterns)
        if path is None:
            statuses.append(SourceStatus(str(source["id"]), "missing", None, None))
            continue
        digest = file_sha256(path)
        known = source.get("sha256")
        state = "new" if known is None else ("unchanged" if known == digest else "changed")
        statuses.append(SourceStatus(str(source["id"]), state, path, digest))
    return statuses


def accept(registry: dict[str, list[dict[str, object]]], statuses: list[SourceStatus], ids: list[str]) -> list[str]:
    """Фиксирует хеш документов, по которым выжимки уже обновлены; возвращает принятые ID."""
    by_id = {status.source_id: status for status in statuses}
    accepted: list[str] = []
    for source in registry["sources"]:
        status = by_id.get(str(source["id"]))
        if str(source["id"]) in ids and status is not None and status.sha256 is not None:
            source["sha256"] = status.sha256
            source["accepted"] = date.today().isoformat()
            accepted.append(str(source["id"]))
    return accepted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Актуальность базы знаний по документам оператора")
    parser.add_argument("--inbox", action="append", type=Path, help="папка со скачанными документами")
    parser.add_argument("--accept", nargs="+", default=[], metavar="ID", help="зафиксировать хеш документов")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args(argv)

    inboxes = [path.expanduser() for path in (args.inbox or [DEFAULT_INBOX])]
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    statuses = check_sources(registry, inboxes)
    titles = {str(s["id"]): s for s in registry["sources"]}

    labels = {"new": "НОВЫЙ", "changed": "ИЗМЕНЁН", "unchanged": "без изменений", "missing": "нет файла"}
    for status in statuses:
        source = titles[status.source_id]
        print(f"{labels[status.state]:>14}  {status.source_id}: {source['title']}")
        if status.path:
            print(f"{'':>16}файл: {status.path.name}")
        if status.state in {"new", "changed"}:
            print(f"{'':>16}пересмотреть: {', '.join(source['topics'])}")  # type: ignore[arg-type]
        if status.state == "missing":
            where = source.get("url") or source.get("site_section")
            print(f"{'':>16}скачать в браузере: {where}")

    if args.accept:
        accepted = accept(registry, statuses, args.accept)
        args.registry.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nЗафиксированы: {', '.join(accepted) or 'ничего (нет файлов с такими ID)'}")
        return 0
    return 1 if any(status.state in {"new", "changed"} for status in statuses) else 0


if __name__ == "__main__":
    sys.exit(main())
