#!/usr/bin/env python3

import csv
import re
import subprocess
from pathlib import Path
from typing import NamedTuple

import typer
from loguru import logger

INSTITUTE = "Институт компьютерных наук"
BOOK_NAME = re.compile(r"(?P<year>20\d{2})-(?P<book>\d+)\.pdf")
AUTHOR = re.compile(
    r"\([^()]*@[^()\n]+?,\s*(?P<group>[A-ZА-ЯЁ]+[-–]\d{2}(?:[-–][A-ZА-ЯЁ0-9]+)+)\s*\)",
)
SUPERVISOR = re.compile(r"^Научн(?:ый|ые) руководител(?:ь|и)\s*[:—–-]\s*")
SIDE_LABEL = re.compile(r"\n(?:[ \t]*\n)*[ \t]{50,}(?:ИНФОРМАЦИОННЫЕ|ТЕХНОЛОГИИ)[ \t]*\n(?:[ \t]*\n)*")
PAGE_NUMBER = re.compile(r"\s*(\d+)\s*")
FIELDS = ("year", "book", "pages", "pdf_page", "group", "topic", "supervisor")

app = typer.Typer()


class Line(NamedTuple):
    text: str
    printed_page: int
    pdf_page: int
    computer_science: bool


class Article(NamedTuple):
    title_start: int
    author_line: int
    group: str


def read_lines(pdf: Path) -> list[Line]:
    command = ["pdftotext", "-layout", str(pdf), "-"]
    logger.info("Running {}", command)
    text = subprocess.run(command, check=True, encoding="utf-8", stdout=subprocess.PIPE).stdout
    lines: list[Line] = []
    for page_index, page in enumerate(text.split("\f"), 1):
        content = SIDE_LABEL.sub("\n", page).strip("\n").splitlines()
        if not content:
            continue
        is_computer_science = content[0].strip() == INSTITUTE
        number = PAGE_NUMBER.fullmatch(content[-1])
        if number:
            content.pop()
        if is_computer_science:
            content = content[2:]
        printed_page = int(number[1]) if number else 0
        lines.extend(Line(line.strip(), printed_page, page_index, is_computer_science) for line in content)
        lines.append(Line("", printed_page, page_index, is_computer_science))
    return lines


def find_articles(lines: list[Line]) -> list[Article]:
    articles: list[Article] = []
    for index, line in enumerate(lines):
        match = AUTHOR.search(line.text)
        if not match:
            continue
        start = index
        while start > 0 and not lines[start - 1].text:
            start -= 1
        while start > 0 and lines[start - 1].text:
            start -= 1
        articles.append(Article(start, index, match["group"]))
    return articles


def find_supervisor(body: list[Line]) -> tuple[str, int]:
    for index, line in enumerate(body):
        if not SUPERVISOR.match(line.text):
            continue
        end = index + 1
        while end < len(body) and body[end].text and body[end].printed_page == line.printed_page:
            end += 1
        name = " ".join(item.text for item in body[index:end])
        return SUPERVISOR.sub("", name), body[end - 1].printed_page
    return "", next((line.printed_page for line in reversed(body) if line.text), 0)


def extract_book(pdf: Path) -> list[tuple[int, int, str, int, str, str, str]]:
    match = BOOK_NAME.fullmatch(pdf.name)
    if match is None:
        raise Exception(f"Expected YEAR-BOOK.pdf, got {pdf.name}")
    year, book = int(match["year"]), int(match["book"])
    lines = read_lines(pdf)
    articles = find_articles(lines)
    rows = []
    for index, article in enumerate(articles):
        author = lines[article.author_line]
        if not author.computer_science or not article.group.startswith(("М", "M")):
            continue
        title_lines = [line for line in lines[article.title_start : article.author_line] if line.text]
        if not title_lines or not title_lines[0].printed_page:
            raise Exception(f"Missing title or printed page in {pdf.name}, PDF page {author.pdf_page}")
        start = title_lines[0]
        boundary = articles[index + 1].title_start if index + 1 < len(articles) else len(lines)
        supervisor, last_page = find_supervisor(lines[article.author_line + 1 : boundary])
        pages = str(start.printed_page)
        if last_page > start.printed_page:
            pages = f"{pages}–{last_page}"
        topic = " ".join(line.text for line in title_lines)
        topic = re.sub(r"(?<=\w)- (?=[а-яёa-z])", "-", topic)
        rows.append((year, book, pages, start.pdf_page, article.group, topic, supervisor))
    logger.info("{}: {} master's entries", pdf.name, len(rows))
    return rows


@app.callback()
def main() -> None:
    pass


@app.command()
def extract(books: list[Path] = typer.Argument(...), output: Path = typer.Option(...)) -> None:
    try:
        rows = [row for pdf in sorted(books) for row in extract_book(pdf)]
        with output.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(FIELDS)
            writer.writerows(rows)
        logger.info("Wrote {} entries to {}", len(rows), output)
    except Exception:
        logger.exception("Could not extract conference abstracts")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
