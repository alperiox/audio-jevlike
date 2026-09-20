from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from prosodia.schema import Example, QuestionSpec


@runtime_checkable
class Corpus(Protocol):
    """Loaders are interchangeable so the build is decoupled from IEMOCAP
    registration lead time (spec §4.3)."""

    name: str

    def question_specs(self) -> list[QuestionSpec]: ...
    def iter_examples(self, split: str) -> Iterator[Example]: ...
