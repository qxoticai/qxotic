"""Structured output: the reply is constrained to the type you ask for, so it always parses.
A dataclass needs no dependency; a Pydantic class, a TypedDict, list[T], Literal and Enum work too."""
from dataclasses import dataclass
from typing import Literal
import jinfer

@dataclass
class Person:
    name: str
    born: int
    city: str
    field: Literal["mathematics", "physics", "literature"]

with jinfer.load("LiquidAI/LFM2.5-350M-GGUF:Q8_0") as model:
    person = model.chat("Ada Lovelace, born 1815 in London, wrote the first program.", schema=Person, max_tokens=64, temperature=0).parsed
    print(person)                                   # Person(name='Ada Lovelace', born=1815, ...)
    print(model.chat("Name three planets.", schema=list[str], max_tokens=32, temperature=0).parsed)
