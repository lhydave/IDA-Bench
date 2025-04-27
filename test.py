from dataclasses import dataclass, asdict


@dataclass
class test:
    a: int | None = None

b = dict()
t = test(**b)
print(t)
asdict(t)