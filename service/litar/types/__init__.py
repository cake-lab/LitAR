from dataclasses import dataclass


@dataclass
class Vector2Int:
    x: int
    y: int

    def __str__(self) -> str:
        return f'({self.x}, {self.y})'


@dataclass
class Vector3Int:
    x: int
    y: int
    z: int

    def __str__(self) -> str:
        return f'({self.x}, {self.y}, {self.z})'
