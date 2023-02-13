class BasePackage:
    identifier: int

    def __init__(self, raw_bytes: bytes) -> None:
        identifier = raw_bytes[0]

        if identifier != self.identifier:
            raise ValueError(
                'Incorrect reconstruction keyframe package header')
