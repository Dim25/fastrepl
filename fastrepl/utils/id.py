from cuid2 import Cuid

CUID_GENERATOR: Cuid = Cuid(length=10)


def get_cuid():  # pragma: no cover
    return CUID_GENERATOR.generate()
