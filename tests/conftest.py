import os

import pytest

from culture.evaluate.pool import Evaluator
from culture.game.hanabi import HanabiParams


@pytest.fixture(scope="session")
def pool_ev():
    ev = Evaluator(HanabiParams(), workers=max(2, min(10, (os.cpu_count() or 4) - 2)))
    yield ev
    ev.close()


@pytest.fixture
def ev():
    return Evaluator(HanabiParams(), workers=0)
