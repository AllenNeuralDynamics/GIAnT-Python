"""giant silo on the synthetic session with its own seeding (seeds in noise).

Seeds placed in noise can end with an all-zero footprint, which made the
activity noise's normal matrix singular (numpy's inv raised where MATLAB's
warns); the run must complete.
"""

import numpy as np
import pytest

from giant_python.pipeline import silo
from giant_python.pipeline.silo_session import SiloSession

# the fixture is used by name
from .test_raster_silo_stages import fake  # noqa: F401
from .test_raster_silo_stages import EX, PARAMS, make_session


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_own_seeding_in_noise_completes(tmp_path, fake):  # noqa: F811
    """Without a seeds file the session seeds itself; the run finishes and
    writes its summaries."""
    table = make_session(tmp_path / "own")
    report = silo.run_silo(table, PARAMS, EX)
    s = SiloSession.open(table)
    assert (s.out_dir / "experiment_summary.h5").exists()
    assert report["total"] > 0
    assert np.isfinite(report["total"])
