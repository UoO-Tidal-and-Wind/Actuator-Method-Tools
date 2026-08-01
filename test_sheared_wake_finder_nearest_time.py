"""Tests for nearest-time sheared-wake data selection."""

import numpy as np

from amtools.input_output.wake_finder import ShearedWakeFinderFile


def test_get_data_at_nearest_time_returns_all_rows(tmp_path):
    file_path = tmp_path / "wake"
    file_path.write_text(
        "\n".join(
            [
                "Time(s) centreX(m) centreY(m) centreZ(m) sigmaY(m) sigmaZ(m) Requiv(m)",
                "0.0 1 2 3 4 5 6",
                "1.0 7 8 9 10 11 12",
                "1.0 13 14 15 16 17 18",
            ]
        ),
        encoding="utf-8",
    )
    reader = ShearedWakeFinderFile(file_path)
    reader.read()

    centre, sigma_y, sigma_z, r_equiv = reader.get_data_at_nearest_time(0.9)

    np.testing.assert_allclose(centre, [[7, 8, 9], [13, 14, 15]])
    np.testing.assert_allclose(sigma_y, [10, 16])
    np.testing.assert_allclose(sigma_z, [11, 17])
    np.testing.assert_allclose(r_equiv, [12, 18])
