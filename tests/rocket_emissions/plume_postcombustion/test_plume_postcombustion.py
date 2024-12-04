"""Tests for the `PlumePostCombustion` class."""

import numpy as np
import pytest

import launch_reentry_emissions.rocket_emissions.plume_postcombustion.plume_emission_index as pei
from launch_reentry_emissions.rocket_emissions.plume_postcombustion import (
    PlumeEmissionIndexes,
    PlumePostcombustion,
    PrimaryEmissionIndexes,
)


def test_plume_emissions_no_burn() -> None:
    """Test the plume emissions with no burn."""
    plume_emission_indexes = PlumeEmissionIndexes(
        primary_emission_indexes=PrimaryEmissionIndexes(nox=0.1, co2=0.5, h2o=0.2)
    )
    plume_postcombustion = PlumePostcombustion(plume_emission_indexes)

    final_emissions = plume_postcombustion.evaluate(h=0.0, mass_burned=0.0)

    assert sum(final_emissions) == pytest.approx(0.0)


def test_plume_emissions_no_ei() -> None:
    """Test the plume emissions with no emission index."""
    plume_emission_indexes = PlumeEmissionIndexes(
        primary_emission_indexes=PrimaryEmissionIndexes()
    )
    plume_postcombustion = PlumePostcombustion(plume_emission_indexes)

    final_emissions = plume_postcombustion.evaluate(h=0.0, mass_burned=1.0)

    # Should be equal to the nitrogen oxides emission index at sea level.
    assert sum(final_emissions) == pytest.approx(pei.EI_NOX_S_SL)


def test_plume_emission_random() -> None:
    """Test the plume emissions with random values."""
    primary_emission_indexes = PrimaryEmissionIndexes(nox=0.1, co2=0.5, h2o=0.4)
    plume_emission_indexes = PlumeEmissionIndexes(
        primary_emission_indexes=primary_emission_indexes
    )
    plume_postcombustion = PlumePostcombustion(plume_emission_indexes)

    rng = np.random.default_rng(42)
    size = 100
    mass_burned = rng.uniform(0.01, 1.0, size=size)
    h = rng.uniform(0.0, 100.0e3, size=size)

    final_emissions = plume_postcombustion.evaluate(h=h, mass_burned=mass_burned)

    # Should apprximately equal the fuel burn, since the emission indexes are 1.0
    # with only difference coming from the emission index of NOx at sea level
    np.testing.assert_allclose(sum(final_emissions), mass_burned, atol=pei.EI_NOX_S_SL)
    np.testing.assert_allclose(
        final_emissions.co2, primary_emission_indexes.co2 * mass_burned, rtol=1e-14
    )
    np.testing.assert_allclose(
        final_emissions.h2o, primary_emission_indexes.h2o * mass_burned, rtol=1e-14
    )
    assert np.all(final_emissions.nox > primary_emission_indexes.nox * mass_burned)


def test_plume_emission_random_no_nox() -> None:
    """Test the plume emissions with random values without nox secondary emissions."""
    primary_emission_indexes = PrimaryEmissionIndexes(nox=0.1, co2=0.5, h2o=0.4)
    plume_emission_indexes = PlumeEmissionIndexes(
        primary_emission_indexes=primary_emission_indexes, nox_s_sl=0.0
    )
    plume_postcombustion = PlumePostcombustion(plume_emission_indexes)

    rng = np.random.default_rng(42)
    size = 100
    mass_burned = rng.uniform(0.01, 1.0, size=size)
    h = rng.uniform(0.0, 100.0e3, size=size)

    final_emissions = plume_postcombustion.evaluate(h=h, mass_burned=mass_burned)

    # Should apprximately equal the fuel burn, since the emission indexes are 1.0
    np.testing.assert_allclose(sum(final_emissions), mass_burned, rtol=1e-14)
    np.testing.assert_allclose(
        final_emissions.co2, primary_emission_indexes.co2 * mass_burned, rtol=1e-14
    )
    np.testing.assert_allclose(
        final_emissions.h2o, primary_emission_indexes.h2o * mass_burned, rtol=1e-14
    )
    np.testing.assert_allclose(
        final_emissions.nox, primary_emission_indexes.nox * mass_burned, rtol=1e-14
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
