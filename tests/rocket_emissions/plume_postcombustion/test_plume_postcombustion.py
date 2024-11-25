import numpy as np
import pytest

from launch_reentry_emissions.rocket_emissions.plume_postcombustion.plume_postcombustion import (
    EI_BC_P,
    EI_BC_RED,
    EI_CO_RED,
    EI_NOX_S_SL,
    M_BC,
    M_CO,
    M_CO2,
    PlumePostcombustionModel,
    PrimaryEmissionIndexes,
)


def test_black_carbon_postcombustion_stratosphere():
    """Test black_carbon_postcombustion_stratosphere function with a single float value."""
    h = 20.0e3
    result = PlumePostcombustionModel.final_black_carbon_stratosphere(h)
    assert EI_BC_P * EI_BC_RED < result < EI_BC_P


def test_black_carbon_postcombustion_troposphere_array():
    """Test black_carbon_postcombustion_troposphere function with a numpy array."""
    h_array = np.array([0.0, 10.0, 16.0, 40.0, 45.0]) * 1e3
    result_array = PlumePostcombustionModel.final_black_carbon(h_array)
    bc_min = EI_BC_RED * EI_BC_P
    np.testing.assert_allclose(result_array[0], bc_min)
    np.testing.assert_allclose(result_array[1], bc_min)
    assert bc_min < result_array[2] < EI_BC_P
    assert bc_min < result_array[3] < EI_BC_P
    np.testing.assert_allclose(result_array[-1], EI_BC_P)


def test_carbon_monoxide_postcombustion():
    """Test final_carbon_monoxide function clamps to maximum for higher altitudes."""
    h = 90.0e3
    ei_co_p = 0.1
    result = PlumePostcombustionModel.final_carbon_monoxide(h, ei_co_p, 0.3)
    assert result == ei_co_p


def test_carbon_monoxide_postcombustion_sea_level():
    """Test final_carbon_monoxide function at sea level."""
    rng = np.random.default_rng(42)
    size = 50
    h = np.zeros(size)
    ei_co_p = rng.uniform(0.1, 0.4, size=size)
    ei_co2_p = rng.uniform(0.1, 0.4, size=size)
    ei_c_expected = ei_co_p + ei_co2_p
    ei_co_f = PlumePostcombustionModel.final_carbon_monoxide(h, ei_co_p, ei_co2_p)
    np.testing.assert_allclose(ei_co_f, EI_CO_RED * ei_c_expected)


def test_carbon_monoxide_postcombustion_srm():
    """Test final_carbon_monoxide function at 30 km gives approx 1.85%."""
    h = 30.0e3
    ei_co_p = 0.17
    ei_com = ei_co_p * 2
    ei_co2_rel_expected = 0.0185
    ei_co_f = PlumePostcombustionModel.final_carbon_monoxide(h, ei_co_p, ei_co_p)
    ei_co_rel_f = ei_co_f / ei_com
    np.testing.assert_allclose(ei_co_rel_f, ei_co2_rel_expected, rtol=1e-2)


def test_carbon_monoxide_postcombustion_array():
    """Test final_carbon_monoxide function within range."""
    rng = np.random.default_rng(42)
    size = 50
    h = rng.uniform(0.0, 90e3, size=size)
    ei_co_p = 0.2
    ei_co2_p = rng.uniform(0.0, 0.4, size=size)
    ei_co_f = PlumePostcombustionModel.final_carbon_monoxide(h, ei_co_p, ei_co2_p)
    assert all(ei_co_f >= EI_CO_RED * (ei_co_p + ei_co2_p))
    assert all(ei_co_f <= ei_co_p)


def test_carbon_monoxide_postcombustion_no_c():
    """Test final_carbon_monoxide function when no carbon is present."""
    rng = np.random.default_rng(42)
    size = 50
    h = rng.uniform(0.0, 90e3, size=size)
    ei_co_p = 0.0
    ei_co2_p = 0.0
    ei_co_f = PlumePostcombustionModel.final_carbon_monoxide(h, ei_co_p, ei_co2_p)
    np.testing.assert_allclose(ei_co_f, 0.0)


def test_nitrogen_oxides_postcombustion_sea_level():
    """Test nitrogen_oxides function at sea level."""
    rng = np.random.default_rng(42)
    size = 50
    h = np.zeros(size)
    ei_nox_p = rng.uniform(0.1, 0.4, size=size)
    ei_nox_s_sl = rng.uniform(0.1, 0.4, size=size)
    ei_n_expected = ei_nox_p + ei_nox_s_sl
    ei_nox_f = PlumePostcombustionModel.final_nitrogen_oxides(h, ei_nox_p, ei_nox_s_sl)
    np.testing.assert_allclose(ei_nox_f, ei_n_expected)


def test_nitrogen_oxides_postcombustion_array():
    """Test nitrogen_oxides function within range."""
    rng = np.random.default_rng(42)
    size = 50
    h = rng.uniform(0.0, 90e3, size=size)
    ei_nox_p = rng.uniform(0.1, 0.4, size=size)
    ei_nox_s_sl = rng.uniform(0.1, 0.4, size=size)
    ei_n_expected = ei_nox_p + ei_nox_s_sl
    ei_n_f = PlumePostcombustionModel.final_nitrogen_oxides(h, ei_nox_p, ei_nox_s_sl)
    assert all(ei_n_f <= ei_n_expected)
    assert all(ei_n_f > 0.0)


def test_nitrogen_oxides_postcombustion_decreasing():
    """Test nitrogen_oxides function decreases emissions at higher altitudes."""

    size = 50
    h = np.linspace(0.0, 90e3, size)
    ei_nox_p = EI_NOX_S_SL
    ei_n_expected = ei_nox_p + EI_NOX_S_SL
    ei_n_f = PlumePostcombustionModel.final_nitrogen_oxides(h, ei_nox_p)
    assert all(ei_n_f <= ei_n_expected)
    assert all(ei_n_f > 0.0)
    assert all(x >= y for x, y in zip(ei_n_f, ei_n_f[1:]))


def test_plume_postcombustion_model_sl():
    """Test PlumePostcombustionModel function with a single float value."""
    h = 0.0e3
    primary_emission_indexes = PrimaryEmissionIndexes(
        bc=EI_BC_P, co=0.1, co2=0.1, nox=0.1, h2o=0.1
    )
    plume_postcombustion = PlumePostcombustionModel(
        h=h, primary_emission_indexes=primary_emission_indexes
    )

    final_emission_indexes = plume_postcombustion.evaluate()

    np.testing.assert_allclose(final_emission_indexes.bc, EI_BC_P * EI_BC_RED)
    np.testing.assert_allclose(final_emission_indexes.co, EI_CO_RED * 0.2)
    np.testing.assert_allclose(
        final_emission_indexes.co2,
        0.1
        + M_CO2 / M_CO * (0.1 - EI_CO_RED * 0.2)
        + M_CO2 / M_BC * (EI_BC_P - EI_BC_P * EI_BC_RED),
    )
    np.testing.assert_allclose(final_emission_indexes.nox, EI_NOX_S_SL + 0.1)
    np.testing.assert_allclose(final_emission_indexes.h2o, 0.1)


if __name__ == "__main__":
    pytest.main()