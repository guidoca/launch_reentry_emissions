"""Class containing the plume postcombustion model for the rocket emissions module.

Based on National Academies of Sciences, Engineering, and Medicine 2021. Commercial
Space Vehicle Emissions Modeling. Washington, DC: The National Academies
Press. https://doi.org/10.17226/26142
"""

import typing

import numpy as np
import numpy.typing as npt

EI_NOX_S_SL = 33.0 / 1000
"""Mean Secondary NOx emissions at sea level for the Atlas V and Delta IV engines, in kg/kg."""
EI_CO_RED = 0.0025
EI_BC_RED = 0.04
EI_BC_P = 0.25

H_STRATO = 15.0
"""Altitude, in km, up to which exponential model is valid."""

METERS_TO_KM = 1e-3
"""Conversion between meters and km"""

M_CO = 28.0
M_CO2 = 44.0
M_BC = 12.0
M_CH4 = 16.0
M_H2O = 18.0
M_H = 1.0
M_H2 = 2.0


class PrimaryEmissionIndexes(typing.NamedTuple):
    """Primary emission indexes for the PlumePostcombustionModel."""

    nox: float = 0.0
    co2: float = 0.0
    co: float = 0.0
    ch4: float = 0.0
    bc: float = 0.0
    h2o: float = 0.0
    h: float = 0.0
    h2: float = 0.0
    oh: float = 0.0


class FinalEmissionIndexes(typing.NamedTuple):
    """Final emission indexes for the PlumePostcombustionModel."""

    nox: npt.NDArray[np.float64] | float
    co2: npt.NDArray[np.float64] | float
    co: npt.NDArray[np.float64] | float
    ch4: float
    bc: npt.NDArray[np.float64] | float
    h2o: npt.NDArray[np.float64] | float
    h: float
    h2: float
    oh: float


class PlumePostcombustionModel:
    """Class containing the plume postcombustion model for the rocket emissions module.

    Based on National Academies of Sciences, Engineering, and Medicine 2021. Commercial
    Space Vehicle Emissions Modeling. Washington, DC: The National Academies
    Press. https://doi.org/10.17226/26142
    """

    __slots__ = "h", "primary_emission_indexes"

    def __init__(
        self,
        h: npt.NDArray[np.float64] | float | None = None,
        primary_emission_indexes: PrimaryEmissionIndexes = None,
    ):
        self.h = h
        self.primary_emission_indexes = primary_emission_indexes

    def evaluate(
        self,
        h: npt.NDArray[np.float64] | float | None = None,
        primary_emission_indexes: PrimaryEmissionIndexes = None,
    ) -> FinalEmissionIndexes:
        """Evaluate the plume postcombustion model.

        Args:
            h: Altitude [m].
            species: Primary emission indexes.

        Returns:
            Final emission indexes.
        """
        if h is not None:
            self.h = h
        else:
            h = self.h
        if primary_emission_indexes is not None:
            self.primary_emission_indexes = primary_emission_indexes
        else:
            primary_emission_indexes = self.primary_emission_indexes
        return FinalEmissionIndexes(
            nox=PlumePostcombustionModel.final_nitrogen_oxides(
                h, primary_emission_indexes.nox
            ),
            co2=PlumePostcombustionModel.final_carbon_dioxide(
                h,
                primary_emission_indexes.co2,
                primary_emission_indexes.co,
                primary_emission_indexes.ch4,
                primary_emission_indexes.bc,
            ),
            co=PlumePostcombustionModel.final_carbon_monoxide(
                h, primary_emission_indexes.co, primary_emission_indexes.co2
            ),
            ch4=primary_emission_indexes.ch4,
            bc=PlumePostcombustionModel.final_black_carbon(
                h, primary_emission_indexes.bc
            ),
            h2o=PlumePostcombustionModel.final_water_vapor(
                primary_emission_indexes.h2o,
                primary_emission_indexes.h,
                primary_emission_indexes.h2,
                primary_emission_indexes.oh,
            ),
            h=primary_emission_indexes.h,
            h2=primary_emission_indexes.h2,
            oh=primary_emission_indexes.oh,
        )

    def final_nitrogen_oxides(
        h: npt.NDArray[np.float64] | float,
        ei_nox_p: float,
        ei_nox_s_sl: float = EI_NOX_S_SL,
    ) -> npt.NDArray[np.float64] | float:
        """Calculate the nitrogen oxides postcombustion emission index.

        Sums up primary and secondary emissions from interaction of the plume with atmosphere.

        Args:
        """
        # Calculate the co2 emissions from carbon monoxide post-combustion
        ei_nox_s = ei_nox_s_sl * np.exp(-0.26 * h * METERS_TO_KM)
        return ei_nox_p + ei_nox_s

    def final_carbon_dioxide(
        h: npt.NDArray[np.float64] | float,
        ei_co2_p: float,
        ei_co_p: float,
        ei_ch4_p: float,
        ei_bc_p: float,
        mw_co2: float = M_CO2,
        mw_co: float = M_CO,
        mw_ch4: float = M_CH4,
        mw_bc: float = M_BC,
    ) -> npt.NDArray[np.float64] | float:
        """Calculate the carbon dioxide postcombustion emission index."""
        # Calculate the co2 emissions from methane post-combustion. Assumed 0, as no model available.
        ei_from_ch4_f = 0.0 * ei_ch4_p
        # Calculate the co2 emissions from black carbon post-combustion
        ei_from_bc_f = ei_bc_p - PlumePostcombustionModel.final_black_carbon(h, ei_bc_p)

        # Calculate the co2 emissions from carbon monoxide post-combustion
        ei_from_co_f = ei_co_p - PlumePostcombustionModel.final_carbon_monoxide(
            h, ei_co_p, ei_co2_p
        )
        return (
            ei_co2_p
            + mw_co2 / mw_co * ei_from_co_f
            + mw_co2 / mw_ch4 * ei_from_ch4_f
            + mw_co2 / mw_bc * ei_from_bc_f
        )

    @staticmethod
    def final_water_vapor(
        ei_h2o_p: float,
        ei_h_p: float,
        ei_h2_p: float,
        ei_oh_p: float,
        mw_h2o: float = M_H2O,
        mw_h: float = M_H,
        mw_h2: float = M_H2,
    ) -> float:
        """Calculate the water vapor postcombustion emission index.

        Atom hydrogen and hydrogen molecules are assumed to post-combust with
        oxygen from the sorrouindg air to form water vapor.
        Note that this assumption might not hold at high altitudes were atmospheric oxygen might be negligible.

        Args:
            ei_h2o_p: Primary emission index of water vapor [M_H2O/M_EX].
            ei_h_p: Primary emission index of hydrogen [M_H/M_EX].
            ei_h2_p: Primary emission index of hydrogen molecules [M_H2/M_EX].
            ei_oh_p: Primary emission index of hydroxyl radicals [M_OH/M_EX].
            mw_h2o: Molecular weight of water [g/mol].
            mw_h: Molecular weight of hydrogen [g/mol].
            mw_h2: Molecular weight of hydrogen molecules [g/mol].

        """
        return ei_h2o_p + mw_h2o / mw_h * ei_h_p + mw_h2o / mw_h2 * ei_h2_p + ei_oh_p

    @staticmethod
    def final_black_carbon_stratosphere(
        h: npt.NDArray[np.float64] | float, ei_bc_p: float = EI_BC_P
    ) -> npt.NDArray[np.float64] | float:
        """Calculate the black carbon postcombustion emission index from 15 km altitude using the exponential model.

        Note that the burned fraction should be added as CO2 emitted with the corresponding molecular weight fraction.

        Args:
            h: Altitude [m]. From 15 km to 42 km.
            ei_bc_p: Primary emission index of black carbon [M_BC/M_EX].

        Returns:
            Final black carbon emission index [M_BC/M_EX].
        """
        return ei_bc_p * EI_BC_RED * np.exp(0.12 * (h * METERS_TO_KM - H_STRATO))

    @staticmethod
    def final_black_carbon(
        h: npt.NDArray[np.float64] | float, ei_bc_p: float = EI_BC_P
    ) -> npt.NDArray[np.float64]:
        """Calculate the black carbon postcombustion emission index.

        Note that the burned fraction should be added as CO2 emitted with the corresponding molecular weight fraction.

        Args:
            h: Altitude [m].
            ei_bc_p: Primary emission index of black carbon [M_BC/M_EX].

        Returns:
            Final black carbon emission index [M_BC/M_EX].
        """

        # Calculate the black carbon postcombustion fraction above 15 km
        bc_high = np.minimum(
            np.full_like(h, ei_bc_p, dtype=np.float64),
            PlumePostcombustionModel.final_black_carbon_stratosphere(h, ei_bc_p),
        )

        # Clamp to the minimum for below 15 km.
        return np.maximum(
            np.full_like(h, ei_bc_p * EI_BC_RED, dtype=np.float64), bc_high
        )

    @staticmethod
    def final_carbon_monoxide_lower(
        h: npt.NDArray[np.float64] | float, ei_co_p: float, ei_co2_p: float
    ) -> npt.NDArray[np.float64] | float:
        """Calculate the carbon monoxide postcombustion emission index at lower altitudes.


        Args:
            h: Altitude [m], valid until co_f>co_p.
            ei_co_p: Primary emission index of carbon monoxide [M_CO/M_EX].
            ei_co2_p: Primary emission index of carbon dioxide [M_CO2/M_EX].

        Returns:
            Final carbon monoxide emission index [M_CO/M_EX].
        """
        return EI_CO_RED * np.exp(0.067 * h * METERS_TO_KM) * (ei_co_p + ei_co2_p)

    @staticmethod
    def final_carbon_monoxide(
        h: npt.NDArray[np.float64] | float, ei_co_p: float, ei_co2_p: float
    ) -> npt.NDArray[np.float64]:
        """Calculate the carbon monoxide postcombustion emission index clamping maximum.


        Args:
            h: Altitude [m], valid until co_f>co_p.
            ei_co_p: Primary emission index of carbon monoxide [M_CO/M_EX].
            ei_co2_p: Primary emission index of carbon dioxide [M_CO2/M_EX].

        Returns:
            Final carbon monoxide emission index [M_CO/M_EX].
        """
        # Calculate the carbon monoxide postcombustion fraction without clamping at lower altitudes
        ei_co_f_lower = PlumePostcombustionModel.final_carbon_monoxide_lower(
            h, ei_co_p, ei_co2_p
        )
        # Clamp maximum value
        return np.minimum(np.full_like(h, ei_co_p, dtype=np.float64), ei_co_f_lower)
