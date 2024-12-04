"""Class containing the plume postcombustion for the rocket emissions module.

Based on National Academies of Sciences, Engineering, and Medicine 2021. Commercial
Space Vehicle Emissions Modeling. Washington, DC: The National Academies
Press. https://doi.org/10.17226/26142
"""

import dataclasses
import pathlib
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd

EI_NOX_S_SL = 33.0 / 1000
"""Mean Secondary NOx emissions at sea level for the Atlas V and Delta IV engines, in kg/kg."""
EI_CO_RED = 0.0025
EI_BC_RED = 0.04
EI_BC_P = 0.25

H_STRATO = 15.0
"""Altitude, in km, up to which exponential model is valid."""

METERS_TO_KM = 1e-3
"""Conversion between meters and km"""


@dataclasses.dataclass(init=False, frozen=True)
class MolarMassSpecies:
    """Molar masses used for the PlumePostcombustion."""

    nox: float
    co2: float = 44.0
    co: float = 28.0
    ch4: float = 16.0
    bc: float = 12.0
    h2o: float = 18.0
    h: float = 1.0
    h2: float = 2.0
    oh: float = 17.0


class PrimaryEmissionIndexes(typing.NamedTuple):
    """Primary emission indexes for the PlumePostcombustion."""

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
    """Final emission indexes for the PlumePostcombustion."""

    nox: npt.NDArray[np.floating] | float
    co2: npt.NDArray[np.floating] | float
    co: npt.NDArray[np.floating] | float
    ch4: npt.NDArray[np.floating] | float
    bc: npt.NDArray[np.floating] | float
    h2o: npt.NDArray[np.floating] | float
    h: npt.NDArray[np.floating] | float
    h2: npt.NDArray[np.floating] | float
    oh: npt.NDArray[np.floating] | float


class FinalEmissions(typing.NamedTuple):
    """Named tuple containing emissions."""

    nox: npt.NDArray[np.floating] | float
    co2: npt.NDArray[np.floating] | float
    co: npt.NDArray[np.floating] | float
    ch4: npt.NDArray[np.floating] | float
    bc: npt.NDArray[np.floating] | float
    h2o: npt.NDArray[np.floating] | float
    h: npt.NDArray[np.floating] | float
    h2: npt.NDArray[np.floating] | float
    oh: npt.NDArray[np.floating] | float


class PlumePostcombustionEmissionIndexes:
    """Class containing the plume postcombustion emission indexes for the rocket emissions module.

    Based on National Academies of Sciences, Engineering, and Medicine 2021. Commercial
    Space Vehicle Emissions Modeling. Washington, DC: The National Academies
    Press. https://doi.org/10.17226/26142
    """

    __slots__ = "h", "primary_emission_indexes"

    def __init__(
        self,
        primary_emission_indexes: PrimaryEmissionIndexes,
        h: npt.NDArray[np.floating] | float | None = None,
    ) -> None:
        """Initialize the PlumePostcombustion model.

        Args:
            primary_emission_indexes: Primary emission indexes.
            h: Altitude [m]. Defaults to None.
        """
        self.h = h
        self.primary_emission_indexes = primary_emission_indexes

    @classmethod
    def from_df(
        cls,
        primary_emission_indexes_df: pd.DataFrame,
        h: npt.NDArray[np.floating] | float | None = None,
    ) -> "PlumePostcombustionEmissionIndexes":
        """Initialize the model from a CSV file.

        Args:
            primary_emission_indexes_df: DataFrame containing the primary emission indexes.
            h: Altitude [m]. Defaults to None.

        Returns:
            PlumePostcombustionEmissionIndexes.
        """
        # Convert the column names of the dataframe to lower case to match the PrimaryEmissionIndexes fields
        primary_emission_indexes_df.columns = (
            primary_emission_indexes_df.columns.str.lower()
        )

        # Create the PrimaryEmissionIndexes instance using the dataframe values
        primary_emission_indexes = PrimaryEmissionIndexes(
            **{
                field: primary_emission_indexes_df[field].iloc[0]
                for field in PrimaryEmissionIndexes._fields
            }
        )
        return cls(primary_emission_indexes, h)

    @classmethod
    def from_csv(
        cls,
        file_path: str | pathlib.Path,
        h: npt.NDArray[np.floating] | float | None = None,
    ) -> "PlumePostcombustionEmissionIndexes":
        """Initialize the PlumePostcombustion model from a CSV file.

        Args:
            file_path: Path to the CSV file.
            h: Altitude [m]. Defaults to None.

        Returns:
            PlumePostcombustionEmissionIndexes.
        """
        return cls.from_df(pd.read_csv(file_path), h)

    def evaluate(
        self,
        h: npt.NDArray[np.floating] | float | None = None,
    ) -> FinalEmissionIndexes:
        """Evaluate the plume postcombustion.

        Args:
            h: Altitude [m].

        Returns:
            Final emission indexes.
        """
        if h is None and self.h is not None:
            h = self.h
        elif h is None:
            raise ValueError("Altitude h must be provided.")
        primary_emission_indexes = self.primary_emission_indexes
        return FinalEmissionIndexes(
            nox=PlumePostcombustionEmissionIndexes.final_nitrogen_oxides(
                h, primary_emission_indexes.nox
            ),
            co2=PlumePostcombustionEmissionIndexes.final_carbon_dioxide(
                h,
                primary_emission_indexes.co2,
                primary_emission_indexes.co,
                primary_emission_indexes.ch4,
                primary_emission_indexes.bc,
            ),
            co=PlumePostcombustionEmissionIndexes.final_carbon_monoxide(
                h, primary_emission_indexes.co, primary_emission_indexes.co2
            ),
            ch4=PlumePostcombustionEmissionIndexes.final_carbon_monoxide(
                h, primary_emission_indexes.ch4, primary_emission_indexes.co2
            ),
            bc=PlumePostcombustionEmissionIndexes.final_black_carbon(
                h, primary_emission_indexes.bc
            ),
            h2o=PlumePostcombustionEmissionIndexes.final_water_vapor(
                h,
                primary_emission_indexes.h2o,
                primary_emission_indexes.h,
                primary_emission_indexes.h2,
                primary_emission_indexes.oh,
            ),
            h=PlumePostcombustionEmissionIndexes.final_carbon_monoxide(
                h,
                primary_emission_indexes.h,
                primary_emission_indexes.h2o,
            ),  # Assumed similar to carbon monoxide
            h2=PlumePostcombustionEmissionIndexes.final_carbon_monoxide(
                h,
                primary_emission_indexes.h2,
                primary_emission_indexes.h2o,
            ),  # Assumed similar to carbon monoxide,
            oh=PlumePostcombustionEmissionIndexes.final_carbon_monoxide(
                h,
                primary_emission_indexes.oh,
                primary_emission_indexes.h2o,
            ),  # Assumed similar to carbon monoxide,,
        )

    @staticmethod
    def final_nitrogen_oxides(
        h: npt.NDArray[np.floating] | float,
        ei_nox_p: npt.NDArray[np.floating] | float,
        ei_nox_s_sl: npt.NDArray[np.floating] | float = EI_NOX_S_SL,
    ) -> npt.NDArray[np.floating] | float:
        """Calculate the nitrogen oxides postcombustion emission index.

        Sums up primary and secondary emissions from interaction of the plume with atmosphere.

        Args:
            h: Altitude [m].
            ei_nox_p: Primary emission index of nitrogen oxides [M_NOx/M_EX].
            ei_nox_s_sl: Secondary emission index of nitrogen oxides at sea level [M_NOx/M_EX].

        Returns:
            Final nitrogen oxides emission index [M_NOx/M_EX].
        """
        # Calculate the co2 emissions from carbon monoxide post-combustion
        ei_nox_s = ei_nox_s_sl * np.exp(-0.26 * h * METERS_TO_KM)
        return ei_nox_p + ei_nox_s

    @staticmethod
    def final_carbon_dioxide(
        h: npt.NDArray[np.floating] | float,
        ei_co2_p: float,
        ei_co_p: float,
        ei_ch4_p: float,
        ei_bc_p: float,
        mw_co2: float = MolarMassSpecies.co2,
        mw_co: float = MolarMassSpecies.co,
        mw_ch4: float = MolarMassSpecies.ch4,
        mw_bc: float = MolarMassSpecies.bc,
    ) -> npt.NDArray[np.floating] | float:
        """Calculate the carbon dioxide postcombustion emission index.

        Args:
            h: Altitude [m].
            ei_co2_p: Primary emission index of carbon dioxide [M_CO2/M_EX].
            ei_co_p: Primary emission index of carbon monoxide [M_CO/M_EX].
            ei_ch4_p: Primary emission index of methane [M_CH4/M_EX].
            ei_bc_p: Primary emission index of black carbon [M_BC/M_EX].
            mw_co2: Molecular weight of carbon dioxide [g/mol].
            mw_co: Molecular weight of carbon monoxide [g/mol].
            mw_ch4: Molecular weight of methane [g/mol].
            mw_bc: Molecular weight of black carbon [g/mol].

        Returns:
            Final carbon dioxide emission index [M_CO2/M_EX].
        """
        # Calculate the co2 emissions from methane post-combustion.
        # We assume similar relation as to carbon monoxide, as no model is available.
        # This is a simplification, as the reaction of methane with oxygen is more complex.
        ei_from_ch4_f = (
            ei_ch4_p
            - PlumePostcombustionEmissionIndexes.final_carbon_monoxide(
                h, ei_ch4_p, ei_co2_p
            )
        )
        # Calculate the co2 emissions from black carbon post-combustion
        ei_from_bc_f = ei_bc_p - PlumePostcombustionEmissionIndexes.final_black_carbon(
            h, ei_bc_p
        )

        # Calculate the co2 emissions from carbon monoxide post-combustion
        ei_from_co_f = (
            ei_co_p
            - PlumePostcombustionEmissionIndexes.final_carbon_monoxide(
                h, ei_co_p, ei_co2_p
            )
        )
        return (
            ei_co2_p
            + mw_co2 / mw_co * ei_from_co_f
            + mw_co2 / mw_ch4 * ei_from_ch4_f
            + mw_co2 / mw_bc * ei_from_bc_f
        )

    @staticmethod
    def final_water_vapor(
        h: npt.NDArray[np.floating] | float,
        ei_h2o_p: float,
        ei_h_p: float,
        ei_h2_p: float,
        ei_oh_p: float,
        mw_h2o: float = MolarMassSpecies.h2o,
        mw_h: float = MolarMassSpecies.h,
        mw_h2: float = MolarMassSpecies.h2,
    ) -> npt.NDArray[np.floating] | float:
        """Calculate the water vapor postcombustion emission index.

        Atom hydrogen and hydrogen molecules are assumed to post-combust with
        oxygen from the sorrouindg air to form water vapor.
        Note that this assumption might not hold at high altitudes were atmospheric oxygen might be negligible.

        Args:
            h: Altitude [m].
            ei_h2o_p: Primary emission index of water vapor [M_H2O/M_EX].
            ei_h_p: Primary emission index of hydrogen [M_H/M_EX].
            ei_h2_p: Primary emission index of hydrogen molecules [M_H2/M_EX].
            ei_oh_p: Primary emission index of hydroxyl radicals [M_OH/M_EX].
            mw_h2o: Molecular weight of water [g/mol].
            mw_h: Molecular weight of hydrogen [g/mol].
            mw_h2: Molecular weight of hydrogen molecules [g/mol].

        Returns:
            Final water vapor emission index [M_H2O/M_EX].

        """
        # Calculate the h2o emissions from atomic hydrogen post-combustion
        # We assume similar relation as to carbon monoxide, as no model is available.
        # This is a simplification, as the reaction of atomic hydrogen with oxygen is more complex.
        ei_from_h_f = ei_h_p - PlumePostcombustionEmissionIndexes.final_carbon_monoxide(
            h,
            ei_h_p,
            ei_h2o_p,
        )
        # Calculate the h2o emissions from hydrogen post-combustion
        # We assume similar relation as to carbon monoxide, as no model is available.
        # This is a simplification, as the reaction of hydrogen with oxygen is more complex.
        ei_from_h2_f = (
            ei_h2_p
            - PlumePostcombustionEmissionIndexes.final_carbon_monoxide(
                h,
                ei_h2_p,
                ei_h2o_p,
            )
        )
        # Calculate the h2o emissions from hydroxil radical post-combustion
        # We assume similar relation as to carbon monoxide, as no model is available.
        # This is a simplification, as the reaction of hydrogen with oxygen is more complex.
        ei_from_oh_f = (
            ei_oh_p
            - PlumePostcombustionEmissionIndexes.final_carbon_monoxide(
                h,
                ei_oh_p,
                ei_h2o_p,
            )
        )
        return (
            ei_h2o_p
            + mw_h2o / mw_h * ei_from_h_f
            + mw_h2o / mw_h2 * ei_from_h2_f
            + ei_from_oh_f
        )

    @staticmethod
    def final_black_carbon_stratosphere(
        h: npt.NDArray[np.floating] | float, ei_bc_p: float = EI_BC_P
    ) -> npt.NDArray[np.floating] | float:
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
        h: npt.NDArray[np.floating] | float, ei_bc_p: float = EI_BC_P
    ) -> npt.NDArray[np.floating]:
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
            PlumePostcombustionEmissionIndexes.final_black_carbon_stratosphere(
                h, ei_bc_p
            ),
        )

        # Clamp to the minimum for below 15 km.
        return np.maximum(
            np.full_like(h, ei_bc_p * EI_BC_RED, dtype=np.float64), bc_high
        )

    @staticmethod
    def final_carbon_monoxide_lower(
        h: npt.NDArray[np.floating] | float,
        ei_co_p: npt.NDArray[np.floating] | float,
        ei_co2_p: npt.NDArray[np.floating] | float,
        ei_co_red: npt.NDArray[np.floating] | float = EI_CO_RED,
    ) -> npt.NDArray[np.floating] | float:
        """Calculate the carbon monoxide postcombustion emission index at lower altitudes.

        Args:
            h: Altitude [m], valid until co_f>co_p.
            ei_co_p: Primary emission index of carbon monoxide [M_CO/M_EX].
            ei_co2_p: Primary emission index of carbon dioxide [M_CO2/M_EX].
            ei_co_red: Reduction of the emission index of carbon monoxide at sea level [M_CO/M_EX].

        Returns:
            Final carbon monoxide emission index [M_CO/M_EX].
        """
        return ei_co_red * np.exp(0.067 * h * METERS_TO_KM) * (ei_co_p + ei_co2_p)

    @staticmethod
    def final_carbon_monoxide(
        h: npt.NDArray[np.floating] | float,
        ei_co_p: npt.NDArray[np.floating] | float,
        ei_co2_p: npt.NDArray[np.floating] | float,
        ei_co_red: npt.NDArray[np.floating] | float = EI_CO_RED,
    ) -> npt.NDArray[np.floating]:
        """Calculate the carbon monoxide postcombustion emission index clamping maximum.

        Args:
            h: Altitude [m], valid until co_f>co_p.
            ei_co_p: Primary emission index of carbon monoxide [M_CO/M_EX].
            ei_co2_p: Primary emission index of carbon dioxide [M_CO2/M_EX].
            ei_co_red: Reduction of the emission index of carbon monoxide at sea level [M_CO/M_EX].

        Returns:
            Final carbon monoxide emission index [MM_CO/M_EX].
        """
        # Calculate the carbon monoxide postcombustion fraction without clamping at lower altitudes
        ei_co_f_lower = PlumePostcombustionEmissionIndexes.final_carbon_monoxide_lower(
            h, ei_co_p, ei_co2_p, ei_co_red
        )
        # Clamp maximum value
        return np.minimum(np.full_like(h, ei_co_p, dtype=np.float64), ei_co_f_lower)


class PlumePostcombustionEmissions:
    """Class for calculating final emissions from the emission index and mass burned."""

    @staticmethod
    def evaluate(
        mass_burned: npt.NDArray[np.floating] | float,
        final_emission_indexes: FinalEmissionIndexes,
    ) -> FinalEmissions:
        """Evaluate the emissions.

        Args:
            mass_burned: Mass burned.
            final_emission_indexes: Final emission indexes.

        Returns:
            Final emissions.
        """
        emissions = []
        for emm in final_emission_indexes:
            emissions.append(emm * mass_burned)
        return FinalEmissions(*emissions)


class PlumePostcombustion:
    """Class for calculating final emissions from the emission index and mass burned."""

    __slots__ = "h", "mass_burned", "plume_postcombustion_emission_indexes"

    def __init__(
        self,
        plume_postcombustion_emission_indexes: PlumePostcombustionEmissionIndexes,
        h: npt.NDArray[np.floating] | float | None = None,
        mass_burned: npt.NDArray[np.floating] | float | None = None,
    ) -> None:
        """Initialize the PlumePostcombustion model.

        Args:
            plume_postcombustion_emission_indexes: PlumePostcombustionEmissionIndexes.
            h: Altitude [m]. Defaults to None.
            mass_burned: Mass burned [kg]. Defaults to None.
        """
        self.plume_postcombustion_emission_indexes = (
            plume_postcombustion_emission_indexes
        )
        self.h = h
        self.mass_burned = mass_burned

    def evaluate(
        self,
        h: npt.NDArray[np.floating] | float | None = None,
        mass_burned: npt.NDArray[np.floating] | float | None = None,
    ) -> FinalEmissions:
        """Evaluate the emissions.

        Args:
            h: Altitude [m]. Defaults to None.
            mass_burned: Altitude [m]. Defaults to None.

        Returns:
            Final emissions.
        """
        if h is None and self.h is not None:
            h = self.h
        elif h is None:
            raise ValueError("Altitude h must be provided.")
        if mass_burned is None and self.mass_burned is not None:
            mass_burned = self.mass_burned
        elif mass_burned is None:
            raise ValueError("Altitude h must be provided.")
        final_emission_indexes = self.plume_postcombustion_emission_indexes.evaluate(
            h=h
        )
        return PlumePostcombustionEmissions.evaluate(
            mass_burned=mass_burned, final_emission_indexes=final_emission_indexes
        )
