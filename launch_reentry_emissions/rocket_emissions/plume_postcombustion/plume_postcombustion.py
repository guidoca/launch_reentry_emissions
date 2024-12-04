"""Class containing the plume postcombustion for the rocket emissions module.

Based on National Academies of Sciences, Engineering, and Medicine 2021. Commercial
Space Vehicle Emissions Modeling. Washington, DC: The National Academies
Press. https://doi.org/10.17226/26142
"""

import typing

import numpy as np
import numpy.typing as npt

from launch_reentry_emissions.rocket_emissions.plume_postcombustion.plume_emission_index import (
    FinalEmissionIndexes,
    PlumeEmissionIndexes,
)


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


class PlumeEmissions:
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
    """Class consisting of the plume emission indexes and plume emissions."""

    __slots__ = "h", "mass_burned", "plume_emission_indexes"

    def __init__(
        self,
        plume_emission_indexes: PlumeEmissionIndexes,
        h: npt.NDArray[np.floating] | float | None = None,
        mass_burned: npt.NDArray[np.floating] | float | None = None,
    ) -> None:
        """Initialize the PlumePostcombustion model.

        Args:
            plume_emission_indexes: PlumePostcombustionEmissionIndexes.
            h: Altitude [m]. Defaults to None.
            mass_burned: Mass burned [kg]. Defaults to None.
        """
        self.plume_emission_indexes = plume_emission_indexes
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
        final_emission_indexes = self.plume_emission_indexes.evaluate(h=h)
        return PlumeEmissions.evaluate(
            mass_burned=mass_burned, final_emission_indexes=final_emission_indexes
        )
