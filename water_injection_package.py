#!/usr/bin/python
# -*- coding: utf-8 -*-

#----------------------------------------------------------------------------
#   Copyright (C) 2017 <Alexandre Vaudrey>                                  |
#                                                                           |
#   This program is free software: you can redistribute it and/or modify    |
#   it under the terms of the GNU General Public License as published by    |
#   the Free Software Foundation, either version 3 of the License, or       |
#   (at your option) any later version.                                     |
#                                                                           |
#   This program is distributed in the hope that it will be useful,         |
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          |
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           |
#   GNU General Public License for more details.                            |
#                                                                           |
#   You should have received a copy of the GNU General Public License       |
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.   |
#---------------------------------------------------------------------------|

import numpy as np
import scipy.optimize as sp

__docformat__ = "restructuredtext en"
__author__ = "Alexandre Vaudrey <alexandre.vaudrey@gmail.com>"
__date__ = "26/12/2016"

# ---- Needed physical datas --------------------------------------------------
# Atomic weights of elements used in the fuel chemical composition
ATOMIC_WEIGHTS = {'C':12.011, 'H':1.008, 'O':15.999, 'N':14.007, 'S':32.06}
# Ratio of nitrogen to oxygen molar quantities in dry air
NITROGEN_OXYGEN_RATIO = 0.79/0.21
# Universal constant of ideal gas
IDEAL_GASES_CONSTANT = 8.3144486 # J/(mol.K)
# Specific heats at constant pressure of dry air and water vapor, in [J/(kg.K)]
DRY_AIR_CP, WATER_VAPOR_CP = 1004., 1805.
# Specific gas constants of dry air and water vapor, in [J/(kg.K)]
DRY_AIR_R, WATER_VAPOR_R = 287., 462.
# Water latent heat of vaporization, in [J/(kg.K)]
WATER_LW = 2501e+3
# Molar mass of the dry air
DRY_AIR_M = 28.9645
# Ratio of the molecular mass of water on the standard dry air one
ALPHAW = (ATOMIC_WEIGHTS['H']*2+ATOMIC_WEIGHTS['O'])/DRY_AIR_M


# The class fuel
class Fuel(dict):
    """
    Chemical fuel represented by its chemical composition (of the type CHONS)
    and by its gaseous specific heat at constant pressure.
    """
    # Attributes --------------------------------------------------------------
    def __init__(self, composition=None, specif_heat=1644.):
        dict.__init__(self)
        composition = composition or {'C':8, 'H':18, 'O':0, 'N':0, 'S':0}
        # Name of the fuel, octane by default
        self.fuel_name = 'octane'
        # Chemical composition of the fuel, represented by a dictionnary
        # containing the numbers of atoms of Carbon (C), Hydrogen (H),
        # Oxygen (O), Nitrogen (N) and Sulfur (S), so the CHONS.
        self.fuel_composition = composition
        # Specific heat at constant pressure, in [J/(kg.K)]
        self.fuel_specif_heat_at_cste_p = specif_heat
    # Methods -----------------------------------------------------------------
    def fuel_molar_mass(self):
        """Calculation of the fuel molar mass thanks to the chemical
        composition."""
        # Initialization
        molar_mass = 0.0
        for comp in ['C', 'H', 'O', 'N', 'S']:
            molar_mass += self.fuel_composition[comp]*ATOMIC_WEIGHTS[comp]
        return molar_mass
    def stoichiometric_air_fuel_ratio(self):
        """Stoichiometric value of the Air-Fuel Ratio (AFR)."""
        # Calculation of the stoichiometric coefficient of oxygen if burnt
        # with this fuel
        oxygen_stoichiometric_coeff = self.fuel_composition['C']\
        +0.25*self.fuel_composition['H']-0.5*self.fuel_composition['O']\
        +self.fuel_composition['S']
        # And of the AFR
        afrs = (1+NITROGEN_OXYGEN_RATIO*ATOMIC_WEIGHTS['N']/\
                (ATOMIC_WEIGHTS['O']))*oxygen_stoichiometric_coeff*2*\
                ATOMIC_WEIGHTS['O']/self.fuel_molar_mass()
        return afrs
    def stoichiometric_fuel_air_ratio(self):
        """Stoichiometric value of the Fuel-Air Ratio (FAR)."""
        return 1/self.stoichiometric_air_fuel_ratio()
    def fuel_ideal_gas_specif_r(self):
        """The specif constant 'r' used in the ideal gas law, in
        [J/(kg.K)]."""
        return IDEAL_GASES_CONSTANT*1e+3/self.fuel_molar_mass()
    def fuel_specif_heat_at_cste_v(self):
        """ Specific heat at constant volume cV [J/(kg.K)] for the fuel
        considered as a gas."""
        # Use of the famous Mayer relation for ideal gases : cV = cp - r
        return self.fuel_specif_heat_at_cste_p-\
                self.fuel_ideal_gas_specif_r()

class FreshMixture(Fuel):
    """
    Chemical composition and specific enthalpy of a fresh mixture aspirated by
    an internal combustion engine and composed of air, water and sometimes fuel.
    """
    # Attributes --------------------------------------------------------------
    def __init__(self, air_fuel_equivalent_ratio=1.0, ambient_temperature=298.,\
                ambient_relative_humidity=0.5, ambient_pressure=1.0):
        # Initialization of the class fuel
        Fuel.__init__(self)
        # Name of the fresh mixture
        self.mix_name = 'fresh_mixture-1'
        # Equivalent air fuel ratio, usually noted lambda. The default value is
        # here corresponding to a stoichiometric combustion.
        self.air_fuel_equivalent_ratio = air_fuel_equivalent_ratio
        # The amount of water vapor at the entrance of the intake system comes
        # from the values of both the ambient temperature (in [K]) and the
        # relative humidity.
        self.ambient_temperature = ambient_temperature
        self.ambient_relative_humidity = ambient_relative_humidity
        # Pressure of the fresh charge, in [bar]. The default value is 1 bar.
        self.ambient_pressure = ambient_pressure
    # Methods -----------------------------------------------------------------
    # ---- General
    @staticmethod
    def equilibrium_vapor_pressure(temperature):
        """ Calculation of the equilibrium water vapor pressure, in [Pa]
        according to the correlation from Wexler (1976). Temperature must
        be entered in Kelvins, from 173.15 K to 473.15 K."""
        if (temperature < 273.) or (temperature > 433.15):
            raise ValueError("'temperature' must be 273.15K < T < 433.15K")
        # Empirical coefficients
        wexler_coeff = np.array([-5.8002206e+03, 1.3914993e+00, -4.8640239e-02,\
                                 4.1764768e-05, -1.4452093e-08, 6.5459673])
        # And the Wexler formula which calculates first the value of ln(p)
        lnp = np.dot(wexler_coeff[:-1], np.power(temperature, range(-1, 4)))\
                +wexler_coeff[-1]*np.log(temperature)
        return np.exp(lnp)
    def specif_humidity(self, pressure, theta, relative_h):
        """ Specific humidity/Moisture content/Humidity ratio, defined
        as the ratio of the water vapor mass on the sole dry air one.
        Pressure 'p' is in [bar], relative temperature 'theta' in [°C] and
        relative humidity 'relative_h' is dimensionless."""
        return ALPHAW/(pressure*1e+5/(relative_h*self.equilibrium_vapor_pressure(theta))-1)
    def specif_enthalpy(self, theta, omega):
        """ Calculation of the specific enthalpy of the fresh mixture, in
        [J/(kg.K)], from the values of relative temperature 'theta' (in [°C])
        and specific humidity 'omega'."""
        return (self.dry_mix_specif_heat_at_cste_p()\
                +omega*WATER_VAPOR_CP)*theta+omega*WATER_LW
    def mass_fractions(self, omega):
        """ Mass fractions of fuel, air and water vapor for a given value of the
        specific humidity 'omega', as a tuple."""
        # Actual Fuel-Air Ratio (FAR)
        far = self.fuel_air_ratio()
        return (far/((1+far)*(1+omega)), 1/((1+far)*(1+omega)), omega/(1+omega))
    # ---- Ambient state/before the water injection process
    def ambient_specif_humidity(self):
        """ Specific humidity/Moisture content/Humidity ratio, defined
        as the ratio of the water vapor mass on the sole dry air one."""
        # Calculation are done outside the intake system, so with no
        # mention of the fuel.
        return self.specif_humidity(self.ambient_pressure,\
                                      self.ambient_temperature,\
                                      self.ambient_relative_humidity)
    def ambient_specif_enthalpy(self):
        """ Specific enthalpy of the fresh mixture in the ambient state."""
        return self.specif_enthalpy(self.ambient_temperature-273.15,\
                                      self.ambient_specif_humidity())
    def entrance_mass_fractions(self):
        """ Mass fractions of fuel, air and water vapor at the entrance
        point, as a tuple."""
        return self.mass_fractions(self.ambient_specif_humidity())
    # ---- Mixture state
    def air_fuel_ratio(self):
        """ Actual Air-Fuel Ratio (AFR) of the fresh mixture."""
        return self.air_fuel_equivalent_ratio*\
                self.stoichiometric_air_fuel_ratio()
    def fuel_air_ratio(self):
        """ Actual Fuel-Air Ratio (FAR) of the fresh mixture."""
        return 1/self.air_fuel_ratio()
    def dry_mix_specif_heat_at_cste_p(self):
        """ Specific heat at constant pressure (cp) of the dry fresh mixture
        (without water vapor), in [J/(kg.K)]."""
        # Actual Fuel-Air Ratio (FAR)
        far = self.fuel_air_ratio()
        # And the specific heat of the blend of dry air and fuel
        return (DRY_AIR_CP+far*self.fuel_specif_heat_at_cste_p)/(1+far)
    def dry_mix_specif_heat_at_cste_v(self):
        """ Specific heat at constant volume (cV) of the dry fresh mixture
        (without water vapor), in [J/(kg.K)]."""
        # Actual Fuel-Air Ratio (FAR)
        far = self.fuel_air_ratio()
        # Specific heats at constant volumes, calculated thanks to the
        # Mayer relation
        cvair, cvwater = DRY_AIR_CP-DRY_AIR_R, WATER_VAPOR_CP-WATER_VAPOR_R
        # And the specific heat at constant volume of the blend
        return (cvair+far*cvwater)/(1+far)
    def dry_mix_heat_capacity_ratio(self):
        """ Heat capacity ratio of the dry fresh mixture."""
        # Specific heat at constant pressure
        c_p = self.dry_mix_specif_heat_at_cste_p()
        # Specific heat at constant pressure
        c_v = self.dry_mix_specif_heat_at_cste_v()
        return c_p/c_v
    def dry_mix_ideal_gas_specif_r(self):
        """ Specific gas constant (r of the ideal gas law) of the dry fresh
        mixture (without water vapor), in [J/(kg.K)]."""
        # Actual Fuel-Air Ratio (FAR)
        far = self.fuel_air_ratio()
        # And the ideal gas constant of the blend of dry air and fuel
        return (DRY_AIR_R+far*self.fuel_ideal_gas_specif_r())/(1+far)
    # ---- Saturation properties
    def maximum_specif_humidity(self, pressure, theta):
        """ Maximum value of the fresh mixture specific humidity at a
        given pressure p in [bar] and a given relative temperature 'theta'
        in [°C]."""
        # Local constant useful for the calculation
        alpha_fuel = self.fuel_molar_mass()/DRY_AIR_M
         # Actual Fuel-Air Ratio (FAR)
        far = self.fuel_air_ratio()
        return ALPHAW*(1+far/alpha_fuel)/(1+far)\
        *1/(pressure*1e+5/self.equilibrium_vapor_pressure(theta+273.15)-1)
    def wet_bulb_temperature(self, pressure, enthalpy):
        """ Wet-bulb temperature corresponding to a given value of the
        fresh mixture pressure in [bar] and specific enthalpy in [J/kg]."""
        # Function of the temperature theta whom the root corresponds
        # to the wet-bulb temperature.
        def f_to_solve(theta):
            result = self.dry_mix_specif_heat_at_cste_p()*theta\
            +self.maximum_specif_humidity(pressure, theta)*(WATER_VAPOR_CP*theta\
                                                      +WATER_LW)-enthalpy
            return result
        # The wet-bulb temperature is obtained thanks to the Newton
        # method applied to the function f, with the ambient temperature
        # as the initial/starting point.
        twb = sp.newton(f_to_solve, self.ambient_temperature-273.15)
        return twb
    def wet_bulb_specif_humidity(self, pressure, enthalpy):
        """ Specific humidity at the wet-bulb point fo a given pressure
        'p' in [bar] and a given specific enthalpy in [J/kg]."""
        # Local constant useful for the calculation
        alpha_fuel = self.fuel_molar_mass()/DRY_AIR_M
        # Actual Fuel-Air Ratio (FAR)
        far = self.fuel_air_ratio()
        # Wet-bulb temperature
        thetawb = self.wet_bulb_temperature(pressure, enthalpy)
        return ALPHAW*(1+far/alpha_fuel)/(1+far)\
        *1/(pressure*1e+5/self.equilibrium_vapor_pressure(thetawb+273.15)-1)
    def max_adiabatic_water_fuel_ratio(self, pressure_i, thetai):
        """ Maximum value of the Water-Fuel Ratio (WFR) if the injection
        process is supposed as adiabatic, for a given value of the
        intake 'pressure_i', in [bar] and temperature thetai, in [°C]."""
        # Actual Fuel-Air Ratio (FAR)
        far = self.fuel_air_ratio()
        # Intake value of the specific enthalpy
        enthalpy_i = self.specif_enthalpy(thetai, self.ambient_specif_humidity())
        # If the process is adiabatic, the maximum value of the specific
        # humidity is the wet-bulb one.
        wmax = self.wet_bulb_specif_humidity(pressure_i, enthalpy_i)
        return (1+far)/far*(wmax-self.ambient_specif_humidity())
    # ---- Moist fresh mixture
    # TODO : to Finish
    def moist_mix_specif_heat_at_cste_p(self, omega):
        """ Specific heat at constant pressure (cp) of the moist fresh mixture
        (with water vapor), in [J/(kg.K)], from a value of the specific
        humidity 'omega'."""
        pass
    def moist_mix_specif_heat_at_cste_v(self, omega):
        """ Specific heat at constant volume (cV) of the moist fresh mixture
        (with water vapor), in [J/(kg.K)], from a value of the specific
        humidity 'omega'."""
        pass
    def moist_mix_heat_capacity_ratio(self, omega):
        """ Heat capacity ratio of the moist fresh mixture (with water vapor),
        in [J/(kg.K)], from a value of the specific humidity 'omega'."""
        pass
    def moist_mix_ideal_gas_specif_r(self, omega):
        """ Specific gas constant (r of the ideal gas law) of the moist fresh
        mixture (with water vapor), in [J/(kg.K)], from a value of the specific
        humidity 'omega'."""
        pass
    def water_fuel_ratio(self, omega):
        """ Value of the Water-Fuel Ratio (WFR) required to obtain the value
        'omega' of the specific humidity."""
        # TODO : Check if the entered value is greater or not to the maximum
        # one corresponding to saturation.
        pass

if __name__ == '__main__':
    # Test of the class fuel using for example ethanol (C2H6O) as fuel
    ethanol = Fuel(composition={'C':2, 'H':6, 'O':1, 'N':0, 'S':0},\
                  specif_heat=1415.)
    ethanol.fuel_name = 'ethanol'
    print('---- Fuel : %s ----' % ethanol.fuel_name)
    print('Molar mass: M = %1.2f g/mol' % ethanol.fuel_molar_mass())
    print('Stoichiometric Air-Fuel Ratio: AFRs = %1.2f' %
          ethanol.stoichiometric_air_fuel_ratio())
    print('Stoichiometric Fuel-Air Ratio: FARs = %1.3f' %
          ethanol.stoichiometric_fuel_air_ratio())
    print('Ideal gas specific constant: r = %2.3f J/(kg.K)' %
          ethanol.fuel_ideal_gas_specif_r())
    print('Specific heat at constant volume: cV = %2.3f J/(kg.K)' %
          ethanol.fuel_specif_heat_at_cste_v())
    # Creation of a fresh mixture
    mixture = FreshMixture(ambient_temperature=293.)
    print('---- Fresh mixture: %s ----' % mixture.mix_name)
