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
ATOMIC_WEIGHTS = {'C':12.011, 'H':1.008, 'O':15.999, 'N':14.007,'S':32.06}
# Ratio of nitrogen to oxygen molar quantities in dry air
NITROGEN_OXYGEN_RATIO = 0.79/0.21
# Universal constant of ideal gas
IDEAL_GASES_CONSTANT = 8.3144486 # J/(mol.K)
# Specific heats at constant pressure of dry air and water vapor, in [J/(kg.K)]
DRY_AIR_CP , WATER_VAPOR_CP = 1004. , 1805.
# Specific gas constants of dry air and water vapor, in [J/(kg.K)]
DRY_AIR_R , WATER_VAPOR_R = 287. , 462.
# Water latent heat of vaporization, in [J/(kg.K)]
WATER_LW = 2501e+3
# Molar mass of the dry air
DRY_AIR_M = 28.9645
# Ratio of the molecular mass of water on the standard dry air one
ALPHAW = (ATOMIC_WEIGHTS['H']*2+ATOMIC_WEIGHTS['O'])/DRY_AIR_M

# The class fuel
class fuel():
    """
    Chemical fuel represented by its chemical composition (of the type CHONS) 
    and by its gaseous specific heat at constant pressure.
    """
    # Attributes --------------------------------------------------------------
    def __init__(self,composition={'C':8, 'H':18, 'O':0, 'N':0, 'S':0},\
                specific_heat=1644.):
        # Name of the fuel, octane by default
        self.fuel_name = 'octane'
        # Chemical composition of the fuel, represented by a dictionnary 
        # containing the numbers of atoms of Carbon (C), Hydrogen (H),
        # Oxygen (O), Nitrogen (N) and Sulfur (S), so the CHONS.
        self.fuel_composition = composition
        # Specific heat at constant pressure, in [J/(kg.K)]
        self.fuel_specific_heat_at_cste_pressure = specific_heat
    # Methods -----------------------------------------------------------------
    def fuel_molar_mass(self):
        """Calculation of the fuel molar mass thanks to the chemical 
        composition."""
        # Initialization
        mm = 0.0
        for c in ['C','H','O','N','S']:
            mm += self.fuel_composition[c]*ATOMIC_WEIGHTS[c]
        return mm
    def stoichiometric_air_fuel_ratio(self):
        """Stoichiometric value of the Air-Fuel Ratio (AFR)."""
        # Calculation of the stoichiometric coefficient of oxygen if burnt 
        # with this fuel
        oxygen_stoichiometric_coeff = self.fuel_composition['C']\
        +0.25*self.fuel_composition['H']-0.5*self.fuel_composition['O']\
        +self.fuel_composition['S']
        # And of the AFR
        AFRs = (1+NITROGEN_OXYGEN_RATIO*ATOMIC_WEIGHTS['N']/\
                (ATOMIC_WEIGHTS['O']))*oxygen_stoichiometric_coeff*2*\
                ATOMIC_WEIGHTS['O']/self.fuel_molar_mass()
        return AFRs
    def stoichiometric_fuel_air_ratio(self):
        """Stoichiometric value of the Fuel-Air Ratio (FAR)."""
        return 1/self.stoichiometric_air_fuel_ratio()
    def ideal_gas_specific_constant(self):
        """The specific constant 'r' used in the ideal gas law, in
        [J/(kg.K)]."""
        return IDEAL_GASES_CONSTANT*1e+3/self.fuel_molar_mass()

class fresh_mixture(fuel):
    """
    Chemical composition and specific enthalpy of a fresh mixture aspirated by 
    an internal combustion engine and composed of air, water and sometimes fuel.
    """
    # Attributes --------------------------------------------------------------
    def __init__(self,air_fuel_equivalent_ratio=1.0,ambient_temperature=298.,\
                ambient_relative_humidity=0.5,ambient_pressure=1.0):
        # Initialization of the class fuel
        fuel.__init__(self)
        # Name of the fresh mixture
        self.mixture_name = 'fresh_mixture-1'
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
    def equilibrium_vapor_pressure(self,temperature):
        """ Calculation of the equilibrium water vapor pressure, in [Pa] 
        according tothe correlation from Wexler (1976). Temperature must 
        be entered in Kelvins, from 173.15 K to 473.15 K."""
        # Empirical coefficients
        wexler_coeff = np.array([-0.60436117e+4,0.189318833e+2,\
                                 -0.28235894e-1,0.17241129e-4,0.2858487e+1])
        # And the Wexler formula which calculates first the value of ln(p)
        lnp = np.dot(wexler_coeff[:-1],np.power(temperature,range(-1,3)))\
            +wexler_coeff[-1]*np.log(temperature)
        return np.exp(lnp)
    def specific_humidity(self,p,theta,HR):
        """ Specific humidity/Moisture content/Humidity ratio, defined
        as the ratio of the water vapor mass on the sole dry air one.
        Pressure 'p' is in [bar], relative temperature 'theta' in [°C] and
        relative humidity 'HR' is dimensionless."""
        return ALPHAW/(p*1e+5/(HR*self.equilibrium_vapor_pressure(theta))-1)
    def specific_enthalpy(self,theta,omega):
        """ Calculation of the specific enthalpy of the fresh mixture, in
        [J/(kg.K)], from the values of relative temperature 'theta' (in [°C])
        and specific humidity 'omega'."""
        return (self.dry_mixture_specific_heat_at_cste_pressure()\
                +omega*WATER_VAPOR_CP)*theta+omega*WATER_LW
    def mass_fractions(self,omega):
        """ Mass fractions of fuel, air and water vapor for a given value of the
        specific humidity 'omega', as a tuple."""
        # Actual Fuel-Air Ratio (FAR)
        FAR = self.fuel_air_ratio()
        return (FAR/((1+FAR)*(1+omega)),1/((1+FAR)*(1+omega)),omega/(1+omega))
    # ---- Ambient state/before the water injection process
    def ambient_specific_humidity(self):
        """ Specific humidity/Moisture content/Humidity ratio, defined
        as the ratio of the water vapor mass on the sole dry air one."""
        # Calculation are done outside the intake system, so with no
        # mention of the fuel.
        return self.specific_humidity(self.ambient_pressure,\
                                      self.ambient_temperature,\
                                      self.ambient_relative_humidity)
    def ambient_specific_enthalpy(self):
        """ Specific enthalpy of the fresh mixture in the ambient state."""
        return self.specific_enthalpy(self.ambient_temperature-273.15,\
                                      self.ambient_specific_humidity())
    def entrance_mass_fractions(self):
        """ Mass fractions of fuel, air and water vapor at the entrance 
        point, as a tuple."""
        return self.mass_fractions(self.ambient_specific_humidity())
    # ---- Mixture state
    def air_fuel_ratio(self):
        """ Actual Air-Fuel Ratio (AFR) of the fresh mixture."""
        return self.air_fuel_equivalent_ratio*self.stoichiometric_air_fuel_ratio()
    def fuel_air_ratio(self):
        """ Actual Fuel-Air Ratio (FAR) of the fresh mixture."""
        return 1/self.air_fuel_ratio()
    def dry_mixture_specific_heat_at_cste_pressure(self):
        """ Specific heat at constant pressure (cp) of the dry fresh mixture
        (without water vapor), in [J/(kg.K)]."""
        # Actual Fuel-Air Ratio (FAR)
        FAR = self.fuel_air_ratio()
        # And the specific heat of the blend of dry air and fuel
        return (DRY_AIR_CP+FAR*self.fuel_specific_heat_at_cste_pressure)/(1+FAR)
    def dry_mixture_specific_heat_at_cste_volume(self):
        """ Specific heat at constant volume (cV) of the dry fresh mixture
        (without water vapor), in [J/(kg.K)]."""
        # Actual Fuel-Air Ratio (FAR)
        FAR = self.fuel_air_ratio()
        # Specific heats at constant volumes, calculated thanks to the
        # Mayer relation
        cVair , cVwater = DRY_AIR_CP-DRY_AIR_R , WATER_VAPOR_CP-WATER_VAPOR_R
        # And the specific heat at constant volume of the blend
        return (cVair+FAR*cVwater)/(1+FAR)
    def dry_mixture_heat_capacity_ratio(self):
        """ Heat capacity ratio of the dry fresh mixture."""
        # Specific heat at constant pressure
        cp = self.dry_mixture_specific_heat_at_cste_pressure()
        # Specific heat at constant pressure
        cV = self.dry_mixture_specific_heat_at_cste_volume()
        return cp/cV
    def dry_mixture_ideal_gas_specific_constant(self):
        """ Specific gas constant (r of the ideal gas law) of the dry fresh
        mixture (without water vapor), in [J/(kg.K)]."""
        # Actual Fuel-Air Ratio (FAR)
        FAR = self.fuel_air_ratio()
        # And the ideal gas constant of the blend of dry air and fuel
        return (DRY_AIR_R+FAR*self.ideal_gas_specific_constant())/(1+FAR)
    # ---- Saturation properties
    def maximum_specific_humidity(self,p,theta):
        """ Maximum value of the fresh mixture specific humidity at a
        given pressure p in [bar] and a given relative temperature 'theta'
        in [°C]."""
        # Local constant useful for the calculation
        alpha_fuel = self.fuel_molar_mass()/DRY_AIR_M
         # Actual Fuel-Air Ratio (FAR)
        FAR = self.fuel_air_ratio()
        return ALPHAW*(1+FAR/alpha_fuel)/(1+FAR)\
        *1/(p*1e+5/self.equilibrium_vapor_pressure(theta+273.15)-1)
    def wet_bulb_temperature(self,p,h):
        """ Wet-bulb temperature corresponding to a given value of the
        fresh mixture pressure 'p' in [bar] and specific enthalpy 'h' in
        [J/kg]."""
        # Function of the temperature theta whom the root corresponds 
        # to the wet-bulb temperature.
        def f(theta):
            result = self.dry_mixture_specific_heat_at_cste_pressure()*theta\
            +self.maximum_specific_humidity(p,theta)*(WATER_VAPOR_CP*theta\
                                                      +WATER_LW)-h
            return result
        # The wet-bulb temperature is obtained thanks to the Newton
        # method applied to the function f, with the ambient temperature
        # as the initial/starting point.
        twb = sp.newton(f,fm.ambient_temperature-273.15)
        return twb
    def wet_bulb_specific_humidity(self,p,h):
        """ Specific humidity at the wet-bulb point fo a given pressure 
        'p' in [bar] and a given specific enthalpy 'h' in [J/kg]."""
        # Local constant useful for the calculation
        alpha_fuel = self.fuel_molar_mass()/DRY_AIR_M
        # Actual Fuel-Air Ratio (FAR)
        FAR = self.fuel_air_ratio()
        # Wet-bulb temperature
        thetawb = self.wet_bulb_temperature(p,h)
        return ALPHAW*(1+FAR/alpha_fuel)/(1+FAR)\
        *1/(p*1e+5/self.equilibrium_vapor_pressure(thetawb+273.15)-1)
    def maximum_adiabatic_water_fuel_ratio(self,pi,thetai):
        """ Maximum value of the Water-Fuel Ratio (WFR) if the injection
        process is supposed as adiabatic, for a given value of the 
        intake pressure pi, in [bar] and temperature thetai, in [°C]."""
        # Actual Fuel-Air Ratio (FAR)
        FAR = self.fuel_air_ratio()
        # Intake value of the specific enthalpy
        hi = self.specific_enthalpy(thetai,self.ambient_specific_humidity())
        # If the process is adiabatic, the maximum value of the specific
        # humidity is the wet-bulb one.
        wmax = self.wet_bulb_specific_humidity(pi,hi)
        return (1+FAR)/FAR*(wmax-self.ambient_specific_humidity())
    # ---- Moist fresh mixture
    # TODO : to Finish
    def moist_mixture_specific_heat_at_cste_pressure(self,omega):
        """ Specific heat at constant pressure (cp) of the moist fresh mixture
        (with water vapor), in [J/(kg.K)], from a value of the specific 
        humidity 'omega'."""
        pass
    def moist_mixture_specific_heat_at_cste_volume(self,omega):
        """ Specific heat at constant volume (cV) of the moist fresh mixture
        (with water vapor), in [J/(kg.K)], from a value of the specific 
        humidity 'omega'."""
        pass
    def moist_mixture_heat_capacity_ratio(self,omega):
        """ Heat capacity ratio of the moist fresh mixture (with water vapor),
        in [J/(kg.K)], from a value of the specific humidity 'omega'."""
        pass
    def moist_mixture_ideal_gas_specific_constant(self,omega):
        """ Specific gas constant (r of the ideal gas law) of the moist fresh
        mixture (with water vapor), in [J/(kg.K)], from a value of the specific
        humidity 'omega'."""
        pass
    def water_fuel_ratio(self,omega):
        """ Value of the Water-Fuel Ratio (WFR) required to obtain the value
        'omega' of the specific humidity."""
        # TODO : Check if the entered value is greater or not to the maximum 
        # one corresponding to saturation.
        pass
