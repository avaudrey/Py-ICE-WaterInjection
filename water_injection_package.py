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

# If CoolProp is installed, it will be used for the calculation of physical
# properties of water
try:
    import CoolProp
    from CoolProp.CoolProp import PropsSI
    is_coolprop_present = True
except ImportError:
    is_coolprop_present = False

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
# Liquid water specific heat at constant pressure, in [J/(kg.K)]
LIQUID_WATER_CP = 4180.
# Water specific enthalpy of vaporization, in [J/(kg.K)]
WATER_LW = 2501e+3
# Molar mass of the dry air
DRY_AIR_M = 28.9645
# Ratio of the molecular mass of water on the standard dry air one
ALPHAW = (ATOMIC_WEIGHTS['H']*2+ATOMIC_WEIGHTS['O'])/DRY_AIR_M

# The class fuel
class Fuel:
    """
    Engine fuel, considered as an ideal gas and represented by its chemical
    composition (of the type CHONS) and by its mass specific heat at constant
    pressure, in [kJ/kg]. It is also possible to adjust the values of the mass
    specif heat at constant volume, of the specific ideal gas constant or of the
    heat capacity ratio.
    """
    # Attributes --------------------------------------------------------------
    def __init__(self):
        # Default chemical composition of the fuel, octane
        composition = {'C':8, 'H':18, 'O':0, 'N':0, 'S':0}
        # Name of the fuel, octane by default
        self.fuel_name = 'octane'
        # Chemical composition of the fuel, represented by a dictionnary
        # containing the numbers of atoms of Carbon (C), Hydrogen (H),
        # Oxygen (O), Nitrogen (N) and Sulfur (S), so the CHONS.
        self.fuel_composition = composition
        # Specific heat at constant pressure, in [J/(kg.K)]
        self._fuel_specif_heat_at_cste_p = 1644.
        # Specific heat at constant volume, in [J/(kg.K)]
        self._fuel_specif_heat_at_cste_V = 1571.
        # Heat capacity ratio
        self._fuel_heat_capacity_ratio = 1.046
    # Attributes defined as properties ----------------------------------------
    @property
    def fuel_specif_heat_at_cste_p(self):
        """Mass specific heat at constant pressure of the fuel, in J/(kg.K),
        considered as an ideal gas."""
        return self._fuel_specif_heat_at_cste_p
    @fuel_specif_heat_at_cste_p.setter
    def fuel_specif_heat_at_cste_p(self, cp):
        """New value of the specific heat at constant pressure, in J/(kg.K)."""
        if cp <= 0.0:
            raise ValueError("The mass specific heat a constant pressure has to "
                             "be positive, in J/(kg.K) ! ")
        # New value of the specific heat at constant volume, thanks to the
        # Mayer's relation
        r = self.fuel_ideal_gas_specif_r()
        cV = cp - r
        if cV <= 0.0:
            raise ValueError("The mass specific heat a constant pressure has to "
                             "be greater than the specific ideal gas constant ! ")
        self._fuel_specif_heat_at_cste_V = cV
        # And of the heat capacity ratio
        self._fuel_heat_capacity_ratio = cp/cV
        # New value
        self._fuel_specif_heat_at_cste_p = cp
        pass
    @property
    def fuel_specif_heat_at_cste_V(self):
        """Mass specific heat at constant volume of the fuel, in J/(kg.K),
        considered as an ideal gas."""
        return self._fuel_specif_heat_at_cste_V
    @fuel_specif_heat_at_cste_V.setter
    def fuel_specif_heat_at_cste_V(self, cV):
        """New value of the specific heat at constant pressure, in J/(kg.K)."""
        if cV <= 0.0:
            raise ValueError("The mass specific heat a constant volume has to "
                             "be positive, in J/(kg.K) !")
        # New value of the specific heat at constant pressure, thanks to the
        # Mayer's relation
        r = self.fuel_ideal_gas_specif_r()
        cp = cV + r
        if cp <= cV:
            raise ValueError("The mass specific heat a constant pressure has to "
                             "be greater than the one at constant volume ! ")
        self._fuel_specif_heat_at_cste_p = cp
        # And of the heat capacity ratio
        self._fuel_heat_capacity_ratio = cp/cV
        # New value
        self._fuel_specif_heat_at_cste_V = cV
        pass
    @property
    def fuel_heat_capacity_ratio(self):
        """The heat capacity ratio of the fuel, so the famous 'gamma = cp/cV',
        dimensionless."""
        return self._fuel_heat_capacity_ratio
    @fuel_heat_capacity_ratio.setter
    def fuel_heat_capacity_ratio(self, gamma):
        """New value of the 'gamma = cp/cV', dimensionless."""
        if gamma <= 0.0:
            raise ValueError("The heat capacity ratio has to be positive.""")
        # New values of the specific heat at constant pressure and constant
        # volume, thanks to the Mayer's relation
        r = self.fuel_ideal_gas_specif_r()
        cp, cV = gamma*r/(gamma-1), r/(gamma-1)
        # Test
        if cp <= cV:
            raise ValueError("The mass specific heat a constant pressure has to "
                             "be greater than the one at constant volume ! ")
        # And new values
        self._fuel_specif_heat_at_cste_p = cp
        self._fuel_specif_heat_at_cste_V = cV
        self._fuel_heat_capacity_ratio = gamma
        pass
    # Methods -----------------------------------------------------------------
    def fuel_molar_mass(self):
        """Calculation of the fuel molar mass thanks to the chemical
        composition."""
        # Initialization
        molar_mass = 0.0
        for comp in ['C', 'H', 'O', 'N', 'S']:
            molar_mass += self.fuel_composition[comp]*ATOMIC_WEIGHTS[comp]
        return molar_mass
    def fuel_alpha_constant(self):
        """ Alpha constant used in the calculations related to mixture with
        water vapour."""
        return self.fuel_molar_mass()/DRY_AIR_M
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

class FreshMixture(Fuel):
    """ Chemical composition and specific enthalpy of a fresh mixture aspirated
    by an internal combustion engine and composed of air, water and sometimes
    fuel. """
    # TODO: Possibility to directly enter the value of the ambient specific
    # humidity.
    # TODO : Method for the computation of the intake valve mix relative
    # humidity
    # Attributes --------------------------------------------------------------
    def __init__(self):
        # Initialization of the class fuel
        super(FreshMixture, self).__init__()
        # Name of the fresh mixture
        self.mix_name = 'fresh_mixture-1'
        # Fuel is actually mixed with fresh air for spark ignition (SI) or Dual
        # Fuel compression ignition (CI) engines. At the contrary, for usual CI
        # engines, fuel is directly injected into the cylinder and not mixed
        # with fresh air in the fresh mixture. In the latter case, this variable
        # must be set as False
        self.fuel_is_present = True
        # Equivalent air fuel ratio, usually noted lambda. The default value is
        # here corresponding to a stoichiometric combustion.
        self.air_fuel_equivalence_ratio = 1.0
        # Amount of water injected per amount of fuel consumed, equal to zero by
        # default, as without any water injection process
        self.water_fuel_ratio = 0.0
        # Temperature and pressure inside the intake duct before the water
        # injection process, respectively expressed in [°C] and [bar]
        self.intake_duct_temperature = 25.0
        self.intake_duct_pressure = 1.0
        # Temperature and pressure in the surroundings outside the engine,
        # respectively expressed in [°C] and [bar]
        self.ambient_temperature = 25.0
        self.ambient_pressure = 1.0
        # The amount of water vapor at the intake_duct of the intake system comes
        # from the values of both the ambient temperature and the relative
        # humidity.
        self.ambient_relative_humidity = 0.5
        # Properties of the fuel, defined as python properties in the Fuel class
        fuel_specif_heat_at_cste_p =\
                property(Fuel.fuel_specif_heat_at_cste_p.__get__)
        fuel_specif_heat_at_cste_V =\
                property(Fuel.fuel_specif_heat_at_cste_V.__get__)
        fuel_heat_capacity_ratio =\
                property(Fuel.fuel_heat_capacity_ratio.__get__)
    # Attributes defined as properties ----------------------------------------
    # Methods ================================================================= 
    # ---- General ------------------------------------------------------------
    @staticmethod
    def equilibrium_vapor_pressure(temperature):
        """ Calculation of the equilibrium water vapor pressure, in [Pa]. If
        CoolProp is not installed on the computer, the Cadiergues correlation is
        used. Temperature must be entered in °C, from 0°C to 100°C."""
        if is_coolprop_present:
            peq = PropsSI('P', 'T', 273.15+temperature, 'Q', 1.0, 'Water')
        else:
            if (temperature < 0.0) or (temperature > 100.):
                raise ValueError("'temperature' must be 0°C < T < 100°C")
            logp = 2.7877+7.625*(temperature)/(temperature+241.)
            peq = np.power(10, logp)
        return peq
    def specif_humidity(self, pressure, theta, relative_h):
        """ Specific humidity/Moisture content/Humidity ratio, defined
        as the ratio of the water vapor mass on the dry fresh mixture one.
        Pressure 'p' is in [bar], relative temperature 'theta' in [°C] and
        relative humidity 'relative_h' is dimensionless."""
        # Parameter introduced to represent the modification of the dry gas
        # composition due to the presence of fuel, if needed.
        if self.fuel_is_present:
            afr = self.air_fuel_ratio()
            alpha_fuel = self.fuel_alpha_constant()
            correction_factor = (1+alpha_fuel*afr)/(alpha_fuel*(1+afr))
        else:
            correction_factor = 1.0
        # And calculation
        return correction_factor*ALPHAW/(pressure*1e+5/\
                       (relative_h*self.equilibrium_vapor_pressure(theta))-1)
    def equilibrium_specif_humidity(self, pressure, theta):
        """ Equilibrium value of the fresh mixture specific humidity at a
        given pressure p in [bar] and a given relative temperature 'theta'
        in [°C]."""
        return self.specif_humidity(pressure, theta, 1.0)
    def specif_enthalpy(self, theta, omega):
        """ Calculation of the specific enthalpy of the fresh mixture, in
        [J/(kg.K)], from the values of relative temperature 'theta' (in [°C])
        and specific humidity 'omega'."""
        return (self.dry_mix_specif_heat_at_cste_p()\
                +omega*WATER_VAPOR_CP)*theta+omega*WATER_LW
    # ---- Ambient state/before the intake process -----------------------------
    # In the rest, the word "Ambient" is related to the fresh air before its
    # suction inside the intake duct, so at point '0', before the injection of
    # fuel and water.
    def ambient_specif_humidity(self):
        """ Specific humidity/Moisture content/Humidity ratio, defined
        as the ratio of the water vapor mass on the sole dry air one, in the
        ambient air."""
        # Calculation are done outside the intake system, so with no
        # mention of the fuel.
        return self.specif_humidity(self.ambient_pressure,
                                    self.ambient_temperature,
                                    self.ambient_relative_humidity)
    def ambient_specif_enthalpy(self):
        """ Specific enthalpy of the fresh mixture in the ambient state."""
        return self.specif_enthalpy(self.ambient_temperature,\
                                    self.ambient_specif_humidity())
    # ---- Intake duct state, after fuel injection but before the water one ---
    # The word 'Intake duct' is related to the fresh mixture after the fuel
    # injection (if the latter exists) but before the water injection.
    def intake_duct_mass_fractions(self):
        """ Mass fractions of fuel, air and water vapour and liquid water, after
        the fuel injection but before the water one, as a tuple."""
        # Actual Air-Fuel Ratio
        afr = self.air_fuel_ratio()
        # Parameter equal to 1 if the fuel is in the fresh mixture and equal to
        # 0 otherwise
        y = self.fuel_is_present*1.0
        # Calculation of the specific water content after the mix of the fuel
        # with fresh air
        w1 = self.intake_duct_specif_humidity()
        # Mass fraction of each component, calculated using the specific water
        # content at point 1
        fractions = np.array([y, afr, (y+afr)*w1, 0.0])/((y+afr)*(1+w1))
        return tuple(fractions)
    def intake_duct_specif_humidity(self):
        """ Specific humidity/Moisture content/Humidity ratio, defined as the
        ratio of the water vapor mass on the sole dry air one, in the moist
        fresh mixture after the fuel injection (if required) but before the
        water one."""
        # Actual Air-Fuel Ratio
        afr = self.air_fuel_ratio()
        # Parameter equal to 1 if the fuel is in the fresh mixture and equal to
        # 0 otherwise
        y = self.fuel_is_present*1.0
        # And calculation
        w1 = afr/(y+afr)*self.ambient_specif_humidity()
        return w1
    def intake_duct_specif_enthalpy(self):
        """ Specific enthalpy of the moist fresh mixture before the water
        injection process."""
        return self.specif_enthalpy(self.intake_duct_temperature,\
                                      self.intake_duct_specif_humidity())
    # ---- Mixture state ------------------------------------------------------
    def air_fuel_ratio(self):
        """ Actual Air-Fuel Ratio (AFR) of the fresh mixture, obtained from fuel
        chemical composition and from the Air-Fuel equivalence Ratio value."""
        return self.air_fuel_equivalence_ratio*\
                self.stoichiometric_air_fuel_ratio()
    def fuel_air_ratio(self):
        """ Actual Fuel-Air Ratio (FAR) of the fresh mixture, obtained from fuel
        chemical composition and from the Air-Fuel equivalence Ratio value."""
        return 1/self.air_fuel_ratio()
    # ---- Properties of the dry mixture, so composed of fuel and air ---------
    # These values are relevant and useful at any point of the system where air
    # and fuel are blended.
    def dry_mix_specif_heat_at_cste_p(self):
        """ Specific heat at constant pressure (cp) of the dry fresh mixture
        (without water vapor), in [J/(kg.K)]."""
        if self.fuel_is_present:
            # Actual Air-Fuel Ratio (FAR)
            afr = self.air_fuel_ratio()
            # And the specific heat of the blend of dry air and fuel
            cp = (self.fuel_specif_heat_at_cste_p+afr*DRY_AIR_CP)/(1+afr)
        else:
            # Without any fuel in the fresh mixture, the dry specific heat is
            # the dry air one
            cp = DRY_AIR_CP
        return cp
    def dry_mix_specif_heat_at_cste_V(self):
        """ Specific heat at constant volume (cV) of the dry fresh mixture
        (without water vapor), in [J/(kg.K)]."""
        # Specific heat at constant volume of air, calculated thanks to the
        # Mayer relation
        cvair = DRY_AIR_CP-DRY_AIR_R
        if self.fuel_is_present:
            # Actual Air-Fuel Ratio (FAR)
            afr = self.air_fuel_ratio()
            # And the specific heat of the blend of dry air and fuel
            cv = (self.fuel_specif_heat_at_cste_V+afr*cvair)/(1+afr)
        else:
            # Without any fuel in the fresh mixture, the dry specific heat is
            # the dry air one
            cv = cvair 
        return cv
    def dry_mix_heat_capacity_ratio(self):
        """ Heat capacity ratio of the dry fresh mixture."""
        # Specific heat at constant pressure
        c_p = self.dry_mix_specif_heat_at_cste_p()
        # Specific heat at constant pressure
        c_V = self.dry_mix_specif_heat_at_cste_V()
        return c_p/c_V
    def dry_mix_ideal_gas_specif_r(self):
        """ Specific gas constant (r of the ideal gas law) of the dry fresh
        mixture (without water vapor), in [J/(kg.K)]."""
        if self.fuel_is_present:
            # Actual Air-Fuel Ratio (AFR)
            afr = self.air_fuel_ratio()
            # And the specific heat of the blend of dry air and fuel
            r = (self.fuel_ideal_gas_specif_r()+afr*DRY_AIR_R)/(1+afr)
        else:
            # Without any fuel in the fresh mixture, the dry gas constant is the
            # dry air one
            r = DRY_AIR_R
        return r
    # ---- Intake valve state, after the water injection ----------------------
    # The word 'Intake valve' is related to the fresh mixture after the water
    # injection process, just before the entrance into the engine cylinder.
    # - When the wet-bulb temperature is reached at the intake point ----------
    def wet_bulb_temperature(self):
        """ Wet-bulb temperature corresponding to the lowest temperature
        attainable at the intake valve point of the engine, after water
        injection."""
        # Function of the temperature theta whom the root corresponds
        # to the wet-bulb temperature.
        def f_to_solve(theta):
            result = self.dry_mix_specif_heat_at_cste_p()*theta\
                    +self.equilibrium_specif_humidity(self.intake_duct_pressure, theta)*\
                    (WATER_VAPOR_CP*theta+WATER_LW)\
                    -self.intake_duct_specif_enthalpy()
            return result
        # The wet-bulb temperature is obtained thanks to the Newton
        # method applied to the function f, with the ambient temperature
        # as the initial/starting point.
        twb = sp.newton(f_to_solve, self.ambient_temperature)
        return twb
    def wet_bulb_specif_humidity(self):
        """ Specific humidity at the intake valve point and at the wet-bulb
        temperature state."""
        # Local constant useful for the calculation
        alpha_fuel = self.fuel_molar_mass()/DRY_AIR_M
        # Actual Fuel-Air Ratio (FAR)
        far = self.fuel_air_ratio()
        # Computation of the Wet-bulb temperature
        thetawb = self.wet_bulb_temperature()
        # And of the specific humidity, at equilibrium at this point, by
        # definition
        return self.equilibrium_specif_humidity(self.intake_duct_pressure,\
                                                thetawb)
    def equilibrium_water_fuel_ratio(self):
        """ Equilibrium value of the Water-Fuel Ratio (WFR) if the injection
        process is supposed as adiabatic. This ratio is calculated using the
        intake_duct pressure and the wet-bulb temperature."""
        # Parameter equal to 1 if the fuel is in the fresh mixture and equal to
        # 0 otherwise
        y = self.fuel_is_present*1.0
        # Actual value of the Air-Fuel Ratio (FAR)
        afr = self.air_fuel_ratio()
        # Wet bulb temperature corresponding to the given intake_duct conditions
        thetawb = self.wet_bulb_temperature()
        return (y+afr)*(self.equilibrium_specif_humidity(self.intake_duct_pressure,\
                                                      thetawb)\
                        -self.intake_duct_specif_humidity())
    def intake_valve_temperature(self):
        """ Temperature at the intake valve point, so after water injection."""
        # Parameter equal to 1 if the fuel is in the fresh mixture and equal to
        # 0 otherwise
        y = self.fuel_is_present*1.0
        # Actual value of the Air-Fuel Ratio (FAR)
        afr = self.air_fuel_ratio()
        # In order to make the difference between a cooling process with a
        # complete vaporisation and an incomplete one, we need to know the
        # equilibrium WFR
        wfreq = self.equilibrium_water_fuel_ratio()
        # Actual Water-Fuel Ratio
        wfr = self.water_fuel_ratio
        # Specific heat of the dry fresh mixture
        cpdfm = self.dry_mix_specif_heat_at_cste_p()
        # Type of vaporisation process, complete or not
        if wfr <= wfreq:
            # If the vaporisation is complete, an explicit formula can be used
            # to calculate the temperature drop
            temperature_drop = ((WATER_VAPOR_CP*self.intake_duct_temperature+\
                                 WATER_LW)*wfr)/((y+afr)*\
                                (cpdfm+self.intake_duct_specif_humidity()*\
                                WATER_VAPOR_CP)+wfr*WATER_VAPOR_CP)
            cooling_temp = self.intake_duct_temperature-temperature_drop
        else:
            # If the vaporisation is incomplete, the computation process is more
            # complicated
            # We start be calculating the specific water content at the intake
            # point
            omegai = self.intake_duct_specif_humidity()+wfr/(y+afr)
            # The constant member of the future equation to solve
            z = self.intake_duct_specif_enthalpy()+omegai*LIQUID_WATER_CP*\
                    self.intake_duct_temperature
            # Function of the temperature theta whom the root corresponds to the
            # intake temperature.
            def f_to_solve(theta):
                return (cpdfm+omegai*LIQUID_WATER_CP)*theta+\
                self.equilibrium_specif_humidity(self.intake_duct_pressure, theta)*\
                        ((WATER_VAPOR_CP-LIQUID_WATER_CP)*theta+\
                        LIQUID_WATER_CP*self.intake_duct_temperature+WATER_LW)-z
            cooling_temp = sp.newton(f_to_solve, self.ambient_temperature)
        return cooling_temp 
    def intake_valve_specif_humidity(self):
        """ Specific water content of the fresh mixture after the water
        injection point, at the intake valve."""
        # Actual Water-Fuel Ratio
        wfr = self.water_fuel_ratio
        # Actual value of the Air-Fuel Ratio (FAR)
        afr = self.air_fuel_ratio()
        # Parameter equal to 1 if the fuel is in the fresh mixture and equal to
        # 0 otherwise
        y = self.fuel_is_present*1.0
        return self.intake_duct_specif_humidity()+wfr/(y+afr)
    def intake_valve_mass_fractions(self):
        """ Mass fractions of fuel, air and water vapour and liquid water, after
        the water injection, as a tuple."""
        # Actual Air-Fuel Ratio
        afr = self.air_fuel_ratio()
        # Parameter equal to 1 if the fuel is in the fresh mixture and equal to
        # 0 otherwise
        y = self.fuel_is_present*1.0
        # Calculation of the specific water content after the mix of the fuel
        # with fresh air
        w1 = self.intake_duct_specif_humidity()
        # Equilibrium Water-Fuel Ratio 
        wfreq = self.equilibrium_water_fuel_ratio()
        # Actual Water-Fuel Ratio
        wfr = self.water_fuel_ratio
        if wfr <= wfreq:
            # For a complete vaporisation
            xvap, xliq = (y+afr)*w1+wfr, 0.0
        else:
            # For an incomplete vaporisation
            xvap, xliq = (y+afr)*w1+wfreq, wfr-wfreq 
        # Mass fraction of each component, calculated using the specific water
        # content at point 1
        fractions = np.array([y, afr, xvap, xliq])/((y+afr)*(1+w1)+wfr)
        return tuple(fractions)
    def intake_valve_mix_specif_heat_at_cste_p(self):
        """ Specific heat at constant pressure (cp) of the moist fresh mixture
        (with water vapor), in [J/(kg.K)], from a value of the specific
        humidity 'omega'."""
        # Specific water content at the intake valve point
        omega2 = self.intake_valve_specif_humidity()
        return self.dry_mix_specif_heat_at_cste_p()+omega2*WATER_VAPOR_CP
    def intake_valve_mix_specif_heat_at_cste_V(self):
        """ Specific heat at constant volume (cV) of the moist fresh mixture
        (with water vapor), in [J/(kg.K)], from a value of the specific
        humidity 'omega'."""
        # Specific water content at the intake valve point
        omega2 = self.intake_valve_specif_humidity()
        return self.dry_mix_specif_heat_at_cste_V()+\
                omega2*(WATER_VAPOR_CP-WATER_VAPOR_R)
    def intake_valve_mix_heat_capacity_ratio(self):
        """ Heat capacity ratio of the moist fresh mixture (with water vapor),
        in [J/(kg.K)], from a value of the specific humidity 'omega'."""
        # Specific heat at constant pressure
        c_p = self.intake_valve_mix_specif_heat_at_cste_p()
        # Specific heat at constant pressure
        c_V = self.intake_valve_mix_specif_heat_at_cste_V()
        return c_p/c_v
    def intake_valve_mix_ideal_gas_specif_r(self):
        """ Specific gas constant (r of the ideal gas law) of the moist fresh
        mixture (with water vapor), in [J/(kg.K)], from a value of the specific
        humidity 'omega'."""
        # Specific water content at the intake valve point
        omega2 = self.intake_valve_specif_humidity()
        return self.dry_mix_ideal_gas_specif_r()+omega2*WATER_VAPOR_R
    def intake_valve_mix_gas_density(self):
        """ Density of the fresh mixture, in [kg/m3], at the intake valve point
        just before entering into the engine cylinder."""
        # The density of the gaseous part of the fresh charge is calculated by
        # the ideal gas law
        r = self.intake_valve_mix_ideal_gas_specif_r()
        T = self.intake_valve_temperature()+273.15
        return self.intake_duct_pressure*1e+5/(r*T)
    def intake_valve_fuel_density(self):
        """ Density of the sole fuel, in [kg/m3], at the intake valve point, so
        just before entering into the engine cylinder."""
        # Mass fraction of the fuel
        xfuel = self.intake_valve_mass_fractions()[0]
        return xfuel*self.intake_valve_mix_gas_density()
    def intake_valve_mix_relative_humidity(self):
        # TODO : To finish.
        pass

class EngineGeometry:
    """Simplified geometrical model of a reciprocating internal combustion\
            of screw compressor."""
    # Attributes ==============================================================
    def __init__(self):
        # Name of the engine, if necessary 
        self.engine_name = 'engine-1'
        # Compression ratio
        self.engine_compression_ratio = 11.
        # Swept volume in m^3, equal to 1 liter by default
        self.engine_swept_volume = 1e-3
    # Methods ================================================================= 
    def engine_clearance_volume(self):
        """Clearance volume of the engine, in m^3."""
        a = self.engine_compression_ratio
        return self.engine_swept_volume/(a-1)
    def engine_maximum_volume(self):
        """Maximum volume inside the engine, in m^3."""
        a = self.engine_compression_ratio
        return self.engine_swept_volume/(1-1/a)

class WetCompression(EngineGeometry):
    """Numerical model of the reversible and adiabatic wet compression process
    of a dry gas within a positive displacement compressor, as a reciprocating
    one for example."""
    # Attributes ==============================================================
    def __init__(self):
        # Calculations related to the engine geometry will be done using the
        # class specifically designed for so.
        super(WetCompression, self).__init__()
        # Name of the process, if necessary 
        self.compression_process_name = 'compression-1'
        # Intake temperature and pressure, in °C and in bar, respectively
        self.intake_temperature = 20.
        self.intake_pressure = 1.
        # Water injection related parameters
        self.intake_specific_water_content = 0.0
        # Physical properties of the dry gas to compress
        self.dry_gas_ideal_specif_r = 287.
        self.dry_gas_specif_heat_at_cste_V = 717.
        # Parameters of the numerical solving process, size of the numerical
        # mesh used to solve the ode problem
        self._compression_numerical_size = 101
    # Attributes defined as properties ----------------------------------------
    @property
    def compression_numerical_size(self):
        """Size of the numerical mesh used to numerically solve the ode problem,
        as an integer."""
        return self._compression_numerical_size
    @compression_numerical_size.setter
    def compression_numerical_size(self, N):
        """Size of the numerical mesh used to numerically solve the ode problem,
        as an integer."""
        if (type(N) != int):
            raise ValueError('The size of the mesh has to be an integer value!')
        pass
    # Methods ================================================================= 
    # Related to the initial state of the gaseous mixture to compress
    def intake_maximum_specific_humidity(self):
        """Maximum value of the specific humidity mvap/mg at the beginning of
        the compression stroke, dimensionless."""
        # Ratio of the dry gas to water vapour molar masses
        a = self.dry_gas_ideal_specif_r/WATER_VAPOR_R
        # Equilibrium vapour pressure before the compression
        peq0 = FreshMixture.equilibrium_vapor_pressure(self.intake_temperature)
        return a*peq0/(self.intake_pressure*1e+5-peq0)
    def liquid_water_specific_volume(self, temperature):
        """Specific volume of the liquid water vliq, in m^3/kg, vs. temperature.
        If installed on the computer, the CoolProp package is used to calculate
        this value, otherwise, a constant value is used instead."""
        if is_coolprop_present:
            vliq = 1/PropsSI('D', 'T', 273.15+temperature, 'Q', 0.0, 'Water')
        else:
            vliq = 1e-3
        return vliq
    def compression_type(self):
        """Is the compression process initially dry, unsaturated or
        saturated?"""
        if (self.intake_specific_water_content >=\
            self.intake_maximum_specific_humidity()):
            type = 'Saturated'
        else:
            if (self.intake_specific_water_content == 0.0):
                type = 'Dry'
            else:
                type = 'Unsaturated'
        return type
    def intake_dry_gas_specific_volume(self):
        """Initial value of the dry gas specific volume, V/mg, in m^3/kg."""
        # Calculation of the gaseous mix initial specific humidity, in using the
        # already existing method if the mix is already saturated.
        type = self.compression_type()
        if (type == 'Saturated'):
            omega = self.intake_maximum_specific_humidity()
        else:
            omega = self.intake_specific_water_content
        # Ideal specific constant of the gaseous mix to compress
        r = self.dry_gas_ideal_specif_r+omega*WATER_VAPOR_R
        # Aspirated mass of dry gas
        v = r*(self.intake_temperature+273.15)/(1e+5*self.intake_pressure)
        # Correction taking into account the mass of liquid water if necessary
        if (type == 'Saturated'):
            vliq = self.liquid_water_specific_volume(self.intake_temperature)
            # The amount of liquid aspirated is proportional to the difference
            # between the specific water content and the specific humidity
            v += (self.intake_specific_water_content-omega)*vliq
        return v
    def dry_gas_aspirated_mass(self):
        """Mass of dry gas, in kg, actually aspirated into the engine."""
        return self.engine_maximum_volume()\
                /self.intake_dry_gas_specific_volume()
    # Related to the numerical problem to solve
    def compression_numerical_volume_step(self):
        """Elementary step of actual volume variation along the numerical
        solving process, in m."""
        return self.engine_swept_volume/(self.compression_numerical_size-1)
    def compression_numerical_volume_mesh(self):
        """The actual mesh of actual volume values used in the ode numerical
        solving process."""
        # The further numerical process being a compression one, with a decrease
        # of the actual volume, the list of volume values is in a decreasing
        # order.
        return np.linspace(self.engine_clearance_volume(),\
                           self.engine_maximum_volume(),\
                           self._compression_numerical_size)[::-1]
    def dry_gas_specific_volume(self):
        """Values of the dry gas specific volume V/mg, used in the ode solving
        process, in m^3/kg."""
        return self.compression_numerical_volume_mesh()\
                /self.dry_gas_aspirated_mass()

if __name__ == '__main__':
    wetcomp1 = WetCompression()
    wetcomp1.intake_temperature = 50.
    wetcomp1.intake_specific_water_content = 0.1
#    pass
