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
# Liquid water specific heat at constant pressure, in [J/(kg.K)]
LIQUID_WATER_CP = 4180.
# Water specific enthalpy of vaporization, in [J/(kg.K)]
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
    # TODO : Is it possible to change directly any physical property, cV, cp,
    # etc. and to calculate directly all the other physical properties?
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
        """ Specific heat at constant volume cV [J/(kg.K)] of the fuel,
        considered as an ideal gas."""
        # Use of the famous Mayer relation for ideal gases : cV = cp - r
        return self.fuel_specif_heat_at_cste_p-\
                self.fuel_ideal_gas_specif_r()
    def fuel_heat_capacity_ratio(self):
        """ Heat capacity ratio of the fuel, considered as an ideal gas."""
        # Specific heat at constant pressure
        c_p = self.fuel_specif_heat_at_cste_p
        # Specific heat at constant pressure
        c_v = self.fuel_specif_heat_at_cste_v()
        return c_p/c_v
    def fuel_alpha_constant(self):
        """ Alpha constant used in the calculations related to mixture with
        water vapour."""
        return self.fuel_molar_mass()/DRY_AIR_M

class FreshMixture(Fuel):
    """
    Chemical composition and specific enthalpy of a fresh mixture aspirated by
    an internal combustion engine and composed of air, water and sometimes fuel.
    """
    # TODO: Possibility to directly enter the value of the ambient specific
    # humidity.
    # Attributes --------------------------------------------------------------
    def __init__(self, air_fuel_equivalence_ratio=1.0, water_fuel_ratio=0.0,\
                 intake_duct_temperature=25., intake_duct_pressure=1.0,\
                 ambient_temperature=25., ambient_pressure=1.0,\
                 ambient_relative_humidity=0.5):
        # Initialization of the class fuel
        Fuel.__init__(self)
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
        self.air_fuel_equivalence_ratio = air_fuel_equivalence_ratio
        # Amount of water injected per amount of fuel consumed, equal to zero by
        # default, as without any water injection process
        self.water_fuel_ratio = water_fuel_ratio
        # Temperature and pressure inside the intake duct before the water
        # injection process, respectively expressed in [°C] and [bar]
        self.intake_duct_temperature = intake_duct_temperature
        self.intake_duct_pressure = intake_duct_pressure
        # Temperature and pressure in the surroundings outside the engine,
        # respectively expressed in [°C] and [bar]
        self.ambient_temperature = ambient_temperature
        self.ambient_pressure = ambient_pressure
        # The amount of water vapor at the intake_duct of the intake system comes
        # from the values of both the ambient temperature and the relative
        # humidity.
        self.ambient_relative_humidity = ambient_relative_humidity
    # Methods ================================================================= 
    # ---- General ------------------------------------------------------------
    @staticmethod
    def equilibrium_vapor_pressure(temperature):
        """ Calculation of the equilibrium water vapor pressure, in [Pa]
        according to the Cadiergues correlation. Temperature must be entered in
        °C, from 0°C to 100°C."""
        if (temperature < 0.0) or (temperature > 100.):
            raise ValueError("'temperature' must be 0°C < T < 100°C")
        logp = 2.7877+7.625*(temperature)/(temperature+241.)
        return np.power(10,logp)
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
        w0 = ALPHAW/(self.ambient_pressure*1e+5/\
                     (self.ambient_relative_humidity*\
                      self.equilibrium_vapor_pressure(self.ambient_temperature))-1)
        return w0
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
        # Mass fraction of each component
        fractions = np.array([y,afr,(y+afr)*w1,0.0])/((y+afr)*(1+w1))
        return tuple(fractions)
    def intake_duct_specif_humidity(self):
        """ Specific humidity/Moisture content/Humidity ratio, defined as the
        ratio of the water vapor mass on the sole dry air one, in the moist
        fresh mixture after the fuel injection (if required) but before the
        water one."""
        # Actual Air-Fuel Ratio
        afr = self.air_fuel_ratio()
        # And calculation
        if self.fuel_is_present:
            # If fuel is present, the specific humidity value is slightly
            # decreased regarding to the value in the ambience, water vapour
            # being more diluted in an increasing amount of dry gas.
            w1 = afr/(1+afr)*self.ambient_specif_humidity()
        else:
            # Without any fuel, the intake duct specific humidity is the one of
            # the ambience
            w1 = self.ambient_specif_humidity()
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
    def dry_mix_specif_heat_at_cste_v(self):
        """ Specific heat at constant volume (cV) of the dry fresh mixture
        (without water vapor), in [J/(kg.K)]."""
        # Specific heat at constant volume of air, calculated thanks to the
        # Mayer relation
        cvair = DRY_AIR_CP-DRY_AIR_R
        if self.fuel_is_present:
            # Actual Air-Fuel Ratio (FAR)
            afr = self.air_fuel_ratio()
            # And the specific heat of the blend of dry air and fuel
            cv = (self.fuel_specif_heat_at_cste_v()+afr*cvair)/(1+afr)
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
        c_v = self.dry_mix_specif_heat_at_cste_v()
        return c_p/c_v
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
        if (wfr <= wfreq):
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
                self.equilibrium_specif_humidity(self.intake_duct_pressure,theta)*\
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
            xvap , xliq = (y+afr)*w1+wfr , 0.0
        else:
            # For an incomplete vaporisation
            xvap , xliq = (y+afr)*w1+wfreq , wfr-wfreq 
        # Mass fraction of each component
        fractions = np.array([y,afr,xvap,xliq])/((y+afr)*(1+w1)+wfr)
        return tuple(fractions)
    def intake_valve_mix_specif_heat_at_cste_p(self):
        """ Specific heat at constant pressure (cp) of the moist fresh mixture
        (with water vapor), in [J/(kg.K)], from a value of the specific
        humidity 'omega'."""
        # Specific water content at the intake valve point
        omega2 = self.intake_valve_specif_humidity()
        return self.dry_mix_specif_heat_at_cste_p()+omega2*WATER_VAPOR_CP
    def intake_valve_mix_specif_heat_at_cste_v(self):
        """ Specific heat at constant volume (cV) of the moist fresh mixture
        (with water vapor), in [J/(kg.K)], from a value of the specific
        humidity 'omega'."""
        # Specific water content at the intake valve point
        omega2 = self.intake_valve_specif_humidity()
        return self.dry_mix_specif_heat_at_cste_v()+\
                omega2*(WATER_VAPOR_CP-WATER_VAPOR_R)
    def intake_valve_mix_heat_capacity_ratio(self):
        """ Heat capacity ratio of the moist fresh mixture (with water vapor),
        in [J/(kg.K)], from a value of the specific humidity 'omega'."""
        # Specific heat at constant pressure
        c_p = self.intake_valve_mix_specif_heat_at_cste_p()
        # Specific heat at constant pressure
        c_v = self.intake_valve_mix_specif_heat_at_cste_v()
        return c_p/c_v
    def intake_valve_mix_ideal_gas_specif_r(self):
        """ Specific gas constant (r of the ideal gas law) of the moist fresh
        mixture (with water vapor), in [J/(kg.K)], from a value of the specific
        humidity 'omega'."""
        # Specific water content at the intake valve point
        omega2 = self.intake_valve_specif_humidity()
        return self.dry_mix_ideal_gas_specif_r()+omega2*WATER_VAPOR_R
#    def water_fuel_ratio(self, omega):
#        """ Value of the Water-Fuel Ratio (WFR) required to obtain the value
#        'omega' of the specific humidity."""
#        # TODO : Check if the entered value is greater or not to the maximum
#        # one corresponding to saturation.
#        pass

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
    mixture = FreshMixture(ambient_temperature=20.)
    mixture.fuel_composition = {'C':2, 'H':6, 'O':1, 'N':0, 'S':0}
    mixture.fuel_specif_heat_at_cste_p = 1415.
    print('---- Fresh mixture: %s ----' % mixture.mix_name)
    print('Ambient temperature: T0 = %2.2f °C' % mixture.ambient_temperature)
    print('Ambient pressure: p0 = %2.5f bar' % mixture.ambient_pressure)
    print('Ambient relative humidity: HR0 = %2.2f' %
          mixture.ambient_relative_humidity)
    print('Ambient specific humidity: w0 = %2.2e' %
          mixture.ambient_specif_humidity())
    print('In such situation, the saturated specific humidity would be')
    print('of %2.2e at most.' % mixture.equilibrium_specif_humidity(mixture.ambient_pressure,\
                                                                    mixture.ambient_temperature))
    print('Ambient specific enthalpy: h0 = %2.2f kJ/kg' %
          (1e-3*mixture.ambient_specif_enthalpy()))
    print('The corresponding wet bulb temperature is thetawb = %2.1f°C.' % mixture.wet_bulb_temperature())
    print('Intake duct fresh mixture composition : %2.2f%% of fuel, %2.2f%%\nof '\
          'air, %2.2f%% of steam and %2.2f%% of liquid water, in mass.' %
          tuple((1e+2*np.array(list(mixture.intake_duct_mass_fractions())))))
    print('With an Air-Fuel equivalence ratio "lambda" of %2.1f, the\nactual '\
          'Air Fuel Ratio is %2.2f, so a Fuel-Air Ratio of\n'\
          '%1.3e.' % (mixture.air_fuel_equivalence_ratio,
                      mixture.air_fuel_ratio(), mixture.fuel_air_ratio()))
    print('The dry mixture specific heat at constant pressure is\n'\
          'cp(dry) = %2.1f J/(kg/K) while at constant volume, it\n'\
          'is cV(dry) = %2.1f J/(kg/K). That gives us a heat\n'\
          'capacity ratio "gamma" of %1.3f and a dry mix ideal\n'\
          'gas constant r(dry) = %2.2f J/(kg/K).' %
          (mixture.dry_mix_specif_heat_at_cste_p(),
           mixture.dry_mix_specif_heat_at_cste_v(),
           mixture.dry_mix_heat_capacity_ratio(),
           mixture.dry_mix_ideal_gas_specif_r()))
    

