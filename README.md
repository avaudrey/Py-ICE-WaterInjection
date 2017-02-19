# Py-ICE-WaterInjection
Numerical modelling of [**water injection processes**](https://en.wikipedia.org/wiki/Water_injection_(engine)) in Internal Combustion Engines (ICE) intake manifolds.

## Introduction

The **"port" injection of water** into the intakes of Internal Combustion Engines (ICE), sometimes called **fumigation**, is an old and well known strategy to improve their performances and decrease some of their tail pipe pollutions, as shortly explained in this video from the [Society of Automotive Engineers (SAE)](http://www.sae.org/) youtube channel:

[![SAE Eye on Engineering: Water Injection Returns](http://img.youtube.com/vi/nXZ4V3-M8xI/0.jpg)](https://www.youtube.com/watch?v=nXZ4V3-M8xI)

In a nutshell, the [_water latent heat of vaporization_](https://en.wikipedia.org/wiki/Enthalpy_of_vaporization) creates a cooling effect of the fresh mixture and increases the engine [_volumetric efficiency_](https://en.wikipedia.org/wiki/Volumetric_efficiency), and so the amount of fuel aspirated. Furthermore, the required compression work and the maximum flame temperature decrease as well.

Hovewer,  previous studies of the effects of water injection on a given ICE were so far only based on experimental results, and then inadequate to untangle its actual effects on different engine parameters, but also to predict its potential improvement when set up on an existing engine. The goal of this project is to develop **a numerical tool usable to theoretically assess** the influence of water injection on the performances of a given ICE.

## How does it work
Firstly, the `water_injection_package` must be loaded as any other python package, for example by:
```python
In [1]: import water_injection_package as wi
```
The two main objects in this package are the classes `Fuel` and `FreshMixture`.
### The `Fuel` class
#### Introduction :
The first practical thing to do in such study is to choose the _fuel_ eventually burnt within the engine. The class `Fuel` has been designed for so. The _chemical composition_ and the _specific heat heat at constant pressure_ (when the fuel is considered as an [**ideal gas**](https://en.wikipedia.org/wiki/Ideal_gas)) are the only fuel properties required so far.
#### Attributes
##### Fuel composition : 
The fuel chemical composition is given to the class as a python dictionary of the type:
```python
In [2]: octane_composition = {'C':8, 'H':18, 'O':0, 'N':0, 'S':0}
```
With each value corresponding to the number of atoms of _carbon_ `'C'`, _hydrogen_ `'H'`, _oxygen_ `'O'`, _nitrogen_ `'N'` and _sulfur_ `'S'`, respectively involved in the fuel composition. In the previous example, the fuel considered is then the usual [_iso-octane_](https://en.wikipedia.org/wiki/2,2,4-Trimethylpentane). If I want for example to consider pure [ethanol](https://en.wikipedia.org/wiki/Ethanol_fuel) as fuel, I just have to type:
```python
In [3]: ethanol_composition = {'C':2, 'H':6, 'O':1, 'N':0, 'S':0}
```
##### Specific heat :
As previously mentioned, the only other required physical property of the fuel is its **gaseous** [_specific heat at constant pressure_](http://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node18.html), expressed in J/(kg.K). Once the latter is known, the _octane_ fuel can be created by:
```python
In [4]: octane = wi.Fuel(composition=octane_composition, specif_heat=1.644e+3)
```
#### Methods
The methods included into the `Fuel` class allow to compute some useful properties, as for example its [**molar mass**](https://en.wikipedia.org/wiki/Molar_mass), in g/mol:
```python
In [5]: octane.fuel_molar_mass()
Out[5]: 114.232
```
Its _specific heat at constant volume_:
```python
In [6]: octane.fuel_specif_heat_at_cste_v()
Out[6]: 1571.2143655017858
```
Or its [_heat capacity ratio_](https://en.wikipedia.org/wiki/Heat_capacity_ratio):
```python
In [7]: octane.fuel_heat_capacity_ratio()
Out[7]: 1.0463244456621101
```
Another interesting parameter, more related to the combustion process itself, is the _stoichiometric [Air-Fuel Ratio (AFR)](https://en.wikipedia.org/wiki/Air%E2%80%93fuel_ratio)_, so the amount of air (in mass) required to burn each amount of fuel consumed. We have for octane:
```python
In [8]: octane.stoichiometric_air_fuel_ratio()
Out[8]: 15.033440717137056
```
Sometimes, it is the _Fuel-Air Ratio (FAR)_ which is used instead of the AFR:
```python
In [9]: octane.stoichiometric_fuel_air_ratio()
Out[9]: 0.06651837186280789
```

### The `FreshMixture` class

The goal of the class `FreshMixture` is to do calculations on _wet fresh mixtures_, so on gaseous blends of **fuel**, **air** and **water**, the latter being present in the fresh mixture because of both the _water injection process_ and the ambient _humidity_ in the surroundings. This is why the attributes of the `FreshMixture` class are related to the blend itself and to its surroundings.

#### Attributes
1. Regarding to the specific type of engine considered, the fuel is not always involved in the fresh mixture in the intake system, as for "usual" [_compression ignition (CI)_](https://en.wikipedia.org/wiki/Diesel_engine) engines for instance, in whom the fuel is directly injected into the cylinder at the end of the compression stroke. In order to consider such situation, we can choose to mix the fuel with the rest of the fresh mixture during the intake process, using the _boolean_ parameter `fuel_is_present = True` or `False`.
2. The second attribute needed is related to the combustion process, it is the [_Air-Fuel Equivalence Ratio_](https://en.wikipedia.org/wiki/Air%E2%80%93fuel_ratio#Air.E2.80.93fuel_equivalence_ratio_.28.CE.BB.29) `air_fuel_equivalent_ratio`, usually called "lambda". This parameter is:
a. `= 1` for a **stoichiometric** combustion.
b. `< 1` for a **rich** combustion, so with more fuel than it is needed.
c. `> 1` for a **lean** combustion, so with less fuel than it is needed.
3. The _water fuel ratio (WFR)_ is the fundamental parameter of engines water injection systems. Its value is usually such as `0.2 < WFR < 1.5`.
4. The other attributes are the ones related to the engine surroundings, so the latter:
a. Temperature `ambient_temperature`.
b. Pressure `ambient_pressure`.
c. And [relative humidity](https://en.wikipedia.org/wiki/Relative_humidity) `ambient_relative_humidity`.

Our fresh mixture can now be created, for example with a stoichiometric combustion (`air_fuel_equivalent_ratio=1.0`) with no water injected (`water_fuel_ratio=0.0`) and within an ambience at a temperature of 25°C, a pressure of 1 bar and a relative humidity of 50%:
```python
mixture = wi.FreshMixture(fuel_is_present=True, air_fuel_equivalent_ratio=1.0,\
                 water_fuel_ratio=0.0, ambient_temperature=298.,\
                 ambient_relative_humidity=0.5, ambient_pressure=1.0)
```

