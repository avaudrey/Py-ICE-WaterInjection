# Py-ICE-WaterInjection
Numerical modelling of water injection processes in Internal Combustion Engines (ICE) intake manifolds.

## Introduction

The **"port" injection of water** into the intakes of Internal Combustion Engines (ICE), sometimes called **fumigation**, is an old and well known strategy to improve their performances and decrease some of their tail pipe pollutions, as shortly explained in this video from the [Society of Automotive Engineers (SAE)](http://www.sae.org/) youtube channel:

[![SAE Eye on Engineering: Water Injection Returns](http://img.youtube.com/vi/nXZ4V3-M8xI/0.jpg)](https://www.youtube.com/watch?v=nXZ4V3-M8xI)

In a nutshell, the _water latent heat of vaporization_ creates a cooling effect of the fresh mixture and increases the engine _volumetric efficiency_, and so the amount of fuel aspirated. Furthermore, the required compression work and the maximum flame temperature decrease as well.

Hovewer,  previous studies of the effects of water injection on a given ICE were so far only based on experimental results, and then inadequate to untangle its actual effects on different engine parameters, but also to predict its potential improvement when set up on an existing engine. The goal of this project is to develop **a numerical tool usable to theoretically assess** the influence of water injection on the performances of a given ICE.

## How does it work

Firstly, the `water_injection_package` must be loaded as any other python package:
```python
In [1]: import water_injection_package as wi
```
### The `Fuel` class

**Introduction :** The first thing to do in such study is to choose the _fuel_ eventually burnt within the engine. The class `Fuel` has been designed for so. The _chemical composition_ and the _specific heat heat at constant pressure_ (when the fuel is considered as an [**ideal gas**](https://en.wikipedia.org/wiki/Ideal_gas)) are the only fuel properties required so far.

**Composition :** The fuel chemical composition is given to the class as a python dictionary of the type:
```python
In [2]: octane_composition = {'C':8, 'H':18, 'O':0, 'N':0, 'S':0}
```
With each value corresponding to the number of atoms of _carbon_ `'C'`, _hydrogen_ `'H'`, _oxygen_ `'O'`, _nitrogen_ `'N'` and _sulfur_ `'S'`, respectively involved in the fuel composition. In the previous example, the fuel considered is then the usual [_iso-octane_](https://en.wikipedia.org/wiki/2,2,4-Trimethylpentane) whom chemical formula is usually noted $C_8H_{18}$.

**Specific heat :** As previously mentioned, the only other required physical property of the fuel is its **gaseous** [_specific heat at constant pressure_](http://web.mit.edu/16.unified/www/FALL/thermodynamics/notes/node18.html), usually noted $c_p$ and expressed in $J/(kg \cdot K)$.

```python
In [3]: octane = wi.Fuel(composition=octane_composition, specif_heat=1.644e+3)
```

The methods included into the `Fuel` class allow to compute some useful property, as for example its [**molar mass**](https://en.wikipedia.org/wiki/Molar_mass) $M$, in $g/mol$:
```python
In [4]: octane.fuel_molar_mass()
Out[4]: 114.232
```
Its _specific heat at constant volume_ $c_V$:
```python
In [5]: octane.fuel_specif_heat_at_cste_v()
Out[5]: 1571.2143655017858
```
Or its _heat capacity ratio_ $\gamma$:
```python
In [6]: octane.fuel_heat_capacity_ratio()
Out[6]: 1.0463244456621101
```

