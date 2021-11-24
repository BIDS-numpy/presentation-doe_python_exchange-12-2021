---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
rise:
  theme: serif
---

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
import numpy as np
import matplotlib.pyplot as plt
%matplotlib notebook
```

+++ {"slideshow": {"slide_type": "slide"}}

# Python: the language for effective scientific computing

[Ross Barnowski](https://bids.berkeley.edu/people/ross-barnowski) `rossbar@berkeley.edu` | [@rossbar](https://github.com/rossbar) on GitHub

DOE Python Exchange | 12/01/2021

+++ {"slideshow": {"slide_type": "slide"}}

## The Question: Why Python

+++ {"slideshow": {"slide_type": "fragment"}}

 - My personal take

+++ {"slideshow": {"slide_type": "fragment"}}

 - Looking ahead...

+++ {"slideshow": {"slide_type": "slide"}}

## A bit about me...

+++ {"slideshow": {"slide_type": "subslide"}}

I began my research career in the national lab system:

+++ {"slideshow": {"slide_type": "fragment"}}

 - Summer 2008: SULI internship at Argonne National Laboratory

+++ {"slideshow": {"slide_type": "fragment"}}

 - Summer 2009: Summer internship with the Nuclear Data Group at LLNL
   * First exposure to Python

+++ {"slideshow": {"slide_type": "fragment"}}

 - 2010-2019: Grad school -> post-doc -> research scientist & lecturer
   * UC Berkeley nuclear engineering department & the Applied Nuclear Physics
     group at LBNL

+++ {"slideshow": {"slide_type": "subslide"}}

I'm a nuclear engineer by training, specializing in radiation instrumentation;
particularly gamma-ray spectroscopy and imaging.

+++ {"slideshow": {"slide_type": "subslide"}}

### Real-time 3D gamma-ray mapping
 - Scene data fusion (SDF): computer vision + gamma-ray imaging
 - Applications in e.g. nuclear contamination remediation

TODO: Fukushima image here

+++ {"slideshow": {"slide_type": "subslide"}}

### Volumetric gamma-ray imaging
 - Imaging modalities for e.g. nuclear safeguards or small-animal imaging

TODO: Nearfield compton imaging example

+++ {"slideshow": {"slide_type": "slide"}}

## My background

+++ {"slideshow": {"slide_type": "fragment"}}

- Systems integration
  * Interfacing with hardware, data acquisition systems
  * Synchronizing & fusing data from disparate sources

+++ {"slideshow": {"slide_type": "fragment"}}

- Real-time acquisition and analysis
  * Performance!
  * "Distributed" computing

+++ {"slideshow": {"slide_type": "fragment"}}

- Workstation-scale computing
  * Mobile (hand-held or cart-based) measurement systems
  * Datasets usually O(GB) - not big data
