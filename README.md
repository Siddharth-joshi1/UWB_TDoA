# UWB TDoA Localization

## Overview

This project implements Ultra-Wideband (UWB) localization using Time Difference of Arrival (TDoA). It estimates the position of a target node based on time differences of signals received at multiple anchors.

## Method

TDoA measurements define hyperbolic equations:

$$
d_i - d_j = c (t_i - t_j)
$$

These are solved using numerical methods such as least squares to estimate position.
## Features

* TDoA-based localization
* Least squares estimation
* Error analysis and visualization

## Usage

```bash
git clone https://github.com/Siddharth-joshi1/UWB_TDoA.git
cd UWB_TDoA
python main.py
```

## Notes

Assumes known anchor positions and synchronized receivers.
