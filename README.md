# Risk-based Path Planning for Autonomous Overtaking Manuevers
This is the github project for the F1Tenth Independent Study Projects 2021. In this project we are focusing on  advanced overtaking techniques that are enable and improvisation of the driver.

## Requirements
- Linux Ubuntu (developed and tested on version 18.04)
- Python 3.6.9
- [f1tenth_gym](https://github.com/DavidDePauw1/f1tenth_gym) package
- packages listed in `requirements.txt`

## Installation
Use the command below in the root directory of this repo, in order to install all required modules

`pip3 install -r /path/to/requirements.txt`

## Running Example Scripts
Use the commands below to run scripts to exemplify the usage and functionality of this project

`python3 ExampleScripts/IntegratedControllerExample.py`

## Code Organization

This project is organized into a number of directories, each of which has its own `README` file going into further detail on its contents.  Top level directories for this repository include.

- `ExampleScripts` - scripts meant to showcase project's functionality and provide insight into how to use the code
- `Overtaking` - Python package implementing controllers using risk-based motion primitives

## Authors
- [Raymond Bjorkman](raybjork@seas.upenn.edu)
- [David DePauw](daviddep@seas.upenn.edu)