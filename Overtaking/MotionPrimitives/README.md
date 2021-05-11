## Motion Primitives

This folder contains definitions for different motion primitive sets.
A primitive set is a 3d tensor of size [N, res+1, res+1], with N being the number of primitives, and res being the resolution of the grid.

### MotionPrimitiveSuper.py 
Superclass for defining other primitive set types. Defines useful methods for calculating risk field given control inputs.

### MotionPrimitives.py
Most basic Motion Primitive set. Takes a set of speeds and a set of steering values and creates a set of primitives with every combination of speed and steering angle for fixed amount of time.

### ExplicitPrimitiveSet.py
Defined multi-control input primitive set from an explicit description of speeds and steering angles for a given number of control inputs per primitive.

### TreeMotionPrimitives.py
Makes a primitive set as a tree of all possible control inputs from discrete lists up to a certain depth.