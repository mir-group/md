# MD for ML

This is a simple NVE molecular dynamics package that is designed to be force-field independent and to couple easily
to a range of machine learned force fields. There are three main components of this package:

1. md.py: the main md engine. takes a force model as input, which can be any function that computes the forces on
all atoms in a structure.

2. struc.py: holds atomic coordinates (in angstrom), periodic cell, species, and atomic masses (in amu), and
automatically folds atoms back into the primary cell to handle diffusive systems.

3. output.py: prints information about the MD trajectory to a text file.
