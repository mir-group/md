import numpy as np
import sys
sys.path.append('../src')
import struc, output, md, env


def lj_force_on_atom(environment, epsilon, sigma):
    force = np.array([0., 0., 0.])
    for bond in environment.bond_array_2:
        dist = bond[0]
        relative_coord = bond[1:]
        force_term = 4 * epsilon * \
            (6 * sigma**6 / dist**7 - 12 * sigma**12 / dist**13)
        force += force_term * relative_coord
    return force


def lj_force_on_structure(structure, cutoff, epsilon, sigma):
    for atom in range(len(structure.positions)):
        env_curr = env.AtomicEnvironment(structure, atom, cutoff)
        structure.forces[atom] = lj_force_on_atom(env_curr, epsilon, sigma)


# set up structure: periodic cell, positions, and species
periodic_cell = np.eye(3) * 10
species = ['H'] * 2
positions = np.array([[0, 0, 0], [0, 0, 3]])
mass_dict = {'H': 1}
structure = struc.Structure(periodic_cell, species, positions, mass_dict)
cutoffs = np.array([5])
epsilon = 1
sigma = 2
force_args = (cutoffs, epsilon, sigma)
dt = 0.001
number_of_steps = 1000

md_object = md.MD(lj_force_on_structure, structure, dt, number_of_steps,
                  force_args=force_args)

md_object.run()
