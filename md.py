from struc import Structure
import numpy as np
import time
import datetime
import concurrent.futures
import md
import output


class MD:
    """Generates NVE dynamics from a general force model."""

    def __init__(self, force_model, dt: float, number_of_steps: int,
                 pos_init: np.ndarray, species, cell, masses,
                 prev_pos_init: np.ndarray=None, skip: int=0,
                 output_name='otf_run.out'):

        self.dt = dt
        self.Nsteps = number_of_steps
        self.force_model = force_model

        self.structure = Structure(cell=cell, species=species,
                                   positions=pos_init,
                                   mass_dict=masses,
                                   prev_positions=prev_pos_init)

        self.noa = self.structure.positions.shape[0]
        self.atom_list = list(range(self.noa))
        self.curr_step = 0
        self.kes = []
        self.output_name = output_name

    def run(self):
        self.start_time = time.time()

        while self.curr_step < self.Nsteps:
            # verlet algorithm follows Frenkel p. 70
            self.force_model(self.structure)  # update forces
            self.update_positions()
            self.record_state()
            self.curr_step += 1

        output.conclude_run(self)

    def update_positions(self):
        dtdt = self.dt ** 2
        new_pos = np.zeros((self.noa, 3))

        # update positions
        for i, pre_pos in enumerate(self.structure.prev_positions):
            mass = self.structure.mass_dict[self.structure.species[i]]
            pos = self.structure.positions[i]
            forces = self.structure.forces[i]

            new_pos[i] = 2 * pos - pre_pos + dtdt * forces / mass

        # update temperature
        KE, temperature, velocities = self.calculate_temperature(new_pos)
        self.KE = KE
        self.temperature = temperature
        self.velocities = velocities

        self.structure.prev_positions = self.structure.positions
        self.structure.positions = new_pos
        self.structure.wrap_positions()

    def record_state(self):
        self.kes.append(self.KE)
        output.write_md_config(self.dt, self.curr_step, self.structure,
                               self.temperature, self.KE,
                               self.start_time, self.output_name,
                               self.velocities)

    def calculate_temperature(self, new_pos):
        # set velocity and temperature information
        velocities = (new_pos - self.structure.prev_positions) / (2 * self.dt)

        KE = 0
        for i in range(len(self.structure.positions)):
            for j in range(3):
                KE += 0.5 * \
                    self.structure.mass_dict[self.structure.species[i]] * \
                    velocities[i][j] * velocities[i][j]

        # see conversions.nb for derivation
        kb = 0.0000861733034

        # see p. 61 of "computer simulation of liquids"
        temperature = 2 * KE / (3 * self.noa * kb)

        return KE, temperature, velocities
