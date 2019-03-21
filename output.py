import time
import datetime
import numpy as np
import multiprocessing


def write_to_output(string: str, output_file: str = 'otf_run.out'):
    with open(output_file, 'a') as f:
        f.write(string)


def write_header(cutoffs, structure, output_name, std_tolerance):

    with open(output_name, 'w') as f:
        f.write(str(datetime.datetime.now()) + '\n')

    # report previous positions
    headerstring = ''
    headerstring += '\nprevious positions (A):\n'
    for i in range(len(structure.positions)):
        headerstring += structure.species[i] + ' '
        for j in range(3):
            headerstring += str("%.8f" % structure.prev_positions[i][j]) + ' '
        headerstring += '\n'
    headerstring += '-' * 80 + '\n'

    write_to_output(headerstring, output_name)


def write_md_config(dt, curr_step, structure, temperature, KE,
                    start_time, output_name, velocities):
    string = ''
    string += "\n*-Frame: " + str(curr_step)
    string += '\nSimulation Time: %.3f ps \n' % (dt * curr_step)

    # Construct Header line
    string += 'El \t\t\t  Position (A) \t\t\t\t\t '
    string += 'Force (ev/A) '
    string += '\t\t\t\t\t\t Std. Dev (ev/A) \t'
    string += '\t\t\t\t\t\t Velocities (A/ps) \n'

    # Construct atom-by-atom description
    for i in range(len(structure.positions)):
        string += structure.species[i] + ' '
        for j in range(3):
            string += str("%.8f" % structure.positions[i][j]) + ' '
        string += '\t'
        for j in range(3):
            string += str("%.8f" % structure.forces[i][j]) + ' '
        string += '\t'
        for j in range(3):
            string += str("%.8e" % structure.stds[i][j]) + ' '
        string += '\t'
        for j in range(3):
            string += str("%.8e" % velocities[i][j]) + ' '
        string += '\n'

    string += '\n'
    string += 'temperature: %.2f K \n' % temperature
    string += 'kinetic energy: %.6f eV \n' % KE

    string += 'wall time from start: %.2f s \n' % \
        (time.time() - start_time)

    write_to_output(string, output_name)


def conclude_run(output_name):
    footer = '▬' * 20 + '\n'
    footer += 'Run complete. \n'

    write_to_output(footer, output_name)
