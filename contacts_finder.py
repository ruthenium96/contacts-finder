import numpy
import sympy
import ast
import CifFile
import sys
import itertools
import re

def parse_data(cif_file_name):
    cif_file = CifFile.ReadCif(cif_file_name)
    if len(cif_file.keys()) == 1:
        return cif_file.items()[0][1]
    else:
        print(f"Multiple blocks have been found: {cif_file.keys()}. Choose block to parse (print a number):")
        data_block_number = int(input())
        print(f"{cif_file.keys()[data_block_number]} has been chosen")
        return cif_file.items()[data_block_number][1]

def extract_cell_information(data):
    a = str_to_float(data["_cell_length_a"])
    b = str_to_float(data["_cell_length_b"])
    c = str_to_float(data["_cell_length_c"])
    alpha = str_to_float(data["_cell_angle_alpha"]) / (180.0 / numpy.pi)
    beta = str_to_float(data["_cell_angle_beta"]) / (180.0 / numpy.pi)
    gamma = str_to_float(data["_cell_angle_gamma"]) / (180.0 / numpy.pi)
    return a, b, c, alpha, beta, gamma

# expression from German Wikipedia:
def fractional_to_cartesian_matrix(a, b, c, alpha, beta, gamma):
    transformation_matrix = numpy.zeros((3, 3))
    transformation_matrix[0, 0] = a
    transformation_matrix[0, 1] = b * numpy.cos(gamma)
    transformation_matrix[1, 1] = b * numpy.sin(gamma)
    transformation_matrix[0, 2] = c * numpy.cos(beta)
    transformation_matrix[1, 2] = c * (numpy.cos(alpha) - numpy.cos(beta) * numpy.cos(gamma)) / numpy.sin(gamma)
    transformation_matrix[2, 2] = c * numpy.sqrt(1 - numpy.cos(alpha)**2 - numpy.cos(beta)**2 - numpy.cos(gamma)**2 + 2*numpy.cos(alpha)*numpy.cos(beta)*numpy.cos(gamma)) / numpy.sin(gamma)
    return transformation_matrix

def extract_symmetry_operations(data):
    symop_str = data["_space_group_symop_operation_xyz"]
    symop_str = [str.lower() for str in symop_str]
    from sympy.abc import x, y, z
    symop_func = [sympy.lambdify([x, y, z], sympy.parsing.sympy_parser.parse_expr(str)) for str in symop_str]
    return dict(zip(symop_str, symop_func))

def generate_shifts(a, b, c, distance):
    shift_a = int(numpy.ceil(distance / a)) + 2
    shift_b = int(numpy.ceil(distance / b)) + 2
    shift_c = int(numpy.ceil(distance / c)) + 2
    return list(itertools.product(range(-shift_a, shift_a + 1), range(-shift_b, shift_b + 1), range(-shift_c, shift_c + 1)))

# delete "(" and ")" from floats:
def str_to_float(str):
    return float(re.sub("\(.*?\)", "", str))

def extract_fragments(data):
    all_fragments = [[label] for label in data["_atom_site_label"]]
    list_of_atom_pairs = list(zip(data["_geom_bond_atom_site_label_1"], data["_geom_bond_atom_site_label_2"]))
    for first, second in list_of_atom_pairs:
        for fragment in all_fragments:
            if first in fragment:
                first_fragment = fragment
            if second in fragment:
                second_fragment = fragment
        if first_fragment != second_fragment:
            first_fragment.extend(second_fragment)
            all_fragments.remove(second_fragment)
    print(f"Found these fragments based on cif bonds:\n {all_fragments}")
    # TODO: allow user to use his own fragmentation
    return all_fragments

def generate_xyz(fragments, all_labels, coordinates_of_all_atoms, types_of_all_atoms, transformation_matrix, label1, label2, shift, func):
    result = ""
    for fragment in fragments:
        if label1 not in fragment:
            continue
        for label in fragment:
            index_of_current_label = all_labels.index(label)
            coordinates_of_current_atom = coordinates_of_all_atoms[index_of_current_label]
            cartesian_coordinates_of_current_atom = transformation_matrix @ coordinates_of_current_atom
            type_of_current_atom = types_of_all_atoms[index_of_current_label]
            result += type_of_current_atom + " " + ' '.join(map(str, cartesian_coordinates_of_current_atom)) + '\n'
    for fragment in fragments:
        if label2 not in fragment:
            continue
        for label in fragment:
            index_of_current_label = all_labels.index(label)
            coordinates_of_current_atom = numpy.array(func(*coordinates_of_all_atoms[index_of_current_label])) + shift
            cartesian_coordinates_of_current_atom = transformation_matrix @ coordinates_of_current_atom
            type_of_current_atom = types_of_all_atoms[index_of_current_label]
            result += type_of_current_atom + " " + ' '.join(map(str, cartesian_coordinates_of_current_atom)) + '\n'
    return result

def extract_coordinates_of_specific_atoms(data, list_of_indexes):
    return numpy.array([numpy.array([str_to_float(data[attr][index]) for attr in ["_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]]) for index in list_of_indexes])

def main():
    cif_file_name = sys.argv[1]
    data = parse_data(cif_file_name)

    fragments = extract_fragments(data)

    coordinates_of_all_atoms = extract_coordinates_of_specific_atoms(data, range(len(data["_atom_site_label"])))
    types_of_all_atoms = data["_atom_site_type_symbol"]

    # extract cell information:
    a, b, c, alpha, beta, gamma = extract_cell_information(data)
    # extract symmetry operations:
    symop = extract_symmetry_operations(data)
    # construct transformation from fractional coordinates to Cartesian:
    transformation_matrix = fractional_to_cartesian_matrix(a, b, c, alpha, beta, gamma)

    # extract labels:
    all_labels = data["_atom_site_label"]
    labels_of_requested_atoms = ast.literal_eval(input(f"Here is the list of all labels {all_labels}.\nPlease, choose labels to find contacts between them. Print list of labels in the Python format: [\"\"*]\n"))

    indexes_of_requsted_atoms = [all_labels.index(label) for label in labels_of_requested_atoms]

    coordinates_of_requsted_atoms = extract_coordinates_of_specific_atoms(data, indexes_of_requsted_atoms)

    dictionary_label_vs_coordinates_requested = dict(zip(labels_of_requested_atoms, coordinates_of_requsted_atoms))

    # ask about distance:
    max_distance = float(input("Print maximum distance (Angstrom):\n"))
    # generate all shifts:
    coordinates_shiftes = generate_shifts(a, b, c, max_distance)

    for label1, coordinates1 in dictionary_label_vs_coordinates_requested.items():
        for label2, coordinates2 in dictionary_label_vs_coordinates_requested.items():
            if label1 < label2:
                continue
            for shift in coordinates_shiftes:
                for func_name, func in symop.items():
                    cartesian_distance = numpy.linalg.norm(transformation_matrix @ (numpy.array(func(*coordinates2)) - numpy.array(coordinates1) + numpy.array(shift)))
                    if cartesian_distance < max_distance and cartesian_distance != 0:
                        print(label1, label2, cartesian_distance, "Å ", shift, func_name)
                        print(generate_xyz(fragments, all_labels, coordinates_of_all_atoms, types_of_all_atoms,
                                           transformation_matrix, label1, label2, shift, func))
                        print("------------------------")


if __name__ == '__main__':
    main()
