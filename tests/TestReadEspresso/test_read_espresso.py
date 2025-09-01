import cellconstructor as CC
import cellconstructor.Structure
import sys, os


def test_read_espresso_ibrav():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the data
    structure = CC.Structure.Structure()
    structure.read_scf("SiC.pwi", read_espresso=True)
    return structure


if __name__ == "__main__":
    struct = test_read_espresso_ibrav()
    struct.fix_coords_in_unit_cell()

    import ase, ase.visualize
    ase.visualize.view(struct.get_ase_atoms())

