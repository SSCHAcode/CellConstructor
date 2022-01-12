import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Methods

import ase, ase.io
import ase.calculators.calculator

import numpy as np

import sys, os



class Calculator:
    def __init__(self):
        """
        CELLCONSTRUCTOR  CALCULATOR
        ===========================
        
        This is an alternative to ASE calculators, which often do not work.
        It is explicitely done for cellconstructor and python-sscha
        
        """

        self.label = "label"
        self.directory = None 
        self.command = None 
        self.properties = {}
        self.structure = None  



def get_energy_forces(calculator, structure):
    """
    Accepts both an ase calculaotr and a calculator of CellConstructor
    """

    if isinstance(calculator, ase.calculators.calculator.Calculator):
        atm = structure.get_ase_atoms()
        atm.set_calculator(calculator)
        return atm.get_total_energy(), atm.get_forces()
    elif isinstance(calculator, Calculator):
        calculator.calculate(structure)
        return calculator.properties["energy"], calculator.properties["forces"]
    else:
        raise ValueError("Error, unknown calculator type")

class FileIOCalculator(Calculator):
    def __init__(self):
        Calculator.__init__(self)

    def write_input(self):
        if self.directory is None:
            self.directory = os.path.abspath(".")

        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

    def calculate(self, structure):
        self.write_input(structure)
        self.execute()
        self.read_results()
    
    def execute(self):
        cmd = "cd {} && {} && cd ..".format(self.directory, self.command.replace("PREFIX", self.label))
        os.system(cmd)

    def read_results(self):
        pass 


class Espresso(FileIOCalculator):
    def __init__(self,  input_data, pseudopotentials, masses = None, command = "pw.x -i PREFIX.pwi > PREFIX.pwo", kpts = (1,1,1), koffset = (0,0,0)):
        """
        ESPRESSO CALCULATOR
        ===================

        parameters
        ----------
            data_input : dict
                Dictionary of the Quantum Espresso PW input namespace
            pseudopotentials : dict
                Dictionary of the file names of the pseudopotentials
            masses : dict
                Dictionary of the masses (in UMA) of the specified atomic species
        """
        FileIOCalculator.__init__(self)

        self.kpts = kpts
        self.koffset = koffset
        self.input_data = input_data
        self.pseudopotentials = pseudopotentials
        if masses is None:
            masses = {}
            for atm in pseudopotentials:
                masses[atm] = 1.000
        self.masses = masses

        assert len(list(self.pseudopotentials)) == len(list(self.masses)), "Error, pseudopotential and masses must match"

    def write_input(self, structure):
        FileIOCalculator.write_input(self)

        typs = np.unique(structure.atoms)

        total_input = self.input_data
        total_input["system"].update({"nat" : structure.N_atoms, "ntyp" : len(typs), "ibrav" : 0})

        scf_text = "".join(CC.Methods.write_namelist(total_input))

        scf_text += """
ATOMIC_SPECIES
"""
        for atm in typs:
            scf_text += "{}  {}   {}\n".format(atm, self.pseudopotentials[atm], self.masses[atm])
        
        scf_text += """
K_POINTS automatic
{} {} {} {} {} {}
""".format(self.kpts[0], self.kpts[1], self.kpts[2],
            self.koffset[0], self.koffset[1], self.koffset[2])
        
        scf_text += structure.save_scf(None, get_text = True)

        filename = os.path.join(self.directory, self.label + ".pwi")

        with open(filename, "w") as fp:
            fp.write(scf_text)
        

    def read_results(self):
        FileIOCalculator.read_results(self)

        filename = os.path.join(self.directory, self.label + ".pwo")

        atm = ase.io.read(filename)

        self.properties = {"energy" : atm.get_total_energy(), "forces" : atm.get_forces()}
        

        

















