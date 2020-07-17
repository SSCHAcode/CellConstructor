import cellconstructor as CC
import cellconstructor.Phonons 
import cellconstructor.ForceTensor


import sys, os
import time
import numpy as np 


def test_eff_charge_signle_q():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons("dyn", 32)

    # Q vector
    q_vector = np.array([0.25, 0, 0]) / dyn.alat

    t2 = CC.ForceTensor.Tensor2(dyn.structure, dyn.structure.generate_supercell(dyn.GetSupercell()), dyn.GetSupercell())
    t2.SetupFromPhonons(dyn)

    t2.Center()


    # Interpolate without effective charges
    efc = t2.effective_charges.copy()
    t2.effective_charges = None

    dynq_short_range = t2.Interpolate(-q_vector, asr = False)

    np.savetxt("DYNQ_shortrange.dat", dynq_short_range, header = "Q = {}".format(q_vector * dyn.alat))

    t2.effective_charges = efc
    dynq_full = t2.Interpolate(-q_vector, asr = False)

    np.savetxt("DYNQ_full.dat", dynq_full, header = "Q = {}".format(q_vector * dyn.alat))

    new_dyn = dyn.Copy()
    new_dyn.q_tot = [q_vector]
    new_dyn.q_stars = [[q_vector]]
    new_dyn.dynmats = [dynq_full]
    new_dyn.nqirr = 1
    new_dyn.save_qe("matdyn_python")
    
    
def test_eff_charge_interpolation():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons("dyn", 32)
    #dyn.Symmetrize()

    # Load the q path
    fp = open("matdyn.in", "r")
    lines = [l.strip() for l in fp.readlines()]
    fp.close()
    reading = False
    q_points = []
    q_prev = None
    n_prev = None
    for l in lines:
        if l == "/":
            reading = True 
            continue 

        if not reading:
            continue 

        # Read the q points
        data = l.split()
        if len(data) != 4:
            continue 
        
        current_q = np.array([ float(data[0]), float(data[1]), float(data[2])])
        if q_prev is not None:
            # Perform the linear interpolations
            q_x = np.linspace(q_prev[0], current_q[0], n_prev)
            q_y = np.linspace(q_prev[1], current_q[1], n_prev)
            q_z = np.linspace(q_prev[2], current_q[2], n_prev)

            for i in range(n_prev):
                q_vector = np.array([q_x[i], q_y[i], q_z[i]])  / dyn.alat# Change the units 
                q_points.append( q_vector )
            
        # Prepare for the next
        q_prev = current_q
        n_prev = int(data[-1]) 
        
    # Initialize the Force tensor
    print("Preparing the interpolation...")
    time1 = time.time()
    t2 = CC.ForceTensor.Tensor2(dyn.structure, dyn.structure.generate_supercell(dyn.GetSupercell()), dyn.GetSupercell())
    t2.SetupFromPhonons(dyn)

    # Center the dynamical matrix
    time2 = time.time()

    print("Time elapsed for preparing interpolation: {} s".format(time2 - time1))

    t2.WriteOnFile("mat2R_cc.d3q", "D3Q")
    t2.Center()
    time3 = time.time()

    # Get the mass matrix
    masses = np.tile( dyn.structure.get_masses_array(), (3,1)).T.ravel()
    _m_ = np.sqrt(np.outer(masses, masses))

    # Interpolate in the q points
    ws = []
    print ("Interpolating...")
    for iq, q in enumerate(q_points):
        dynq = t2.Interpolate( -q )

        # Diagonalize the dynamical matrix
        fc = dynq / _m_ 
        w2 = np.linalg.eigvalsh(fc)
        freqs = np.sign(w2) * np.sqrt(np.abs(w2))
        freqs *= CC.Units.RY_TO_CM
        ws.append( freqs )

    time4 = time.time()

    print("Time elapsed for interpolating: {} s".format(time4 - time3))

    # Save the frequencies
    # To be compared with matdyn
    np.savetxt("cc.freqs.gp", np.array(ws), header = "Frequencies interpolated with cellconstructor [cm-1]")


if __name__ == "__main__":
    #test_eff_charge_signle_q()
    test_eff_charge_interpolation()
