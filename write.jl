include("functions.jl")
using HDF5

C6 = 5.42 * 10^6;
Omega = 15.8;  
Rb = (C6/Omega)^(1/6);

# Varying the lattice spacing
ls_range = [4, 6.8]                                   # 4 is the smallest possible distance, 6.8 is checkerboard<->disordered
Delta_perturbation = 0.05*Omega                      # Should be adjusted as required   
Delta_range = range(-3.0, stop=3.0, length=61) * Omega  
N_range = [5]      

for N in N_range
    for ls in ls_range
        for Delta in Delta_range

            # Performing calculations with unperturbed Hamiltonian
            Nx, Ny = N, N;
            ax, ay = ls, ls;           
            sites, H_mpo = quera_hamiltonian(Nx, Ny, Omega, Delta, 0, zeros(Nx*Ny), 0, C6, ax, ay)
            energy, psi = dmrg_engine(sites, H_mpo, 10, 100, [5, 20, 100, 200])

            Omega_str = replace(string(Omega), "." => "_")
            Delta_str = replace(string(Delta), "." => "_")
            ls_str = replace(string(ls), "." => "_")

            # Saving to HDF5
            f = h5open("data/N$(N)/a$(ls_str)/Delta_$(Delta_str).h5", "w")
            write(f, "psi", psi)
            write(f, "energy", energy)
            write(f, "Omega", Omega)
            write(f, "Delta", Delta)
            write(f, "ls", ls)
            close(f)

            # Performing calculations with perturbed hamiltonian
            sites, H_mpo = quera_hamiltonian(Nx, Ny, Omega, Delta + Delta_perturbation, 0, zeros(Nx*Ny), 0, C6, ax, ay)
            energy, psi = dmrg_engine(sites, H_mpo, 10, 100, [5, 20, 100, 200])

            Omega_str = replace(string(Omega), "." => "_")
            Delta_str = replace(string(Delta), "." => "_")
            ls_str = replace(string(ls), "." => "_")

            # Saving to HDF5
            f = h5open("data/N$(N)/a$(ls_str)/Delta_$(Delta_str)_perturbed.h5", "w")
            write(f, "psi", psi)
            write(f, "energy", energy)
            write(f, "Omega", Omega)
            write(f, "Delta", Delta)
            write(f, "ls", ls)
            close(f)
        end
    end
end

                        