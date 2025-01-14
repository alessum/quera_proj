using ITensors, ITensorMPS, Plots, ProgressBars
using Base.Iterators
using Statistics
using Distributed

# declare custom spin-1/2 effective number operator
function ITensors.op(::OpName"N",::SiteType"S=1/2")
    M = zeros(2,2)
    M[2,2] = 1.0
    return M
  end

# Creates Hamiltonian MPO for a Rydberg system, with OB along x axis and PB along y axis - open cylinder
# Mostly used to reproduce the results of the paper; phase diagram of the Rydberg system
function gen_rydberg_system(Nx, Ny, Omega, Delta, C6, lattice_spacing, coupling_cutoff = 1/10000, y_periodic = false)
    
    # ALL HAIL THE ZIGZAG GEOMETRY
    # 
    #     (1)-(Ny+1)- ... -( (Nx-1)*Ny + 1 )
    #      |    |                  |
    #     (2)-(Ny+2)- ... -( (Nx-1)*Ny + 2 )
    #      |    |                  |
    #      .    .                  .
    #      |    |                  |
    #     (Ny)-(2*Ny)- ... -    ( Nx*Ny )
    #
    # ITS SUPER LONG RANGE EVEN FOR NN BUT I CANT FIND ANYTHING BETTER RIGHT NOW

    N = Nx*Ny

    coords = collect(flatten([[Int64[_a, _b] for _b in 1:Ny] for _a in 1:Nx]))
    #print(coords)

    mpo_ops = AutoMPO()

    # Global X and Z fields
    for j in 1:N

        mpo_ops += Omega/2, "X", j # global X
        mpo_ops += -Delta, "N", j # global N
        
    end

    # Long-range interactions
    for i in 1:N
        ci = coords[i]
        for j in 1:N
            cj = coords[j]

            if i != j # exclude self-interactions
                # compute interaction strength
                if y_periodic
                    Y1, Y2, Y3 = abs(ci[2]-cj[2]), abs(ci[2]-cj[2] + Ny), abs(cj[2]-ci[2] + Ny)
                    Y = min(Y1, Y2, Y3)
                else
                    Y = abs(ci[2]-cj[2])
                end
                X = abs(ci[1]-cj[1])

                prefactor = sqrt(X^2 + Y^2)                  
                if prefactor < 1/coupling_cutoff                # check if interaction is within cutoff 
                    x = prefactor*lattice_spacing
                    coeff = 0.5 * C6 / (x^6);                   
                    mpo_ops += coeff, "N", i, "N", j
                end
            end  
        end
    end

    sites = siteinds("S=1/2",N)

    return sites, MPO(mpo_ops, sites);

end

# Creates Hamiltonian MPO for a Rydberg system corresponding to the full [time-independent] QuEra Aquila Hamiltonian
# Note that the geometry can be even more generic, here we constrain it to rectamgular lattice
# Used to compare the results of the paper with the Hamiltonian that we work with
function quera_hamiltonian(Nx, Ny, Omega, Delta_Global, Delta_Local, Hj, phi, C6, ax, ay, coupling_cutoff = 1/800)
    # Note that we again use ZIGZAG labeling 
    N = Nx*Ny
    coords = collect(flatten([[Int64[_a, _b] for _b in 1:Ny] for _a in 1:Nx]))
    mpo_ops = AutoMPO()

    # Global X and Z fields
    for j in 1:N

        mpo_ops += Omega/2 * exp(-im * phi), "S+", j                # Global X with phase    
        mpo_ops += Omega/2 * exp(+im * phi), "S-", j 
        mpo_ops += (-Delta_Global - Delta_Local * Hj[j]), "N", j    # global N

    end

    # Long-range interactions
    for i in 1:N
        ci = coords[i]
        for j in 1:N
            cj = coords[j]

            if i != j # exclude self-interactions
                # compute interaction strength, using open BC
                X, Y = (ci[1]-cj[1]) * ax, (ci[2]-cj[2]) * ay
                Rij2 = X^2 + Y^2;                 
                if Rij2 < 1/coupling_cutoff * ax               # check if interaction is within cutoff 
                    coeff = C6 / (Rij2^3);                   # Note factor of 0.5 difference from the paper
                    mpo_ops += coeff, "N", i, "N", j
                end
            end  
        end
    end

    sites = siteinds("S=1/2",N)

    return sites, MPO(mpo_ops, sites);
end

function dmrg_engine(sites, H_mpo, linkdims=10, nsweeps=10, maxdim=[5, 20, 100], cutoff=[1e-10])
    
    psi0 = randomMPS(sites; linkdims)
    return dmrg(H_mpo, psi0; nsweeps,maxdim,cutoff)

end