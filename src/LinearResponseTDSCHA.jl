module LinearResponseTDSCHA

using Bumper
using LinearAlgebra
using AtomicSymmetries

struct SCHAData{T}
    avg_structure :: AbstractMatrix{T}
    atoms_types :: AbstractVector{Int}
    supercell_size :: AbstractVector{Int}
    dynmat :: AbstractArray{Complex{T}, 3}
end

struct EnsembleData{T}
    original_dynmat :: AbstractArray{Complex{T}, 3}
    ensemble_structures_sc :: AbstractArray{T, 3}
    ensemble_forces_sc :: AbstractArray{T, 3}
    u_disp_sc :: AbstractMatrix{T}
    u_disp_fourier :: AbstractArray{Complex{T}, 3}
    f_fourier :: AbstractArray{Complex{T}, 3}
end


inlcude("perturbation_fourier.jl")

end # module LinearResponseTDSCHA
