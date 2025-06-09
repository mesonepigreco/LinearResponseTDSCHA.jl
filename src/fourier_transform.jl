@doc raw"""
    vector_r2q!(
        v_q :: Array{Complex{T}, 3},
        v_sc :: Matrix{T},
        q_tot :: Matrix{T})


Fourier transform a vector from real space and q space.

$$
v_k(\vec q) = \frac{1}{\sqrt{N_q}} \sum_{R} e^{-i 2\pi \vec R\cdot \vec q} v_k(\vec R)
$$


## Parameters

- v_q : (n_configs, 3nat, nq) 
    The target vector in Fourier space.
- v_sc : (n_configs, 3*nat_sc)
    The original vector in real space
- q_tot : (3, nq)
    The list of q vectors
- itau : (nat_sc)
    The correspondance for each atom in the supercell with the atom in the primitive cell.
- R_lat : (3, nat_sc)
    The origin coordinates of the supercell in which the atom is
"""
function vector_r2q!(
        v_q :: Array{Complex{T}, 3},
        v_sc :: Matrix{T},
        q :: Matrix{T},
        itau :: Vector{I},
        R_lat :: Matrix{T}
    ) where {T <: AbstractFloat, I <: Integer}

    nq = size(q, 2)
    n_random = size(v_sc, 1)
    nat_sc = size(v_sc, 2) ÷ 3
    nat = size(v_q, 2)

    v_q .= 0

    for jq ∈ 1:nq
        for k ∈ 1:nat_sc
            @views q_dot_R = q[:, jq]' * R_lat[:, k]
            exp_value = exp(- 1im * 2π * q_dot_R)

            for α in 1:3
                index_sc = 3 * (k - 1) + α
                index_uc = 3 * (itau[k] - 1) + α
                @simd for i ∈ 1:n_random
                    v_q[i, index_uc, jq] += exp_value * v_sc[i, index_sc]
                end
            end
        end
    end

    v_q ./= √(nq)
end

@doc raw"""
    vector_q2r!(
        v_sc :: Matrix{T},
        v_q :: Array{Complex{T}, 3},
        q_tot :: Matrix{T},
        itau :: Vector{I},
        R_lat :: Matrix{T}) where {T <: AbstractFloat, I <: Integer}


Fourier transform a vector from q space to real space.

$$
v_k(\vec R) = \frac{1}{\sqrt{N_q}} \sum_{R} e^{+i 2\pi \vec R\cdot \vec q} v_k(\vec q)
$$


## Parameters


- v_sc : (n_configs, 3*nat_sc)
    The target vector in real space
- v_q : (n_configs, nq, 3*nat) 
    The original vector in Fourier space. 
- q_tot : (3, nq)
    The list of q vectors
- itau : (nat_sc)
    The correspondance for each atom in the supercell with the atom in the primitive cell.
- R_lat : (3, nat_sc)
    The origin coordinates of the supercell in which the atom is
"""
function vector_q2r!(
        v_sc :: Matrix{T},
        v_q :: Array{Complex{T}, 3},
        q :: Matrix{T},
        itau :: Vector{I},
        R_lat :: Matrix{T}
    ) where {T <: AbstractFloat, I <: Integer}

    nq = size(q, 2)
    n_random = size(v_sc, 1)
    nat_sc = size(v_sc, 2) ÷ 3
    tmp_vector = zeros(Complex{T}, (n_random, 3*nat_sc))

    v_sc .= 0
    for jq ∈ 1:nq
        for k ∈ 1:nat_sc
            @views q_dot_R = q[:, jq]' * R_lat[:, k]
            exp_value = exp(1im * 2π * q_dot_R)

            for α in 1:3
                index_sc = 3 * (k - 1) + α
                index_uc = 3 * (itau[k] - 1) + α
                @simd for i ∈ 1:n_random
                    tmp_vector[i, index_sc] += exp_value * v_q[i, index_uc, jq]
                end
            end
        end
    end

    v_sc .= real(tmp_vector)
    v_sc ./= √(nq)
end

@doc raw"""
    matrix_r2q!(
        matrix_q :: Array{Complex{T}, 3},
        matrix_r :: AbstractMatrix{T},
        q :: Matrix{T},
        itau :: Vector{I},
        R_lat :: Matrix{T})

Fourier transform a matrix from real to q space

$$
M_{ab}(\vec q) = \sum_{\vec R} e^{2\pi i \vec q\cdot \vec R}\Phi_{a;b + \vec R}
$$

Where ``\Phi_{ab}`` is the real space matrix, the ``b+\vec R`` indicates the corresponding atom in the supercell displaced by ``\vec R``. 


## Parameters

- matrix_q : (3nat, 3nat, nq) 
    The target matrix in Fourier space.
- matrix_r : (3*nat_sc, 3*nat)
    The original matrix in real space (supercell)
- q_tot : (3, nq)
    The list of q vectors
- itau : (nat_sc)
    The correspondance for each atom in the supercell with the atom in the primitive cell.
- R_lat : (3, nat_sc)
    The origin coordinates of the supercell in which the corresponding atom is
"""
function matrix_r2q!(
        matrix_q :: Array{Complex{T}, 3},
        matrix_r :: AbstractMatrix{T},
        q :: Matrix{T},
        itau :: Vector{I},
        R_lat :: Matrix{T}; buffer = default_buffer())
    nq = size(q, 2)
    ndims = size(q, 1)
    nat_sc = size(matrix_r, 1) ÷ ndims
    nat = size(matrix_q, 1) ÷ ndims

    matrix_q .= T(0.0) 

    @no_escape buffer begin
        ΔR⃗ = @alloc(T, ndim)

        phase_i = Complex(T)(-2π * 1im)

        for iq in 1:nq
            for k_i in 1:nat
                k = ndims*(k_i - 1) + k_α
                @simd for h_i in 1:nat_sc
                    @views ΔR⃗ .= R_lat[:, k_i]
                    @views ΔR⃗ .-= R_lat[:, h_i]
                    @views q_dot_R = ΔR⃗' * q[:, iq]

                    h_i_uc = itau[h_i]

                    exp_factor = exp(phase_i * q_dot_R)
                    @views matrix_q[ndims*(h_i_uc - 1) : ndims * h_i_uc, ndims*(k_i - 1) : ndims*k_i, iq] .= matrix_r[ndims*(h_i - 1) : ndims * h_i, ndims*(k_i - 1) : ndims*k_i, iq]
                    matrix_q[ndims*(h_i_uc - 1) : ndims * h_i_uc, ndims*(k_i - 1) : ndims*k_i, iq] .*= exp_factor
                end
            end
        end
        nothing
    end
end

@doc raw"""
    matrix_q2r!(
        matrix_r :: AbstractMatrix{T},
        matrix_q :: Array{Complex{T}, 3},
        q :: Matrix{T},
        itau :: Vector{I},
        R_lat :: Matrix{T})

Fourier transform a matrix from q space into r space

$$
\Phi_{ab} = \frac{1}{N_q} \sum_{\vec q}
M_{ab}(\vec  q) e^{2i\pi \vec q\cdot[\vec R(a) - \vec R(b)]}
$$

Where ``\Phi_{ab}`` is the real space matrix, ``M_{ab}(\vec q)`` is the q space matrix.


## Parameters


- matrix_r : (3*nat_sc, 3*nat)
    The target matrix in real space (supercell)
- matrix_q : (3nat, 3nat, nq) 
    The original matrix in Fourier space.
- q_tot : (3, nq)
    The list of q vectors
- itau : (nat_sc)
    The correspondance for each atom in the supercell with the atom in the primitive cell.
- R_lat : (3, nat_sc)
    The origin coordinates of the supercell in which the corresponding atom is
"""
function matrix_r2q!(
        matrix_q :: Array{Complex{T}, 3},
        matrix_r :: AbstractMatrix{T},
        q :: Matrix{T},
        itau :: Vector{I},
        R_lat :: Matrix{T}; buffer = default_buffer())
    nq = size(q, 2)
    ndims = size(q, 1)
    nat_sc = size(matrix_r, 1) ÷ ndims
    nat = size(matrix_q, 1) ÷ ndims

    matrix_r .= T(0.0) 

    @no_escape buffer begin
        ΔR⃗ = @alloc(T, ndim)

        phase_i = Complex(T)(2π * 1im)

        for iq in 1:nq
            for k_i in 1:nat
                k = ndims*(k_i - 1) + k_α
                @simd for h_i in 1:nat_sc
                    @views ΔR⃗ .= R_lat[:, k_i]
                    @views ΔR⃗ .-= R_lat[:, h_i]
                    @views q_dot_R = ΔR⃗' * q[:, iq]

                    h_i_uc = itau[h_i]

                    exp_factor = exp(phase_i * q_dot_R)
                    @views matrix_r[ndims*(h_i - 1) : ndims * h_i, ndims*(k_i - 1) : ndims*k_i, iq] .= matrix_q[ndims*(h_i_uc - 1) : ndims * h_i_uc, ndims*(k_i - 1) : ndims*k_i, iq]
                    @views matrix_r[ndims*(h_i - 1) : ndims * h_i, ndims*(k_i - 1) : ndims*k_i, iq] .*= exp_factor
                end
            end
        end
        nothing
    end
end
