@doc raw"""
    get_d2V_dR2_Rperturb!(d2V_dR2_pert, u_disp, v_disp, δf, w_eq)

Compute the perturbed average

$$
\left<\frac{d^2V}{dR_adR_b}\right>_{\rho^{(1)}}
$$

where the perturbation is on the centroid $R^{(1)}$.

## Parameters

- `d2V_dR2_pert` : The target value of the perturbed average to be computed
- `R1` : The perturbation ``R^{(1)}`` (Gamma, n_modes)
- `u_disp` : The ionic displacements (n_configs, n_modes, n_q)
- `v_disp` : The ionic displacements multiplied by the inverse covariance matrix
- `δf` : The force (- sscha forces) (n_configs, n_modes, n_q)
- `w_q` : The weights of the configurations (at equilibrium)
- `minus_q_index` : For each q point, who is the (-q + G) associated? 
"""
function get_d2V_dR2_Rperturb!(d2V_dR2 :: AbstractArray{Complex{T}, 3}, R1 :: AbstractVector{T}, u_disp :: AbstractArray{Complex{T}, 3}, v_disp :: AbstractArray{Complex{T}, 3}, δf :: AbstractArray{Complex{T}, 3}, w_eq :: AbstractVector{T}, minus_q_index :: AbstractVector{Int}; buffer=default_buffer())
    # Get the dimension
    n_random = size(u_disp, 1)
    n_modes = size(u_disp, 2)
    nq = size(u_disp, 3)

    d2V_dR2 .= T(0.0)

    @assert size(R1) == n_modes, "Error, R1 must be of size $n_modes, got $(size(R1)) instead."


    @no_escape buffer begin
        # Allocate on the stack the weights
        w_pert = @alloc(T, n_random)
        tmp_matrix = @alloc(T, n_random)
        δ0 = @alloc(T, n_random, n_modes)
        @view δ0 .= real.(v_disp[:, :, 1])

        w_pert .= T(0.0)

        for i in 1:n_modes
            mul!(w_pert, R1', δ0[:, i], 1.0, 1.0)
        end
        w_pert ./= n_random
            
        # 2/3 of the permutation symmetry
        for jq in 1:nq
            for k in 1:n_modes
                for j in 1:n_modes
                    @simd for i in 1:n_random
                        tmp = v_disp[i, j, jq] * conj(δf[i, k, jq])
                        d2V_dR2[j, k, jq] -= tmp * w_eq[i] * w_pert[i]
                    end
                end
            end
            # Impose symmetry
            @views d2V_dR2[:, :, jq] += d2V_dR2[:, :, jq]'
        end
        d2V_dR2 ./= T(3 * n_random) 

        # The last 1/3 permutation symmetry
        # Compute the perturbation on the ensemble
        @view δ0 .= real.(δf_disp[:, :, 1])
        w_pert .= T(0.0)

        for i in 1:n_modes
            mul!(w_pert, R1', δ0[:, i], 1.0, 1.0)
        end
        w_pert ./= n_random
            
        # Compute d2V_dR2
        divide_by = T(1.0/ ( n_random * 3.0))
        for jq in 1:nq
            for k in 1:n_modes
                for j in 1:n_modes
                    @simd for i in 1:n_random
                        tmp = v_disp[i, j, jq] * conj(v_disp[i, k, jq])
                        d2V_dR2[j, k, jq] -= tmp * w_eq[i] * w_pert[i] * divide_by
                        # TODO: Here the time could be divided by 2 as k, j are symmetric
                    end
                end
            end
        end
        
        # Impose time reversal
        for iq in 1:nq
            @views d2V_dR2[:, :, iq] .+= conj.(d2V_dR2[:, :, minus_q_index[iq]]')
        end
        d2V_dR2 ./= T(2)
        nothing
    end
end


@doc raw"""
    get_d2V_dR2_Rperturb!(d2V_dR2_pert, u_disp, v_disp, δf, w_eq)

Compute the perturbed average

$$
\left<\frac{dV}{dR_a}\right>_{\rho^{(1)}}
$$

where the perturbation is on the inverse covariance matrix ``\Upsilon^{(1)}``.

## Parameters

- `dV_dR_pert` : The target value of the perturbed average to be computed
- `dV2_dR2_pert` : The target value of the perturbed average to be computed (only if v4)
- `Y1` : The perturbation ``R^{(1)}`` (Gamma, n_modes)
- `u_disp` : The ionic displacements (n_configs, n_modes, n_q)
- `v_disp` : The ionic displacements multiplied by the inverse covariance matrix
- `δf` : The force (- sscha forces) (n_configs, n_modes, n_q)
- `w_q` : The weights of the configurations (at equilibrium)
- `minus_q_index` : For each q point, who is the (-q + G) associated? 
- `Ψ` : For each q, the covariance matrix
- `compute_v4` : If true, then `d2V_dR2_pert` is computed [Default true]
- `buffer` : Bumper.jl buffer for stack allocation (caching).
"""
function get_avgs_Yperturb!(dV_dR :: AbstractVector{T}, dV2_dR2 :: AbstractArray{Complex{T}, 3}, Y1 :: AbstractArray{Complex{T}, 3}, u_disp :: AbstractArray{Complex{T}, 3}, v_disp :: AbstractArray{Complex{T}, 3}, δf :: AbstractArray{Complex{T}, 3}, w_eq :: AbstractVector{T}, minus_q_index :: AbstractVector{Int}, Ψ :: AbstractArray{Complex{T}, 3}; buffer=default_buffer(), compute_v4 :: Bool = true)
    # Get the dimension
    n_random = size(u_disp, 1)
    n_modes = size(u_disp, 2)
    nq = size(u_disp, 3)

    dV_dR .= T(0.0)

    @assert size(Y1)[1] == n_modes, "Error, Y1 must be of size $n_modes ^2 x $nq, got $(size(Y1)) instead."
    @assert size(Y1)[2] == n_modes, "Error, Y1 must be of size $n_modes ^2 x $nq, got $(size(Y1)) instead."
    @assert size(Y1)[3] == nq, "Error, Y1 must be of size $n_modes ^2 x $nq, got $(size(Y1)) instead."

    @no_escape buffer begin
        # Allocate on the stack the weights
        w_pert = @alloc(T, n_random)

        for jq in 1:nq
            for k in 1:n_modes
                for j in 1:n_modes
                    @views mul!(w_pert, u_disp[:, k, jq]', u_disp[:, j, jq], Y1[k, j, jq], 1.0)
                    # @simd for i in 1:n_random
                    #     w_pert[i] = Y1[k, j, jq] * conj(u_disp[i, k, jq]) * u_disp[i, j, jq]
                    # end
                end
            end
        end
        w_pert ./= n_random
        w_pert .*= -w_eq # multiply by -1

        δ = @alloc(T, n_random, n_modes)
        @view δ .= real.(δf[:, :, 1])
        
        # Add to the 1/3 of the perturbation
        mul!(dV_dR, w_pert', δ, T(1.0/3.0), T(0.0))

        # Compute the v4 contribution
        if compute_v4
            dV2_dR2 .= 0.0
            for iq in 1:nq
                for j in 1:n_modes
                    for k in 1:n_modes
                        @simd for i in 1:n_random
                            d2V_dR2[k, j, iq] -= w_pert[i] * conj(δf[i, k, iq]) * v_disp[i, j, iq]
                        end
                    end
                end

                @views d2V_dR2[:, :, iq] .+= d2V_dR2[:, :, iq]'
            end
            d2V_dR2 ./= T(n_random * 2)
        end

        w_pert .= T(0.0)
        # Compute the Ψ⋅δf
        Ψδf = @alloc(Complex{T}, n_random, n_modes, nq)
        for iq in 1:nq
            @views mul!(Ψδf, δf[:, :, iq], Ψ[:, :, iq])
        end

        # Compute the 2/3 of the weight
        for jq in 1:nq
            for k in 1:n_modes
                for j in 1:n_modes
                    @simd for i in 1:n_random
                        w_pert[i] -= conj(u_disp[i, k, jq]) * Y1[k, j, jq] * Ψδf[i, j, jq]
                    end
                end
            end
        end
        w_pert ./= n_random
        w_pert .*= w_eq

        # Add the last 2/3 of the symmetrized force
        @views δ .= real.(v_disp[:,:, 1])
        mul!(dV_dR, w_pert', δ, T(2.0/3.0), T(1.0))

        if compute_v4
            divide_by = T(1.0 / n_random)
            for iq in 1:nq
                for j in 1:n_modes
                    for k in 1:n_modes
                        @simd for i in 1:n_random
                            d2V_dR2[k, j, iq] -= w_pert[i] * conj(v_disp[i, k, iq]) * v_disp[i, j, iq] * divide_by
                        end
                    end
                end
            end

            # Apply time reversal
            for iq in 1:nq
                @views d2V_dR2[:, :, iq] .+= conj.(d2V_dR2[:, :, minus_q_index[iq]]')
            end
            d2V_dR2 ./= T(2)
        end
        nothing
    end
end
