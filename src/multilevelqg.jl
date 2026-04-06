module MultiLevelQG

export
  fwdtransform!,
  invtransform!,
  streamfunctionfrompv!,
  pvfromstreamfunction!,
  bfromstreamfunction!,
  omegaeqn!,
  updatevars!,

  set_q!,
  set_ψ!,
  energies,
  fluxes

using
  FFTW,
  CUDA,
  LinearAlgebra,
  StaticArrays,
  Reexport,
  DocStringExtensions,
  KernelAbstractions

@reexport using FourierFlows

using FourierFlows: parsevalsum, parsevalsum2, superzeros, plan_flows_rfft, CPU, GPU
using KernelAbstractions.Extras.LoopInfo: @unroll

nothingfunction(args...) = nothing

"""
    Problem(nlevels :: Int,
                        dev = CPU();
                         nx = 128,
                         ny = nx,
                         Lx = 2π,
                         Ly = Lx,
                         f₀ = 1.0,
                          β = 0.0,
                          H₀ = 1.0,
                          U = zeros(nlevels),
                          N² = pi .* ones(nlevels),
                        eta = nothing,
       topographic_gradient = (0, 0),
                          r = 0,
                          ν = 0,
                         nν = 1,
                         dt = 0.01,
                    stepper = "RK4",
                     calcFq = nothingfunction,
                 stochastic = false,
                     linear = false,
           aliased_fraction = 1/3,
                          T = Float64)

Construct a multi-level quasi-geostrophic problem with `nlevels` levels (including surface buoyancy levels) on device `dev`.
The vertical is discretized using the Chebyshev collocation method. The vertical grid is thus defined by the Chebyshev grid,
which depends on H₀ and nlevels.

Arguments
=========
- `nlevels`: (required) Number of levels.
- `dev`: (required) `CPU()` (default) or `GPU()`; computer architecture used to time-step `problem`.

Keyword arguments
=================
  - `nx`: Number of grid points in ``x``-domain.
  - `ny`: Number of grid points in ``y``-domain.
  - `Lx`: Extent of the ``x``-domain.
  - `Ly`: Extent of the ``y``-domain.
  - `f₀`: Constant planetary vorticity.
  - `β`:  Planetary vorticity ``y``-gradient.
  - `H₀`: Extent of the ``z''-domain.
  - `U`:  Background constant zonal flow ``U(y)`` at Chebyshev levels.
  - `N²`: Background stratification at Chebyshev levels.
  - `eta`: Periodic component of the bathymetry.
  - `topographic_gradient`: The ``(x, y)`` components of the topographic large-scale gradient.
  - `r`: Linear bottom drag coefficient.
  - `ν`: Small-scale (hyper)-viscosity coefficient.
  - `nν`: (Hyper)-viscosity order, `nν```≥ 1``.
  - `dt`: Time-step.
  - `stepper`: Time-stepping method.
  - `calcF`: Function that calculates the Fourier transform of the forcing, ``F̂``.
  - `stochastic`: `true` or `false` (default); boolean denoting whether `calcF` is temporally stochastic.
  - `linear`: `true` or `false` (default); boolean denoting whether the linearized equations of motions are used.
  - `aliased_fraction`: the fraction of high wavenumbers that are zero-ed out by `dealias!()`.
  - `T`: `Float32` or `Float64` (default); floating point type used for `problem` data.
"""
function Problem(nlevels::Int,                                     # number of levels
                          dev = CPU();
              # Numerical parameters
                           nx = 128,
                           ny = nx,
                           Lx = 2π,
                           Ly = Lx,
              # Physical parameters
                           f₀ = 1.0,                               # Coriolis parameter
                            β = 0.0,                               # y-gradient of Coriolis parameter
                           H₀ = 1.0,                               # extent of the ``z''-domain.
                            U = zeros(nlevels),                    # background constant zonal flow ``U(y)`` at Chebyshev levels
                           N² = pi .* ones(nlevels),               # background stratification at Chebyshev levels
                          eta = nothing,                           # periodic component of the bathymetry
         topographic_gradient = (0, 0),                            # tuple with the ``(x, y)`` components of topographic large-scale gradient
              # Bottom Drag and/or (hyper)-viscosity
                            r = 0,
                            ν = 0,
                           nν = 1,
              # Timestepper and equation options
                           dt = 0.01,
                      stepper = "RK4",
                       calcFq = nothingfunction,
                   stochastic = false,
                       linear = false,
              # Float type and dealiasing
             aliased_fraction = 1/3,
                            T = Float64)

  # bathymetry
  eta === nothing && (eta = zeros(dev, T, (nx, ny)))

  grid = TwoDGrid(dev; nx, Lx, ny, Ly, aliased_fraction, T)

  params = Params(nlevels, f₀, β, H₀, N², U, eta, topographic_gradient, r, ν, nν, grid; calcFq)

  vars = calcFq == nothingfunction ? DecayingVars(grid, params) : (stochastic ? StochasticForcedVars(grid, params) : ForcedVars(grid, params))

  equation = linear ? LinearEquation(params, grid) : Equation(params, grid)

  FourierFlows.Problem(equation, stepper, dt, grid, vars, params)
end

"""
    struct Params{T, Aphys3D, Aphys2D, Atrans4D, Atrans4D_int, Trfft} <: AbstractParams

The parameters for the `MultiLevelQG` problem.

$(TYPEDFIELDS)
"""
struct Params{T, Aphys3D, Aphys2D, Atrans4D, Atrans4D_int, Trfft} <: AbstractParams
  # prescribed params
    "number of levels"
   nlevels :: Int
    "constant planetary vorticity"
        f₀ :: T
    "planetary vorticity ``y``-gradient"
         β :: T
    "extent of the ``z``-domain"
         H₀ :: T
    "tuple with background stratification at Chebyshev levels"
         N² :: Tuple
    "array with background constant zonal flow ``U(y)`` at Chebyshev levels"
         U :: Aphys3D
    "array containing the bathymetry"
       eta :: Aphys2D
    "tuple containing the ``(x, y)`` components of topographic large-scale gradient"
    topographic_gradient :: Tuple{T, T}
    "linear bottom drag coefficient"
         r :: T
    "small-scale (hyper)-viscosity coefficient"
         ν :: T
    "(hyper)-viscosity order, `nν```≥ 1``"
        nν :: Int
    "function that calculates the Fourier transform of the forcing, ``F̂``"
   calcFq! :: Function

  # derived params
      "tuple of Chebyshev levels"
         z :: Tuple
    "array containing ``x``-gradient of upper surface buoyancy, interior PV, and lower surface buoyancy due to topography"
        Qx :: Aphys3D
    "array containing ``y``-gradient of upper surface buoyancy, interior PV, and lower surface buoyancy due to ``β``, ``U``, and topography"
        Qy :: Aphys3D
    "array containing coefficients for getting PV from streamfunction"
         S :: Atrans4D
    "array containing coefficients for inverting PV to streamfunction"
       S⁻¹ :: Atrans4D
    "array containing coefficients for inverting the interior omega equation for vertical velocity"
       M⁻¹ :: Atrans4D_int
    "array containing Chebyshev differentiation matrix, which discretizes ``∂z``"
         D :: Aphys2D
    "rfft plan for FFTs"
  rfftplan :: Trfft
end

function convert_U_to_U3D(dev, nlevels, grid, U::AbstractArray{TU, 1}) where TU
  T = eltype(grid)

  if length(U) == nlevels
    U_2D = zeros(dev, T, (1, nlevels))
    U_2D[:] = U
    U_2D = repeat(U_2D, outer=(grid.ny, 1))
  else
    U_2D = zeros(dev, T, (grid.ny, 1))
    U_2D[:] = U
  end

  U_3D = zeros(dev, T, (1, grid.ny, nlevels))
  @views U_3D[1, :, :] = U_2D

  return U_3D
end

function convert_U_to_U3D(dev, nlevels, grid, U::AbstractArray{TU, 2}) where TU
  T = eltype(grid)
  U_3D = zeros(dev, T, (1, grid.ny, nlevels))
  @views U_3D[1, :, :] = U

  return U_3D
end

function convert_U_to_U3D(dev, nlevels, grid, U::Number)
  T = eltype(grid)
  A = device_array(dev)
  U_3D = reshape(repeat([T(U)], outer=(grid.ny, 1)), (1, grid.ny, nlevels))

  return A(U_3D)
end

function Params(nlevels::Int, f₀, β, H₀, N², U, eta, topographic_gradient, r, ν, nν, grid::TwoDGrid;
                calcFq=nothingfunction, effort=FFTW.MEASURE)
  dev = grid.device
  T = eltype(grid)
  A = device_array(dev)

  ny, nx = grid.ny, grid.nx
  nkr, nl = grid.nkr, grid.nl
  kr, l  = grid.kr, grid.l

  # Chebyshev grid on [–H₀, 0]
  ξ = [cos((i - 1) * pi / (nlevels - 1)) for i in 1 : nlevels] # Chebyshev grid on [-1, 1]
  z = H₀ / 2 .* (ξ .- 1)                                       # maps [-1, 1] -> [-H₀, 0]

  U = convert_U_to_U3D(dev, nlevels, grid, U)

  Uyy = real.(ifft(-l.^2 .* fft(U[:, :, 2 : end - 1]))) # only calculate curvature of shear for interior PV part
  Uyy = CUDA.@allowscalar repeat(Uyy, outer=(nx, 1, 1))

  # Calculate the periodic components of the bathymetry gradients
  eta = A(eta)
  etah = rfft(eta)
  etax = irfft(im * kr .* etah, nx)   # ∂η/∂x
  etay = irfft(im * l  .* etah, nx)   # ∂η/∂y

  # Add large-scale topographic gradient
  topographic_gradient = T.(topographic_gradient)
  @. etax += topographic_gradient[1]
  @. etay += topographic_gradient[2]

  # Add everything to background buoyancy/PV gradients, except part coming from vertical shear   
  Qx = zeros(dev, T, (nx, ny, nlevels))
  @views @. Qx[:, :, end] += N²[end] * etax

  Qy = zeros(dev, T, (nx, ny, nlevels))
  β_T = T(β) # T(β) ensures that Qy remains same type as U
  @views @. Qy[:, :, 2 : end - 1] = β_T - Uyy 
  @views @. Qy[:, :, end] += N²[end] * etay

  rfftplanlayered = plan_flows_rfft(A{T, 3}(undef, grid.nx, grid.ny, nlevels), [1, 2]; flags=effort)

  # Compute vertical derivative matrix
  D = zeros(T, (nlevels, nlevels))
  calcD!(D, H₀, nlevels)

  # Compute vertical part of PV inversion matrix
  F = zeros(T, (nlevels, nlevels))
  calcF!(F, D, f₀, N²)

  # Subtract the vertical shear part from background buoyancy/PV: i.e., Qy -= F*U
  Qy .-= A(reshape(permutedims(F * permutedims(Array(U)[1, :, :], (2, 1)), (2, 1)), 1, ny, nlevels)) 

  # Compute PV inversion matrix
  typeofSkl = SArray{Tuple{nlevels, nlevels}, T, 2, nlevels^2} # StaticArrays of type T and dims = (nlevels, nlevels)

  S = Array{typeofSkl, 2}(undef, (nkr, nl))    # Array of StaticArrays
  calcS!(S, F, nlevels, grid)

  S⁻¹ = Array{typeofSkl, 2}(undef, (nkr, nl))  # Array of StaticArrays
  calcS⁻¹!(S⁻¹, F, nlevels, grid)

  # Compute omega equation inversion matrix
  typeofMkl = SArray{Tuple{nlevels - 2, nlevels - 2}, T, 2, (nlevels - 2)^2} # StaticArrays of type T and dims = (nlevels - 2, nlevels - 2)

  M⁻¹ = Array{typeofMkl, 2}(undef, (nkr, nl))    # Array of StaticArrays
  calcM⁻¹!(M⁻¹, D, f₀, N², nlevels, grid)

  # Convert to appropriate ArrayType
  S, S⁻¹ = A(S), A(S⁻¹)
  M⁻¹ = A(M⁻¹)
  D = A(D)

  return Params(nlevels, T(f₀), T(β), T(H₀), Tuple(T.(N²)), U, eta, topographic_gradient, T(r), T(ν), nν, calcFq, Tuple(T.(z)), Qx, Qy, S, S⁻¹, M⁻¹, D, rfftplanlayered)
end

numberoflevels(params) = params.nlevels

# ---------
# Equations
# ---------

"""
    hyperviscosity(params, grid)

Return the linear operator `L` that corresponds to (hyper)-viscosity of order ``n_ν`` with
coefficient ``ν`` on the ``nlevels'' interior and surface levels
```math
L_j = - ν |𝐤|^{2 n_ν}, \\ j = 1, ..., n .
```
"""
function hyperviscosity(params, grid)
  dev = grid.device
  T = eltype(grid)

  L = device_array(dev){T}(undef, (grid.nkr, grid.nl, numberoflevels(params)))
  @. L = - params.ν * grid.Krsq^params.nν
  @views @. L[1, 1, :] = 0

  return L
end

"""
    LinearEquation(params, grid)

Return the equation for a multi-level quasi-geostrophic problem with `params` and `grid`.
The linear operator ``L`` includes only (hyper)-viscosity and is computed via
`hyperviscosity(params, grid)`.

The nonlinear term is computed via [`calcNlinear!`](@ref).
"""
function LinearEquation(params, grid)
  L = hyperviscosity(params, grid)

  return FourierFlows.Equation(L, calcNlinear!, grid)
end

"""
    Equation(params, grid)

Return the equation for a multi-level quasi-geostrophic problem with `params` and `grid`.
The linear operator ``L`` includes only (hyper)-viscosity and is computed via
`hyperviscosity(params, grid)`.

The nonlinear term is computed via [`calcN!`](@ref GeophysicalFlows.MultiLayerQG.calcN!).
"""
function Equation(params, grid)
  L = hyperviscosity(params, grid)

  return FourierFlows.Equation(L, calcN!, grid)
end


# ----
# Vars
# ----

"""
    struct Vars{Aphys, Atrans, F, P} <: AbstractVars

The variables for multi-level QG problem.

$(FIELDS)
"""
struct Vars{Aphys, Atrans, F, P} <: AbstractVars
    "upper surface buoyancy, interior PV, lower surface buoyancy"
        q :: Aphys
    "streamfunction"
        ψ :: Aphys
    "``x``-component of velocity"
        u :: Aphys
    "``y``-component of velocity"
        v :: Aphys
    "Fourier transform of generalized PV"
       qh :: Atrans
    "Fourier transform of streamfunction"
       ψh :: Atrans
    "Fourier transform of ``x``-component of velocity"
       uh :: Atrans
    "Fourier transform of ``y``-component of velocity"
       vh :: Atrans
    "Fourier transform of forcing"
      Fqh :: F
    "`sol` at previous time-step"
  prevsol :: P
end

const DecayingVars = Vars{<:AbstractArray, <:AbstractArray, Nothing, Nothing}
const ForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, Nothing}
const StochasticForcedVars = Vars{<:AbstractArray, <:AbstractArray, <:AbstractArray, <:AbstractArray}

"""
    DecayingVars(grid, params)

Return the variables for an unforced multi-level QG problem with `grid` and `params`.
"""
function DecayingVars(grid, params)
  Dev = typeof(grid.device)
  T = eltype(grid)
  nlevels = numberoflevels(params)

  @devzeros Dev T (grid.nx, grid.ny, nlevels) q ψ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, nlevels) qh ψh uh vh

  return Vars(q, ψ, u, v, qh, ψh, uh, vh, nothing, nothing)
end

"""
    ForcedVars(grid, params)

Return the variables for a forced multi-level QG problem with `grid` and `params`.
"""
function ForcedVars(grid, params)
  Dev = typeof(grid.device)
  T = eltype(grid)
  nlevels = numberoflevels(params)

  @devzeros Dev T (grid.nx, grid.ny, nlevels) q ψ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, nlevels) qh ψh uh vh Fqh

  return Vars(q, ψ, u, v, qh, ψh, uh, vh, Fqh, nothing)
end

"""
    StochasticForcedVars(grid, params)

Return the variables for a forced multi-level QG problem with `grid` and `params`.
"""
function StochasticForcedVars(grid, params)
  Dev = typeof(grid.device)
  T = eltype(grid)
  nlevels = numberoflevels(params)

  @devzeros Dev T (grid.nx, grid.ny, nlevels) q ψ u v
  @devzeros Dev Complex{T} (grid.nkr, grid.nl, nlevels) qh ψh uh vh Fqh prevsol

  return Vars(q, ψ, u, v, qh, ψh, uh, vh, Fqh, prevsol)
end

"""
    fwdtransform!(varh, var, params)

Compute the Fourier transform of `var` and store it in `varh`.
"""
fwdtransform!(varh, var, params::AbstractParams) = mul!(varh, params.rfftplan, var)

"""
    invtransform!(var, varh, params)

Compute the inverse Fourier transform of `varh` and store it in `var`.
"""
invtransform!(var, varh, params::AbstractParams) = ldiv!(var, params.rfftplan, varh)

"""
    calcD!(D, H₀, nlevels)

Construct the `nlevels` x `nlevels` Chebyshev differentiation matrix ``D``, which discretizes ``∂z``
"""
function calcD!(D, H₀, nlevels)
    # Chebyshev nodes
    ξ = [cos((i - 1) * pi / (nlevels - 1)) for i in 1 : nlevels] # Chebyshev grid on [-1, 1]
    z = H₀ / 2 .* (ξ .- 1)                                       # maps [-1, 1] -> [-H₀, 0]
    
    # Chebyshev differentiation matrix D
    c = ones(nlevels)
    c[1] = 2
    c[nlevels] = 2
    for i in 1 : nlevels, j in 1 : nlevels
        if i ≠ j
            @views D[i, j] = (c[i] / c[j]) * (-1)^(i + j) / (ξ[i] - ξ[j])
        end
    end
    # Diagonal entries to ensure that rows sum to zero (-> constant vectors in null space)
    for i in 1 : nlevels
        @views D[i, i] = -sum(D[i, j] for j in 1 : nlevels if j ≠ i)
    end
    
    # Scale to [-H₀, 0] grid
    D .= 2 / H₀ * D

    return nothing
end

"""
    calcF!(F, D, f₀, N²)

Construct the `nlevels` x `nlevels` array ``F`` that discretizes the vertical component of the PV inversion.
"""
function calcF!(F, D, f₀, N²)
    
    @views F[1, :] = f₀ * D[1, :]
    @views F[end, :] = f₀ * D[end, :]
    @views F[2 : end - 1, :] = (D .* (f₀^2 ./ N²)' * D)[2 : end - 1, :]
    
    return nothing
end

"""
    calcS!(S, F, nlevels, grid)

Construct the array ``𝕊``, which consists of `nlevels` x `nlevels` static arrays ``𝕊_𝐤`` that
relate the ``q̂_j``'s and ``ψ̂_j``'s for every wavenumber: ``q̂_𝐤 = 𝕊_𝐤 ψ̂_𝐤``.
"""
function calcS!(S, F, nlevels, grid)

  for n=1:grid.nl, m=1:grid.nkr
    k² = CUDA.@allowscalar grid.Krsq[m, n]
    Skl = SMatrix{nlevels, nlevels}(diagm([0; fill(-k², nlevels - 2); 0]) + F)  # subtracts off vorticity part only in the interior
    S[m, n] = Skl
  end

  return nothing
end

"""
    calcS⁻¹!(S, F, nlevels, grid)

Construct the array ``𝕊⁻¹``, which consists of `nlevels` x `nlevels` static arrays ``(𝕊_𝐤)⁻¹``
that relate the ``q̂_j``'s and ``ψ̂_j``'s for every wavenumber: ``ψ̂_𝐤 = (𝕊_𝐤)⁻¹ q̂_𝐤``.
"""
function calcS⁻¹!(S⁻¹, F, nlevels, grid)

  for n=1:grid.nl, m=1:grid.nkr
    k² = CUDA.@allowscalar grid.Krsq[m, n] == 0 ? 1 : grid.Krsq[m, n]
    Skl = diagm([0; fill(-k², nlevels - 2); 0]) + F                         # subtracts off vorticity part only in the interior
    S⁻¹[m, n] = SMatrix{nlevels, nlevels}(I / Skl)
  end

  T = eltype(grid)
  S⁻¹[1, 1] = SMatrix{nlevels, nlevels}(zeros(T, (nlevels, nlevels)))

  return nothing
end

"""
    inversion_kernel!(y, M, x, ::Val{N}) where N

Kernel for matrix-vector multiplication at given wavenumber. The kernel performs the matrix multiplication

```math
y = M x
```

for every wavenumber, where ``y`` and ``x`` are column-vectors of length `N`.
This can be used to perform the PV inversion, e.g., `qh = params.S * ψh` or `ψh = params.S⁻¹ qh`.
It is also used to find the vertical velocity via the omega equation, e.g., `wh = params.M⁻¹ rhsh`.

StaticVectors are used to efficiently perform the matrix-vector multiplication.
"""
@kernel function inversion_kernel!(y, M, x, ::Val{N}) where N
  i, j = @index(Global, NTuple)

  x_tuple = ntuple(Val(N)) do n
    @inbounds x[i, j, n]
  end

  T = eltype(x)
  x_sv = SVector{N, T}(x_tuple)
  y_sv = @inbounds M[i, j] * x_sv

  ntuple(Val(N)) do n
    @inbounds y[i, j, n] = y_sv[n]
  end
end

"""
    pvfromstreamfunction!(qh, ψh, params, grid)

Obtain the Fourier transform of the PV from the streamfunction `ψh` at each level using
`qh = params.S * ψh`.

The matrix multiplications are done via launching a kernel. We use a work layout over
which the kernel is launched.
"""
function pvfromstreamfunction!(qh, ψh, params, grid)
  # Larger workgroups are generally more efficient. For more generality, we could put an
  # if statement that incurs different behavior when either nkl or nl are less than 8.
  workgroup = 8, 8

  # The worksize determines how many times the kernel is run
  worksize = grid.nkr, grid.nl

  # Instantiates the kernel for relevant backend device
  backend = KernelAbstractions.get_backend(qh)
  kernel! = inversion_kernel!(backend, workgroup, worksize)

  # Launch the kernel
  S, nlevels = params.S, params.nlevels
  kernel!(qh, S, ψh, Val(nlevels))

  # Ensure that no other operations occur until the kernel has finished
  KernelAbstractions.synchronize(backend)

  return nothing
end

"""
    streamfunctionfrompv!(ψh, qh, params, grid)

Invert the PV to obtain the Fourier transform of the streamfunction `ψh` at each level from
`qh` using `ψh = params.S⁻¹ * qh`.

The matrix multiplications are done via launching a kernel. We use a work layout over
which the kernel is launched.
"""
function streamfunctionfrompv!(ψh, qh, params, grid)
  # Larger workgroups are generally more efficient. For more generality, we could put an
  # if statement that incurs different behavior when either nkl or nl are less than 8.
  workgroup = 8, 8

  # The worksize determines how many times the kernel is run
  worksize = grid.nkr, grid.nl

  # Instantiates the kernel for relevant backend device
  backend = KernelAbstractions.get_backend(ψh)
  kernel! = inversion_kernel!(backend, workgroup, worksize)

  # Launch the kernel
  S⁻¹, nlevels = params.S⁻¹, params.nlevels
  kernel!(ψh, S⁻¹, qh, Val(nlevels))

  # Ensure that no other operations occur until the kernel has finished
  KernelAbstractions.synchronize(backend)

  return nothing
end

"""
    bfromstreamfunction!(b, ψ, params, grid)

Obtain the buoyancy `b` from the streamfunction `ψ` at each level using
`b = params.f₀ * params.D * ψ`,
i.e., matrix-vector multiplication at each horizontal grid point.
"""
function bfromstreamfunction!(b, ψ, params, grid)
  f₀ = params.f₀
  D = params.D
  nlevels = params.nlevels
  nx, ny = grid.nx, grid.ny

  b .= f₀ * permutedims(reshape(D * reshape(permutedims(ψ, (3, 1, 2)), nlevels, nx * ny), nlevels, nx, ny), (2, 3, 1))

  return nothing
end

"""
    calcM⁻¹!(M⁻¹, D, f₀, N², nlevels, grid)

Construct the array ``M⁻¹``, which consists of `nlevels` x `nlevels` static arrays ``(M_𝐤)⁻¹``
that relate the ``ŵ_j``'s and ``f̂_j``'s for every wavenumber: ``ŵ_𝐤 = (M_𝐤)⁻¹ f̂_𝐤``,
where ``f̂'' represents the rhs interior forcing + boundary conditions in the omega equation. 
"""
function calcM⁻¹!(M⁻¹, D, f₀, N², nlevels, grid)
  T = eltype(grid)

  D²_int = (D * D)[2 : end - 1, 2 : end - 1]
  N²_int = N²[2 : end - 1]

  Mkl = zeros(T, (nlevels, nlevels))
  CUDA.@allowscalar Mkl[1, :] = [1; zeros(T, nlevels - 1)]
  CUDA.@allowscalar Mkl[end, :] = [zeros(T, nlevels - 1); 1]

  for n=1:grid.nl, m=1:grid.nkr
    k² = CUDA.@allowscalar grid.Krsq[m, n] == 0 ? 1 : grid.Krsq[m, n]
    Mkl[2 : end - 1, 2 : end - 1] = -k² * diagm(N²_int) + (f₀^2 * D²_int)
    M⁻¹[m, n] = SMatrix{nlevels, nlevels}(I / Mkl)
  end

  M⁻¹[1, 1] = SMatrix{nlevels, nlevels}(zeros(T, (nlevels, nlevels)))

  return nothing
end

"""
    omegaeqn!(wh, rhsh, params, grid)

Obtain the Fourier transform of the vertical velocity `wh` at each level from the omega equation given
  - the Fourier transform of the upper boundary condition at `z = 0`,
  - the Fourier transform of the forcing in the interior `–H < z < 0`, and
  - the Fourier transform of the bottom boundary condition at `z = -H`
by doing `wh = params.M⁻¹ * rhsh`, where rhsh has
  - the Fourier transform of the upper boundary condition at `z = 0` in the first vertical level
  - the Fourier transform of the interior forcing in the second-penultimate vertical levels
  - the Fourier transform of the bottom boundary condition at `z = -H` in the last vertical level

The matrix multiplications are done via launching a kernel. We use a work layout over
which the kernel is launched.
"""
function omegaeqn!(wh, rhsh, params, grid)
  # Larger workgroups are generally more efficient. For more generality, we could put an
  # if statement that incurs different behavior when either nkl or nl are less than 8.
  workgroup = 8, 8

  # The worksize determines how many times the kernel is run
  worksize = grid.nkr, grid.nl

  # Instantiates the kernel for relevant backend device
  backend = KernelAbstractions.get_backend(wh)
  kernel! = inversion_kernel!(backend, workgroup, worksize)

  # Launch the kernel
  M⁻¹, nlevels = params.M⁻¹, params.nlevels
  kernel!(view(wh, :, :, :), M⁻¹, rhsh, Val(nlevels))

  # Ensure that no other operations occur until the kernel has finished
  KernelAbstractions.synchronize(backend)

  return nothing
end

"""
    omegaeqn!(wh, prob)

Obtain the Fourier transform of the vertical velocity `wh` at each level from the full omega equation
(interior forcing and non-zero boundary conditions) by computing terms from variables stored in prob.

"""
function omegaeqn!(wh, prob) 
  sol, vars, params, grid = prob.sol, prob.vars, prob.params, prob.grid
  A = device_array(grid.device)
  nkr = grid.nkr
  nl = grid.nl
  nlevels = params.nlevels

  # Update and compute relevant variables
  @. vars.qh = sol

  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  invtransform!(vars.ψ, vars.ψh, params)

  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  invtransform!(vars.u, vars.uh, params)
  invtransform!(vars.v, vars.vh, params)

  b = vars.q      # use vars.q as scratch variable
  bfromstreamfunction!(b, vars.ψ, params, grid)

  ζh = vars.uh    # use vars.uh as scratch variable
  @. ζh = -grid.Krsq * vars.ψh
  ζ = vars.ψ      # use vars.ψ as scratch variable
  invtransform!(ζ, ζh, params)

  ### RHS
  rhsh = similar(vars.qh, nkr, nl, nlevels)

  ## Upper BC: w = 0 at z = 0
  @views rhsh[:, :, 1] .= A(zeros(eltype(vars.qh), nkr, nl))

  ## Lower BC: w = rζ + J(ψ, h) at z = -H
  @views rhsh[:, :, end]  .= params.r * ζh[:, :, end]
  @views rhsh[:, :, end] .+= im * grid.kr .* rfft(vars.u[:, :, end] .* params.eta) .+
                             im * grid.l  .* rfft(vars.v[:, :, end] .* params.eta)

  ## Interior RHS forcing for -H < z < 0
  # Scratch variables
  Fx = similar(vars.u)
  Fy = similar(vars.v)

  # Vorticity part
  @. Fx = vars.u * ζ
  uζh = vars.uh  # use vars.uh as scratch varaible
  fwdtransform!(uζh, Fx, params)

  @. Fy = vars.v * ζ
  vζh = vars.vh  # use vars.vh as scratch varaible
  fwdtransform!(vζh, Fy, params)

  @views rhsh[:, :, 2 : end - 1] .= params.f₀ * permutedims(reshape(params.D * reshape(permutedims(im * grid.kr .* uζh .+ im * grid.l .* vζh, (3, 1, 2)), nlevels, nkr * nl), nlevels, nkr, nl), (2, 3, 1))[:, :, 2 : end - 1]

  # Buoyancy part
  @. Fx = vars.u * b
  ubh = vars.uh  # use vars.uh as scratch varaible
  fwdtransform!(ubh, Fx, params)

  @. Fy = vars.v * b
  vbh = vars.vh  # use vars.vh as scratch varaible
  fwdtransform!(vbh, Fy, params)

  @views @. rhsh[:, :, 2 : end - 1] .+= grid.Krsq * (im * grid.kr * ubh[:, :, 2 : end - 1] + im * grid.l * vbh[:, :, 2 : end - 1])

  return omegaeqn!(wh, rhsh, params, grid)
end

# -------
# Solvers
# -------

"""
    calcN!(N, sol, t, clock, vars, params, grid)

Compute the nonlinear term, that is the advection term, the bottom drag, and the forcing:

```math
N_j = - \\widehat{𝖩(ψ_j, q_j)} - \\widehat{U_j ∂_x Q_j} - \\widehat{U_j ∂_x q_j}
 + \\widehat{(∂_y ψ_j)(∂_x Q_j)} - \\widehat{(∂_x ψ_j)(∂_y Q_j)} + δ_{j, n} N² r |𝐤|^2 ψ̂_n + F̂_j .
```
"""
function calcN!(N, sol, t, clock, vars, params, grid)
  nlevels = numberoflevels(params)

  dealias!(sol, grid)

  calcN_advection!(N, sol, vars, params, grid)

  @views @. N[:, :, end] += params.N²[end] * params.r * grid.Krsq * vars.ψh[:, :, end]   # bottom linear drag

  addforcing!(N, sol, t, clock, vars, params, grid)

  return nothing
end

"""
    calcNlinear!(N, sol, t, clock, vars, params, grid)

Compute the nonlinear term of the linearized equations:

```math
N_j = - \\widehat{U_j ∂_x Q_j} - \\widehat{U_j ∂_x q_j} + \\widehat{(∂_y ψ_j)(∂_x Q_j)}
- \\widehat{(∂_x ψ_j)(∂_y Q_j)} + δ_{j, n} N² r |𝐤|^2 ψ̂_n + F̂_j .
```
"""
function calcNlinear!(N, sol, t, clock, vars, params, grid)
  nlevels = numberoflevels(params)

  calcN_linearadvection!(N, sol, vars, params, grid)
  @views @. N[:, :, end] += params.N²[end] * params.r * grid.Krsq * vars.ψh[:, :, end]   # bottom linear drag
  addforcing!(N, sol, t, clock, vars, params, grid)

  return nothing
end

"""
    calcN_advection!(N, sol, vars, params, grid)

Compute the advection term and store it in `N`:

```math
N_j = - \\widehat{𝖩(ψ_j, q_j)} - \\widehat{U_j ∂_x Q_j} - \\widehat{U_j ∂_x q_j}
 + \\widehat{(∂_y ψ_j)(∂_x Q_j)} - \\widehat{(∂_x ψ_j)(∂_y Q_j)} .
```
"""
function calcN_advection!(N, sol, vars, params, grid)
  @. vars.qh = sol

  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  invtransform!(vars.u, vars.uh, params)
  @. vars.u += params.U                    # add the imposed zonal flow U (on upper boundary, in interior, and on lower boundary)

  uQx, uQxh = vars.q, vars.uh              # use vars.q and vars.uh as scratch variables
  @. uQx = vars.u * params.Qx              # (U+u)*∂Q/∂x
  fwdtransform!(uQxh, uQx, params)
  @. N = - uQxh                            # -\hat{(U+u)*∂Q/∂x}

  invtransform!(vars.v, vars.vh, params)

  vQy, vQyh = vars.q, vars.vh              # use vars.q and vars.vh as scratch variables
  @. vQy = vars.v * params.Qy              # v*∂Q/∂y
  fwdtransform!(vQyh, vQy, params)
  @. N -= vQyh                             # -\hat{v*∂Q/∂y}

  invtransform!(vars.q, vars.qh, params)

  uq, vq  = vars.u, vars.v                 # use vars.u and vars.v as scratch variables
  uqh, vqh = vars.uh, vars.vh              # use vars.uh and vars.vh as scratch variables
  @. uq *= vars.q                          # (U+u)*q
  @. vq *= vars.q                          # v*q

  fwdtransform!(uqh, uq, params)
  fwdtransform!(vqh, vq, params)

  @. N -= im * grid.kr * uqh + im * grid.l * vqh    # -\hat{∂[(U+u)q]/∂x} - \hat{∂[vq]/∂y}

  return nothing
end


"""
    calcN_linearadvection!(N, sol, vars, params, grid)

Compute the advection term of the linearized equations and store it in `N`:

```math
N_j = - \\widehat{U_j ∂_x Q_j} - \\widehat{U_j ∂_x q_j}
 + \\widehat{(∂_y ψ_j)(∂_x Q_j)} - \\widehat{(∂_x ψ_j)(∂_y Q_j)} .
```
"""
function calcN_linearadvection!(N, sol, vars, params, grid)
  @. vars.qh = sol

  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  invtransform!(vars.u, vars.uh, params)

  @. vars.u += params.U                    # add the imposed zonal flow U
  uQx, uQxh = vars.q, vars.uh              # use vars.q and vars.uh as scratch variables
  @. uQx  = vars.u * params.Qx             # (U+u)*∂Q/∂x
  fwdtransform!(uQxh, uQx, params)
  @. N = - uQxh                            # -\hat{(U+u)*∂Q/∂x}

  invtransform!(vars.v, vars.vh, params)

  vQy, vQyh = vars.q, vars.vh              # use vars.q and vars.vh as scratch variables

  @. vQy = vars.v * params.Qy              # v*∂Q/∂y
  fwdtransform!(vQyh, vQy, params)
  @. N -= vQyh                             # -\hat{v*∂Q/∂y}

  invtransform!(vars.q, vars.qh, params)

  @. vars.u  = params.U
  Uq, Uqh  = vars.u, vars.uh               # use vars.u and vars.uh as scratch variables
  @. Uq *= vars.q                          # U*q

  fwdtransform!(Uqh, Uq, params)

  @. N -= im * grid.kr * Uqh               # -\hat{∂[U*q]/∂x}

  return nothing
end


"""
    addforcing!(N, sol, t, clock, vars, params, grid)

When the problem includes forcing, calculate the forcing term ``F̂`` at each level and add
it to the nonlinear term ``N``.
"""
addforcing!(N, sol, t, clock, vars::Vars, params, grid) = nothing

function addforcing!(N, sol, t, clock, vars::ForcedVars, params, grid)
  params.calcFq!(vars.Fqh, sol, t, clock, vars, params, grid)
  @. N += vars.Fqh

  return nothing
end


# ----------------
# Helper functions
# ----------------

"""
    updatevars!(vars, params, grid, sol)
    updatevars!(prob)

Update all problem variables using `sol`.
"""
function updatevars!(vars, params, grid, sol)
  dealias!(sol, grid)

  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  @. vars.uh = -im * grid.l  * vars.ψh
  @. vars.vh =  im * grid.kr * vars.ψh

  invtransform!(vars.q, deepcopy(vars.qh), params)
  invtransform!(vars.ψ, deepcopy(vars.ψh), params)
  invtransform!(vars.u, deepcopy(vars.uh), params)
  invtransform!(vars.v, deepcopy(vars.vh), params)

  return nothing
end

updatevars!(prob) = updatevars!(prob.vars, prob.params, prob.grid, prob.sol)


"""
    set_q!(sol, params, vars, grid, q)
    set_q!(prob, q)

Set the solution `prob.sol` as the transform of `q` and update variables.
"""
function set_q!(sol, params, vars, grid, q)
  A = typeof(vars.q)
  fwdtransform!(vars.qh, A(q), params)
  @. vars.qh[1, 1, :] = 0
  @. sol = vars.qh
  updatevars!(vars, params, grid, sol)

  return nothing
end

set_q!(prob, q) = set_q!(prob.sol, prob.params, prob.vars, prob.grid, q)


"""
    set_ψ!(params, vars, grid, sol, ψ)
    set_ψ!(prob, ψ)

Set the solution `prob.sol` to the transform `qh` that corresponds to streamfunction `ψ`
and update variables.
"""
function set_ψ!(sol, params, vars, grid, ψ)
  A = typeof(vars.q)
  fwdtransform!(vars.ψh, A(ψ), params)
  pvfromstreamfunction!(vars.qh, vars.ψh, params, grid)
  invtransform!(vars.q, vars.qh, params)

  set_q!(sol, params, vars, grid, vars.q)

  return nothing
end

set_ψ!(prob, ψ) = set_ψ!(prob.sol, prob.params, prob.vars, prob.grid, ψ)


# """
#     energies(vars, params, grid, sol)
#     energies(prob)

# Return the kinetic energy of each fluid layer KE``_1, ...,`` KE``_{n}``, and the
# potential energy of each fluid interface PE``_{3/2}, ...,`` PE``_{n-1/2}``, where ``n``
# is the number of layers in the fluid. (When ``n=1``, only the kinetic energy is returned.)

# The kinetic energy at the ``j``-th fluid layer is

# ```math
# 𝖪𝖤_j = \\frac{H_j}{H} \\int \\frac1{2} |{\\bf ∇} ψ_j|^2 \\frac{𝖽x 𝖽y}{L_x L_y} = \\frac1{2} \\frac{H_j}{H} \\sum_{𝐤} |𝐤|² |ψ̂_j|², \\ j = 1, ..., n ,
# ```

# while the potential energy that corresponds to the interface ``j+1/2`` (i.e., the interface
# between the ``j``-th and ``(j+1)``-th fluid layer) is

# ```math
# 𝖯𝖤_{j+1/2} = \\int \\frac1{2} \\frac{f₀^2}{g'_{j+1/2} H} (ψ_j - ψ_{j+1})^2 \\frac{𝖽x 𝖽y}{L_x L_y} = \\frac1{2} \\frac{f₀^2}{g'_{j+1/2} H} \\sum_{𝐤} |ψ̂_j - ψ̂_{j+1}|², \\ j = 1, ..., n-1 .
# ```
# """
# function energies(vars, params, grid, sol)
#   nlevels = numberoflevels(params)
#   KE, PE = zeros(nlevels), zeros(nlevels-1)

#   @. vars.qh = sol
#   streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

#   abs²∇𝐮h = vars.uh        # use vars.uh as scratch variable
#   @. abs²∇𝐮h = grid.Krsq * abs2(vars.ψh)

#   V = grid.Lx * grid.Ly * sum(params.H)  # total volume of the fluid

#   for j = 1:nlevels
#     view(KE, j) .= 1 / (2 * V) * parsevalsum(view(abs²∇𝐮h, :, :, j), grid) * params.H[j]
#   end

#   for j = 1:nlevels-1
#     view(PE, j) .= 1 / (2 * V) * params.f₀^2 ./ params.g′[j] .* parsevalsum(abs2.(view(vars.ψh, :, :, j) .- view(vars.ψh, :, :, j+1)), grid)
#   end

#   return KE, PE
# end

# energies(prob) = energies(prob.vars, prob.params, prob.grid, prob.sol)

# """
#     fluxes(vars, params, grid, sol)
#     fluxes(prob)

# Return the lateral eddy fluxes within each fluid layer, lateralfluxes``_1,...,``lateralfluxes``_n``
# and also the vertical eddy fluxes at each fluid interface,
# verticalfluxes``_{3/2},...,``verticalfluxes``_{n-1/2}``, where ``n`` is the total number of layers in the fluid.
# (For a single fluid layer, i.e., when ``n=1``, only the lateral fluxes are returned.)

# The lateral eddy fluxes within the ``j``-th fluid layer are

# ```math
# \\textrm{lateralfluxes}_j = \\frac{H_j}{H} \\int U_j v_j ∂_y u_j
# \\frac{𝖽x 𝖽y}{L_x L_y} , \\  j = 1, ..., n ,
# ```

# while the vertical eddy fluxes at the ``j+1/2``-th fluid interface (i.e., interface between
# the ``j``-th and ``(j+1)``-th fluid layer) are

# ```math
# \\textrm{verticalfluxes}_{j+1/2} = \\int \\frac{f₀²}{g'_{j+1/2} H} (U_j - U_{j+1}) \\,
# v_{j+1} ψ_{j} \\frac{𝖽x 𝖽y}{L_x L_y} , \\ j = 1, ..., n-1.
# ```
# """
# function fluxes(vars, params, grid, sol)

#   nlevels = numberoflevels(params)

#   lateralfluxes, verticalfluxes = zeros(nlevels), zeros(nlevels-1)

#   updatevars!(vars, params, grid, sol)

#   ∂u∂yh = vars.uh           # use vars.uh as scratch variable
#   ∂u∂y  = vars.u            # use vars.u  as scratch variable

#   @. ∂u∂yh = im * grid.l * vars.uh
#   invtransform!(∂u∂y, ∂u∂yh, params)

#   V = grid.Lx * grid.Ly * sum(params.H)  # total volume of the fluid

#   lateralfluxes = params.H .* (sum(@. params.U * vars.v * ∂u∂y; dims=(1, 2)))[1, 1, :]
#   lateralfluxes *= grid.dx * grid.dy / V

#   for j = 1:nlevels-1
#     Uⱼ, Uⱼ₊₁ = view(params.U, :, :, j), view(params.U, :, :, j+1)
#     ψⱼ = view(vars.ψ, :, :, j)
#     vⱼ₊₁ = view(vars.v, :, :, j+1)
#     verticalfluxes[j] = sum(@. params.f₀^2 / params.g′[j] * (Uⱼ - Uⱼ₊₁) * vⱼ₊₁ * ψⱼ)
#   end
#   verticalfluxes *= grid.dx * grid.dy / V

#   return lateralfluxes, verticalfluxes
# end

# fluxes(prob) = fluxes(prob.vars, prob.params, prob.grid, prob.sol)

end # module
