import Pkg
Pkg.activate(@__DIR__)

using MAT
using SweepContractor

const STATE_FILE = joinpath(
    @__DIR__,
    "data_for_QEC",
    "h_z_0_2",
    "TC_non_sym_hz_0.2_hx=0.280_to_0.350_bond_dim_2_chi_40+state.mat",
)

const GRID_SIZE = (8, 8)
const PARAMETER_INDEX = 1
const SWEEP_CHI = 8
const SWEEP_TAU = 32

"""
Load one iPEPS tensor from `List_T`.

The returned tensor has index order
`(left, up, right, down, horizontal_physical, vertical_physical)`.
"""
function load_ipeps(path::AbstractString, parameter_index::Integer)
    isfile(path) || error("iPEPS state file not found: $path")
    data = matread(path)

    haskey(data, "List_T") || error("The MAT file has no variable named List_T")
    haskey(data, "List_h_x") || error("The MAT file has no variable named List_h_x")

    list_t = data["List_T"]
    h_x_values = vec(data["List_h_x"])

    ndims(list_t) == 7 || error(
        "Expected List_T to have 7 indices, but its size is $(size(list_t))",
    )
    1 <= parameter_index <= size(list_t, 1) || error(
        "parameter_index must lie between 1 and $(size(list_t, 1))",
    )
    length(h_x_values) == size(list_t, 1) || error(
        "List_h_x and the parameter index of List_T have different lengths",
    )

    ipeps = Array(selectdim(list_t, 1, parameter_index))
    size(ipeps) == (2, 2, 2, 2, 2, 2) || error(
        "Expected (left, up, right, down, physical, physical) dimensions " *
        "(2, 2, 2, 2, 2, 2), got $(size(ipeps))",
    )

    return ipeps, h_x_values[parameter_index]
end

"""
Create a periodic `width × height` PEPS suitable for `SweepContractor`.

Each entry of `physical_configuration[x, y]` is `(horizontal_bit, vertical_bit)`.
The bits use qubit notation 0/1 and are converted to Julia indices 1/2.
"""
function build_periodic_peps(
    ipeps::AbstractArray{<:Real,6},
    physical_configuration::AbstractMatrix{<:Tuple},
)
    width, height = size(physical_configuration)
    width >= 3 && height >= 3 || error(
        "A periodic grid must have width and height of at least 3",
    )
    network = LabelledTensorNetwork{Tuple{Int,Int}}()

    for x in 1:width, y in 1:height
        horizontal_bit, vertical_bit = physical_configuration[x, y]
        horizontal_bit in 0:1 || error("Horizontal qubit at ($x, $y) is not 0 or 1")
        vertical_bit in 0:1 || error("Vertical qubit at ($x, $y) is not 0 or 1")

        # Virtual-leg order: left, up, right, down.
        neighbours = [
            (mod1(x - 1, width), y), #left leg
            (x, mod1(y + 1, height)), #up leg
            (mod1(x + 1, width), y),  #right leg
            (x, mod1(y - 1, height)), #down leg
        ]

        site_tensor = Array(@view ipeps[
            :, :, :, :, horizontal_bit + 1, vertical_bit + 1
        ])

        # Wrapped bonds overlap straight grid bonds at exact integer positions.
        # Small deterministic offsets let SweepContractor planarise the torus.
        x_position = x + 0.01 * sin(17x + 31y)
        y_position = y + 0.01 * cos(29x - 13y)
        network[(x, y)] = Tensor(neighbours, site_tensor, x_position, y_position)
    end

    return network
end


function main()
    ipeps, h_x = load_ipeps(STATE_FILE, PARAMETER_INDEX)

    # Two physical qubits per lattice site: (horizontal, vertical).
    physical_configuration = fill((0, 0), GRID_SIZE)

    peps = build_periodic_peps(ipeps, physical_configuration)

    # This internal validation gives a clear error before a costly contraction.
    SweepContractor.checkvalid(peps)

    println("Loaded parameter $PARAMETER_INDEX with h_x = $h_x")
    println("iPEPS tensor size: ", size(ipeps))
    println(
        "Created periodic $(GRID_SIZE[1])×$(GRID_SIZE[2]) PEPS with " *
        "$(length(peps)) tensors",
    )

    # Periodic wraparound bonds require SweepContractor's planarisation step.
    contraction_result = @timed sweep_contract(
        peps,
        SWEEP_CHI,
        SWEEP_TAU;
        fast=false,
    )
    scaled_amplitude = contraction_result.value
    amplitude = ldexp(scaled_amplitude...)
    println("Contraction time: $(round(contraction_result.time; digits=3)) seconds")
    println("All-zero physical-configuration amplitude: $amplitude")
    println("SweepContractor scaled result: $scaled_amplitude")

    return peps, scaled_amplitude
end


# Run both as a terminal script and through an editor's `include_string`/Run Code action.
peps, scaled_amplitude = main()
