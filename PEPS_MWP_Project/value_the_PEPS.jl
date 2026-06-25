module PEPSValues

using MAT
using SweepContractor

export LegDimensions, PEPSValueCalculator, tensor_network, scaled_value, value
export configuration_values

"Dimensions in the order left, up, right, down, horizontal physical, vertical physical."
Base.@kwdef struct LegDimensions
    left::Int = 2
    up::Int = 2
    right::Int = 2
    down::Int = 2
    horizontal_physical::Int = 2
    vertical_physical::Int = 2
end

Base.Tuple(dimensions::LegDimensions) = (
    dimensions.left,
    dimensions.up,
    dimensions.right,
    dimensions.down,
    dimensions.horizontal_physical,
    dimensions.vertical_physical,
)

"""
    PEPSValueCalculator(peps; grid_size, leg_dimensions, parameter_index,
                        sweep_chi, sweep_tau)

Store one iPEPS tensor and all settings needed to value physical configurations.

`peps` may have shape
`(parameter, left, up, right, down, physical_horizontal, physical_vertical)`
or may already have its parameter axis removed.
"""
mutable struct PEPSValueCalculator
    peps::Array{Float64,6}
    grid_size::Tuple{Int,Int}
    leg_dimensions::LegDimensions
    sweep_chi::Int
    sweep_tau::Int
    prepared_network::TensorNetwork
    site_indices::Matrix{Int}
    workspaces::Vector{TensorNetwork}

    function PEPSValueCalculator(
        peps::AbstractArray{<:Real};
        grid_size::Tuple{Int,Int}=(8, 8),
        leg_dimensions::Union{LegDimensions,Nothing}=nothing,
        parameter_index::Int=1,
        sweep_chi::Int=16,
        sweep_tau::Int=32,
    )
        all(grid_size .>= 3) || error(
            "Periodic grid dimensions must both be at least 3",
        )
        sweep_chi > 0 || error("sweep_chi must be positive")
        sweep_tau >= sweep_chi || error("sweep_tau must be at least sweep_chi")

        selected_peps = if ndims(peps) == 7
            1 <= parameter_index <= size(peps, 1) || error(
                "parameter_index must lie between 1 and $(size(peps, 1))",
            )
            Array(selectdim(peps, 1, parameter_index))
        elseif ndims(peps) == 6
            Array(peps)
        else
            error("PEPS must have 6 or 7 dimensions, got size $(size(peps))")
        end

        expected_dimensions = isnothing(leg_dimensions) ? size(selected_peps) :
                              Tuple(leg_dimensions)
        size(selected_peps) == expected_dimensions || error(
            "PEPS size $(size(selected_peps)) does not match configured leg " *
            "dimensions $expected_dimensions",
        )
        selected_dimensions = isnothing(leg_dimensions) ?
            LegDimensions(;
                left=size(selected_peps, 1),
                up=size(selected_peps, 2),
                right=size(selected_peps, 3),
                down=size(selected_peps, 4),
                horizontal_physical=size(selected_peps, 5),
                vertical_physical=size(selected_peps, 6),
            ) :
            leg_dimensions

        calculator = new(
            Float64.(selected_peps),
            grid_size,
            selected_dimensions,
            sweep_chi,
            sweep_tau,
            TensorNetwork(),
            zeros(Int, grid_size),
            TensorNetwork[],
        )
        prepare_network!(calculator)
        return calculator
    end
end

"Load the PEPS from a MAT file and construct a `PEPSValueCalculator`."
function PEPSValueCalculator(
    mat_path::AbstractString;
    tensor_name::AbstractString="List_T",
    kwargs...,
)
    isfile(mat_path) || error("PEPS MAT file not found: $mat_path")
    data = matread(mat_path)
    haskey(data, tensor_name) || error(
        "MAT file has no tensor named '$tensor_name'. Available names: $(keys(data))",
    )
    return PEPSValueCalculator(data[tensor_name]; kwargs...)
end

"Check and return the two zero-based physical indices at site `(x, y)`."
function physical_indices(
    calculator::PEPSValueCalculator,
    configuration::AbstractMatrix{<:Tuple},
    x::Int,
    y::Int,
)
    horizontal, vertical = configuration[x, y]
    dimensions = calculator.leg_dimensions
    horizontal in 0:(dimensions.horizontal_physical - 1) || error(
        "Horizontal physical index at ($x, $y) is out of range",
    )
    vertical in 0:(dimensions.vertical_physical - 1) || error(
        "Vertical physical index at ($x, $y) is out of range",
    )
    return horizontal, vertical
end

"Coordinates with small offsets that prevent overlapping periodic wraparound bonds."
function site_position(x::Int, y::Int)
    return (
        x + 0.01 * sin(17x + 31y),
        y + 0.01 * cos(29x - 13y),
    )
end

"""
    tensor_network(calculator, configuration)

Apply a physical configuration and return the periodic labelled tensor network.
Each `configuration[x, y]` entry is `(horizontal_qubit, vertical_qubit)` using 0/1.
"""
function tensor_network(
    calculator::PEPSValueCalculator,
    configuration::AbstractMatrix{<:Tuple},
)
    size(configuration) == calculator.grid_size || error(
        "Configuration size $(size(configuration)) does not match grid " *
        "$(calculator.grid_size)",
    )
    width, height = calculator.grid_size
    network = LabelledTensorNetwork{Tuple{Int,Int}}()

    for x in 1:width, y in 1:height
        horizontal, vertical = physical_indices(calculator, configuration, x, y)

        # This order matches the tensor axes: left, up, right, down.
        neighbours = [
            (mod1(x - 1, width), y),
            (x, mod1(y + 1, height)),
            (mod1(x + 1, width), y),
            (x, mod1(y - 1, height)),
        ]
        site_tensor = Array(@view calculator.peps[
            :, :, :, :, horizontal + 1, vertical + 1
        ])

        # Avoid geometrically overlapping wraparound bonds during planarisation.
        x_position, y_position = site_position(x, y)
        network[(x, y)] = Tensor(
            neighbours,
            site_tensor,
            x_position,
            y_position,
        )
    end

    SweepContractor.checkvalid(network)
    return network
end

"Prepare and cache the configuration-independent periodic network topology."
function prepare_network!(calculator::PEPSValueCalculator)
    empty_configuration = fill((0, 0), calculator.grid_size)
    labelled_network = tensor_network(calculator, empty_configuration)

    # Cache the preprocessing normally repeated by `sweep_contract(...; fast=false)`.
    prepared = SweepContractor.delabel(labelled_network)
    SweepContractor.planarise!(prepared)
    SweepContractor.connect!(SweepContractor.hull!(prepared))
    SweepContractor.sort!(prepared)

    width, height = calculator.grid_size
    for x in 1:width, y in 1:height
        x_position, y_position = site_position(x, y)
        site_index = findfirst(
            tensor -> tensor.x == x_position && tensor.y == y_position,
            prepared,
        )
        isnothing(site_index) && error("Could not locate prepared PEPS site ($x, $y)")
        calculator.site_indices[x, y] = site_index
    end

    calculator.prepared_network = prepared
    calculator.workspaces = [
        deepcopy(prepared) for _ in 1:Threads.nthreads()
    ]
    return calculator
end

"Reset a reusable workspace and insert tensors for one physical configuration."
function configured_network!(
    calculator::PEPSValueCalculator,
    configuration::AbstractMatrix{<:Tuple},
)
    size(configuration) == calculator.grid_size || error(
        "Configuration size $(size(configuration)) does not match grid " *
        "$(calculator.grid_size)",
    )

    # Each Julia thread owns one workspace, so batch contractions can run safely in parallel.
    network = calculator.workspaces[Threads.threadid()]
    for index in eachindex(network)
        network[index].arr = copy(calculator.prepared_network[index].arr)
    end
    width, height = calculator.grid_size

    for x in 1:width, y in 1:height
        horizontal, vertical = physical_indices(calculator, configuration, x, y)
        site_tensor = Array(@view calculator.peps[
            :, :, :, :, horizontal + 1, vertical + 1
        ])

        prepared_tensor = network[calculator.site_indices[x, y]]
        auxiliary_leg_count = ndims(prepared_tensor.arr) - ndims(site_tensor)
        auxiliary_leg_count >= 0 || error("Invalid cached tensor at site ($x, $y)")
        auxiliary_dimensions = ntuple(_ -> 1, auxiliary_leg_count)
        prepared_tensor.arr = reshape(
            site_tensor,
            (size(site_tensor)..., auxiliary_dimensions...),
        )
    end

    return network
end

"Return the contraction as `(mantissa, exponent)`, representing `mantissa * 2^exponent`."
function scaled_value(
    calculator::PEPSValueCalculator,
    configuration::AbstractMatrix{<:Tuple},
)
    network = configured_network!(calculator, configuration)
    return sweep_contract!(
        network,
        calculator.sweep_chi,
        calculator.sweep_tau;
        fast=true,
    )
end

"Return the floating-point contraction value for a physical configuration."
function value(
    calculator::PEPSValueCalculator,
    configuration::AbstractMatrix{<:Tuple},
)
    return ldexp(scaled_value(calculator, configuration)...)
end

"""
    configuration_values(calculator, configurations; threaded=false)

Contract several configurations without reopening the MAT file or repeating periodic
network preprocessing. For parallel evaluation, start Julia with `--threads=auto` and set
`threaded=true`.
"""
function configuration_values(
    calculator::PEPSValueCalculator,
    configurations::AbstractVector;
    threaded::Bool=false,
)
    results = Vector{Float64}(undef, length(configurations))

    if threaded && Threads.nthreads() > 1
        Threads.@threads for index in eachindex(configurations)
            results[index] = value(calculator, configurations[index])
        end
    else
        for index in eachindex(configurations)
            results[index] = value(calculator, configurations[index])
        end
    end

    return results
end

end # module PEPSValues


# Example usage:
#
# include(joinpath(@__DIR__, "PEPS_MWPM", "value_the_PEPS.jl"))
# using .PEPSValues
#
# calculator = PEPSValueCalculator(
#     "path/to/state.mat";
#     parameter_index=1,
#     grid_size=(8, 8),
#     leg_dimensions=LegDimensions(
#         left=2, up=2, right=2, down=2,
#         horizontal_physical=2, vertical_physical=2,
#     ),
#     sweep_chi=16,
#     sweep_tau=32,
# )
# configuration = fill((0, 0), (8, 8))
# contraction = value(calculator, configuration)
