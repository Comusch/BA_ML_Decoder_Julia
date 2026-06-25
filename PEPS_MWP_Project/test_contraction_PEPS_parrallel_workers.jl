import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "value_the_PEPS.jl"))
include(joinpath(@__DIR__, "S_measurment_MWPM.jl"))

using .PEPSValues
using Distributed
using Random
using Plots
using SweepContractor

const PHASE_TRANSITION_PREPARED_WORKERS = Set{Int}()

function load_path_samples(filename::String, width::Int, height::Int)
    samples = Vector{Tuple{Matrix{Tuple{Int,Int}},Tuple{Float64,Int}}}()

    open(filename, "r") do IO
        for line in eachline(IO)
            amplitude_str, config_str = split(line, ": "; limit=2)
            amplitude_match = match(r"^\(([^,]+),\s*([^\)]+)\)$", amplitude_str)
            isnothing(amplitude_match) && error("invalid amplitude in sample line: $line")
            amplitude = (
                parse(Float64, amplitude_match.captures[1]),
                parse(Int, amplitude_match.captures[2]),
            )

            configuration_values = Tuple{Int,Int}[]
            for bit_match in eachmatch(r"\((-?\d+),\s*(-?\d+)\)", config_str)
                push!(configuration_values, (
                    parse(Int, bit_match.captures[1]),
                    parse(Int, bit_match.captures[2]),
                ))
            end
            length(configuration_values) == width * height || error(
                "sample contains $(length(configuration_values)) sites; expected $(width * height)",
            )
            # Configurations are stored as matrix[row=y, column=x].
            configuration = reshape(configuration_values, height, width)
            push!(samples, (configuration, amplitude))
        end
    end
    return samples
end

"Calculate for a given configuration the corresponding topological sector
Returns new configuration: (s = 1 (no flip), s = 2 (flip first row), s = 3 (flip first column), s = 4 (flip first row and column))"
function apply_logical_sector(configuration::Matrix{Tuple{Int, Int}}, sector::Int)
    modified_configuration = copy(configuration)
    width, height = size(configuration, 2), size(configuration, 1)

    if sector == 1
        return modified_configuration
    elseif sector == 3
        for y in 1:height
            modified_configuration[y, 1] = (
                modified_configuration[y, 1][1],
                1- modified_configuration[y, 1][2],
            )
        end
    elseif sector == 2
        for x in 1:width
            modified_configuration[1, x] = (
                1- modified_configuration[1, x][1],
                modified_configuration[1, x][2],
            )
        end
    elseif sector == 4
        for y in 1:height
            modified_configuration[y, 1] = (
                modified_configuration[y, 1][1],
                1- modified_configuration[y, 1][2],
            )
        end
        for x in 1:width
            modified_configuration[1, x] = (
                1 - modified_configuration[1, x][1],
                modified_configuration[1, x][2],
            )
        end
    else
        error("Invalid sector: $sector. Must be between 0 and 3.")
    end

    return modified_configuration
end

"Coordinates with small offsets that prevent overlapping periodic wraparound bonds."
function periodic_peps_site_position(x::Int, y::Int)
    return (
        x + 0.01 * sin(17x + 31y),
        y + 0.01 * cos(29x - 13y),
    )
end

"Rank-3 equality tensor enforcing that three binary virtual variables are equal."
function copy3_tensor()
    copy_tensor = zeros(Float64, 2, 2, 2)
    for t1 in 1:2, t2 in 1:2, t3 in 1:2
        copy_tensor[t1, t2, t3] = (t1 == t2 && t2 == t3) ? 1.0 : 0.0
    end
    return copy_tensor
end

"""
    product_state_peps(p)

Create a bond-dimension-1 PEPS tensor for a product distribution with two
independent physical qubits per site. Each qubit is flipped, i.e. has value `1`,
with probability `p`; it has value `0` with probability `1 - p`.

The returned tensor has axis order
`(left, up, right, down, horizontal_physical, vertical_physical)`.
"""
function product_state_peps(p::Real)
    0 <= p <= 1 || throw(ArgumentError("p must lie in [0, 1]"))

    ipeps = zeros(Float64, 1, 1, 1, 1, 2, 2)
    for h_index in 1:2, v_index in 1:2
        horizontal_bit = h_index - 1
        vertical_bit = v_index - 1
        horizontal_weight = horizontal_bit == 1 ? p : 1 - p
        vertical_weight = vertical_bit == 1 ? p : 1 - p
        ipeps[1, 1, 1, 1, h_index, v_index] =
            horizontal_weight * vertical_weight
    end
    return ipeps
end

"Exact product probability for a fixed physical configuration."
function product_state_probability(
    configuration::AbstractMatrix{<:Tuple},
    p::Real,
)
    probability = 1.0
    for (horizontal_bit, vertical_bit) in configuration
        horizontal_bit in 0:1 || error("Horizontal qubit is not 0 or 1")
        vertical_bit in 0:1 || error("Vertical qubit is not 0 or 1")
        probability *= horizontal_bit == 1 ? p : 1 - p
        probability *= vertical_bit == 1 ? p : 1 - p
    end
    return probability
end

"""
    build_periodic_peps_with_extra_virtuals(ipeps, physical_configuration)

Create a periodic PEPS where each site has three additional binary virtual legs
`t1`, `t2`, and `t3`. The physical PEPS indices are replaced by

    h_new = (t1 + t2 + h_c) % 2
    v_new = (t1 + t3 + v_c) % 2

for the configured physical bits `(h_c, v_c)`.

The configuration is indexed as `physical_configuration[y, x]`, matching the
sample-loading and MWPM helpers in this file.
"""
function build_periodic_peps_with_extra_virtuals(
    ipeps::AbstractArray{<:Real,6},
    physical_configuration::AbstractMatrix{<:Tuple},
)
    ndims(ipeps) == 6 || error(
        "Expected ipeps with 6 dimensions " *
        "(left, up, right, down, h_physical, v_physical), got $(ndims(ipeps))",
    )
    left_dim, up_dim, right_dim, down_dim, h_physical_dim, v_physical_dim = size(ipeps)
    h_physical_dim == 2 && v_physical_dim == 2 || error(
        "Expected ipeps size (left, up, right, down, h_physical, v_physical) " *
        "with binary physical dimensions, got $(size(ipeps))",
    )

    height, width = size(physical_configuration)
    width >= 3 && height >= 3 || error(
        "A periodic grid must have width and height of at least 3",
    )

    site(x, y) = (x, y, 0)
    copylabel(x, y) = (x, y, 1)

    network = LabelledTensorNetwork{Tuple{Int,Int,Int}}()
    equality_tensor = copy3_tensor()

    for x in 1:width, y in 1:height
        h_c, v_c = physical_configuration[y, x]
        h_c in 0:1 || error("Horizontal qubit at ($x, $y) is not 0 or 1")
        v_c in 0:1 || error("Vertical qubit at ($x, $y) is not 0 or 1")

        left = mod1(x - 1, width)
        right = mod1(x + 1, width)
        up = mod1(y + 1, height)
        down = mod1(y - 1, height)

        # Axis order: left, up, right, down, t1, t2, t3.
        site_tensor = zeros(
            Float64,
            left_dim,
            up_dim,
            right_dim,
            down_dim,
            2,
            2,
            2,
        )
        for t1_index in 1:2, t2_index in 1:2, t3_index in 1:2
            t1 = t1_index - 1
            t2 = t2_index - 1
            t3 = t3_index - 1

            h_new = mod(t1 + t2 + h_c, 2)
            v_new = mod(t1 + t3 + v_c, 2)

            site_tensor[:, :, :, :, t1_index, t2_index, t3_index] =
                ipeps[:, :, :, :, h_new + 1, v_new + 1]
        end

        x_position, y_position = periodic_peps_site_position(x, y)

        network[site(x, y)] = Tensor(
            [
                site(left, y),          # PEPS left virtual leg
                site(x, up),            # PEPS up virtual leg
                site(right, y),         # PEPS right virtual leg
                site(x, down),          # PEPS down virtual leg
                copylabel(x, y),        # t1
                copylabel(right, y),    # t2 equals t1 of the right neighbour
                copylabel(x, down),     # t3 equals t1 of the bottom neighbour
            ],
            site_tensor,
            x_position,
            y_position,
        )

        network[copylabel(x, y)] = Tensor(
            [
                site(x, y),             # t1 of this site
                site(left, y),          # t2 of the left neighbour
                site(x, up),            # t3 of the top neighbour
            ],
            equality_tensor,
            x_position - 0.18,
            y_position + 0.18,
        )
    end
    
    SweepContractor.checkvalid(network)
    return network
end

function Calculate_partition_function(
    configuration_values::Matrix{Tuple{Int,Int}},
    calculator::PEPSValues.PEPSValueCalculator,
)
    network = build_periodic_peps_with_extra_virtuals(calculator.peps, configuration_values)
    partition_function = sweep_contract(
        network,
        calculator.sweep_chi,
        calculator.sweep_tau;
        fast=false,
    )
    return partition_function
end

function load_peps_calculator()
    state_file = joinpath(
        @__DIR__,
        "..",
        "data_for_QEC",
        "h_z_0_2",
        "TC_non_sym_hz_0.2_hx=0.280_to_0.350_bond_dim_2_chi_40+state.mat",
    )

    grid_size = (8, 8)
    calculator = PEPSValues.PEPSValueCalculator(
        state_file;
        parameter_index=1,
        grid_size=grid_size,
        sweep_chi=8,
        sweep_tau=16,
    )
    return calculator
end

function find_max(partition_functions::Vector{Tuple{Float64,Int}})
    max_mantissa = -Inf
    max_exponent = -Inf
    max_index = -1
    for (index, (mantissa, exponent)) in enumerate(partition_functions)
        if exponent > max_exponent || (exponent == max_exponent && mantissa > max_mantissa)
            max_mantissa = mantissa
            max_exponent = exponent
            max_index = index
        end
    end
    if max_index == -1
        error("No valid partition function found.")
    end
    return max_index
end

function calculate_ML(reference_configuration::Matrix{Tuple{Int,Int}}, calculator::PEPSValues.PEPSValueCalculator; output=true::Bool)

    if output
        println(" -> Calculating partition functions for the four topological sectors...")
    end
    modifyed_configurations = [
        apply_logical_sector(reference_configuration, sector)
        for sector in 1:4
    ]
    partition_functions = [
        Calculate_partition_function(modified_configuration, calculator)
        for modified_configuration in modifyed_configurations
    ]
    max_index = find_max(partition_functions)
    if output
        println("Partition functions for the four sectors: ", partition_functions)
        println("Most likely topological sector (1, 2, 3, 4) possible: ", max_index)
    end

    return partition_functions[max_index], max_index, modifyed_configurations[max_index]
end


function calculate_new_sample(p, calculator, grid_size)
    configuration = [
        (rand() < p ? 1 : 0, rand() < p ? 1 : 0)
        for _ in 1:grid_size[1], _ in 1:grid_size[2]
    ]

    syndroms_test = calculate_syndromes(configuration)
    reference_configuration = decoded_bit_MWPM(syndroms_test, grid_size[1], grid_size[2])

    result_partition_function, max_index, modified_configuration = calculate_ML(reference_configuration, calculator, output=false)
    W_x_o, W_y_o = measurement_wilson_loops(configuration, reference_configuration)
    W_x, W_y = measurement_wilson_loops(configuration, modified_configuration)
    ml_changed = max_index == 1 ? 0 : 1
    return W_x, W_y, W_x_o, W_y_o, ml_changed

end

function phase_transition_worker_ids()
    return filter(process_id -> process_id != myid(), workers())
end

function prepare_phase_transition_workers!(worker_ids)
    isempty(worker_ids) && return

    project_dir = abspath(joinpath(@__DIR__, ".."))
    peps_value_file = joinpath(@__DIR__, "value_the_PEPS.jl")
    mwpm_file = joinpath(@__DIR__, "S_measurment_MWPM.jl")
    source = read(@__FILE__, String)
    helper_start = findfirst("function apply_logical_sector", source)
    helper_end = findfirst("function phase_transition_worker_ids()", source)
    isnothing(helper_start) && error("Could not find ML helper source start")
    isnothing(helper_end) && error("Could not find ML helper source end")
    helper_source = source[first(helper_start):(first(helper_end) - 1)]

    for worker_id in worker_ids
        worker_id in PHASE_TRANSITION_PREPARED_WORKERS && continue

        remotecall_wait(
            worker_id,
            project_dir,
            peps_value_file,
            mwpm_file,
            helper_source,
        ) do project_dir, peps_value_file, mwpm_file, helper_source
            Core.eval(Main, :(import Pkg))
            Core.eval(Main, :(Pkg.activate($project_dir)))
            include(peps_value_file)
            include(mwpm_file)
            Core.eval(Main, :(using .PEPSValues))
            Core.eval(Main, :(using SweepContractor))
            include_string(Main, helper_source, "phase_transition_worker_helpers.jl")

            Core.eval(Main, quote
                function calculate_phase_transition_sample_worker(p, calculator, grid_size)
                    configuration = [
                        (rand() < p ? 1 : 0, rand() < p ? 1 : 0)
                        for _ in 1:grid_size[1], _ in 1:grid_size[2]
                    ]

                    syndromes = calculate_syndromes(configuration)
                    reference_configuration = decoded_bit_MWPM(
                        syndromes,
                        grid_size[1],
                        grid_size[2],
                    )
                    result_partition_function, max_index, modified_configuration =
                        calculate_ML(reference_configuration, calculator, output=false)
                    W_x_o, W_y_o = measurement_wilson_loops(
                        configuration,
                        reference_configuration,
                    )
                    W_x, W_y = measurement_wilson_loops(
                        configuration,
                        modified_configuration,
                    )
                    ml_changed = max_index == 1 ? 0 : 1
                    return W_x, W_y, W_x_o, W_y_o, ml_changed
                end
            end)
        end
        push!(PHASE_TRANSITION_PREPARED_WORKERS, worker_id)
    end
end

function calculate_phase_transition_samples_serial(
    p,
    calculator,
    grid_size,
    number_samples::Int;
    progress_every::Int=1000,
)
    W_x_total = 0
    W_y_total = 0
    W_x_o_total = 0
    W_y_o_total = 0
    ml_changed_total = 0
    start_time = time_ns()

    elapsed_time = @elapsed begin
        for i in 1:number_samples
            W_x, W_y, W_x_o, W_y_o, ml_changed = calculate_new_sample(p, calculator, grid_size)
            W_x_total += W_x
            W_y_total += W_y
            W_x_o_total += W_x_o
            W_y_o_total += W_y_o
            ml_changed_total += ml_changed

            if progress_every > 0 && i % progress_every == 0
                println(
                    "grid size: ",
                    grid_size[1],
                    ", samples: ",
                    i,
                    ", wall time: ",
                    round((time_ns() - start_time) / 1e9; digits=3),
                    " seconds",
                )
            end
        end
    end

    return W_x_total, W_y_total, W_x_o_total, W_y_o_total, ml_changed_total, elapsed_time
end

function calculate_phase_transition_samples(
    p,
    calculator,
    grid_size,
    number_samples::Int;
    progress_every::Int=1000,
    parallel::Symbol=:distributed,
)
    number_samples > 0 || throw(ArgumentError("number_samples must be positive"))
    parallel in (:distributed, :serial) || throw(ArgumentError(
        "parallel must be :distributed or :serial",
    ))

    if parallel == :distributed
        worker_ids = phase_transition_worker_ids()
        if !isempty(worker_ids)
            prepare_phase_transition_workers!(worker_ids)
            println(
                "Running ",
                number_samples,
                " samples on ",
                length(worker_ids),
                " worker processes",
            )
            totals = [0, 0, 0, 0, 0]
            elapsed_time = @elapsed begin
                totals = @distributed (+) for _ in 1:number_samples
                    W_x, W_y, W_x_o, W_y_o, ml_changed = calculate_phase_transition_sample_worker(
                        p,
                        calculator,
                        grid_size,
                    )
                    [W_x, W_y, W_x_o, W_y_o, ml_changed]
                end
            end
            return (
                totals[1] / number_samples,
                totals[2] / number_samples,
                totals[3] / number_samples,
                totals[4] / number_samples,
                totals[5],
                elapsed_time,
            )
        end

        println("No distributed workers found; running samples serially.")
    end

    W_x_total, W_y_total, W_x_o_total, W_y_o_total, ml_changed_total, elapsed_time = calculate_phase_transition_samples_serial(
        p,
        calculator,
        grid_size,
        number_samples;
        progress_every=progress_every,
    )
    return (
        W_x_total / number_samples,
        W_y_total / number_samples,
        W_x_o_total / number_samples,
        W_y_o_total / number_samples,
        ml_changed_total,
        elapsed_time,
    )
end

function plot_average_w_total(
    theta_over_pi,
    average_w_totals_by_size,
    grid_sizes;
    filename="average_W_total_vs_theta_by_size.png",
)
    figure = plot(
        xlabel="theta / pi",
        ylabel="average W_total",
        title="Average W_total against theta",
        legend=:bottomleft,
    )
    for size in grid_sizes
        plot!(
            figure,
            theta_over_pi,
            average_w_totals_by_size[size];
            marker=:circle,
            linewidth=2,
            label="L = $size",
        )
    end
    savefig(figure, filename)
    println("Saved plot to ", filename)
end

function sanity_check_PS_whole_phase_transition()

    thetas = [0.15, 0.175, 0.18, 0.19, 0.195, 0.2, 0.205, 0.21, 0.215, 0.22, 0.225]
    grid_sizes = [8]

    average_w_totals_by_size = Dict(
        grid_size => Float64[]
        for grid_size in grid_sizes
    )
    number_samples_per_theta = 10000

    for linear_size in grid_sizes
        grid_size = (linear_size, linear_size)
        println("Running grid size ", grid_size)

        for theta_without_pi in thetas
            p = sin(theta_without_pi*pi/2)^2

            ipeps = product_state_peps(p)
            calculator = PEPSValues.PEPSValueCalculator(
                ipeps;
                grid_size=grid_size,
                sweep_chi=4,
                sweep_tau=8,
            )
            println("theta_without_pi = ", theta_without_pi, " => p = ", p)
            W_x_total, W_y_total, W_x_o_total, W_y_o_total, ml_changed_total, time = calculate_phase_transition_samples(
                p,
                calculator,
                grid_size,
                number_samples_per_theta;
                progress_every=1000,
                parallel=:distributed,
            )
            W_total = (W_x_total + W_y_total) / 2
            W_o_total = (W_x_o_total + W_y_o_total) / 2
            println("Sampling time: ", round(time; digits=3), " seconds")
            println("Average W_total (ML): ", W_total)
            println("Average W_total (MWPM): ", W_o_total)
            println(
                "ML changed sector: ",
                ml_changed_total,
                " / ",
                number_samples_per_theta,
                " samples (",
                round(100 * ml_changed_total / number_samples_per_theta; digits=2),
                "%)",
            )
            push!(average_w_totals_by_size[linear_size], W_total)
            println("-------")
        end
    end

    plot_average_w_total(thetas, average_w_totals_by_size, grid_sizes, filename="MWPM+ML_line_8_16.png")
    return thetas, average_w_totals_by_size
end

function sanity_check_simple_Product_state()
    ## here i want to define manually a PEPS for the product state to check how good the contraction is working.
    ## the product state is a PEPS with bond dimension 1
    p = sin(0.1*pi/2)^2
    println("product state with p = ", p)
    grid_size = (8, 8)
    configuration = [
        (rand() < p ? 1 : 0, rand() < p ? 1 : 0)
        for _ in 1:grid_size[1], _ in 1:grid_size[2]
    ]

    ipeps = product_state_peps(p)
    calculator = PEPSValues.PEPSValueCalculator(
        ipeps;
        grid_size=grid_size,
        sweep_chi=4,
        sweep_tau=8,
    )

    exact_probability = product_state_probability(configuration, p)
    contracted_probability = PEPSValues.value(calculator, configuration)

    println("Product-state PEPS size: ", size(ipeps))
    println("Exact product probability: ", exact_probability)
    println("Contracted product probability: ", contracted_probability)
    println("Absolute error: ", abs(contracted_probability - exact_probability))

    println("-------")
    println("Calculating the recovery error and the topological sector for the product state...")
    syndroms_test = calculate_syndromes(configuration)
    reference_configuration = decoded_bit_MWPM(syndroms_test, 8, 8)

    result_partition_function, max_index, modified_configuration = calculate_ML(reference_configuration, calculator, output=true)
    println("resulting Partition function: ", result_partition_function)
    println("-------")

end

function test_loading_and_MWPM()
    path = joinpath(@__DIR__, "generated_configurations/sampled_p=1_n_20.txt")
    samples = load_path_samples(path, 8, 8)
    println("Loaded $(length(samples)) samples from $path") 

    PEPS_calc = load_peps_calculator()

    for (sample_index, (configuration, amplitude)) in enumerate(samples)
        println("Sample $sample_index: amplitude = $amplitude")
        syndroms_test = calculate_syndromes(configuration)
        #println("Syndromes for the first sample: ", syndroms_test)
        reference_configuration = decoded_bit_MWPM(syndroms_test, 8, 8)
        #println("Reference configuration: ", reference_configuration)

        result_partition_function, max_index, modified_configuration = calculate_ML(reference_configuration, PEPS_calc)
        println("resulting Partition function: ", result_partition_function)
        println("-------")
    end
end


#sanity_check_simple_Product_state()
sanity_check_PS_whole_phase_transition()
"""
timer = @timed test_loading_and_MWPM()
println("Total time taken: ", timer.time, " seconds")
"""
