#in this file, we want to implement the maximum likelihood decoder
#The idea is to load the samples, then predict a reference MWPM configuration
#         Creating four different configurations for each topological sector
#         Then sample over the probabilesten configuration and sume up the probabilities for each topological sector

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "value_the_PEPS.jl"))
include(joinpath(@__DIR__, "S_measurment_MWPM.jl"))

using .PEPSValues
using Random


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

function apply_star_operator!(x, y, configuration::Matrix{Tuple{Int, Int}})
    width, height = size(configuration, 2), size(configuration, 1)
    configuration[y, x] = (
        1 - configuration[y, x][1],
        1 - configuration[y, x][2],
    )
    configuration[y, mod1(x - 1, width)] = (
        1 - configuration[y, mod1(x - 1, width)][1],
        configuration[y, mod1(x - 1, width)][2],
    )
    configuration[mod1(y - 1, height), x] = (
        configuration[mod1(y - 1, height), x][1],
        1 - configuration[mod1(y - 1, height), x][2],
    )
end

function add_closed_loops(configuration::Matrix{Tuple{Int, Int}}, closed_loops::Matrix{Int})
    width, height = size(configuration, 2), size(configuration, 1)
    modified_configuration = copy(configuration)

    for x in 1:width, y in 1:height
        #switch the star operator at (x, y)
        if closed_loops[y, x] == 1
            apply_star_operator!(x, y, modified_configuration)
        end
    end
    return modified_configuration
end

function probability_from_scaled_amplitude(
    scaled_amplitude::Tuple{<:Real,<:Integer},
    reference_exponent::Integer,
)
    mantissa, exponent = scaled_amplitude
    return abs(mantissa)^2 * 2.0^(2 * (exponent - reference_exponent))
end

"Logarithm of the unnormalised Born weight `|amplitude|^2`."
function log_born_weight(scaled_amplitude::Tuple{<:Real,<:Integer})
    mantissa, exponent = scaled_amplitude
    iszero(mantissa) && return -Inf
    return 2 * (log(abs(mantissa)) + exponent * log(2.0))
end

function sampling_throw_logical_classes(
    configuration::Matrix{Tuple{Int, Int}},
    calculator::PEPSValues.PEPSValueCalculator;
    num_samples::Int,
)
    n_sweeping_steps = 5
   
    weidth, height = size(configuration, 2), size(configuration, 1)

    current_amplitude = PEPSValues.scaled_value(calculator, configuration)
    println("Initial amplitude: ", current_amplitude)

    list_of_configurations = Vector{Tuple{Matrix{Tuple{Int, Int}},typeof(current_amplitude)}}(undef, num_samples)
    watchlist = Matrix{Int}[]

    modified_configuration = copy(configuration)

    for i in 1:num_samples
        random_closed_loops = [rand(0:1) for _ in 1:weidth, _ in 1:height]
        while random_closed_loops in watchlist
            random_closed_loops = [rand(0:1) for _ in 1:weidth, _ in 1:height]
        end
        push!(watchlist, random_closed_loops)
        modified_configuration = add_closed_loops(configuration, random_closed_loops)

        current_amplitude = PEPSValues.scaled_value(calculator, modified_configuration)
        current_log_weight = log_born_weight(current_amplitude)

        for _ in 1:n_sweeping_steps
            acception = 0
            #this correction process i not very efficient!
            #need to find a way to calculate better the amplitude
            for x in 1:weidth, y in 1:height
                proposed_configuration = copy(modified_configuration)
                apply_star_operator!(x, y, proposed_configuration)
                proposed_amplitude = @timed PEPSValues.scaled_value(calculator, proposed_configuration)
                candidate_log_weight = log_born_weight(proposed_amplitude.value)
                log_ratio = candidate_log_weight - current_log_weight

                if !isnan(log_ratio) && (log_ratio >= 0)
                    modified_configuration = proposed_configuration
                    current_amplitude = proposed_amplitude.value
                    current_log_weight = candidate_log_weight
                    acception += 1
                end
            end
            println("Acceptance rate: $acception, current_amplitude: $current_amplitude")
            acception == 0 && break
        end
        print("--")
        if i%10 == 0
            println("Sampled $i configurations of one logical class")
        end
        list_of_configurations[i] = (modified_configuration, current_amplitude) 
    end
    return list_of_configurations

end

function calculate_decoded_configuration(
    syndromes,
    width,
    height,
    calculator::PEPSValues.PEPSValueCalculator;
    print_out::Bool=false,
)
    refereenz_value = decoded_bit_MWPM(
        syndromes,
        width,
        height,
    )
    config_logical_classes = [apply_logical_sector(refereenz_value, sector) for sector in 1:4]
    # Sample every logical class before choosing one common exponent.  The same
    # scale must be used for all four partition functions so they are comparable.
    sampled_logical_classes = [
        @timed sampling_throw_logical_classes(
            config_logical_classes[i],
            calculator;
            num_samples=100,
        ) for i in 1:4
    ]
    average_time = mean(sampled_logical_classes[i].time for i in 1:4)
    reference_exponent = maximum(
        amplitude[2]
        for sampled_configurations in sampled_logical_classes
        for (_, amplitude) in sampled_configurations.value
    )
    approx_partition_function = [
        sum(
            probability_from_scaled_amplitude(amplitude, reference_exponent)
            for (_, amplitude) in sampled_configurations.value
        ) for sampled_configurations in sampled_logical_classes
    ]

    #This is the maximum likelihood decoding step: we choose the logical class with the highest partition function as the decoded configuration.
    max_index = argmax(approx_partition_function)
    if print_out
        println("Approximate partition functions for the logical classes: ", approx_partition_function)
        println("Chosen logical class: ", max_index)
    end 
    decoded_configuration = config_logical_classes[max_index]

    return decoded_configuration
end

function evaluate_samples_Wz_indicator(samples, calculator::PEPSValues.PEPSValueCalculator)
    isempty(samples) && throw(ArgumentError("samples must not be empty"))

    w_z_x = 0
    w_z_y = 0
    counter = 0
    for (configuration_values, _) in samples
        syndromes = calculate_syndromes(configuration_values)
        # Use MWPM first as a baseline decoder for the sampled configurations.
        decoded_configuration = calculate_decoded_configuration(syndromes, size(configuration_values, 2), size(configuration_values, 1), calculator)


        current_w_z_x, current_w_z_y = measurement_wilson_loops(
            configuration_values,
            decoded_configuration,
        )
        println("Current Wz_x: ", current_w_z_x, " Current Wz_y: ", current_w_z_y)
        w_z_x += current_w_z_x
        w_z_y += current_w_z_y
        counter += 1
        if counter % 10 == 0
            println("Evaluated $counter samples")
        end
    end
    average_w_z_x = w_z_x / length(samples)
    average_w_z_y = w_z_y / length(samples)
    return (average_w_z_x + average_w_z_y) / 2
end

function load_calculator()
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

function test_ML_decoder()
    calculator = load_calculator()

    #load some samples
    filename = joinpath(@__DIR__, "generated_configurations/sampled_p=1_n_400.txt")
    samples = load_path_samples(filename, 8, 8)
    println("Loaded $(length(samples)) samples")
    println("-------")
    for _ in 1:5
        random_number_of_samples = rand(1:length(samples))
        configuration_values, amplitude = samples[random_number_of_samples]
        println("Randomly selected sample ($random_number_of_samples) with amplitude: ", amplitude)
        syndromes = calculate_syndromes(configuration_values)
        decoded_configuration = calculate_decoded_configuration(syndromes, size(configuration_values, 2), size(configuration_values, 1), calculator, print_out=true)
        println("Decoded configuration: ", decoded_configuration)
        println("---------")
    end
    println("---------")

end

function main_ml_decoder(
    filename::String=joinpath(@__DIR__, "generated_configurations/sampled_p=1_n_400.txt");
    width::Int=8,
    height::Int=8,
)
    calculator = load_calculator()

    samples = load_path_samples(filename, width, height)
    println("Loaded $(length(samples)) samples")
    println("-------")
    println("Start calculating Wz indicator for the samples")
    timed_indicator = @timed evaluate_samples_Wz_indicator(samples, calculator)
    println("Wz indicator for the samples: ", timed_indicator.value)
    println("Time taken: ", timed_indicator.time)
    return timed_indicator.value
end

#TODO_for_conradin: Currently the MWPM decoder does not calculate the right W_z of the chosen configurations!

test_ML_decoder()
