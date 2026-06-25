import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "value_the_PEPS.jl"))

using .PEPSValues
using Random


function print_configuration(configuration::AbstractMatrix{<:Tuple})
    for y in axes(configuration, 2)
        for x in axes(configuration, 1)
            horizontal_bit, vertical_bit = configuration[x, y]
            print("($horizontal_bit, $vertical_bit) ")
        end
        println()
    end
end


"Flip physical qubit `spin` (1=horizontal, 2=vertical) in place."
function flip_spin!(
    configuration::AbstractMatrix{<:Tuple},
    x::Int,
    y::Int,
    spin::Int,
)
    checkbounds(configuration, x, y)
    horizontal, vertical = configuration[x, y]

    if spin == 1
        configuration[x, y] = (1 - horizontal, vertical)
    elseif spin == 2
        configuration[x, y] = (horizontal, 1 - vertical)
    else
        throw(ArgumentError("spin must be 1 (horizontal) or 2 (vertical)"))
    end

    return configuration
end


"Compare magnitudes represented as `(mantissa, binary_exponent)`."
function scaled_magnitude_is_greater(
    candidate::Tuple{<:Real,<:Integer},
    current::Tuple{<:Real,<:Integer},
)
    candidate_mantissa, candidate_exponent = candidate
    current_mantissa, current_exponent = current

    iszero(candidate_mantissa) && return false
    iszero(current_mantissa) && return true
    candidate_exponent != current_exponent && return candidate_exponent > current_exponent
    return abs(candidate_mantissa) > abs(current_mantissa)
end


"Greedy local search over horizontal and vertical single-qubit flips."
function sweeping_initialization(
    initial_configuration::AbstractMatrix{<:Tuple},
    calculator::PEPSValues.PEPSValueCalculator,
    max_sweeps::Int,
    initial_amplitude::Tuple{<:Real,<:Integer},
)
    max_sweeps >= 0 || throw(ArgumentError("max_sweeps must be nonnegative"))

    configuration = copy(initial_configuration)
    current_amplitude = initial_amplitude

    #sweeping just one flip at a time
    for sweep in 1:max_sweeps
        accepted_flips = 0

        for x in axes(configuration, 1), y in axes(configuration, 2), spin in 1:2
            flip_spin!(configuration, x, y, spin)
            candidate_amplitude = PEPSValues.scaled_value(calculator, configuration)

            if scaled_magnitude_is_greater(candidate_amplitude, current_amplitude)
                current_amplitude = candidate_amplitude
                accepted_flips += 1
            else
                # Reject the proposal by flipping the same bit back.
                flip_spin!(configuration, x, y, spin)
            end
        end

        #flipping four bits at a time, to escape local minima
        for x in axes(configuration, 1), y in axes(configuration, 2)
            flip_spin!(configuration, x, y, 1)
            flip_spin!(configuration, x, y, 2)
            flip_spin!(configuration, mod1(x + 1, size(configuration, 1)), y, 1)
            flip_spin!(configuration, x, mod1(y + 1, size(configuration, 2)), 2)

            candidate_amplitude = PEPSValues.scaled_value(calculator, configuration)

            if scaled_magnitude_is_greater(candidate_amplitude, current_amplitude)
                current_amplitude = candidate_amplitude
                accepted_flips += 4
            else
                # Reject the proposal by flipping the same bits back.
                flip_spin!(configuration, x, y, 1)
                flip_spin!(configuration, x, y, 2)
                flip_spin!(configuration, mod1(x + 1, size(configuration, 1)), y, 1)
                flip_spin!(configuration, x, mod1(y + 1, size(configuration, 2)), 2)
            end
        end

        println(
            "Sweep $sweep: accepted $accepted_flips flips; " *
            "scaled amplitude = $current_amplitude",
        )
        accepted_flips == 0 && break
        accepted_flips = 0
    end
    
    return configuration, current_amplitude
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

"Perform one Metropolis sweep and return the new amplitude and acceptance count."
function sample_configuration!(
    rng::AbstractRNG,
    current_configuration::AbstractMatrix{<:Tuple},
    calculator::PEPSValues.PEPSValueCalculator,
    current_amplitude::Tuple{<:Real,<:Integer},
)
    current_log_weight = log_born_weight(current_amplitude)
    accepted = 0

    for x in axes(current_configuration, 1), y in axes(current_configuration, 2), spin in 1:2
        flip_spin!(current_configuration, x, y, spin)
        candidate_amplitude = PEPSValues.scaled_value(calculator, current_configuration)
        candidate_log_weight = log_born_weight(candidate_amplitude)
        log_ratio = candidate_log_weight - current_log_weight

        # This is equivalent to min(1, |candidate/current|^2), without ever
        # constructing numbers that underflow or overflow.
        if !isnan(log_ratio) && (log_ratio >= 0 || log(rand()) < log_ratio)
            current_amplitude = candidate_amplitude
            current_log_weight = candidate_log_weight
            accepted += 1
        else
            flip_spin!(current_configuration, x, y, spin)
        end
    end

    return current_amplitude, accepted
end

"Draw PEPS Born-distribution samples after burn-in, saving distinct array copies."
function mcmc_samples(
    rng::AbstractRNG,
    initial_configuration::AbstractMatrix{<:Tuple},
    calculator::PEPSValues.PEPSValueCalculator;
    n_samples::Int,
    burn_in::Int=1,
    thinning::Int=1,
    initial_amplitude=PEPSValues.scaled_value(calculator, initial_configuration),
)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    burn_in >= 0 || throw(ArgumentError("burn_in must be nonnegative"))
    thinning > 0 || throw(ArgumentError("thinning must be positive"))

    configuration = copy(initial_configuration)
    amplitude = initial_amplitude
    samples = Vector{Tuple{typeof(configuration),typeof(amplitude)}}(undef, n_samples)
    total_accepted = 0
    total_proposals = 0

    for sweep in 1:(burn_in + n_samples * thinning)
        amplitude, accepted = sample_configuration!(
            rng, configuration, calculator, amplitude,
        )
        total_accepted += accepted
        total_proposals += 2 * length(configuration)

        if sweep > burn_in && (sweep - burn_in) % thinning == 0
            sample_index = (sweep - burn_in) ÷ thinning
            samples[sample_index] = (copy(configuration), amplitude)
        end
         
        if sweep%10 ==0
            println(
                "MCMC sweep $sweep: accepted $accepted flips; " *
                "current scaled amplitude = $amplitude",
            )
        end
    end

    return samples, total_accepted / total_proposals
end


function main()
    println("----------Start simulation---------------")
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
        parameter_index=24,
        grid_size=grid_size,
        sweep_chi=8,
        sweep_tau=16,
    )

    # A single sweep evaluates 2 * prod(grid_size) = 128 contractions. Keep the
    # max_sweeps is small, because with that seeper it gets really fast in a local minimum.
    max_sweeps = 1
    rng = MersenneTwister(1234)
    initial_configuration = [
        (rand(0:1), rand(0:1))
        for _ in 1:grid_size[1], _ in 1:grid_size[2]
    ]

    println("Initial configuration:")
    print_configuration(initial_configuration)

    first_evaluation = @timed PEPSValues.scaled_value(
        calculator,
        initial_configuration,
    )
    println(
        "Initial scaled amplitude: $(first_evaluation.value) " *
        "($(round(first_evaluation.time; digits=3)) seconds)",
    )
    println("-------Starting sweeping initialization-------")

    configuration, scaled_amplitude = sweeping_initialization(
        initial_configuration,
        calculator,
        max_sweeps,
        first_evaluation.value,
    )

    println("Final configuration:")
    print_configuration(configuration)
    println("Final scaled amplitude: $scaled_amplitude")

    println("-----")
    #now we want to use Markov Chain Monte Carlo to sample from the distribution defined by the PEPS amplitudes
    n_samples = 20
    total_mcmc_information = @timed mcmc_samples(
        rng,
        configuration,
        calculator;
        n_samples=n_samples,
        burn_in=1,
        thinning=1,
        initial_amplitude=scaled_amplitude,
    )
    samples, acceptance_rate = total_mcmc_information.value
    time = total_mcmc_information.time
    println("MCMC sampling time: $(round(time; digits=3)) seconds")
    println("Collected $(length(samples)) configurations")
    println("Metropolis acceptance rate: $(round(acceptance_rate; digits=3))")

    return samples, acceptance_rate
end


function write_samples_to_file(samples::Vector{<:Tuple}, filename::String)
    open(filename, "w") do IO
        for (configuration, amplitude) in samples
            print(IO, amplitude, ": ")
            for y in axes(configuration, 2)
                for x in axes(configuration, 1)
                    horizontal_bit, vertical_bit = configuration[y, x]
                    print(IO, "($horizontal_bit, $vertical_bit) ")
                end
            end
            println(IO)
        end
    end
end

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
            configuration = reshape(configuration_values, width, height)
            push!(samples, (configuration, amplitude))
        end
    end
    return samples
end

function test_write_and_load_samples()
    test_samples = [
        ([(0, 0) (0, 0); (0, 0) (0, 0)], (1.0, 0)),
        ([(1, 0) (0, 1); (1, 1) (0, 0)], (0.5, -1)),
        ([(1, 1) (1, 1); (1, 1) (1, 1)], (0.25, -2)),
    ]
    write_samples_to_file(test_samples, joinpath(@__DIR__, "generated_configurations/test_samples.txt"))

    samples_loaded = load_path_samples(joinpath(@__DIR__, "generated_configurations/test_samples.txt"), 2, 2)
    println("Loaded samples:")
    for (config, amplitude) in samples_loaded
        println("Amplitude: $amplitude, Configuration:")
        print_configuration(config)
    end
end

#With this programm, samples are generated and written to a file, which have a large probability under the PEPS distribution
#Problem there are samples, that are equivalent and can be distinguished, if we want just unique samples
sampled_configurations, acceptance_rate = main()
write_samples_to_file(sampled_configurations, joinpath(@__DIR__, "generated_configurations/sampled_p=1_n_20.txt"))
