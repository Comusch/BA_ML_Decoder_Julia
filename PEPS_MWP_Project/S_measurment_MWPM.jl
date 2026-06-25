using Graphs
using GraphsMatching
using PyQDecoders: pm, sps
using PythonCall: pyconvert

function calculate_syndromes(configuration::AbstractMatrix{<:Tuple})
    height, width = size(configuration)
    syndromes = zeros(Int, width, height)

    for x in 1:width, y in 1:height
        horizontal_bit, vertical_bit = configuration[y, x]
        right_neighbor = configuration[mod1(y + 1, height), x][1]
        down_neighbor = configuration[y, mod1(x + 1, width)][2]
        syndromes[x, y] = (horizontal_bit + vertical_bit +right_neighbor + down_neighbor) % 2
    end

    return syndromes
end

"Convert a plaquette coordinate to its graph vertex."
plaquette_vertex(x::Int, y::Int, width::Int) = x + (y - 1) * width

"Convert a graph vertex back to a plaquette coordinate."
plaquette_position(vertex::Int, width::Int) =
    CartesianIndex(mod1(vertex, width), div(vertex - 1, width) + 1)

"Column index of a horizontal (`spin == 1`) or vertical (`spin == 2`) bit."
function physical_bit_column(x::Int, y::Int, spin::Int, width::Int, height::Int)
    site_index = x + (y - 1) * width
    if spin == 1
        return site_index
    elseif spin == 2
        return width * height + site_index
    else
        throw(ArgumentError("spin must be 1 for horizontal or 2 for vertical"))
    end
end

function flip_check_entry!(check_matrix, row::Int, column::Int)
    check_matrix[row, column] = 1 - check_matrix[row, column]
end

"""
    toric_mwpm_check_matrix(width, height)

Build the binary parity-check matrix matching `calculate_syndromes`.
Columns `1:width*height` are horizontal bits, and the remaining columns are
vertical bits. Rows use the same `plaquette_vertex(x, y, width)` ordering as the
syndrome matrix.
"""
function toric_mwpm_check_matrix(width::Int, height::Int)
    number_plaquettes = width * height
    check_matrix = zeros(Int, number_plaquettes, 2 * number_plaquettes)

    for x in 1:width, y in 1:height
        current = plaquette_vertex(x, y, width)

        horizontal_column = physical_bit_column(x, y, 1, width, height)
        flip_check_entry!(check_matrix, current, horizontal_column)
        flip_check_entry!(
            check_matrix,
            plaquette_vertex(x, mod1(y - 1, height), width),
            horizontal_column,
        )

        vertical_column = physical_bit_column(x, y, 2, width, height)
        flip_check_entry!(check_matrix, current, vertical_column)
        flip_check_entry!(
            check_matrix,
            plaquette_vertex(mod1(x - 1, width), y, width),
            vertical_column,
        )
    end

    return sps.csc_matrix(check_matrix)
end

function syndrome_vector(syndromes::AbstractMatrix{<:Integer}, width::Int, height::Int)
    return [Int(syndromes[x, y]) for y in 1:height for x in 1:width]
end

function decoded_vector_to_configuration(
    decoded_vector::AbstractVector{<:Integer},
    width::Int,
    height::Int,
)
    expected_length = 2 * width * height
    length(decoded_vector) == expected_length || throw(DimensionMismatch(
        "decoded vector has length $(length(decoded_vector)); expected $expected_length",
    ))

    decoded_bits = Matrix{Tuple{Int,Int}}(undef, height, width)
    for y in 1:height, x in 1:width
        horizontal_column = physical_bit_column(x, y, 1, width, height)
        vertical_column = physical_bit_column(x, y, 2, width, height)
        decoded_bits[y, x] = (
            Int(decoded_vector[horizontal_column]),
            Int(decoded_vector[vertical_column]),
        )
    end
    return decoded_bits
end


function print_syndromes(syndromes::AbstractMatrix{<:Integer}, size::Tuple{Int,Int})
    width, height = size
    for y in 1:height
        for x in 1:width
            print(syndromes[x, y], " ")
        end
        println()
    end
end


function resulting_configuration(
    physical_configuration::AbstractMatrix{<:Tuple},
    MWPM_configuration::AbstractMatrix{<:Tuple},
)
    size(physical_configuration) == size(MWPM_configuration) || throw(DimensionMismatch(
        "physical and MWPM configurations must have the same size",
    ))

    combined_configuration = Matrix{Tuple{Int,Int}}(undef, size(physical_configuration))
    for row in axes(physical_configuration, 1), column in axes(physical_configuration, 2)
        physical_horizontal, physical_vertical = physical_configuration[row, column]
        MWPM_horizontal, MWPM_vertical = MWPM_configuration[row, column]
        combined_configuration[row, column] = (
            mod(physical_horizontal + MWPM_horizontal, 2),
            mod(physical_vertical + MWPM_vertical, 2),
        )
    end
    return combined_configuration
end

function measurement_wilson_loops(physical_configuration::AbstractMatrix{<:Tuple}, MWPM_configuration::AbstractMatrix{<:Tuple})
    height, width = size(physical_configuration)
    combined_configuration = resulting_configuration(physical_configuration, MWPM_configuration)
 
    W_x = 0
    for x in 1:width
        W_x = W_x + combined_configuration[1, x][1]
    end

    W_y = 0
    for y in 1:height
        W_y = W_y + combined_configuration[y, 1][2]
    end

    W_x = W_x % 2
    W_y = W_y % 2


    W_x = W_x == 0 ? 1 : -1
    W_y = W_y == 0 ? 1 : -1

    return W_x, W_y
end

function random_0_1(p::Real)
    0 <= p <= 1 || throw(ArgumentError("p must lie in [0, 1]"))
    return rand() < p ? 1 : 0
end

function decoded_bit_MWPM(
    syndromes::AbstractMatrix{<:Integer},
    width::Int,
    height::Int,
)
    size(syndromes) == (width, height) || throw(DimensionMismatch(
        "syndrome matrix has size $(size(syndromes)); expected $((width, height))",
    ))
    all(syndrome -> syndrome in (0, 1), syndromes) || throw(ArgumentError(
        "syndromes must contain only zeros and ones",
    ))
    iseven(sum(syndromes)) || throw(ArgumentError(
        "minimum-weight perfect matching requires an even number of syndromes",
    ))

    matching = pm.Matching(toric_mwpm_check_matrix(width, height))
    decoded_vector = pyconvert(
        Vector{Int},
        matching.decode(syndrome_vector(syndromes, width, height)),
    )
    decoded_bits = decoded_vector_to_configuration(decoded_vector, width, height)
    calculate_syndromes(decoded_bits) == syndromes || error(
        "PyQDecoders MWPM produced a bit string with the wrong syndrome",
    )
    return decoded_bits
end

function test_ML(p::Float64=0.5)
    """configuration = [
        (0, 0) (1, 1) (0, 1);
        (1, 1) (0, 0) (1, 0);
        (0, 1) (0, 0) (1, 0);
    ]"""
    configuration= [ (random_0_1(p), random_0_1(p)) for _ in 1:8, _ in 1:8 ]

    syndromes = calculate_syndromes(configuration)

    println("Syndromes:")
    print_syndromes(syndromes, size(configuration))

    """result = MWPM_decoding(syndromes)
    println("MWPM pairs: ", result.pairs)
    println("MWPM total weight: ", result.total_weight)"""

    decoded_bits = @timed decoded_bit_MWPM(syndromes, size(configuration, 2), size(configuration, 1))
    println("Most probable bit string (time: $(decoded_bits.time)): ", decoded_bits.value)
    println("Decoded syndromes:")
    print_syndromes(calculate_syndromes(decoded_bits.value), size(configuration))

    println("Measurement of Wilson loops: ", measurement_wilson_loops(configuration, decoded_bits.value))
end

#test_ML((sin(0.1*pi))^2)

"""
test(0.3)
println("Test completed.")
println("------------------------------------")"""
