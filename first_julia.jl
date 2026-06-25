println("Hello, World!")

x = 10
y = 20
total = x + y
println("The sum of $x and $y is $total.")
function greet(name)
    return "Hello, $(name)!"
end
namee = "Alice"
greeting = greet(namee)
println(greeting)

for i in 1:5
    println("Iteration $i")
end

include(joinpath(@__DIR__, "test.jl"))
hello()
