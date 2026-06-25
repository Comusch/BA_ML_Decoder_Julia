import Pkg
Pkg.activate(@__DIR__)


using SweepContractor

# Define the tensors
LTN = LabelledTensorNetwork{Char}()
LTN['A'] = Tensor(['D','B'], [i^2-2j for i=0:2, j=0:2], 0, 1)
LTN['B'] = Tensor(['A','D','C'], [-3^i*j+k for i=0:2, j=0:2, k=0:2], 0, 0)
LTN['C'] = Tensor(['B','D'], [j for i=0:2, j=0:2], 1, 0)
LTN['D'] = Tensor(['A','B','C'], [i*j*k for i=0:2, j=0:2, k=0:2], 1, 1)

value = sweep_contract(LTN, 2, 4)
println("The value of the contracted tensor network is: $(value)")