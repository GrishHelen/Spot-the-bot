# code for supercomputer
using LinearAlgebra
using Plots
using Ripserer
using NPZ

data_rus = npzread(pwd() * "/SVD_embeddings/rus_100to2_tsne.npy")
println("here")
flush(stdout::IO)

# smaller center lower part
data_r = Array{Tuple{Float32,Float32},1}()
for i in 1:size(data_rus)[1]
    if -45 < data_rus[i, 1] < 30 && data_rus[i, 2] < -50
        global data_r = push!(data_r, Tuple(Float32(x) for x in data_rus[i, :]))
        # println(Tuple(x for x in data_rus[i, :]))
    end
end
data_r = unique!(data_r)
println(size(data_r))
flush(stdout::IO)

scatter(data_r; label = "data_rus", markersize = 0.5, legend = false)
savefig(pwd()  * "/ripserer (julia)/trash.png")

diagram_rus = ripserer(data_r; alg = :involuted, modulus = 2)

scatter(data_r; label = "data_rus", markersize = 0.5, legend = false)
N = length(diagram_rus[2])
println(N)
for n in (N - 50):N
    plot!(diagram_rus[2][n], data_r; label = "$n")
savefig(pwd() * "/julia_code/results/rus_centerlowerpart_cycles$n.png")
end
plot!()
savefig(pwd() * "/julia_code/results/rus_centerlowerpart_cycles.png")
