using SparseArrays, Laplacians

function solve_systems(n1, n2, m1, m2, C1, R1, V1, C2, R2, V2)
    # create csc matrices
    M = SparseMatrixCSC(n1, m1, C1, R1, V1)
    B = SparseMatrixCSC(n2, m2, C2, R2, V2)

    # output matrix
    X = Array{Float64}(undef, n2, m2);

    sol = approxchol_sddm(M, verbose=false)

    for i in 1:m2
        b = Vector(B[:,i])
        X[:,i] = sol(b, verbose=false)#, tol=1e-16)
    end

    return X
end