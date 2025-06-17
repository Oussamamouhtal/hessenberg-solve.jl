using LinearAlgebra

# 1. Modifiez la fonction suivante pour qu'elle renvoie la solution x
#    du système triangulaire supérieur Rx = b.
#    Votre fonction ne doit modifier ni R ni b.
function backsolve(R::UpperTriangular, b)
    # La matrice R doit etre inversible
    x = similar(b)
    n = length(b)
    for i in n:-1:1
        x[i] = (b[i] - R[i, i+1:end]' * x[i+1:end]) / R[i, i]
    end
    return x
end

# 2. Modifiez la fonction suivante pour qu'elle renvoie la solution x
#    du système Hessenberg supérieur Hx = b ou du problème aux moindres
#    carrés min ‖Hx - b‖ à l'aide de rotations ou
#    réflexions de Givens et d'une remontée triangulaire.
#    Votre fonction peut modifier H et b si nécessaire.
#    Il n'est pas nécessaire de garder les rotations en mémoire et la
#    fonction ne doit pas les renvoyer.
#    Seul le cas réel sera testé ; pas le cas complexe.
function hessenberg_solve(H::UpperHessenberg, b)
    # La matrice H doit etre de rang plein
    m, n = size(H)
    index = min(m-1,n) # Manière optimisé permet de distinguer les deux cas (m=n+1 ou m=n)

    for i in 1:index
        x, y = H[i, i], H[i+1, i]
        r = norm([x, y])
        c = x / r
        s = y / r
        H[i, i] = r
        H[i+1, i] = 0
        for j in i+1:n 
            inter = H[i, j]     # Variable intermédiaire 
            H[i, j] = c * H[i, j] + s * H[i+1, j]
            H[i+1, j] = -s * inter + c * H[i+1, j]     
        end

        # Appliquer la rotation à b 
        inter = b[i]        # Variable intermédiaire 
        b[i] = c * b[i] + s * b[i+1]
        b[i+1] = -s * inter + c * b[i+1]
    end

    # Résoudre le système triangulaire supérieur avec "backsolve"
    R = UpperTriangular(H[1:n, 1:n])
    x = backsolve(R, b[1:n])
    return x
end

# vérification
using Test
for n ∈ (10, 20, 30)
    # square system
    A = rand(n, n)
    A[diagind(A)] .+= 1
    b = rand(n)
    R = UpperTriangular(A)
    x = backsolve(R, b)
    @test norm(R * x - b) ≤ sqrt(eps()) * norm(b)
    H = UpperHessenberg(A)
    x = hessenberg_solve(copy(H), copy(b)) 
    @test norm(H * x - b) ≤ sqrt(eps()) * norm(b)
    # slightly overdetermined least squares
    A = rand(n + 1, n)
    A[diagind(A)] .+= 1
    H = UpperHessenberg(A)
    b = rand(n + 1)
    x_ls = hessenberg_solve(copy(H), copy(b))
    x_qr = H \ b
    @test norm(x_ls - x_qr) ≤ sqrt(eps()) * norm(x_qr)
end
