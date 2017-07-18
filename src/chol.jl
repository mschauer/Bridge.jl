## Compatibility for versions of julia where this function is not defined
import Base.LinAlg: chol, chol! 
 
function _chol!(J::UniformScaling, uplo)
    c, info = Base.LinAlg._chol!(J.λ, uplo)
    UniformScaling(c), info
end

chol!(J::UniformScaling, uplo) = ((J, info) = _chol!(J, uplo); Base.LinAlg.@assertposdef J info)
chol(J::UniformScaling, args...) = ((C, info) = _chol!(J, nothing); Base.LinAlg.@assertposdef C info)
 