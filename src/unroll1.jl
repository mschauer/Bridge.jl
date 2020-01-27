"""
    @unroll1 for-loop

Unroll the first iteration of a `for`-loop.
Set `$first` to true in the first iteration.

Example:
```
    @unroll1 for i in 1:10
        if $first
            a, state = iterate('A':'Z')
        else
            a, state = iterate('A':'Z', state)
        end
        println(i => a)
    end
```
"""
macro unroll1(expr)
    @assert expr isa Expr
    @assert expr.head == :for
    iterspec = expr.args[1]

    @assert iterspec isa Expr
    @assert  iterspec.head == :(=)

    i_ = iterspec.args[1]
    i = esc(i_)
    iter = esc(iterspec.args[2])
    body = esc(expr.args[2])

    body_1 = eval(Expr(:let, :(first = true), Expr(:quote, body)))
    body_i = eval(Expr(:let, :(first = false), Expr(:quote, body)))

    if i_ isa Expr && i_.head == :tuple
        decl = esc(quote local $(i_.args...) end)
    else
        decl = quote local $i end
    end
    
    quote
        local st
        $decl
        @goto enter
        while true
            @goto exit
            @label enter
                ϕ = iterate($iter)
                ϕ === nothing && break
                $i, st = ϕ
                $(body_1)
            @label exit
            while true
                ϕ = iterate($iter, st)
                ϕ === nothing && break
                $i, st = ϕ
                $(body_i)
            end
            break
        end
    end
end
