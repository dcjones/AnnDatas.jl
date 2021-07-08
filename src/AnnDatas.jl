module AnnDatas

using HDF5
using SparseArrays
using DataFrames


export AnnData


struct AnnData
    X::SparseMatrixCSC
    obsm::Union{Nothing, Dict{String, Any}}
    obsp::Union{Nothing, Dict{String, Any}}
    uns::Union{Nothing, Dict{String, Any}}
    obs::Union{Nothing, DataFrame}
    var::Union{Nothing, DataFrame}
end


"""
Read a CSR matrix in a SparseMatrixCSC
"""
function read_csr_matrix(g::HDF5.Group)
    attr = attributes(g)
    @assert read(attr["encoding-type"]) == "csr_matrix"
    m, n = read(attr["shape"])

    V           = read(g["data"])
    csr_indices = read(g["indices"]) .+ 1
    csr_indptr  = read(g["indptr"]) .+ 1
    nnz = length(V)

    # easiest way to do this is to go CSR -> COO -> CSC
    I = Vector{Int32}(undef, nnz)
    J = Vector{Int32}(undef, nnz)

    for i in 1:m
        for k in csr_indptr[i]:csr_indptr[i+1]-1
            I[k] = i
            J[k] = csr_indices[k]
        end
    end

    return sparse(I, J, V, m, n)
end


"""
Read a serialized data frame into a DataFrame
"""
function read_dataframe(input::HDF5.File, path::String)
    if !haskey(input, path)
        return nothing
    end
    g = input[path]

    columns = Dict{String, Any}()
    attr = attributes(g)
    @assert read(attr["encoding-type"]) == "dataframe"
    columnorder = read(attr["column-order"])

    for key in keys(g)
        columns[key] = read(g[key])
    end
    df = DataFrame(Dict(key => columns[key] for key in columnorder))
    df[!,"_index"] = read(g["_index"])

    return df
end


"""
General purpose function to read a groups into a dictionary tree structure.
"""
function read_group(input::HDF5.File, path::String)
    if !haskey(input, path)
        return nothing
    end
    g = input[path]

    data = Dict{String, Any}()
    for key in keys(g)
        dataset = g[key]

        if isa(dataset, HDF5.Group)
            attr = attributes(dataset)
            if length(attr) == 0
                # data[key] = read_group(dataset)
                data[key] = read(dataset)
            elseif haskey(attr, "encoding-type") && read(attr["encoding-type"]) == "csr_matrix"
                data[key] = read_csr_matrix(dataset)
            else
                data[key] = read(dataset)
            end
            # TODO: special cases for other types of data
        else
            data[key] = read(dataset)
        end
    end

    return data
end


"""
Read an AnnData struct from the given h5ad filename.
"""
function Base.read(filename::AbstractString, ::Type{AnnData})
    input = h5open(filename)

    if isa(input["X"], HDF5.Group)
        X = read_csr_matrix(input["X"])
    else
        X = read(input["X"])
    end
    obsm = read_group(input, "obsm")
    obsp = read_group(input, "obsp")
    uns  = read_group(input, "uns")
    obs  = read_dataframe(input, "obs")
    var  = read_dataframe(input, "var")

    close(input)

    return AnnData(X, obsm, obsp, uns, obs, var)
end


end # module
