module AnnDatas

using HDF5
using SparseArrays
using DataFrames


export AnnData


struct AnnData
    X::AbstractMatrix
    obsm::Union{Nothing, Dict{String, Any}}
    obsp::Union{Nothing, Dict{String, Any}}
    uns::Union{Nothing, Dict{String, Any}}
    obs::Union{Nothing, DataFrame}
    var::Union{Nothing, DataFrame}
end


# h5py will read variable length strings as an 'str', but fixed length strings
# as bytes. Julia writes fixed length string, and this ends up causing problems
# sometimes.
#= function vlenstr(s::String)
    # [HDF5.UTF8Char(c) for c in Vector{UInt8}("csr_matrix")]
    [HDF5.UTF8Char(c) for c in Vector{Char}(Vector{UInt8}("csr_matrix")]
    # return Vector{Char}("csr_matrix")
    # return String["csr_matrix"]
end =#


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


function write_csc_matrix(output, X::SparseMatrixCSC)
    grp = create_group(output, "X")
    attr = attributes(grp)
    # attr["encoding-type"] = vlenstr("csr_matrix")
    attr["encoding-type"] = "csr_matrix"
    attr["encoding-version"] = "0.1.0"
    attr["shape"] = Int[size(X,1), size(X,2)]

    Xt = SparseMatrixCSC(transpose(X))

    grp["data"] = Xt.nzval
    grp["indices"] = Xt.rowval
    grp["indptr"] = Xt.colptr
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
    columnorder = Vector{String}(read(attr["column-order"]))

    for key in keys(g)
        if key ∉ columnorder
            pushfirst!(columnorder, key)
        end
        columns[key] = read(g[key])
    end
    df = DataFrame(Dict(key => columns[key] for key in columnorder))
    if haskey(g, "_index")
        df[!,"_index"] = read(g["_index"])
    end

    if haskey(g, "__categories")
        cats = g["__categories"]
        for key in keys(cats)
            values = read(cats[key])
            df[!,key] = [i < 1 ? missing : values[i] for i in df[!,key]]
        end
    end

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


function write_anndata_group(output, name, df::Nothing)
end


function write_anndata_group(output, name, df::DataFrame)
    grp = create_group(output, name)
    attr = attributes(grp)
    attr["encoding-version"] = "0.1.0"
    attr["encoding-type"] = "dataframe"

    has_index = false
    column_order = String[]
    for name in names(df)
        grp[name] = df[!,name]
        if name != "_index"
            push!(column_order, name)
        else
            has_index = true
        end
    end
    attr["column-order"] = column_order
    attr["_index"] = "_index"

    if !has_index
        n = size(df, 1)
        grp["_index"] = String[string(i-1) for i in 1:n]
    end

    return grp
end


function write_anndata_group(output, name, df::Dict)
    grp = create_group(output, name)
    for (k, v) in df
        grp[k] = v
    end
end


function Base.write(filename::AbstractString, adata::AnnData)
    h5open(filename, "w") do output
        if isa(adata.X, SparseMatrixCSC)
            write_csc_matrix(output, adata.X)
        else
            output["X"] = adata.X
        end
        write_anndata_group(output, "obsm", adata.obsm)
        write_anndata_group(output, "obsp", adata.obsp)
        write_anndata_group(output, "uns", adata.uns)
        write_anndata_group(output, "obs", adata.obs)
        write_anndata_group(output, "var", adata.var)
    end
end


end # module
