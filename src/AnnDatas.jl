module AnnDatas

using HDF5
using SparseArrays
using DataFrames


export AnnData


struct AnnData
    X::Union{Nothing, AbstractMatrix}
    obsm::Union{Nothing, Dict{String, Any}}
    obsp::Union{Nothing, Dict{String, Any}}
    uns::Union{Nothing, Dict{String, Any}}
    obs::Union{Nothing, DataFrame}
    var::Union{Nothing, DataFrame}
    layers::Union{Nothing, Dict{String, AbstractMatrix}}
end



function Base.copy(adata::AnnData)
    return AnnData(adata.X, adata.obsm, adata.obsp, adata.uns, adata.obs, adata.var)
end


function Base.size(adata::AnnData)
    return size(adata.X)
end

function Base.size(adata::AnnData, i::Integer)
    return size(adata.X, i)
end


function show_keys(io::IO, d::DataFrame)
    join(io, names(d), ", ")
end

function show_keys(io::IO, d::Dict)
    join(io, keys(d), ", ")
end


function Base.show(io::IO, adata::AnnData)
    m, n = size(adata)
    println(io, "AnnData with n_obs × n_var = $(m) × $(n)")
    for field in fieldnames(AnnData)
        val = getfield(adata, field)
        if field != :X && val !== nothing
            print(io, "  ", string(field), ": ")
            show_keys(io, val)
            println(io)
        end
    end
end


# Indexing helper to index dictionary fields
function _getindex(d::AbstractMatrix, idx)
    return d[:,idx]
end

function _getindex(d::Dict, idx)
    return Dict(k => _getindex(v, idx) for (k,v) in d)
end

function _getindex(d::Dict, rows, cols)
    return Dict(k => v[rows, cols] for (k,v) in d)
end

function _getindex(d::DataFrame, idx)
    return d[idx,:]
end

function _getindex(d::Nothing, idx)
    return nothing
end

function Base.getindex(adata::AnnData, rows, cols_)
    # index by gene name
    if eltype(cols_) <: String
        index = Dict(g => i for (i,g) in enumerate(adata.var._index))
        cols = [index[g] for g in cols_]
    else
        cols = cols_
    end

    return AnnData(
        adata.X[rows, cols],
        _getindex(adata.obsm, rows),
        _getindex(adata.obsp, rows),
        adata.uns,
        _getindex(adata.obs, rows),
        _getindex(adata.var, cols),
        _getindex(adata.layers, rows, cols))
end


function write_vlenstr_attribute(parent, name, s::String)
    dtype = HDF5.API.h5t_copy(HDF5.API.H5T_C_S1)
    HDF5.API.h5t_set_size(dtype, HDF5.API.H5T_VARIABLE)
    HDF5.API.h5t_set_cset(dtype, HDF5.API.H5T_CSET_UTF8)
    dspace = HDF5.API.h5s_create(HDF5.API.H5S_SCALAR)
    dset = create_attribute(parent, name, HDF5.Datatype(dtype), HDF5.Dataspace(dspace))

    strbuf = Base.cconvert(Cstring, s)
    ptr = pointer([pointer(strbuf)])

    HDF5.h5a_write(dset, dtype, ptr)
end


"""
Read a CSR matrix in a SparseMatrixCSC
"""
function read_csr_matrix(g::HDF5.Group)
    attr = attributes(g)
    # @assert read(attr["encoding-type"]) == "csr_matrix"
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

    write_vlenstr_attribute(grp, "encoding-type", "csr_matrix")
    write_vlenstr_attribute(grp, "encoding-version", "0.1.0")
    attr = attributes(grp)
    attr["shape"] = Int[size(X,1), size(X,2)]

    Xt = SparseMatrixCSC(transpose(X))

    grp["data"] = Xt.nzval
    grp["indices"] = Xt.rowval .- 1
    grp["indptr"] = Xt.colptr .- 1
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
        if key != "__categories" && key ∉ columnorder
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
            df[!,key] = [i+1 < 1 ? missing : values[i+1] for i in df[!,key]]
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


function read_matrix(parent, path::String)
    if isa(parent[path], HDF5.Group)
        enc = read(attributes(parent[path])["encoding-type"])
        if enc == "csr_matrix"
            X = read_csr_matrix(parent[path])
        elseif enc == "csc_matrix"
            # X = transpose(read_csr_matrix(parent["X"]))
            # TODO: support CSC matrix
            X = nothing
        else
            error("Unsupported matrix encoding $(enc)")
        end
    else
        X = Matrix(transpose(read(parent[path])))
    end
end


function read_layers(parent, path::String)
    layers = Dict{String, AbstractMatrix}()
    g = parent[path]
    for key in keys(g)
        layers[key] = read_matrix(g, key)
    end
    return layers
end

# TODO: writing `layers`



"""
Read an AnnData struct from the given h5ad filename.
"""
function Base.read(filename::AbstractString, ::Type{AnnData})
    input = h5open(filename)
    X = read_matrix(input, "X")

    obsm   = read_group(input, "obsm")
    obsp   = read_group(input, "obsp")
    uns    = read_group(input, "uns")
    obs    = read_dataframe(input, "obs")
    var    = read_dataframe(input, "var")
    layers = read_layers(input, "layers")

    close(input)

    return AnnData(X, obsm, obsp, uns, obs, var, layers)
end


function write_anndata_group(output, name, df::Nothing)
end


function write_anndata_group(output, name, df::DataFrame)
    grp = create_group(output, name)
    attr = attributes(grp)

    # TODO: because pyh5 is dumb as shit, we need to write these as variable
    # length strings. Othrewise they are read as bytes trings an a bunch
    # of equality tests fail.

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
        if isa(v, Dict)
            write_anndata_group(grp, k, v)
        else
            grp[k] = v
        end
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
