using Documenter
using BurgersJoint

# Get all submodules of BurgersJoint
function get_submodules(mod)
    submodules = []
    for name in names(mod; all=true)
        if Base.isexported(mod, name) && 
           isdefined(mod, name) && 
           getfield(mod, name) isa Module &&
           name != :BurgersJoint
            push!(submodules, name)
        end
    end
    return submodules
end

# Create the api.md file
open(joinpath(@__DIR__, "src", "api.md"), "w") do io
    # Write header
    write(io, """
    # API reference

    ```@meta
    CurrentModule = BurgersJoint
    ```

    This page documents all modules and functions in BurgersJoint.

    """)
    
    # Write section for each submodule
    for mod_name in get_submodules(BurgersJoint)
        write(io, """
        ## $(mod_name)

        ```@autodocs
        Modules = [BurgersJoint.$(mod_name)]
        ```

        """)
    end
end

println("API documentation successfully generated!")

makedocs(
    sitename = "BurgersJoint documentation",
    modules = [BurgersJoint],
    format = Documenter.HTML(),
    pages = [
        "Home" => "index.md",
        "API reference" => "api.md"
    ],
    checkdocs = :none,
    clean = true
)

deploydocs(
    repo = "github.com/octionary/BurgersJoint.jl.git",
    devbranch = "main",
    deploy_config = Documenter.GitHubActions()
)