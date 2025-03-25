#!/usr/bin/env julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
#Pkg.build("BurgersJoint")
include("docs/make.jl")
using LiveServer
serve(dir="docs/build")
