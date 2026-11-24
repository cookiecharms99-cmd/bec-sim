default:
	@just --lists

# compiling thingies

cxx := "g++"
std := "-std=c++20"
inc := "-I../selus-math/include"
warn := "-Wall -Wextra -Wpedantic"
build_dir := "build"


run:
	{{cxx}} main.cpp {{std}} {{inc}} {{warn}} -o {{build_dir}}/main
	./{{build_dir}}/main


