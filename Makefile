CC=icpc
#FLAGS=-std=c++11 #-mkl -I ./include/ -Wl,-rpath,${MKLROOT}/lib 
FLAGS=-std=c++11 -mkl -Wl,-rpath,${MKLROOT}/lib,-rpath,/opt/intel/compilers_and_libraries_2017.4.181/mac/compiler/lib/
src = $(wildcard ./src/*.cpp)

all: compile run

compile:
	${CC} ${FLAGS} ${src} -o bin/a.out
run:
	./bin/a.out
