/** \addtogroup examples 
  * @{ 
  * \defgroup matmul Matrix multiplication
  * @{ 
  * \brief Multiplication of two matrices with user-defined attributes of symmetry and sparsity
  */

#include <ctf.hpp>
#include <float.h>
#include <chrono>
#include <vector>

using namespace CTF;

/**
 * \brief (if test) tests and (if bench) benchmarks m*n*k matrix multiplication with matrices of specified symmetry and sparsity fraction
 * \param[in] m number of rows in C, A
 * \param[in] n number of cols in C, B
 * \param[in] k number of rows in A, cols in B
 * \param[in] dw set of processors on which to execute matmul
 * \param[in] sym_A in {NS, SY, AS, SH} symmetry attributes of A 
 * \param[in] sym_B in {NS, SY, AS, SH} symmetry attributes of B 
 * \param[in] sym_C in {NS, SY, AS, SH} symmetry attributes of C 
 * \param[in] sp_A fraction of nonzeros in A (if 1. A stored as dense)
 * \param[in] sp_B fraction of nonzeros in B (if 1. B stored as dense)
 * \param[in] sp_C fraction of nonzeros in C (if 1. C stored as dense)
 * \param[in] test whether to test
 * \param[in] bench whether to benchmark
 * \param[in] niter how many iterations to compute
 */
int matmul(int     m,
           int     n,
           int     k,
           World & dw,
           int     sym_A=NS, 
           int     sym_B=NS, 
           int     sym_C=NS, 
           double  sp_A=1.,
           double  sp_B=1.,
           double  sp_C=1.,
           bool    test=true,
           bool    bench=false,
           int     niter=10){
  int sA = sp_A < 1. ? SP : 0;
  int sB = sp_B < 1. ? SP : 0;
  int sC = sp_C < 1. ? SP : 0;

  /* initialize matrices with attributes */
  Matrix<> A(m, k, sym_A|sA, dw);
  Matrix<> B(k, n, sym_B|sB, dw);
  Matrix<> C(m, n, sym_C|sC, dw, "C");

  /* fill with random data */
  srand48(dw.rank);
  if (sp_A < 1.)
    A.fill_sp_random(0.0,1.0,sp_A);
  else
    A.fill_random(0.0,1.0);
  if (sp_B < 1.)
    B.fill_sp_random(0.0,1.0,sp_B);
  else
    B.fill_random(0.0,1.0);
  if (sp_C < 1.)
    C.fill_sp_random(0.0,1.0,sp_C);
  else
    C.fill_random(0.0,1.0);

  bool pass = true;
  bench = true;

  std::vector<long long> times;

  if (bench){
    if (dw.rank == 0){
      printf("Starting %d benchmarking iterations of matrix multiplication with specified attributes...\n", niter);
    }
    for (int i=0; i<niter; i++){
      if (sp_A < 1.)
        A.fill_sp_random(0.0,1.0,sp_A);
      else
        A.fill_random(0.0,1.0);
      if (sp_B < 1.)
        B.fill_sp_random(0.0,1.0,sp_B);
      else
        B.fill_random(0.0,1.0);
      if (sp_C < 1.)
        C.fill_sp_random(0.0,1.0,sp_C);
      else
        C.fill_random(0.0,1.0);

      MPI_Barrier(MPI_COMM_WORLD);
      auto start = std::chrono::steady_clock::now();
      C["ik"] = A["ij"]*B["jk"];
      MPI_Barrier(MPI_COMM_WORLD);
      auto end = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      times.push_back(duration);
    }

    if (dw.rank == 0){
        std::cout << "CYCLOPS TIMES [ms] = ";

      for (int i=0; i<niter; i++){
          std::cout << times[i] << " ";
      }
      std::cout << std::endl;
    }
  
  }

  return true;
} 


#ifndef TEST_SUITE
char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int rank, np, m, n, k, pass, niter, bench, sym_A, sym_B, sym_C, test;
  double sp_A, sp_B, sp_C;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m < 0) m = 17;
  } else m = 17;

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 32;
  } else n = 32;

  if (getCmdOption(input_str, input_str+in_num, "-k")){
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k < 0) k = 9;
  } else k = 9;

  if (getCmdOption(input_str, input_str+in_num, "-sym_A")){
    sym_A = atoi(getCmdOption(input_str, input_str+in_num, "-sym_A"));
    if (sym_A != AS && sym_A != SY && sym_A != SH) sym_A = NS;
  } else sym_A = NS;

  if (getCmdOption(input_str, input_str+in_num, "-sym_B")){
    sym_B = atoi(getCmdOption(input_str, input_str+in_num, "-sym_B"));
    if (sym_B != AS && sym_B != SY && sym_B != SH) sym_B = NS;
  } else sym_B = NS;

  if (getCmdOption(input_str, input_str+in_num, "-sym_C")){
    sym_C = atoi(getCmdOption(input_str, input_str+in_num, "-sym_C"));
    if (sym_C != AS && sym_C != SY && sym_C != SH) sym_C = NS;
  } else sym_C = NS;

  if (getCmdOption(input_str, input_str+in_num, "-sp_A")){
    sp_A = atof(getCmdOption(input_str, input_str+in_num, "-sp_A"));
    if (sp_A < 0.0 || sp_A > 1.0) sp_A = .2;
  } else sp_A = .2;

  if (getCmdOption(input_str, input_str+in_num, "-sp_B")){
    sp_B = atof(getCmdOption(input_str, input_str+in_num, "-sp_B"));
    if (sp_B < 0.0 || sp_B > 1.0) sp_B = .2;
  } else sp_B = .2;

  if (getCmdOption(input_str, input_str+in_num, "-sp_C")){
    sp_C = atof(getCmdOption(input_str, input_str+in_num, "-sp_C"));
    if (sp_C < 0.0 || sp_C > 1.0) sp_C = .2;
  } else sp_C = .2;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 10;
  } else niter = 10;

  if (getCmdOption(input_str, input_str+in_num, "-bench")){
    bench = atoi(getCmdOption(input_str, input_str+in_num, "-bench"));
    if (bench != 0 && bench != 1) bench = 1;
  } else bench = 1;

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Multiplying A (%d*%d sym %d sp %lf) and B (%d*%d sym %d sp %lf) into C (%d*%d sym %d sp %lf) \n",m,k,sym_A,sp_A,k,n,sym_B,sp_B,m,n,sym_C,sp_C);
    }
    pass = matmul(m, n, k, dw, sym_A, sym_B, sym_C, sp_A, sp_B, sp_C, false, true, niter);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
