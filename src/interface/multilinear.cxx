#include "vector.h"
#include "timer.h"
#include "../mapping/mapping.h"

namespace CTF {
  template<typename dtype>
  void TTTP(Tensor<dtype> * T, int num_ops, int const * modes, Tensor<dtype> ** mat_list, bool aux_mode_first){
    Timer t_tttp("TTTP");
    t_tttp.start();
    int k = -1;
    bool is_vec = mat_list[0]->order == 1;
    if (!is_vec)
      k = mat_list[0]->lens[1-aux_mode_first];
    dtype ** arrs = (dtype**)malloc(sizeof(dtype*)*num_ops);
    int64_t * ldas = (int64_t*)malloc(num_ops*sizeof(int64_t));
    int * op_lens = (int*)malloc(num_ops*sizeof(int));
    int * phys_phase = (int*)malloc(T->order*sizeof(int));
    int * mat_strides = NULL;
    if (!is_vec)
      mat_strides = (int*)malloc(2*num_ops*sizeof(int));
    for (int i=0; i<T->order; i++){
      phys_phase[i] = T->edge_map[i].calc_phys_phase();
    }

    int64_t npair;
    Pair<dtype> * pairs;
    if (T->is_sparse){
      pairs = (Pair<dtype>*)T->data;
      npair = T->nnz_loc;
    } else
      T->get_local_pairs(&npair, &pairs, true, false);

    for (int i=0; i<num_ops; i++){
      //printf("i=%d/%d %d %d %d\n",i,num_ops,modes[i],mat_list[i]->lens[aux_mode_first], T->lens[modes[i]]);
      if (i>0) IASSERT(modes[i] > modes[i-1] && modes[i]<T->order);
      if (is_vec){
        IASSERT(mat_list[i]->order == 1);
      } else {
        IASSERT(mat_list[i]->order == 2);
        IASSERT(mat_list[i]->lens[1-aux_mode_first] == k);
        IASSERT(mat_list[i]->lens[aux_mode_first] == T->lens[modes[i]]);
      }
      int last_mode = 0;
      if (i>0) last_mode = modes[i-1];
      op_lens[i] = T->lens[modes[i]];///phys_phase[modes[i]];
      ldas[i] = 1;//phys_phase[modes[i]];
      for (int j=last_mode; j<modes[i]; j++){
        ldas[i] *= T->lens[j];
      }
/*      if (i>0){
        ldas[i] = ldas[i] / phys_phase[modes[i-1]];
      }*/
    }

    int64_t max_memuse = CTF_int::proc_bytes_available();
    int64_t tot_size = 0;
    int div = 1;
    if (is_vec){
      for (int i=0; i<num_ops; i++){
        tot_size += mat_list[i]->lens[0]/phys_phase[modes[i]];
      }
      if (tot_size*(int64_t)sizeof(dtype) > max_memuse){
        printf("CTF ERROR: insufficeint memory for TTTP");
      }
    } else {
      do {
        tot_size = 0;
        int kd = (k+div-1)/div;
        for (int i=0; i<num_ops; i++){
          tot_size += mat_list[i]->lens[aux_mode_first]*kd/phys_phase[modes[i]];
        }
        if (div > 1)
          tot_size += npair;
        if (T->wrld->rank == 0)
          printf("tot_size = %ld max_memuse = %ld\n", tot_size*(int64_t)sizeof(dtype), max_memuse);
        if (tot_size*(int64_t)sizeof(dtype) > max_memuse){
          if (div == k)
            printf("CTF ERROR: insufficeint memory for TTTP");
          else
            div = std::min(div*2, k);
        } else
          break;
      } while(true);
    }
    MPI_Allreduce(MPI_IN_PLACE, &div, 1, MPI_INT, MPI_MAX, T->wrld->comm);
    if (T->wrld->rank == 0)
      printf("In TTTP, chosen div is %d\n",div);
    dtype * acc_arr = NULL;
    if (!is_vec && div>1){
      acc_arr = (dtype*)T->sr->alloc(npair);
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int64_t i=0; i<npair; i++){
        acc_arr[i] = 1.;
      }
    } 
    Tensor<dtype> ** redist_mats = (Tensor<dtype>**)malloc(sizeof(Tensor<dtype>*)*num_ops);
    Partition par(T->topo->order, T->topo->lens);
    char * par_idx = (char*)malloc(sizeof(char)*T->topo->order);
    for (int i=0; i<T->topo->order; i++){
      par_idx[i] = 'a'+i+1;
    }
    char mat_idx[2];
    int slice_st[2];
    int slice_end[2];
    int k_start = 0;
    int kd = 0;
    for (int d=0; d<div; d++){
      k_start += kd;
      kd = k/div + (d < k%div);
      int k_end = k_start + kd;

      for (int i=0; i<num_ops; i++){
        int64_t size;
        if (phys_phase[modes[i]] == 1 && div==1){
          if (is_vec){
            size = mat_list[i]->lens[0];
          } else {
            size = mat_list[i]->lens[0]*mat_list[i]->lens[1];
            if (aux_mode_first){
              mat_strides[2*i+0] = k;
              mat_strides[2*i+1] = 1;
            } else {
              mat_strides[2*i+0] = 1;
              mat_strides[2*i+1] = mat_list[i]->lens[0];
            }
          }
          arrs[i] = (dtype*)T->sr->alloc(size);
          mat_list[i]->read_all(arrs[i], true);
          redist_mats[i] = NULL;
        } else {
          if (phys_phase[modes[i]] == 1){
            if (aux_mode_first){
              slice_st[0] = k_start;
              slice_st[1] = 0;
              slice_end[0] = k_end;
              slice_end[1] = T->lens[modes[i]];
              mat_strides[2*i+0] = kd;
              mat_strides[2*i+1] = 1;
            } else {
              slice_st[1] = k_start;
              slice_st[0] = 0;
              slice_end[1] = k_end;
              slice_end[0] = T->lens[modes[i]];
              mat_strides[2*i+0] = 1;
              mat_strides[2*i+1] = T->lens[modes[i]];
            }
            Matrix<dtype> mat = mat_list[i]->slice(slice_st, slice_end);
            arrs[i] = (dtype*)T->sr->alloc(T->lens[modes[i]]*kd);
            mat.read_all(arrs[i], true);
            redist_mats[i] = NULL;
          } else {
            int nrow, ncol;
            int topo_dim = T->edge_map[modes[i]].cdt;
            IASSERT(T->edge_map[modes[i]].type == CTF_int::PHYSICAL_MAP);
            IASSERT(!T->edge_map[modes[i]].has_child || T->edge_map[modes[i]].child->type != CTF_int::PHYSICAL_MAP);
            int comm_lda = 1;
            for (int l=0; l<topo_dim; l++){
              comm_lda *= T->topo->dim_comm[l].np;
            }
            CTF_int::CommData cmdt(T->wrld->rank-comm_lda*T->topo->dim_comm[topo_dim].rank,T->topo->dim_comm[topo_dim].rank,T->wrld->cdt);
            if (is_vec){
              Vector<dtype> * v = new Vector<dtype>(mat_list[i]->lens[0], par_idx[topo_dim], par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
              v->operator[]("i") += mat_list[i]->operator[]("i");
              redist_mats[i] = v;
              arrs[i] = (dtype*)v->data;
              cmdt.bcast(v->data,v->size,T->sr->mdtype(),0);
            } else {
              if (aux_mode_first){
                nrow = kd;
                ncol = T->lens[modes[i]];
                mat_idx[0] = 'a';
                mat_idx[1] = par_idx[topo_dim];
              } else {
                nrow = T->lens[modes[i]];
                ncol = kd;
                mat_idx[0] = par_idx[topo_dim];
                mat_idx[1] = 'a';
              }
              Matrix<dtype> * m = new Matrix<dtype>(nrow, ncol, mat_idx, par[par_idx], Idx_Partition(), 0, *T->wrld, *T->sr);
              m->operator[]("ij") += mat_list[i]->operator[]("ij");
              redist_mats[i] = m;
              arrs[i] = (dtype*)m->data;

              cmdt.bcast(m->data,m->size,T->sr->mdtype(),0);
              if (aux_mode_first){
                mat_strides[2*i+0] = kd;
                mat_strides[2*i+1] = 1;
              } else {
                mat_strides[2*i+0] = 1;
                mat_strides[2*i+1] = m->pad_edge_len[0]/phys_phase[modes[i]];
              }
            }
          }
        }
      }
    if (T->wrld->rank == 0)
      printf("Completed redistribution in TTTP\n");
  #ifdef _OPENMP
      #pragma omp parallel
  #endif
      {
        if (is_vec){
  #ifdef _OPENMP
          #pragma omp for
  #endif
          for (int64_t i=0; i<npair; i++){
            int64_t key = pairs[i].k;
            for (int j=0; j<num_ops; j++){
              //printf("i=%ld, j=%d\n",i,j);
              key = key/ldas[j];
              //FIXME: handle general semiring
              pairs[i].d *= arrs[j][(key%op_lens[j])/phys_phase[modes[j]]];
            }
          }
        } else {
          int * inds = (int*)malloc(num_ops*sizeof(int));
  #ifdef _OPENMP
          #pragma omp for
  #endif
          for (int64_t i=0; i<npair; i++){
            int64_t key = pairs[i].k;
            for (int j=0; j<num_ops; j++){
              key = key/ldas[j];
              inds[j] = (key%op_lens[j])/phys_phase[j];
            }
            dtype acc = 0;
            for (int kk=0; kk<k; kk++){
              dtype a = arrs[0][inds[0]*mat_strides[0]+kk*mat_strides[1]];
              for (int j=1; j<num_ops; j++){
                a *= arrs[j][inds[j]*mat_strides[2*j]+kk*mat_strides[2*j+1]];
              }
              acc += a;
            }
            if (acc_arr == NULL)
              pairs[i].d *= acc;
            else
              acc_arr[i] += acc;
          }
          free(inds);
        }
      }
      for (int j=0; j<num_ops; j++){
        if (redist_mats[j] != NULL){
          if (redist_mats[j]->data != (char*)arrs[j])
            T->sr->dealloc((char*)arrs[j]);
          delete redist_mats[j];
        } else
          T->sr->dealloc((char*)arrs[j]);
      }
    }
    if (acc_arr != NULL){
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int64_t i=0; i<npair; i++){
        pairs[i].d *= acc_arr[i];
      }
      T->sr->dealloc((char*)acc_arr);
    }

    if (!T->is_sparse){
      T->write(npair, pairs);
      T->sr->pair_dealloc((char*)pairs);
    }
    if (T->wrld->rank == 0)
      printf("Completed TTTP\n");
    free(redist_mats);
    if (mat_strides != NULL) free(mat_strides);
    free(phys_phase);
    free(ldas);
    free(op_lens);
    free(arrs);
    t_tttp.stop();
    
  }
}