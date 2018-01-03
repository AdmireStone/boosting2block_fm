#coding:utf8

import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse.linalg import LinearOperator, eigsh,eigs
import time
import traceback
import gc
import math
from  process.data_load import *
from utils.data_path import *
from  eval.auc import *
from sklearn.metrics.pairwise import check_paired_arrays,polynomial_kernel



class Quadratic_Solver(object):

    def __init__(self,maxiters,reg_V,w_eta,w_epoc,X_ci,X_cj,batch_size):
        self.mat_weight_list = []
        self.maxiters = maxiters
        self.eigenvec_list=[]
        self.reg_v=reg_V
        # 1-index
        self.current_j=0
        self.eta = w_eta
        self.w_epoc = w_epoc
        self.batch_size=batch_size
        self.sample_weight= self.init_u()
        self.X_ci = X_ci
        self.X_cj = X_cj
        self.eigsh_kwargs={'tol':0.001,'maxiter':5000}


    def predict_quadratic(self, X_ci, X_cj,P=None, lams=None):
        """Prediction from the quadratic term of the factorization machine.
        Returns <Z, XX'>.
        Note： the P should be a list,the elements of which are the eigenvectors p.ravel()
        """
        if P is None:
            P = self.eigenvec_list
            lams = np.array(self.mat_weight_list)

        if not len(lams):
            return 0

        K_ci = polynomial_kernel(X_ci, np.array(P), degree=2, gamma=1, coef0=0)
        K_cj = polynomial_kernel(X_cj, np.array(P), degree=2, gamma=1, coef0=0)
        return (np.dot(K_ci-K_cj, lams.T)).ravel()



    def calutlate_linear_exp_loss(self):
        '''

        :return: exp(-W^TX)
        '''
        W = np.array(self.linear_weight)
        X_ci, W = check_paired_arrays(self.X_cj, W)
        self.l_exp_loss = np.exp(-safe_sparse_dot(self.X_ci - self.X_cj, W.T, dense_output=True))
        self.l_exp_loss = self.l_exp_loss.ravel()
        print 'self.l_exp_loss:', type(self.l_exp_loss), self.l_exp_loss.shape

    def calutlate_redidual_matrix(self):
        '''

        :return:[[(v_c^Tx_r^+)-(v_c^Tx_r^-)]]_m_s, r=1,2,...,m ;c=1,2,...,s
        '''

        P = self.eigenvec_list
        lams = self.mat_weight_list

        K_ci = polynomial_kernel(self.X_ci, np.array(P), degree=2, gamma=1, coef0=0)
        K_cj = polynomial_kernel(self.X_cj, np.array(P), degree=2, gamma=1, coef0=0)
        self.psi = K_ci - K_cj



    def getEigenvector(self,A,**eigsh_kwargs):
        '''

        :param A: the matrix to be decomposed，A should be symmetric
        :param eigsh_kwargs: by default,{'tol'=0.1,'maxiter':5000}
        :return:
        '''
        '''
        :param A: 
        :return:
        '''
        if 'maxiter' not in self.eigsh_kwargs:
            self.eigsh_kwargs['maxiter'] = 5000
        if 'tol' not in self.eigsh_kwargs:
            self.eigsh_kwargs['tol'] = self.tol

        # todo: use power method
        eigenval, eigenvec = eigsh(A, k=1,**eigsh_kwargs)
        return eigenval, eigenvec

    def getComponentZ_eigval(self,U,X_uv, X_uf):
        '''
        :param s_dim: 输入向量的维度
        return: vv^T、eigenval
        '''
        U_arr = np.array(U).reshape(-1,1)
        P = X_uv.multiply(U_arr)
        Q = X_uf.multiply(U_arr)

        A = safe_sparse_dot(P.T, X_uv) - safe_sparse_dot(Q.T, X_uf)
        # print 'A.shape,type',A.shape,type(A)
        eigenval, eigenvec = self.getEigenvector(A)
        #     print eigenvec.shape
        return eigenvec.ravel(), eigenval

    def getDia(self,Z, X):
        '''
        :注意,这里的Z有问题  必须是 numpy.matrixlib.defmatrix.matrix'
        :return (XZX.T).diagonal()
        '''

        H_P_diag_syn = None
        batch_size = 10000
        total_sample = X.shape[0]
        if total_sample < batch_size:
            start = 0
            end = total_sample
            batch_numbers = 1
        else:
            start = 0
            end = batch_size
            batch_numbers = int(math.ceil((total_sample * 1.) / batch_size))

        batch_diag_list = []
        for i in range(batch_numbers):
            H_P_batch = safe_sparse_dot(safe_sparse_dot(X[start:end], Z), X[start:end].T)
            dia_batch = H_P_batch.diagonal()
            start = end
            end += batch_size
            if end > total_sample:
                end = total_sample

            batch_diag_list.append(dia_batch)

        H_P_diag_syn = np.concatenate((batch_diag_list))

        return H_P_diag_syn

    def getHj(self,Zj, P, Q):

        '''
        这里会遇到严重的内存问题，当8w条数据参与计算的时候，会产生(8W，8W)的矩阵，这是会导致内存溢出
        :return H matrix (N,1)
        '''
        H_P_dia = self.getDia(Zj, P)
        H_Q_dia = self.getDia(Zj, Q)

        H = H_P_dia - H_Q_dia

        rst = np.array(H)

        # H 为列向量
        return rst

class Totally_Corr(Quadratic_Solver):


    def __init__(self,maxiters,reg_V,w_eta,w_epoc,X_ci,X_cj,batch_size,linear_weight):
        self.mat_weight_list = []
        self.maxiters = maxiters
        self.eigenvec_list=[]
        self.reg_v=reg_V
        # 1-index
        self.current_j=0
        self.eta = w_eta
        self.w_epoc = w_epoc
        self.batch_size=batch_size
        self.sample_weight= self.init_u()
        self.X_ci = X_ci
        self.X_cj = X_cj
        self.eigsh_kwargs={'tol':0.001,'maxiter':5000}
        self.linear_weight = linear_weight





    def init_u(self):
        # datapath_bpr = data_path.ml_100k
        datapath_bpr = '/home/zju/dgl/source/project/boosting2block_fm/data/data_set/ml-100k/'
        train_file = datapath_bpr + 'ml_100k_occf_training.txt'
        test_file = datapath_bpr + 'ml_100k_occf_testing.txt'
        train_data, Tr, Tr_neg, Te = data_process(train_file, test_file)
        D0 = np.array([1.0 / float(Tr[u]['num']) for u, i in train_data])
        D = (D0 / np.sum(D0))*len(D0)
        return D


    def update_mat_weight(self):
        '''

        <H,W>
        current_j 1-index
        :param H:
        :return:
        '''
        # <H,W>,(n,1)

        self.calutlate_redidual_matrix()

        batch_count = 0
        p_start = 0
        p_end = self.batch_size
        total_samples = self.X_cj.shape[0]

        # shuffle
        idx=range(self.X_ci.shape[0])
        np.random.shuffle(idx)
        diorder_X_ci=self.X_ci[idx]
        diorder_X_cj = self.X_cj[idx]

        for epoc in range(self.w_epoc):
            while p_start < total_samples:
                if p_end > total_samples:
                    p_end = total_samples
                # print 'p_start:',p_start,'p_end:',p_end
                batch_P = diorder_X_ci[p_start:p_end]
                batch_Q = diorder_X_cj[p_start:p_end]

                # epsilon
                batch_l_exp_loss = self.l_exp_loss[p_start:p_end]
                # epsilon^-1
                batch_l_exp_loss_1=1./batch_l_exp_loss

                batch_psi = self.psi[p_start:p_end]

                lams = np.array(self.mat_weight_list)
                temp_qudratic=np.dot(batch_psi,lams.T)
                temp_qudratic.ravel()
                # 这里可能会出现 数据大小溢出，因为temp_qudratic可能有比较大的值
                temp_A = np.multiply(batch_l_exp_loss_1,np.exp(temp_qudratic))
                temp_A = 1./(1.+temp_A)

                deta = np.dot(temp_A,-batch_psi)
                deta = deta.ravel()
                deta = deta/(batch_P.shape[0])
                assert len(deta) == len(self.mat_weight_lis)
                self.mat_weight_list -= self.eta*(deta+self.reg_v)

                p_start = p_end
                p_end = p_end + self.batch_size

    def update_sample_weight(self,P,Q):

        quadratic = self.predict_quadratic(self.X_ci,self.X_cj)

        reciprocal = 1./self.l_exp_loss
        self.sample_weight = 1./(1 + np.multiply(reciprocal,np.exp(quadratic)))

    def getZ(self):
        Z=None
        for j in range(self.current_j):
            w = self.mat_weight_list[j]
            if Z is None:
                Z=w*self.z_list[j]
            else:
                Z+=w*self.z_list[j]
            # j+=1
        return Z

    def fit(self):

        for iter in range(self.maxiters):
            # 注意这里的sample_weight 每次要不要放大，第一次必须放大
            start=time.time()
            print '####iter-{0}######'.format(iter)
            self.current_j = iter+1

            print '####特征分解.....######'
            eigenvec, eigenval = self.getComponentZ_eigval(self.sample_weight,self.X_ci,self.X_cj)
            print 'eigenval={0}'.format(eigenval)
            if eigenval < self.reg_v:
                Z=self.getZ()
                auc = predic_auc(np.zeros(Z.shape[0]), Z)
                print "auc:", auc
                break

            self.mat_weight_list.append(0.)
            self.eigenvec_list.append(eigenvec)

            print '####更新模型权重.....######'
            time_u=time.time()
            self.update_mat_weight()
            print '耗时={0}'.format(time.time()-time_u)


            eigenvec_list = [p for p, lam in zip(self.eigenvec_list, self.mat_weight_list) if np.abs(lam) > 0]
            lams = [lam for lam in self.mat_weight_list if np.abs(lam) > 0]

            self.eigenvec_list = eigenvec_list
            self.mat_weight_list = lams

            print '####更新样本权重.....######'
            time_u = time.time()
            self.update_sample_weight(self.X_ci,self.X_cj)
            print '耗时={0}'.format(time.time() - time_u)

            if iter!=0 and iter%2 == 0:
                Z=self.getZ()
                auc = predic_auc(np.zeros(Z.shape[0]), Z)
                print "auc:", auc

            print 'len(W)={0}'.format(len(self.mat_weight_list))
            print self.mat_weight_list
            print '总共耗时={0} s'.format(time.time()-start)

    # def fit(self):





#
# class Totally_Corr_with_linear(Totally_Corr):
#
#
#     def __init__(self,maxiters,reg_V,w_eta,w_epoc,X_ci,X_cj,batch_size,linear_weight):
#         self.mat_weight_list = np.zeros(maxiters)
#         self.maxiters = maxiters
#         self.z_list=[]
#         self.reg_v=reg_V
#         # 1-index
#         self.current_j=0
#         self.eta = w_eta
#         self.w_epoc = w_epoc
#         self.batch_size=batch_size
#         self.sample_weight= self.init_u()
#         self.X_ci = X_ci
#         self.X_cj = X_cj
#         self.linear_weight = linear_weight
#
#     def get_linear_value(self,P,Q):
#         '''
#         return exp(W.Tx)
#         :param P:
#         :param Q:
#         :return:
#         '''
#         L=np.mat(self.linear_weight.reshape(-1,1))
#         return np.exp(safe_sparse_dot((P - Q), L))
#
#     def update_sample_weight(self,P,Q,linear_values):
#
#
#         # # Rou=
#         # start = time.time()
#         #  H=self.getH(P,Q)
#         # print 'H矩阵耗时:{0}'.format(time.time()-start)
#         # start = time.time()
#         Rou = self.get_Rou(P,Q)
#         start = time.time()
#         self.sample_weight = 1./(1 + np.multiply(linear_values,np.exp(Rou)))
#
#     def update_mat_weight(self):
#         '''
#         <H,W>
#         current_j 1-index
#         :param H:
#         :return:
#         '''
#         # <H,W>,(n,1)
#         batch_count = 0
#         p_start = 0
#         p_end = self.batch_size
#         total_samples = self.X_cj.shape[0]
#
#         # shuffle
#         idx=range(self.X_ci.shape[0])
#         np.random.shuffle(idx)
#         diorder_X_ci=self.X_ci[idx]
#         diorder_X_cj = self.X_cj[idx]
#
#         for epoc in range(self.w_epoc):
#             while p_start < total_samples:
#                 if p_end > total_samples:
#                     p_end = total_samples
#                 # print 'p_start:',p_start,'p_end:',p_end
#                 batch_P = diorder_X_ci[p_start:p_end]
#                 batch_Q = diorder_X_cj[p_start:p_end]
#                 H = self.getH(batch_P, batch_Q)
#                 Rou = self.get_Rou(batch_P,batch_Q)
#                 #todo 可以优化，在外面计算好，但是考虑数据乱序
#                 Beta = self.get_linear_value(batch_P, batch_Q)
#                 Beta_1 = 1. / Beta
#                 M = -1./(Beta_1 + np.exp(Rou))
#
#                 # temp_q=np.dot(M.T, H)
#                 # temp_b=self.reg_v + np.asarray(np.dot(M.T, H))[0]
#                 self.mat_weight_list[:self.current_j] -= self.eta * ((self.reg_v + np.asarray(np.dot(M.T, H))))[0]
#
#                 p_start = p_end
#                 p_end = p_end + self.batch_size
#
#
#
#
#     def fit(self):
#         linear_values=self.get_linear_value(self.X_ci,self.X_cj)
#         for iter in range(self.maxiters):
#             # 注意这里的sample_weight 每次要不要放大，第一次必须放大
#             start=time.time()
#             print '####iter-{0}######'.format(iter)
#             self.current_j = iter+1
#
#             print '####特征分解.....######'
#             z_t, eigenval = self.getComponentZ_eigval(self.sample_weight,self.X_ci,self.X_cj)
#             print 'eigenval={0}'.format(eigenval)
#             if eigenval < self.reg_v:
#                 Z=self.getZ()
#                 auc = predic_auc(np.zeros(Z.shape[0]), Z)
#                 print "auc:", auc
#                 break
#
#             self.z_list.append(z_t)
#
#             print '####更新模型权重.....######'
#             time_u=time.time()
#             self.update_mat_weight()
#             print '耗时={0}'.format(time.time()-time_u)
#
#             print '####更新样本权重.....######'
#             time_u = time.time()
#             self.update_sample_weight(self.X_ci,self.X_cj,linear_values=linear_values)
#             print '耗时={0}'.format(time.time() - time_u)
#
#             if iter!=0 and iter%10 == 0:
#                 Z=self.getZ()
#                 auc = predic_auc(self.linear_weight, Z)
#                 print "auc:", auc
#
#             print 'len(W)={0}'.format(len(self.mat_weight_list))
#             print self.mat_weight_list
#             print '总共耗时={0} s'.format(time.time()-start)