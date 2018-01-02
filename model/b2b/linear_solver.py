# -*-coding:utf8-*-

import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse.linalg import LinearOperator, eigsh,eigs
import time
import traceback
from quadratic_solver_matrix import *
import math


###########################Linear term update#####################################
class LinearSolver(object):

    def __init__(self,batch_size,epoc,p_data,n_data,quadratic_term,linear_reg_para,l_rate):
        '''
        :param batch_size: 每个批度的大小
        :param epoc: 迭代的周期
        :param p_data: 正样本，即X_uv或者X_ci. type：csr_matrix; shape:(n,dim);每一行一个样本
        :param n_data: 负样本，即X_uf或者X_cj. type：csr_matrix; shape:(n,dim);每一行一个样本
        :param quadratic_term: 二次项的参数，即Z
        :param linear_reg_para: 超参数：公式中的a_1,线性正则项参数
        :param l_rate:超参数，公式中的eat,梯度下降的学习率
        '''
        self.batch_size = batch_size
        self.epoc = epoc
        self.p_data = p_data      
        self.n_data = n_data
        self.quadratic_term = quadratic_term
        self.linear_reg_para = linear_reg_para
        self.l_rate = l_rate

                
    def getPAI(self,X_uv,X_uf,Z):
        '''
        :param X_uv the positive context-tag matrix
        :param X_uf the negative context-tag matrix
        :param Z the interaction matrix
        :return [exp(-<Z,x_uvx_uv^T-x_ufx_uf^T>)],the return should be one-dim array
        '''
        # 下面两个代码也许可以合并，只用和Z计算一次
        #UFZ=safe_sparse_dot(safe_sparse_dot(X_uf,Z),X_uf.T) # negative
        UFZ_dia=QuadraticSolver.getDia(Z,X_uf)

        #UVZ=safe_sparse_dot(safe_sparse_dot(X_uv,Z),X_uv.T) # positive
        UVZ_dia=QuadraticSolver.getDia(Z,X_uv)
        
        PAI=np.exp(UFZ_dia-UVZ_dia)
        if isinstance(PAI,np.matrixlib.defmatrix.matrix):
            return np.asarray(PAI)[0]
        return PAI

    def getA(self,PAI, B, W):
        '''
        The shape of A: 1 x N
        A=[pai_1exp(-W^Tb^(1)),...,pai_1exp(-W^Tb^(1))]

        :param PAI :  a row vector, just like the A
        :param B : a (n,d) matrix,where d is the dimension of input vector
        :param W : the weight of linear term.Note: W should be a colum vector
        '''

        ## 这里有点的问题，为什么第一遍没有发生异常，而第二遍发生了？
        ## 第一遍循环PAI的类型是'numpy.ndarray'，而第二遍PAI的类型为numpy.matrixlib.defmatrix.matrix
        ## 原因：计算PAI需要Z,第一遍boosting的时候，Z是空的，计算出来的PAI是ndarray,而第二遍的时候，Z非空
        try:
            #data = longfloat(np.exp(-safe_sparse_dot(B, W).data))
            data = np.exp(-safe_sparse_dot(B, W).todense())
            A = sp.csr_matrix(np.multiply(np.mat(PAI).T,data)).T
           
        except Exception:
            print '**********debug*************'
            print traceback.format_exc()
            exc_info=open('getA.pkl','wb')
            pickle.dump(B,exc_info)
            pickle.dump(W,exc_info)
            exc_info.close()
            print 'PAI:', PAI.shape, type(PAI)
            print 'B:', B.shape, type(B)
            print 'W:', W.shape, type(W)
            print 'data:', data.shape, type(data)
            raise Exception("抛出一个异常")
        return A

    def getB(self,X_uv,X_uf):
        '''
        B is a matrix,in which each row dnotes b_i.T
        where b_i=x_uv-x_uf
        so B is sparse
        B=[
        ------b^(1)^T-------
        ------b^(2)^T-------
        ....
        ------b^(n)^T-------
        ]
        :param X_uv the positive context-tag matrix
        :param X_uf the negative context-tag matrix
        '''
        return X_uv-X_uf

    def getLinerloss(self,batch_A,W,a_1,beta):
        regular_term=0.5*a_1*np.sum((W.data)**2)
        loss_term=np.log(np.sum((batch_A/(beta)).data))
        return loss_term+regular_term


    def updateW(self,X_uv, X_uf, epoc, Z, batch_size, a_1, eta):
        '''
        更新线性项，更新方法，梯度下降：
            1. SGD
            2. BGD
            3. MBGD
        目前先实现MBGD
        :param X_uv the complete positive context-tag matrix
        :param X_uf the complete negative context-tag matrix
        :param epoc 训练周期，超参数
        :param Z 二次项稀疏
        :param batch_size batch的大小，超参数
        :param a_1 超参数, 是线性参数正则项的权重
        :param eta ，超参数，更新的步长
        :return W  type: scipy.sparse.csc.csc_matrix
        '''
        #print 'eta 蛋疼126',type(eta)
        ## 统计loss
        final_loss = 0.

        total_samples = X_uv.shape[0]
        d_dim = X_uv.shape[1]
        ## 随机初始化
        # NOTE: in this way,the W is a colum vector naturally
        #W = sp.csr_matrix(np.array([1.]*d_dim).reshape(d_dim,1))
        W = sp.csr_matrix(np.array([0.]*d_dim).reshape(d_dim,1))
        p_start = 0
        p_end = batch_size
        total_batches = int(math.ceil((1.*total_samples) / batch_size))
        for ep in range(epoc):
            batch_count = 0
            start_time = time.time()
            p_start = 0
            p_end = batch_size
            while p_start < total_samples:
                if p_end > total_samples:
                    p_end = total_samples
                # print 'p_start:',p_start,'p_end:',p_end
                batch_uv = X_uv[p_start:p_end]
                batch_uf = X_uf[p_start:p_end]
                # batch_B is csr_matrix,(n,dim)
                batch_B = self.getB(batch_uv, batch_uf)
                batch_pai = self.getPAI(X_uv=batch_uv, X_uf=batch_uf, Z=Z)
                # batch_A is csr_matrix,(1,n)
                batch_A = self.getA(batch_pai, batch_B, W)
                sum_M = np.sum(batch_A)
                #           print 'get batch_A finished,total time:',(time.time()-start_time)/60,'min'
                # 负号不要忘记
                W = W - eta * (-safe_sparse_dot(batch_A, batch_B).T / (sum_M) + a_1 * W)
                #             print 'get W finished,total time:',(time.time()-start_time)/60,'min'
                batch_count += 1
                p_start = p_end
                p_end = p_end + batch_size
                eta=0.9*eta
                
                ########
                #print 'eta 蛋疼166',type(eta)
                #########
                
                #             if batch_count==(total_samples/batch_size)-1:
                #                 print 'epoc:',ep,' totoal batches:',(total_samples/batch_size) ,' current batch:',batch_count,'batch_count,loss:', getLinerloss(batch_A,W,a_1,1)
                #             if(batch_count==(total_samples/batch_size)-1):
                if batch_count == total_batches - 1 and ep == epoc - 1:
                    final_loss = self.getLinerloss(batch_A, W, a_1, 1)
                    #             if batch_count > 3:
                    #                 break;
                    #     print '(a_1,eta):',a_1,eta,' loss:',final_loss
        return W

    def fit(self):
        linear_weight=self.updateW(batch_size=self.batch_size,epoc=self.epoc,X_uv=self.p_data,X_uf=self.n_data,
                       Z=self.quadratic_term,a_1=self.linear_reg_para,eta=self.l_rate)
        return linear_weight
###########################Linear term update#####################################


class Linear_Solver_logit(LinearSolver):

    def __init__(self,batch_size,epoc,p_data,n_data,quadratic_term,linear_reg_para,l_rate):
        '''
        :param batch_size: 每个批度的大小
        :param epoc: 迭代的周期
        :param p_data: 正样本，即X_uv或者X_ci. type：csr_matrix; shape:(n,dim);每一行一个样本
        :param n_data: 负样本，即X_uf或者X_cj. type：csr_matrix; shape:(n,dim);每一行一个样本
        :param quadratic_term: 二次项的参数，即Z
        :param linear_reg_para: 超参数：公式中的a_1,线性正则项参数
        :param l_rate:超参数，公式中的eat,梯度下降的学习率
        '''
        self.batch_size = batch_size
        self.epoc = epoc
        self.p_data = p_data
        self.n_data = n_data
        self.quadratic_term = quadratic_term
        self.linear_reg_para = linear_reg_para
        self.l_rate = l_rate


    def updateW(self,X_uv, X_uf, epoc, Z, batch_size, a_1, eta):

        total_samples = X_uv.shape[0]
        d_dim = X_uv.shape[1]
        self.W = np.mat(np.array([0.] * d_dim).reshape(d_dim, 1))
        B = self.getB(self.p_data, self.n_data)


        total_batches = int(math.ceil((1. * total_samples) / batch_size))
        for ep in range(epoc):
            batch_count = 0
            start_time = time.time()
            p_start = 0
            p_end = batch_size
            while p_start < total_samples:
                if p_end > total_samples:
                    p_end = total_samples
                # print 'p_start:',p_start,'p_end:',p_end
                batch_uv = X_uv[p_start:p_end]
                batch_uf = X_uf[p_start:p_end]
                # batch_B is csr_matrix,(n,dim)
                batch_B = self.getB(batch_uv, batch_uf)
                batch_lambda = self.getPAI(X_uv=batch_uv, X_uf=batch_uf, Z=Z)

                batch_L = np.exp(-safe_sparse_dot(batch_B, self.W))
                batch_L = np.ravel(batch_L)
                batch_F = 1 + 1 / batch_lambda * batch_L

                assert batch_F.shape == batch_L.shape
                batch_F_1 = 1./batch_F

                self.W = self.W - eta * (-safe_sparse_dot(batch_B.T, batch_F_1.reshape([-1, 1])) + a_1 * self.W)

                batch_count += 1
                p_start = p_end
                p_end = p_end + batch_size
                eta = 0.9 * eta
        return self.W