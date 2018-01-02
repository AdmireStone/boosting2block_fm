# -*——coding:utf8-*-


import sys
if not '/home/zju/dgl/source/project/boosting2block_fm/utils/' in sys.path:
    sys.path.append('/home/zju/dgl/source/project/boosting2block_fm/')

from utils import data_path
from quadratic_slolver import  *
from linear_solver import  *

from  eval.auc import *



class Algorithm(object):
    'from_synthetic_data_csv.pkl'

    def __init__(self,train_data_file):

        datapath = data_path.ml_100k
        train_data_file = datapath + train_data_file
        self.X_ci,self.X_cj=self.load_data_file(train_data_file)

    def set_args(self):

        # 两区块，迭代次数
        self.total_iters = 20

        # 控制二次项迭代次数，基模型的个数
        self.maxiters_2 = 100

        self.reg_V = 0.001
        self.reg_linear = 0.001

        self.w_eta = 0.01
        self.w_epoc = 20
        self.batch_size_2 = 100 #求二次项权重

        self.batch_size_linear = 100 # 线性项
        self.linear_eat = 0.01
        self.linear_epoc = 10


    def load_data_file(self,train_data_file):
        '''
        从文件加载处理好的数据
        '''
        fi = open(train_data_file, 'rb')
        X_ci = pickle.load(fi)
        X_cj = pickle.load(fi)
        fi.close()
        X_ci = sp.csr_matrix(X_ci)
        X_cj = sp.csr_matrix(X_cj)
        return X_ci, X_cj




    def only_qudratic(self):
        '''
        只使用二次项
        :return:
        '''
        qs = Totally_Corr(self.maxiters_2,self.reg_V,self.w_eta,self.w_epoc,self.X_ci,self.X_cj,self.batch_size_2)
        qs.fit()
        return qs.getZ()


    def two_block_algortihm(self):
        '''
        :return:
        '''
        # exp(-Rou)
        quadratic_term = 1.
        for iter in range(self.total_iters):

            ls = Linear_Solver_logit(self.batch_size_linear,self.linear_epoc,
                                 self.X_ci,self.X_cj,quadratic_term,self.linear_epoc,self.linear_epoc)

            linear_weight = ls.fit()

            qs = Totally_Corr_with_linear(self.maxiters_2,self.reg_V,self.w_eta,self.w_epoc,self.X_ci,self.X_cj,
                                          self.batch_size_2,linear_weight)
            qs.fit()

            return (linear_weight,qs.getZ())


if __name__=='__main__':


    alg=Algorithm('from_synthetic_data_csv.pkl')
    alg.set_args()
    # alg.only_qudratic()
    alg.two_block_algortihm()


    # def load_data_file(train_data_file):
    #     '''
    #     从文件加载处理好的数据
    #     '''
    #     fi = open(train_data_file, 'rb')
    #     X_ci = pickle.load(fi)
    #     X_cj = pickle.load(fi)
    #     fi.close()
    #     X_ci = sp.csr_matrix(X_ci)
    #     X_cj = sp.csr_matrix(X_cj)
    #     return X_ci, X_cj
    #
    # datapath = data_path.ml_100k
    # # datapath = '/home/zju/dgl/source/project/boosting2block_fm/data/data_set/ml-100k/'
    # train_data_file = datapath + 'from_synthetic_data_csv.pkl'
    # X_ci, X_cj = load_data_file(train_data_file)
    #
    # maxiters = 100
    # reg_V = 0.001
    # w_eta = 0.01
    # w_epoc = 20
    # batch_size = 100
    #
    # start=time.time()
    # qs = Totally_Corr(maxiters,reg_V,w_eta,w_epoc,X_ci,X_cj,batch_size)
    # qs.fit()
    #
    # with open('model.pkl','wb') as fo:
    #     pickle.dump(qs,fo)
    # print "train_end! time:={0}".format(time.time()-start)