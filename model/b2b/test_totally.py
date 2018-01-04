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
        self.total_iters = 10

        # 控制二次项迭代次数，基模型的个数
        self.maxiters_2 = 5

        self.reg_V = 10
        self.reg_linear = 0.002

        self.w_eta = 0.001
        self.w_epoc = 20
        self.batch_size_2 = 100 #求二次项权重时候，SGD要用

        self.batch_size_linear = 100 # 线性项
        self.linear_eat = 0.00001
        self.linear_epoc = 10

    def reset_quadratic_args(self,reg_v,componets):
        self.reg_V = reg_v
        self.maxiters_2 = componets

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

    def predict(self,linear_weight,quadratic_solver_instance):

        sub = self.X_ci - self.X_cj

        linear_term = safe_sparse_dot(sub, linear_weight)

        linear_term = np.ravel(linear_term)

        # quadratic_term = quadratic_solver_instance.getHj(Z, self.X_ci, self.X_cj)

        quadratic_term = quadratic_solver_instance.predict_quadratic(self.X_ci,self.X_cj)

        # print 'predict:linear_term.shape',linear_term.shape
        # print 'predict:quadratic_term.shape', quadratic_term.shape
        assert linear_term.shape == quadratic_term.shape

        predicts = linear_term + quadratic_term

        return predicts

    def loss(self,linear_weight,Z,element_z_weitghts,quadratic_solver_instance):
        '''

        :param linear_weight:  列向量 [n,1]
        :param Z: 合成的Z,e.q. sum(w_jz_j)
        :param element_z_weitghts: list
        :return:
        '''
        regular = 0.5*self.reg_linear*np.dot(linear_weight.T,linear_weight) + self.reg_V*np.sum(element_z_weitghts)

        predicts = self.predict(linear_weight,quadratic_solver_instance)

        ## objective function loss

        a = 1+np.exp(-predicts)
        b = np.log(a)
        loss = np.sum(b)+regular

        return  loss


    def loss_simple(self,linear_weight,Z,element_z_weitghts,quadratic_solver_instance):
        '''

        :param linear_weight:  列向量 [n,1]
        :param Z: 合成的Z,e.q. sum(w_jz_j)
        :param element_z_weitghts: list
        :return:
        '''
        regular = 0.5*self.reg_linear*np.dot(linear_weight.T,linear_weight) + self.reg_V*np.sum(element_z_weitghts)

        predicts = self.predict(linear_weight,quadratic_solver_instance)
        # print np.sum(predicts > 0.)
        return  np.sum(predicts<0.)*1./len(predicts)


    def two_block_algortihm(self,isfit_linear=True):
        '''
        :return:
        '''
        # exp(-Rou)
        Z = np.zeros((self.X_cj.shape[1],self.X_cj.shape[1]))
        quadratic_predicts=np.zeros(self.X_ci.shape[0])
        linear_weight = np.zeros(self.X_cj.shape[1])
        for iter in range(self.total_iters):

            # ls = Linear_Solver_logit(self.batch_size_linear,self.linear_epoc,
            #                      self.X_ci,self.X_cj,quadratic_term,self.linear_epoc,self.linear_epoc)

            if isfit_linear:
                ls = Linear_Solver_logit(self.batch_size_linear, self.linear_epoc, self.X_ci,
                                         self.X_cj,quadratic_predicts, self.reg_linear, self.linear_eat)
                linear_weight = ls.fit()


            qs = Totally_Corr(self.maxiters_2,self.reg_V,self.w_eta,self.w_epoc,self.X_ci,self.X_cj,
                                          self.batch_size_2,linear_weight.ravel())
            qs.fit()

            quadratic_predicts = qs.predict_quadratic(self.X_ci,self.X_cj)
            # Z = qs.getZ()
            print 'epoc={0},loss={1}'.format(iter,self.loss_simple(linear_weight,Z,qs.mat_weight_list,qs))

        # return (linear_weight,Z)


if __name__=='__main__':



    # alg.only_qudratic()
    # [100,50,10,5,1,0.1,0.01]
    # for reg_v in [0.006,0.003,0.001,0.0006,0.0003,0.0001]:
    #     cmp_args_list = [2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 50]
    #     cmp_args_list.reverse()
    #     for componets in cmp_args_list:
    #         print '********************reg_v={0},rank={1}**************'.format(reg_v,componets)
    #         alg = Algorithm('from_synthetic_data_csv.pkl')
    #         alg.set_args()
    #         alg.reset_quadratic_args(reg_v,componets)
    #         alg.two_block_algortihm(isfit_linear=True)



    for reg_v in [0.0006,0.0003,0.0001]:
        cmp_args_list = [2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 50]
        cmp_args_list.reverse()
        for componets in cmp_args_list:
            print '********************reg_v={0},rank={1}**************'.format(reg_v,componets)
            alg = Algorithm('from_synthetic_data_csv.pkl')
            alg.set_args()
            alg.reset_quadratic_args(reg_v,componets)
            alg.two_block_algortihm(isfit_linear=False)


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