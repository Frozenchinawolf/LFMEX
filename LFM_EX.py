from collections import defaultdict
import networkx as nx
import numpy as np

'''
    基于LFM的改进版
'''

class CommunityEX():
    ''' 代表一个局部小社区 '''

    def __init__(self, G, theta):
        self._G = nx.freeze(G)
        self._theta = theta
        self._nodes = set()
        self._k_in = 0
        self._k_out = 0
        self._log = []
        self._fitness_list = []

    def get_log(self):
        return self._log

    def get_fitness(self):
        return self._fitness_list

    def get_neighbors(self):
        neighbors = set()
        for node in self._nodes:
            neighbors.update(set(self._G.neighbors(node)) - self._nodes)
        return neighbors

    def add_node(self, node):
        neighbors = set(self._G.neighbors(node))
        # 邻居中在当前社团里的节点
        node_k_in = len(neighbors & self._nodes)
        # 邻居中不再当前社团里的节点，也就是边缘节点
        node_k_out = len(neighbors) - node_k_in
        # 把一个节点加入社区
        self._nodes.add(node)
        # 社团里的度要增加2倍的node_k_in
        self._k_in += 2*node_k_in
        # 社团边缘的外度要加上边缘节点再减去内里节点
        self._k_out = self._k_out+node_k_out-node_k_in
        # 记录操作
        add_log = "+ {}".format(node)
        self._log.append(add_log)
        self._fitness_list.append(1.0*self._k_in/(self._k_in+self._k_out))

    def whether_add(self, node, node2comms):
        neighbors = set(self._G.neighbors(node))
        old_k_in = self._k_in
        old_k_out = self._k_out
        node_k_in = len(neighbors & self._nodes)
        node_k_out = len(neighbors) - node_k_in
        new_k_in = old_k_in + 2*node_k_in
        new_k_out = old_k_out + node_k_out-node_k_in

        theta = self._theta

        new_fitness = 1.0*new_k_in/(new_k_in+new_k_out)
        old_fitness = 1.0*old_k_in/(old_k_in+old_k_out)

        
        if new_fitness-old_fitness > theta:
            degree_u = len(neighbors)
            temp=[len(node2comms[node])for node in neighbors]
            # print(existed_comms_set)
            # 代表和待选社区的联系
            x = node_k_in
            # 代表和其他已经存在社区的联系
            z = np.sum(temp)
            # 代表还剩余的联系
            y = degree_u-z-x
            # 说明node周围已经全是分配过社区的节点了
            if y < 0:
                return None

            num_nodes = len(self._nodes)
            max_d_k_in=max(degree_u,old_k_in)
            beta = np.exp(-num_nodes)
            alpha = 1.0/(np.sqrt(num_nodes)+1)
            score = 1.0*(x/max_d_k_in)*alpha+1.0*(y/degree_u) * \
                beta-1.0*(z/degree_u)*(1-alpha-beta)
            return score
        else:
            return None

class UnionFind():
    """ 专门用来合并社区的，维持等价社区的类
    """
    def __init__(self, G,comms):
        self.G=nx.freeze(G)
        self.index = [i for i, c in enumerate(comms)]
        self.size = [1 for i, c in enumerate(comms)]
        self.expaned_comms = {i: set(c) for i, c in enumerate(comms)}
        self.degree_comms={i:self.eval_comm_degree(c) for i,c in enumerate(comms)}

    def find(self, comm):
        # 路径压缩
        while comm != self.index[comm]:
            # 把当前节点往上提一层
            self.index[comm] = self.index[self.index[comm]]
            comm = self.index[comm]
        return comm

    def union(self, c1, c2):
        root1 = self.find(c1)
        root2 = self.find(c2)
        if root1 != root2:
            # 把小树挂到大树上去
            if self.size[root1] < self.size[root2]:
                self.index[root1] = root2
                self.expaned_comms[root2].update(self.expaned_comms[root1])
                self.expaned_comms.pop(root1)
                self.degree_comms[root2]+=self.degree_comms[root1]
                self.degree_comms.pop(root1)
                self.size[root2] = self.size[root2]+self.size[root1]
                return root2
            else:
                self.index[root2] = root1
                self.expaned_comms[root1].update(self.expaned_comms[root2])
                self.expaned_comms.pop(root2)
                self.degree_comms[root1]+=self.degree_comms[root2]
                self.degree_comms.pop(root2)
                self.size[root1] = self.size[root2]+self.size[root1]
                return root1
        else:
            return root1

    def get_partition(self):
        return self.expaned_comms

    def get_degree_distri(self):
        return self.degree_comms

    def get_component(self):
        tree = defaultdict(lambda: set())
        for i in range(len(self.index)):
            root = self.find(i)
            tree[root].add(i)
        return tree

    def eval_comm_degree(self,comm):
        degree_list=[ self.G.degree[node] for node in comm]
        degree_sum=np.sum(degree_list)
        return degree_sum

    def __repr__(self):
        return self.get_component().__repr__()

class LFM_EX():
    """
    不改变原图结构
    """

    def __init__(self, G, theta,eps, max_comms=100):
        self._G = nx.freeze(G)
        self._theta = theta
        self._max_comms = max_comms
        self._log = []
        self._fitness_list = []

        self._eps=eps

    def get_log(self):
        return self._log

    def get_fitness(self):
        return self._fitness_list

    def execute(self):
        communities = []
        # 没有分配到社区的节点
        node_not_include = list(self._G.nodes)
        node2comms=defaultdict(lambda:set())
        count = 0
        
        while(len(node_not_include) != 0):
            c = CommunityEX(self._G, self._theta)

            seed = np.random.choice(node_not_include)

            # print('seed:{}'.format(seed))
            # 新行成一个社区
            c.add_node(seed)
            # 要检查这个社区外围的一阶邻居节点
            to_be_examined = c.get_neighbors()
            while(to_be_examined):
                # largest fitness to be added
                m = {}
                for node in to_be_examined:
                    is_add = c.whether_add(node, node2comms)
                    m[node] = is_add
                m = {k: v for k, v in m.items() if v is not None}
                # fitness增量全小于等于0，停止
                if len(m) == 0:
                    break
                to_be_add = sorted(
                    m.items(), key=lambda x: x[1], reverse=True)[0]
                c.add_node(to_be_add[0])

                to_be_examined = c.get_neighbors()

            # 把已经分配过社区的节点从未分配列表中去掉
            for node in c._nodes:
                node2comms[node].add(count)
                if(node in node_not_include):
                    node_not_include.remove(node)
            communities.append(list(c._nodes))
            self._log.append(c.get_log())
            self._fitness_list.append(c.get_fitness())
            count += 1
            if count == self._max_comms:
                if len(node_not_include) == 0:
                    pass
                else:
                    communities.append(node_not_include)
                    self._log.append(['no action'])
                    print('LFM_EX allocate not finished')
                break
        # # 让社区递增排序
        # comms_sorted=[sorted(map(int, comm)) for comm in communities]
        return communities

    def merge(self,max_iter=100):
        m=len(self._G.edges)
        comms = self.execute()
        uf=UnionFind(self._G,comms)
        adj_comms=LFM_EX.get_adj_comms(self._G,comms,uf)
        done=False
        count=0
        while not done and count<max_iter:
            # 当下面没有发生合并操作，也就意味着无社区可并，此时退出
            done=True
            # 建立社区之间的模块度增长合并关系
            merge_comms_single_arrow={}
            partition=uf.get_partition()
            degree_distri=uf.get_degree_distri()
            for root_i in partition.keys():
                neigh_dict=adj_comms[root_i]
                # j 都是根节点
                max_sim=0
                max_root_j=None
                for root_j,num_edge in neigh_dict.items():
                    d_i=degree_distri[root_i]
                    d_j=degree_distri[root_j]
                    sim=(num_edge-d_i*d_j/(2*m))/min(d_i,d_j)
                    if sim>=1:
                        print('sim not little than 1 equal:{:.4f}'.format(sim))
                        sim=1
                    # sim 相似度至少大于0
                    if sim>max_sim:
                        max_sim=sim
                        max_root_j=root_j
                # sim 相似度大于eps才是有效的相似社区
                if (max_root_j is not None) and max_sim>self._eps:
                    merge_comms_single_arrow[root_i]=max_root_j
            
            # 寻找双向依赖的两个社区进行合并
            log_key=set()
            merge_comms_biarrow={}
            for k,v in merge_comms_single_arrow.items():
                if (not v in log_key) and merge_comms_single_arrow[v]==k:
                    merge_comms_biarrow[k]=v
                    # 记录合并过的键值，防止重复合并
                    log_key.add(k)
            for w,u in merge_comms_biarrow.items():
                # 在合并的过程中，原来的社区号由于合并可能变得不是根社区
                root_w=uf.find(w)
                root_u=uf.find(u)
                # 如果已经在一个社区，则不用合并
                if root_w!=root_u:
                    self.merge_pair(uf,adj_comms,root_w,root_u)
                    done=False
            
            # for k,v in merge_comms_single_arrow.items():
            #     root_k=uf.find(k)
            #     root_v=uf.find(v)
            #     if root_k!=root_v:
            #         num_edge=adj_comms[root_k][root_v]
            #         d_k=degree_distri[root_k]
            #         d_v=degree_distri[root_v]
            #         sim=(num_edge-d_k*d_v/(2*m))/(2*m)
            #         if sim>self._eps:
            #             self.merge_pair(uf,adj_comms,root_k,root_v)
            #             done=False

            count+=1
        if count==max_iter:
            if done==False:
                print('it has reached max iter,so exit and merge has not finished!')

        result_comms=[list(v) for k,v in uf.get_partition().items()]
        return result_comms

    def merge_pair(self,uf,adj_comms,c1,c2):
        """ 合并两个社区的同时，修改社区的邻接关系
        """
        component=uf.get_component()
        if not( c1 in component and c2 in component):
            raise RuntimeError('c1 and c2 are not root community')

        root=uf.union(c1,c2)
        adj_comms[c1].pop(c2)
        adj_comms[c2].pop(c1)
        c1_keys=set(adj_comms[c1].keys())
        c2_keys=set(adj_comms[c2].keys())
        inter=c1_keys&c2_keys
        if root==c1:
            outer=c2_keys-inter
            # i是共有的，所以i有c1,c2
            for i in inter:
                adj_comms[c1][i]=adj_comms[c1][i]+adj_comms[c2][i]
                adj_comms[i][c1]=adj_comms[c1][i]
                adj_comms[i].pop(c2)
            # j是c2独有的，所以j有c2,但没有c1
            for j in outer:
                adj_comms[c1][j]=adj_comms[c2][j]
                adj_comms[j][c1]=adj_comms[c1][j]
                adj_comms[j].pop(c2)
            adj_comms.pop(c2)
        else:
            outer=c1_keys-inter
            for i in inter:
                adj_comms[c2][i]=adj_comms[c1][i]+adj_comms[c2][i]
                adj_comms[i][c2]=adj_comms[c2][i]
                adj_comms[i].pop(c1)
            for j in outer:
                adj_comms[c2][j]=adj_comms[c1][j]
                adj_comms[j][c2]=adj_comms[c2][j]
                adj_comms[j].pop(c1)
            adj_comms.pop(c1)
        
        return root
    # 从这里开始是类方法
    def is_overlapping(comms):
        """ 判断社区划分是否有重叠部分
        """
        vertex_community = defaultdict(lambda: set())
        for i, c in enumerate(comms):
            for v in c:
                vertex_community[v].add(i)

        for k, v in vertex_community.items():
            if len(v) != 1:
                return True
        return False
    
    def get_adj_comms(G,comms,uf):
        """ 可以根据comms，和uf的状态构建出现在comms的邻接关系
        """
        # 节点到社区的映射
        node2comms=defaultdict(lambda: set())
        for i,comm in enumerate(comms):
            for v in comm:
                node2comms[v].add(i)
        # print(node2comms)
        # 建立社区之间的邻接关系
        adj_comms=defaultdict(lambda:defaultdict(lambda:0))
        for edge in G.edges:
            u,v=edge
            for c_u in node2comms[u]:
                for c_v in node2comms[v]:
                    root_u=uf.find(c_u)
                    root_v=uf.find(c_v)
                    # 一条边连接两个不同的组成，才要统计
                    if root_u!=root_v:
                        adj_comms[root_u][root_v]+=1
                        adj_comms[root_v][root_u]+=1
        # print_adj_comms(adj_comms)
        return adj_comms

if __name__ == "__main__":
    graph=nx.karate_club_graph()
    algorithm=LFM_EX(graph,0.01,0)
    comms=algorithm.merge()
    print(comms)