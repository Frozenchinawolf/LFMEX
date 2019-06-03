import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import numpy as np
import sys
sys.path.append('..')
import util


class Community():
    ''' use set operation to optimize calculation '''
    def __init__(self,G,alpha=1.0):
        self._G = G
        self._alpha = alpha
        self._nodes = set()
        self._k_in = 0
        self._k_out = 0
        self._log=[]

    def get_log(self):
        return self._log

    def add_node(self,node):
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
        add_log="+ {}".format(node)
        self._log.append(add_log)

        
    def remove_node(self,node):
        neighbors = set(self._G.neighbors(node))
        node_k_in = len(neighbors &  self._nodes)
        node_k_out = len(neighbors) - node_k_in
        self._nodes.remove(node)
        self._k_in -= 2*node_k_in
        self._k_out = self._k_out - node_k_out+node_k_in
        # 记录操作
        remove_log="- {}".format(node)
        self._log.append(remove_log)

    def cal_add_fitness(self,node):
        neighbors = set(self._G.neighbors(node))
        old_k_in = self._k_in
        old_k_out = self._k_out
        vertex_k_in = len(neighbors & self._nodes)
        vertex_k_out = len(neighbors) - vertex_k_in 
        new_k_in = old_k_in + 2*vertex_k_in
        new_k_out = old_k_out + vertex_k_out-vertex_k_in
        new_fitness = new_k_in/(new_k_in+new_k_out)**self._alpha
        old_fitness = old_k_in/(old_k_in+old_k_out)**self._alpha
        return new_fitness-old_fitness
    
    def cal_remove_fitness(self,node):
        neighbors = set(self._G.neighbors(node))
        old_k_in = self._k_in
        old_k_out = self._k_out
        vertex_k_in = len(neighbors & self._nodes)
        vertex_k_out = len(neighbors) - vertex_k_in
        new_k_in = old_k_in - 2*vertex_k_in
        new_k_out = old_k_out - vertex_k_out + vertex_k_in
        old_fitness = old_k_in/(old_k_in+old_k_out)**self._alpha 
        new_fitness = new_k_in/(new_k_in+new_k_out)**self._alpha
        return old_fitness-new_fitness
    # 从社区里面剔除一些不好的点
    def recalculate(self):
        for vid in self._nodes:
            fitness = self.cal_remove_fitness(vid)
            # 如果适应性函数小于0，则返回这个节点，之后要删除
            if fitness < 0.0:
                return vid
        return None
    # 返回社区周围的一阶邻居节点
    def get_neighbors(self):
        neighbors = set()
        for node in self._nodes:
            neighbors.update(set(self._G.neighbors(node)) - self._nodes)
        return neighbors
    # 整个社区的适应性
    def get_fitness(self):
        return float(self._k_in)/((self._k_in+self._k_out) ** self._alpha)

class LFM():
    
    def __init__(self, G, alpha,max_iter):
        self._G = G
        self._alpha = alpha
        self._log=[]
        self._max_iter=max_iter

    def get_log(self):
        return self._log

    def execute(self):
        communities = []
        # 没有分配到社区的节点
        node_not_include = list(self._G.nodes)
        count = 0
        while(len(node_not_include) != 0):
            c = Community(self._G, self._alpha)
            # randomly select a seed node
            seed = random.choice(node_not_include)
            # print('seed:{}'.format(seed))
            # 新行成一个社区
            c.add_node(seed)
            # 要检查这个社区外围的一阶邻居节点
            to_be_examined = c.get_neighbors()
            while(to_be_examined):
                #largest fitness to be added
                m = {}
                for node in to_be_examined:
                    fitness = c.cal_add_fitness(node)
                    m[node] = fitness
                to_be_add = sorted(m.items(),key=lambda x:x[1],reverse = True)[0]
                 
                #stop condition
                # 周围社区节点的适应性都小于0，所以这个社区不能再增大了，退出
                if(to_be_add[1] < 0.0):
                    break
                c.add_node(to_be_add[0])
                
                # 计算要剔除的节点
                # 有加入就要有删除
                to_be_remove = c.recalculate()
                while(to_be_remove != None):
                    c.remove_node(to_be_remove)
                    to_be_remove = c.recalculate()
                    
                to_be_examined = c.get_neighbors()
            # 把加入社区的节点从未分配列表中删除                      
            for node in c._nodes:
                if(node in node_not_include):
                    node_not_include.remove(node)
            communities.append(c._nodes)
            self._log.append(c.get_log())
            count += 1
            if count == self._max_iter:
                if len(node_not_include) == 0:
                    pass
                else:
                    communities.append(set(node_not_include))
                    self._log.append(['no action'])
                    print('LFM allocate not finished')
                break
        return communities

def draw_comm_process(graph,layout,fig_size,colors,comm_log,path):
    """ 画出单独社区生成拓展的过程 """
    files=os.listdir(path)
    # 保存图片之前，看看是否为空，不然的话会和上次的图片混在一起
    if len(files)==0:
        pass
    else:
        for file in files:
            os.remove(os.path.join(path,file))
    # 开始遍历每个社区和对应的操作
    for i,(comm,log) in enumerate(comm_log):
        # 检查生成这个社区的操作和这个社区能不能对的上
        if check(comm,log):
            # 用来保存要画的节点
            nodes_to_draw=[]
            for j,action in enumerate(log):
                act,node=action.split(' ')
                if act=='+':
                    nodes_to_draw.append(node)
                else:
                    nodes_to_draw.remove(node)
                file_name='{}_{}.png'.format(i,j)
                file_path=os.path.join(path,file_name)
                # 如果社区数目和颜色数目对不上，这里数组会溢出
                real_draw_nodes(G,layout,fig_size,nodes_to_draw,file_path,color[i])
        # 直接出错     
        else:
            raise RuntimeError('log behaviour error!')

def draw_comm_distri(graph,layout,fig_size,colors,communities,path):
    """ 画出最后的社区分布图，重叠的节点用单独的颜色表示 """
    if len(colors) <= len(communities):
        raise RuntimeError('颜色不足，每个社区都标注颜色')
    # 得到每个节点对应的社区
    vertex_community = defaultdict(lambda:list())
    for i,c in enumerate(communities):
        for v in c:
            vertex_community[v].append(i)
    # 按照节点的社区，划分对应的颜色
    color_node=[color[-1] if len(c_list)>1 else color[c_list[0]] for v, c_list in vertex_community.items()]
    # print(colors)
    # 得到节点的列表
    node_list=[ v for v in vertex_community.keys()]
    files=os.listdir(path)
    file_name='whole.png'
    file_path=os.path.join(path,file_name)
    real_draw_nodes(G,layout,fig_size,node_list,file_path,color_node)

def real_draw_nodes(G,layout,fig_size,nodes_to_draw,path,color):
    """ 真正画图的函数 """
    plt.figure()
    plt.axis('off')
    plt.xlim(tuple(map(lambda x: x*1.2,fig_size[0])))
    plt.ylim(tuple(map(lambda x: x*1.2,fig_size[1])))
    nx.draw_networkx(G,layout,nodelist=nodes_to_draw,node_color=color)
    plt.savefig(path)

def cal_size(layout):
    """ 计算包含网络图的最小长方形 """
    x_value=[x for x,y in layout.values()]
    y_value=[y for x,y in layout.values()]
    x_min,x_max=np.min(x_value),np.max(x_value)
    y_min,y_max=np.min(y_value),np.max(y_value)
    return ((x_min,x_max),(y_min,y_max))

def check(comm,log):
    """ 检查社区和对应的生成操作是否匹配 """
    len_comm=len(comm)
    count_add=0
    count_remove=0
    for i in log:
        if i[0]=='+':
            count_add+=1
        elif i[0]=='-':
            count_remove+=1
        else:
            raise RuntimeError('log error')
    if (count_add-count_remove)==len_comm:
        return True
    else:
        return False

def draw_karate_club_real_community(graph,layout,fig_size,path):
    """ 画出空手道俱乐部的真实社区 """
    Hi_com=[vertex  for vertex,club in graph.nodes(data='club') if club=='Mr. Hi']
    Officer_com=[vertex  for vertex,club in graph.nodes(data='club') if club=='Officer']

    plt.figure()
    plt.axis('off')
    plt.xlim(tuple(map(lambda x: x*1.2,fig_size[0])))
    plt.ylim(tuple(map(lambda x: x*1.2,fig_size[1])))
    nx.draw_networkx(graph,layout,nodelist=Hi_com,node_color='red',node_shape='o')
    nx.draw_networkx(graph,layout,nodelist=Officer_com,node_color='green',node_shape='s')
    file_path=os.path.join(path,'real_comm.png')
    plt.savefig(file_path)
    return [Hi_com,Officer_com]

if(__name__ == "__main__"):
    # G = nx.karate_club_graph()
    graph_name = 'zachary'
    source_path = './raw_data/'+graph_name
    target_path = './new_data/'
    G = util.init_graph(source_path, target_path, graph_name)
    # algorithm = LFM(G,2,100)
    # communities = algorithm.execute()
    # log=algorithm.get_log()
    # comm_log=zip(communities,log)

    # path='./code/others_code/lfm_img'
    # layout=nx.layout.kamada_kawai_layout(G)
    # color=['red','green','yellow','cyan','blue','magenta','violet','pink']
    # fig_size=cal_size(layout)

    # draw_comm_process(G,layout,fig_size,color,comm_log,path)
    # draw_comm_distri(G,layout,fig_size,color,communities,path)
    # real_comm=draw_karate_club_real_community(G,layout,fig_size,path)
    # modu=util.cal_EQ(G,communities)
    # real_modu=util.cal_EQ(G,real_comm)
    # # real_modu=0.35
    # print('模块度-LFM:{}-真实{}'.format(modu,real_modu))
    # print('finish')
    comms = [["32", "15", "30", "29", "18", "9", "23", "20", "33", "27", "8", "22", "14", "26"],
            ["17", "11", "10", "21", "16", "6", "12", "0", "5", "4"],
            ["17", "11", "21", "30", "9", "3", "2", "12",
                "1", "19", "28", "0", "7", "8", "13"],
            ["25", "28", "24", "31"]]
    path='./code/others_code/'
    layout=nx.layout.kamada_kawai_layout(G)
    color=['red','green','yellow','cyan','blue','magenta','violet','pink']
    fig_size=cal_size(layout)
    draw_comm_distri(G,layout,fig_size,color,comms,path)

