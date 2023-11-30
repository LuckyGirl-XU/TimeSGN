import torch
import dgl
import math
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from modules.TimeEncode import *


   
class DTMPLayer(torch.nn.Module):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, seperate = False):
        super(DTMPLayer, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat
        self.dim_edge_feat = dim_edge_feat
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.seperate = seperate
        self.att_bias = torch.nn.Parameter(torch.zeros(self.dim_out))
        nn.init.normal_(self.att_bias, mean = 0, std = 0.01)
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if seperate:
            if dim_edge_feat > 0:
                self.w_q_t = torch.nn.Linear(dim_time, dim_out)
                self.w_k_t = torch.nn.Linear(dim_time , dim_out)
                self.w_v_t = torch.nn.Linear(dim_time, dim_out)
                self.w_q_e = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_k_e = torch.nn.Linear(dim_node_feat + dim_edge_feat, dim_out)
                self.w_v_e = torch.nn.Linear(dim_node_feat + dim_edge_feat, dim_out) 
            else:
                self.w_q_t = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_k_t = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_v_t = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_q_e = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_k_e = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_v_e = torch.nn.Linear(dim_node_feat, dim_out)        
        else:
            if dim_node_feat + dim_time > 0:
                self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
            self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
            self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)
        self.w_com = torch.nn.Linear(dim_out*2, dim_out)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        
    def forward(self, b):
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device('cuda:0'))
        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])
            zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device('cuda:0')))
        if self.seperate:
            if self.dim_edge_feat > 0:
                Q_t = self.w_q_t(zero_time_feat)[b.edges()[1]]
                K_t = self.w_k_t(time_feat)
                V_t = self.w_v_t(time_feat)
                Q_t = torch.reshape(Q_t, (Q_t.shape[0], self.num_head, -1))
                K_t = torch.reshape(K_t, (K_t.shape[0], self.num_head, -1))
                V_t = torch.reshape(V_t, (V_t.shape[0], self.num_head, -1))
                att_t = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q_t*K_t, dim=2)))
                att_t = self.att_dropout(att_t)
                V_t = torch.reshape(V_t*att_t[:, :, None], (V_t.shape[0], -1))  

                Q_e = self.w_q_e(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K_e = self.w_k_e(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
                V_e = self.w_v_e(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
                Q_e = torch.reshape(Q_e, (Q_e.shape[0], self.num_head, -1))
                K_e = torch.reshape(K_e, (K_e.shape[0], self.num_head, -1))
                V_e = torch.reshape(V_e, (V_e.shape[0], self.num_head, -1))
                att_e = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q_e*K_e, dim=2)))
                att_e = self.att_dropout(att_e)
                V_e = torch.reshape(V_e*att_e[:, :, None], (V_e.shape[0], -1))  
            else:
                Q_t = self.w_q_t(zero_time_feat)[b.edges()[1]]
                K_t = self.w_k_t(time_feat)
                V_t = self.w_v_t(time_feat)
                Q_t = torch.reshape(Q_t, (Q_t.shape[0], self.num_head, -1))
                K_t = torch.reshape(K_t, (K_t.shape[0], self.num_head, -1))
                V_t = torch.reshape(V_t, (V_t.shape[0], self.num_head, -1))
                att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q_t*K_t + self.att_bias, dim=2)))
                att = self.att_dropout(att)
                V_t = torch.reshape(V_t*att[:, :, None], (V_t.shape[0], -1))  

                Q_e = self.w_q_e(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K_e = self.w_k_e(b.srcdata['h'][b.num_dst_nodes():])
                V_e = self.w_v_e(b.srcdata['h'][b.num_dst_nodes():])
                Q_e = torch.reshape(Q_e, (Q_e.shape[0], self.num_head, -1))
                K_e = torch.reshape(K_e, (K_e.shape[0], self.num_head, -1))
                V_e = torch.reshape(V_e, (V_e.shape[0], self.num_head, -1))
                att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q_e*K_e + self.att_bias, dim=2)))
                att = self.att_dropout(att)
                V_e = torch.reshape(V_e*att[:, :, None], (V_e.shape[0], -1))       
            
            V = self.w_com(torch.cat([V_t, V_e], dim = 1))
            b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
            b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
        else:
            Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
            K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
            V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K + self.att_bias, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
            b.update_all(dgl.function.copy_src('v', 'm'), dgl.function.sum('m', 'h'))
        
        if self.dim_node_feat != 0:
            rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
        else:
            rst = b.dstdata['h']
        rst = self.w_out(rst)
        rst = torch.nn.functional.relu(self.dropout(rst))
        return self.layer_norm(rst)
    
