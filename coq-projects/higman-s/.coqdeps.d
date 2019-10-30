higman_aux.vo higman_aux.glob higman_aux.v.beautified: higman_aux.v inductive_wqo.vo tree.vo list_embeding.vo
higman_aux.vio: higman_aux.v inductive_wqo.vio tree.vio list_embeding.vio
tree.vo tree.glob tree.v.beautified: tree.v
tree.vio: tree.v
higman.vo higman.glob higman.v.beautified: higman.v inductive_wqo.vo tree.vo higman_aux.vo
higman.vio: higman.v inductive_wqo.vio tree.vio higman_aux.vio
inductive_wqo.vo inductive_wqo.glob inductive_wqo.v.beautified: inductive_wqo.v
inductive_wqo.vio: inductive_wqo.v
list_embeding.vo list_embeding.glob list_embeding.v.beautified: list_embeding.v inductive_wqo.vo
list_embeding.vio: list_embeding.v inductive_wqo.vio
