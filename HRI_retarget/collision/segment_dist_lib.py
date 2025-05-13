import torch
# def calc_point2seg_dist(P0, P1, P2):
#     s = P2 - P1
#     lmbda = torch.norm((P0 - P1),s) / (torch.norm(s) * torch.norm(s))
#     if 0 <= lmbda and lmbda <= 1:
#         dist = torch.norm(P0-(P1+lmbda * s))
#     else:
#         dist = torch.min(torch.norm(P0-P1),torch.norm(P0-P2))

def calc_point2seg_dist(P0, P1, P2):
    # 向量 s 表示线段方向
    s = P2 - P1
    s_norm_sq = torch.sum(s * s, dim=-1, keepdim=True)  # 避免重复 norm 计算
    
    # 计算 lambda（投影比例因子），clamp 到 [0, 1] 区间
    proj = torch.sum((P0 - P1) * s, dim=-1, keepdim=True) / (s_norm_sq + 1e-12)  # 避免除0
    lmbda = proj.clamp(0.0, 1.0)
    
    # 投影点 = P1 + lambda * s
    proj_point = P1 + lmbda * s
    
    # 最终距离 = 点到投影点的距离
    dist = torch.norm(P0 - proj_point, dim=-1)
    return dist

def calc_seg2seg_dist(P1, P2, Q1, Q2):
    """
    Efficient, branchless computation of distance between segment P1P2 and Q1Q2
    Supports broadcasting across batch dimensions.
    """
    s1 = P2 - P1  # (N, D)
    s2 = Q2 - Q1
    r = P1 - Q1

    M2 = torch.sum(s1 * s1, dim=-1, keepdim=True)  # ||s1||^2
    N2 = torch.sum(s2 * s2, dim=-1, keepdim=True)  # ||s2||^2
    P = torch.sum(s1 * s2, dim=-1, keepdim=True)   # dot(s1, s2)
    r1 = torch.sum(r * s1, dim=-1, keepdim=True)   # dot(r, s1)
    r2 = torch.sum(r * s2, dim=-1, keepdim=True)   # dot(r, s2)

    C = M2 * N2 - P ** 2 + 1e-12  # 防止除以 0
    lambda_1 = (P * r2 - N2 * r1) / C
    lambda_2 = (M2 * r2 - P * r1) / C

    # Clamp 投影点在 [0, 1] 区间
    lambda_1_clamped = lambda_1.clamp(0.0, 1.0)
    lambda_2_clamped = lambda_2.clamp(0.0, 1.0)

    # 最近点坐标
    closest_P = P1 + lambda_1_clamped * s1
    closest_Q = Q1 + lambda_2_clamped * s2

    # 计算最近距离
    dist_direct = torch.norm(closest_P - closest_Q, dim=-1)

    # 针对不在区间 [0, 1] 内的情况，回退为点到线段距离
    # 构造 4 种情况的距离
    dists = torch.stack([
        calc_point2seg_dist(P1, Q1, Q2),
        calc_point2seg_dist(P2, Q1, Q2),
        calc_point2seg_dist(Q1, P1, P2),
        calc_point2seg_dist(Q2, P1, P2),
    ], dim=-1)

    dist_fallback = torch.min(dists, dim=-1).values

    # 判断 λ 是否都在 [0, 1] 内
    in_range = ((lambda_1 >= 0) & (lambda_1 <= 1) &
                (lambda_2 >= 0) & (lambda_2 <= 1)).squeeze(-1)

    # 用 where 选择最终距离（无 if）
    final_dist = torch.where(in_range, dist_direct, dist_fallback)

    return final_dist
