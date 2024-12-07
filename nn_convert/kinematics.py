import math


l1=0.1
l2=0.34
l3=0.34


def getVirtualLegLength(theta_kp):
    theta_knee = math.pi + theta_kp;#大腿和小腿所夹的角: theta_knee、膝关节的转角: theta_kp
    l_vir = math.sqrt(pow(l2, 2) + pow(l3, 2) - 2 * l2 * l3 * math.cos(theta_knee));#余弦定理: c² = a² + b² -2abcosθ
    return l_vir
  
  
def getThetaVirUp(theta_kp, l_vir):
    return math.asin(l3 * math.sin(theta_kp) / l_vir)
  
  
def IK(p,q):
    for i in range(4):
        if (i==0 or i==1):
            l_sign = -1
        else:
            l_sign = 1
        l_vir_square = pow(p[i, 0], 2) + pow(p[i, 1], 2) + pow(p[i, 2], 2) - pow(l1, 2)
        theta_knee = math.acos(max(-1, min((l_vir_square - pow(l2, 2) - pow(l3, 2)) / (-2 * l2 * l3), 1)))
        theta_kp = -(math.pi - theta_knee)#theta_knee: 大腿和小腿所夹的角、theta_kp: 膝关节的转角
        l_vir = getVirtualLegLength(theta_kp)
        theta_virp = math.asin(max(-1, min(-p[i][0] / l_vir, 1)))#虚拟腿俯仰角
        theta_vir_up = getThetaVirUp(theta_kp, l_vir)#虚拟腿与大腿的夹角
        theta_hp = theta_virp - theta_vir_up
        s1 = l_vir * math.cos(theta_hp + theta_vir_up) * p[i][1] + l_sign * l1 * p[i][2]
        c1 = l_sign * l1 * p[i][1] - l_vir * math.cos(theta_hp + theta_vir_up) * p[i][2]
        theta_hr = math.atan2(s1, c1)
        q[i][0] = theta_hr
        q[i][1] = theta_hp
        q[i][2] = theta_kp
        
        
# 正运动学：由关节角度得到足端位置
def getLegFK(q,p):
    theta0 = -0.039
    for i in range(4):
        if (i==0 or i==1):
            l_sign = -1
        else:
            l_sign = 1
        p[i][0] = -l2 * math.sin(q[i][1]) - l3 * math.sin(q[i][1] + q[i][2] + theta0)
        p[i][1] = l_sign * l1 * math.cos(q[i][0])+ (l2 * math.cos(q[i][1]) + l3 * math.cos(q[i][1] + q[i][2] + theta0)) * math.sin(q[i][0])
        p[i][2] = l_sign * l1 * math.sin(q[i][0])- (l2 * math.cos(q[i][1]) + l3 * math.cos(q[i][1] + q[i][2] + theta0)) * math.cos(q[i][0])

