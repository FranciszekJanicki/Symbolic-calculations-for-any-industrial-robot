import numpy as np
import sympy as sp
import roboticstoolbox as rtb
from spatialmath import *
from spatialmath.base import *
from spatialmath.base.symbolic import *


# careful! these symbols are lists indexed from 0, so theta1 = t[0], etc
# why lists instead of one big sp.symbols list and each element pushed into each next variable? because it looks funny with this many variables

theta = sp.symbols('theta1:7') # theta angles
dtheta = sp.symbols('dtheta1:7') # theta derivatives 
s1, s2, s3, s4, s5, s6 = [sp.sin(theta[0]), sp.sin(theta[1]), sp.sin(theta[2]), sp.sin(theta[3]), sp.sin(theta[4]), sp.sin(theta[5])] # theta sinuses
c1, c2, c3, c4, c5, c6 = [sp.cos(theta[0]), sp.cos(theta[1]), sp.cos(theta[2]), sp.cos(theta[3]), sp.cos(theta[4]), sp.cos(theta[5])] # theta cosinuses

d = sp.symbols('d1:7') # d offsets
dd = sp.symbols('dd1:7') # d derivatives

alpha = sp.symbols('alpha1:7') # alpha angles are symbolic constants that may vary in different manipulators
a = sp.symbols('a1:7') # a offsets
m = sp.symbols('m1:7') # links masses
px, py, pz = sp.symbols('px, py, pz') # position vector 
g = sp.symbols('g') # g force

Ixx = sp.symbols('Ixx1:7') # xx elements of inertia tensors
Iyy = sp.symbols('Iyy1:7') # yy elements of inertia tensors
Izz = sp.symbols('Izz1:7') # zz elements of inertia tensors
I = [
    np.diag([Ixx[0], Iyy[0], Izz[0]]),
    np.diag([Ixx[1], Iyy[1], Izz[1]]),
    np.diag([Ixx[2], Iyy[2], Izz[2]])
]


def printDynamicsEquation(D, C, G):
    print("Dynamics equation:", D, "* qdd + ", C, "* qd + ", G, "= tau")


def createRobot():
    r = rtb.DHRobot([
        rtb.RevoluteDH(alpha=-pi()/2, d=d[0]), # symbolic manipulator's links
        rtb.RevoluteDH(alpha=pi()/2, d=d[1]),
        rtb.PrismaticDH(a=a[2], qlim=[0,3])], 
                        name="DH_enjoyer", symbolic=True)
    r.addconfiguration_attr("mycfg", [theta[0], theta[1], d[2]]) # symbolic configuration
    r.q = r.mycfg
    r.qd = [dtheta[0], dtheta[1], dd[2]] # symbolic velocities
    return r


def fkine(robot, q):
    T_all = robot.fkine_all(q)
    T13 = T_all[1].inv() * T_all[-1]
    p1 = T_all[1].inv().A @ np.transpose([px, py, pz, 1])

    # print("Position of 3_in_1 from fkine: ", simplify(T13.A[:4, 3]))
    # print("Position of 3_in_1 from given position: ", p1[:4])
    # print("T matrixes: ", T_all)
    return T_all


def ikine(pos):
    px, py, pz = pos
    ik_sol1 = []
    ik_sol2 = []
    
    s1_1, s1_2 = [-px+sp.sqrt(px**2+py**2-d[1]**2), -px-sp.sqrt(px**2+py**2-d[1]**2)]
    c1_ = py+d[1]
    q1_1 = sp.atan2(s1_1, c1_)
    q1_2 = sp.atan2(s1_2, c1_)
    ik_sol1.append(q1_1)
    ik_sol2.append(q1_2)
    
    s2_, c2_ = [c1*px+s1*py, pz]
    q2 = sp.atan2(s2_, c2_)
    ik_sol1.append(q2)
    ik_sol2.append(q2)
    
    q3 = sp.sqrt((c1*px+s1*py)**2+pz**2)
    ik_sol1.append(q3)
    ik_sol2.append(q3)
    
    print("First inverse kinematics solution:", ik_sol1)
    print("Second inverse kinematics solution:", ik_sol2, end="\n\n")
    return ik_sol1, ik_sol2
    
    
def jacobe(fk, robot):
    T_list = fk
    J_columns = []
    T_i_last = T_list[robot.n].A

    for i in range(robot.n):
        T_i = T_list[i].A

        if robot.links[i].isrevolute:
            J_columns.append(np.array([np.cross(T_i[:3, 2], T_i_last[:3, 3]-T_i[:3, 3]), T_i[:3, 2]]).reshape(6, 1))
        else:
            J_columns.append(np.array([T_i[:3, 2], np.zeros(3)]).reshape(6, 1))
        print("J[", i+1, "]: ", simplify(J_columns[i]))

    Jacobe = np.hstack(J_columns)
    if robot.jacob0(robot.q).all() == Jacobe.all():
        print("Jacobe of TCP:", Jacobe, end="\n\n")
        return Jacobe
    

def getOC(robot):
    OC = []

    for i in range(robot.n):
        robotCopy = createRobot()

        if robotCopy.links[i].isrevolute:
            robotCopy.links[i].d /= 2
        else:
            robotCopy.q[i] /= 2
        T_OC_list = fkine(robotCopy, robotCopy.q)
        print("T_OC", i, " :", T_OC_list)
        T_OC = T_OC_list[i+1]
        OC_current = T_OC.A[:3, 3]
        OC.append(OC_current)

    print("Positions of centers of masses: ", OC, end="\n\n")
    return OC


def jacobeOC(fk, robot):
    OC = getOC(robot)
    T_list = fk
    J_OC = []

    for i in range(robot.n):
        J_columns = []
        OC_i = OC[i]

        for j in range(robot.n):
            T_j = T_list[j].A
            z_j = T_j[:3, 2]
            O_j = T_j[:3, 3]

            if robot.links[j].isrevolute:
                J_columns.append(np.array([np.cross(z_j, OC_i - O_j), 
                                        z_j]).reshape(6, 1))
            else:
                J_columns.append(np.array([z_j, np.zeros(3)]).reshape(6, 1))  
        J_OC_current = []
        numNon0Cols = i+1  

        for j in range(numNon0Cols):
            J_OC_current.append((J_columns[j]))
            
        for j in range(robot.n - numNon0Cols):
            J_OC_current.append(np.zeros(6).reshape(6, 1))
            
        J_OC_current = np.hstack(J_OC_current) 
        print("Jacobe of OC", i+1, ":", J_OC_current, end="\n\n")
        J_OC.append(J_OC_current)
        
    return J_OC


def getDmatrix(fk, J, robot, m):
    T_list = fk
    D_acc = 0

    for i in range(robot.n):
        J_OC_i = J[i]
        Jv_i = J_OC_i[:3, :]
        Jw_i = J_OC_i[3:, :]
        T_i = T_list[i+1].A
        R_i = T_i[:3, :3]
        D_i = np.transpose(Jv_i) @ Jv_i * m[i] + np.transpose(Jw_i) @ R_i @ I[i] @ np.transpose(R_i) @ Jw_i
        D_acc += D_i
    
    D = sp.simplify(D_acc)
    # if robot.inertia(robot.q).all() == D.all():
    print("D (inertia) matrix:", D, end="\n\n")
    return D


def getGmatrix(robot, m, g):
    OC = getOC(robot)
    Ep = 0

    for i in range(robot.n):
        OC_i = OC[i]
        Epi = m[i] * OC_i[2] * g
        Ep += Epi

    print("Potential energy:", Ep, end="\n\n")
    G = []

    for q in robot.q:
        G.append(sp.simplify(sp.diff(Ep, q)))
    
    G = np.hstack(G)      
    # if robot.gravload(robot.q) == G:
    print("G (gravity) matrix:", G, end="\n\n")
    return G


def getCmatrix(func, robot):
    D = func
    N = robot.n
    q = robot.q
    qd = robot.qd
    christoffel = sp.tensor.array.MutableDenseNDimArray.zeros(N, N, N) # I have no idea what variable is that, but chatGPT suggested that and it works...

    for k in range(N):
        for j in range(N):
            for i in range(N):
                c_ijk = 0.5 * (sp.diff(D[j, k], q[i]) + sp.diff(D[i, k], q[j]) - sp.diff(D[j, i], q[k]))
                christoffel[i,j,k] = sp.simplify(c_ijk)

    print("Christoffel (c_ijk) symbols: ")
    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(f'C{i+1}{j+1}{k+1} =', christoffel[i, j, k])

    C_rows = []
    for k in range(N):
        C_sum = 0
        for j in range(N):
            for i in range(N):
                C_sum += christoffel[i, j, k] * qd[i] * qd[j]
        C_rows.append(sp.simplify(C_sum))

    C = np.vstack(C_rows)
    print("\n", "C (coriolis) matrix: ", sp.simplify(C), end="\n\n")
    return C
         
         
if __name__ == "__main__":
    myRobot = createRobot()

    FK = fkine(myRobot, myRobot.q)
    IK = ikine([px, py, pz])
    J = jacobe(FK, myRobot)
    J_OC = jacobeOC(FK, myRobot)
    D = getDmatrix(FK, J_OC, myRobot, [m[0], m[1], m[2]])
    G = getGmatrix(myRobot, [m[0], m[1], m[2]], g)
    C = getCmatrix(D, myRobot)

    printDynamicsEquation(D, C, G)
