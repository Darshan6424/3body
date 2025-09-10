import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#solutions initial conditions

y01 =  [ 0.536387073390   ,  # x1
         0.054088605008   ,  # y1
        -0.252099126491   ,  # x2
         0.694527327749   ,  # y2
        -0.275706601688   ,  # x3
        -0.335933589318   ,  # y3
         0               ,  # x4
         0               ,  # y4
        -0.569379585581   ,  # vx1
         1.255291102531   ,  # vy1
         0.079644615252   ,  # vx2
        -0.458625997341   ,  # vy2
         0.489734970329   ,  # vx3
        -0.796665105189   ,  # vy3
         0               ,  # vx4
         0               ]  # vy4

 #oval,catfish and starship
 
y02 =  [ 0.419698802831   ,  # x1
         1.190466261252   ,  # y1
         0.076399621771   ,  # x2
         0.296331688995   ,  # y2
         0.100310663856   ,  # x3
        -0.729358656127   ,  # y3
         0               ,  # x4
         0               ,  # y4
         0.102294566003   ,  # vx1
         0.687248445943   ,  # vy1
         0.148950262064   ,  # vx2
         0.240179781043   ,  # vy2
        -0.251244828060   ,  # vx3
        -0.927428226977   ,  # vy3
         0               ,  # vx4
         0               ]  # vy4
#skinny pineapple

y03 =  [ 0.716248295713   ,  # x1
         0.384288553041   ,  # y1
         0.086172594591   ,  # x2
         1.342795868577   ,  # y2
         0.538777980808   ,  # x3
         0.481049882656   ,  # y3
         0               ,  # x4
         0               ,  # y4
         1.245268230896   ,  # vx1
         2.444311951777   ,  # vy1
        -0.675224323690   ,  # vx2
        -0.962879613630   ,  # vy2
        -0.570043907206   ,  # vx3
        -1.481432338147   ,  # vy3
         0               ,  # vx4
         0               ]  # vy4
#oval with flourishes

y04=  [-1                ,  # x1
         0                ,  # y1
         1                ,  # x2
         0                ,  # y2
         0                ,  # x3
         0                ,  # y3
         0                ,  # x4
         0                ,  # y4
         0.383444         ,  # vx1  =  p1
         0.377364         ,  # vy1  =  p2
         0.383444         ,  # vx2  =  p1
         0.377364         ,  # vy2  =  p2
        -2*0.383444       ,  # vx3  = -2*p1
        -2*0.377364       ,  # vy3  = -2*p2
         0                ,  # vx4
         0                ]  # vy4


#moth 3

y05 =  [ 0.8083106230     ,  # x1
         0.0000000000     ,  # y1
        -0.4954148566     ,  # x2
         0.0000000000     ,  # y2
        -0.3128957664     ,  # x3
         0.0000000000     ,  # y3
         0               ,  # x4
         0               ,  # y4
         0.0000000000     ,  # vx1
         0.9901979166     ,  # vy1
         0.0000000000     ,  # vx2
        -2.7171431768     ,  # vy2
         0.0000000000     ,  # vx3
         1.7269452602     ,  # vy3
         0               ,  # vx4
         0               ]  # vy4
#R1


y06 =  [-1.0207041786   ,  # x1
         0.0000000000   ,  # y1
         2.0532718983   ,  # x2
         0.0000000000   ,  # y2
        -1.0325677197   ,  # x3
         0.0000000000   ,  # y3
         0             ,  # x4
         0             ,  # y4
         0.0000000000   ,  # vx1
         9.1265693140   ,  # vy1
         0.0000000000   ,  # vx2
         0.0660238922   ,  # vy2
         0.0000000000   ,  # vx3
        -9.1925932061   ,  # vy3
         0             ,  # vx4
         0             ]  # vy4

#HENON 2

y07 =  [-0.8124282691   ,  # x1
         0.0000000000   ,  # y1
         1.8146415679   ,  # x2
         0.0000000000   ,  # y2
        -1.0022132988   ,  # x3
         0.0000000000   ,  # y3
         0             ,  # x4
         0             ,  # y4
         0.0000000000   ,  # vx1
         2.1152813609   ,  # vy1
         0.0000000000   ,  # vx2
         0.2107937022   ,  # vy2
         0.0000000000   ,  # vx3
        -2.3260750631   ,  # vy3
         0             ,  # vx4
         0             ]  # vy4


#HENON 11

y08 =  [ 0.906009977921   ,  # x1
         0.347143444587   ,  # y1
        -0.263245299491   ,  # x2
         0.140120037700   ,  # y2
        -0.252150695248   ,  # x3
        -0.661320078799   ,  # y3
         0               ,  # x4
         0               ,  # y4
         0.242474965162   ,  # vx1
         1.045019736387   ,  # vy1
        -0.360704684300   ,  # vx2
        -0.807167979922   ,  # vy2
         0.118229719138   ,  # vx3
        -0.237851756465   ,  # vy3
         0               ,  # vx4
         0               ]  # vy4
# hand in hand in oval

solutionList = [y01,y02,y03,y04,y05,y06,y07,y08]
inp = int(input("Choose a solution from 1 to 8"))
y0 = solutionList[inp - 1]

# The masses of the 4 bodies

m1 = m2 = m3 = 1
m4 = 0.0053

#Time thingy

N= 670000
T = 0.001

#Defining the fucntion

def ThreeBodyProb(t,y,m1,m2,m3,m4):
    f = np.zeros(16)
    #velocities of the 3bodies
    f[0] = y[8]
    f[1] = y[9]
    f[2] = y[10]
    f[3] = y[11]
    f[4] = y[12]
    f[5] = y[13]
    f[6] = y[14]
    f[7] = y[15]

        #now calculating the acceleration for each body respectively
    # The x and y accelerations of each object respectively
    f[8]  = -m2*(y[0]-y[2])/(((y[0]-y[2])**2+(y[1]-y[3])**2)**(3/2)) \
            -m3*(y[0]-y[4])/(((y[0]-y[4])**2+(y[1]-y[5])**2)**(3/2)) \
            -m4*(y[0]-y[6])/(((y[0]-y[6])**2+(y[1]-y[7])**2)**(3/2)) 
    f[9]  = -m2*(y[1]-y[3])/(((y[0]-y[2])**2+(y[1]-y[3])**2)**(3/2)) \
            -m3*(y[1]-y[5])/(((y[0]-y[4])**2+(y[1]-y[5])**2)**(3/2)) \
            -m4*(y[1]-y[7])/(((y[0]-y[6])**2+(y[1]-y[7])**2)**(3/2)) 

    f[10] = -m1*(y[2]-y[0])/(((y[2]-y[0])**2+(y[3]-y[1])**2)**(3/2)) \
            -m3*(y[2]-y[4])/(((y[2]-y[4])**2+(y[3]-y[5])**2)**(3/2)) \
            -m4*(y[2]-y[6])/(((y[2]-y[6])**2+(y[3]-y[7])**2)**(3/2)) 
    f[11] = -m1*(y[3]-y[1])/(((y[2]-y[0])**2+(y[3]-y[1])**2)**(3/2)) \
            -m3*(y[3]-y[5])/(((y[2]-y[4])**2+(y[3]-y[5])**2)**(3/2)) \
            -m4*(y[3]-y[7])/(((y[2]-y[6])**2+(y[3]-y[7])**2)**(3/2)) 

    f[12] = -m1*(y[4]-y[0])/(((y[4]-y[0])**2+(y[5]-y[1])**2)**(3/2)) \
            -m2*(y[4]-y[2])/(((y[4]-y[2])**2+(y[5]-y[3])**2)**(3/2)) \
            -m4*(y[4]-y[6])/(((y[4]-y[6])**2+(y[5]-y[7])**2)**(3/2)) 
    f[13] = -m1*(y[5]-y[1])/(((y[4]-y[0])**2+(y[5]-y[1])**2)**(3/2)) \
            -m2*(y[5]-y[3])/(((y[4]-y[2])**2+(y[5]-y[3])**2)**(3/2)) \
            -m4*(y[5]-y[7])/(((y[4]-y[6])**2+(y[5]-y[7])**2)**(3/2)) 

    f[14] = -m1*(y[6]-y[0])/(((y[6]-y[0])**2+(y[7]-y[1])**2)**(3/2)) \
            -m2*(y[6]-y[2])/(((y[6]-y[2])**2+(y[7]-y[3])**2)**(3/2)) \
            -m3*(y[6]-y[4])/(((y[6]-y[4])**2+(y[7]-y[5])**2)**(3/2)) 
    f[15] = -m1*(y[7]-y[1])/(((y[6]-y[0])**2+(y[7]-y[1])**2)**(3/2)) \
            -m2*(y[7]-y[3])/(((y[6]-y[2])**2+(y[7]-y[3])**2)**(3/2)) \
            -m3*(y[7]-y[5])/(((y[6]-y[4])**2+(y[7]-y[5])**2)**(3/2)) 

    return f

# Solution for the positions at all times
t = np.linspace(0,N*T,N)
sol = solve_ivp(ThreeBody,[0,670],y0,t_eval=t,rtol=1e-9,
                args=(m1, m2, m3, m4))

unstable_solutions_x = []
unstable_solutions_y = []
stable_solutions_px  = []
unstable_solutions_px = []

stable_solutions_py = []
unstable_solutions_py = []
stable_solutions_vx = []
unstable_solutions_vx = []
stable_solutions_vy = []
unstable_solutions_vy = []

func_inpt = ['m','px','py','vx','vy']

def plt_4bodies():
    # Positions body 1
    plt.plot(solution.y[0], solution.y[1], '-g', label = 'Body 1')
    # Positions body 2
    plt.plot(solution.y[2], solution.y[3], '-b', label = 'Body 2')
    # Positions body 3
    plt.plot(solution.y[4], solution.y[5], '-r', label = 'Body 3')
    # Positions body 4
    plt.plot(solution.y[6], solution.y[7], '-y', label = 'Body 4')
    plt.legend()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.show()

    # Position-x body 1
    plt.plot(t, solution.y[0], 'black', label = 'Body 1')
    # Position-x body 2
    plt.plot(t, solution.y[2], 'b', label = 'Body 2')
    # Position-x body 3
    plt.plot(t, solution.y[4], 'r', label = 'Body 3')
    # Position-x body 4
    plt.plot(t, solution.y[6], 'y', label = 'Body 4')
    plt.legend(loc='upper right')
    plt.ylabel("Position(x)")
    plt.xlabel("time")
    plt.xlim(0, 30)
    plt.ylim(-5, 5)
    plt.show()

    # Position-y body 1
    plt.plot(t, solution.y[1], 'black', label = 'Body 1')
    # Position-y body 2
    plt.plot(t, solution.y[3], 'b', label = 'Body 2')
    # Position-y body 3
    plt.plot(t, solution.y[5], 'r', label = 'Body 3')
    # Position-y body 4
    plt.plot(t, solution.y[7], 'y', label = 'Body 4')

    plt.legend(loc='upper right')
    plt.ylabel("Position(y)")
    plt.xlabel("time")
    plt.xlim(0, 30)
    plt.ylim(-5, 5)
    plt.show()


g = input("Choose one of the following arguments to do your testing: m, px, py, vx, vy")

if g in func_inpt:
    print("We'll process your order...")

stable_masses = np.arange(0.001, 0.01, 0.0001)
print(stable_masses)

positions_x = np.arange(0.1, 1, 0.01)
print(positions_x)

positions_y = np.arange(0.1, 1, 0.01)
print(positions_y)

velocities_x = np.arange(0.1, 1, 0.01)
print(velocities_x)

velocities_y = np.arange(0.1, 1, 0.01)
print(velocities_y)


def VectorStability(g):
    x = 50
    distance = []
    if g == 'm':
        for i in stable_masses:
            used_mass = i
            print(i)
            solution = solve_ivp(ThreeBody, [0,670], y0, t_eval=t, rtol=1e-9,
                                 args = (m1, m2, m3, i))

            # d12
            distance.append(np.sqrt((solution.y[2][N-1] - solution.y[0][N-1])**2 +
                                    (solution.y[3][N-1] - solution.y[1][N-1])**2))

            # d13
            distance.append(np.sqrt((solution.y[4][N-1] - solution.y[0][N-1])**2 +
                                    (solution.y[5][N-1] - solution.y[1][N-1])**2))

            # d23
            distance.append(np.sqrt((solution.y[4][N-1] - solution.y[2][N-1])**2 +
                                    (solution.y[5][N-1] - solution.y[3][N-1])**2))

            plt_4bodies()
            for i in distance:
                if i > 10:
                    output = print("This system is unstable")
                    unstable_solutions_m.append(used_mass)
                    break
                else:
                    output = print("This system is stable")
                    stable_solutions_m.append(used_mass)
                    break
        return output

    elif g == 'px':
        for i in positions_x:
            used_pos_x = i
            print(i)
            y0[6] = i
            solution = solve_ivp(ThreeBody, [0,670], y0, t_eval=t, rtol=1e-9,
                             args = (m1, m2, m3, m4))

        # d12
            distance.append(np.sqrt((solution.y[2][N-1] - solution.y[0][N-1])**2 +
                                (solution.y[3][N-1] - solution.y[1][N-1])**2))
        # d13
            distance.append(np.sqrt((solution.y[4][N-1] - solution.y[0][N-1])**2 +
                                (solution.y[5][N-1] - solution.y[1][N-1])**2))
        # d23
            distance.append(np.sqrt((solution.y[4][N-1] - solution.y[2][N-1])**2 +
                                (solution.y[5][N-1] - solution.y[3][N-1])**2))

            plt_4bodies()
            for i in distance:
                if i > 10:
                    output = print("This system is unstable")
                    unstable_solutions_px.append(used_pos_x)
                    break
                else:
                    output = print("This system is stable")
                    stable_solutions_px.append(used_pos_x)
                    break
        return output

    elif g == 'py':
        for i in positions_y:
            used_pos_y = i
            print(i)
            y0[7] = i
            solution = solve_ivp(ThreeBody, [0,670], y0, t_eval=t, rtol=1e-9,
                                args = (m1, m2, m3, m4))

            # d12
            distance.append(np.sqrt((solution.y[2][N-1] - solution.y[0][N-1])**2 +
                                    (solution.y[3][N-1] - solution.y[1][N-1])**2))
            # d13
            distance.append(np.sqrt((solution.y[4][N-1] - solution.y[0][N-1])**2 +
                                    (solution.y[5][N-1] - solution.y[1][N-1])**2))
            # d23
            distance.append(np.sqrt((solution.y[4][N-1] - solution.y[2][N-1])**2 +
                                    (solution.y[5][N-1] - solution.y[3][N-1])**2))

            plt_4bodies()
            for i in distance:
                if i > 10:
                    output = print("This system is unstable")
                    unstable_solutions_py.append(used_pos_y)
                    break
                else:
                    output = print("This system is stable")
                    stable_solutions_py.append(used_pos_y)
                    break
        return output

    elif g == 'vx':
        for i in velocities_x:
            used_vel_x = i
            print(i)
            y0[14] = i
            solution = solve_ivp(ThreeBody, [0,670], y0, t_eval=t, rtol=1e-9,
                                args = (m1, m2, m3, m4))

            # d12
            distance.append(np.sqrt((solution.y[2][N-1] - solution.y[0][N-1])**2 +
                                    (solution.y[3][N-1] - solution.y[1][N-1])**2))
            # d13
            distance.append(np.sqrt((solution.y[4][N-1] - solution.y[0][N-1])**2 +
                                    (solution.y[5][N-1] - solution.y[1][N-1])**2))
            # d23
            distance.append(np.sqrt((solution.y[4][N-1] - solution.y[2][N-1])**2 +
                                    (solution.y[5][N-1] - solution.y[3][N-1])**2))

            plt_4bodies()
            for i in distance:
                if i > 10:
                    output = print("This system is unstable")
                    unstable_solutions_vx.append(used_vel_x)
                    break
                else:
                    output = print("This system is stable")
                    stable_solutions_vx.append(used_vel_x)
                    break
        return output
        
    elif g == 'vy':
        for i in velocities_y:
            used_vel_y = i
            print(i)
            y0[15] = i
            solution = solve_ivp(ThreeBody, [0,670], y0, t_eval=t, rtol=1e-9, args = (m1, m2, m3, m4))
            distance.append(np.sqrt((solution.y[2][N-1] - solution.y[0][N-1])**2 + (solution.y[3][N-1] - solution.y[1][N-1])**2)) #r12
            distance.append(np.sqrt((solution.y[4][N-1] - solution.y[0][N-1])**2 + (solution.y[5][N-1] - solution.y[1][N-1])**2)) #r13
            distance.append(np.sqrt((solution.y[4][N-1] - solution.y[2][N-1])**2 + (solution.y[5][N-1] - solution.y[3][N-1])**2)) #r23
            plt_4bodies()
            for i in distance:
                if i > 10:
                    output = print("This system is unstable")
                    unstable_solutions_vy.append(used_vel_y)
                    break
                else:
                    output = print("This system is stable")
                    stable_solutions_vy.append(used_vel_y)
                    break
        return output

VectorStability(g)
if g == 'm':
    print(f"stable solutions:{stable_solutions_m}")
    print(f"unstable solutions:{unstable_solutions_m}")
elif g == 'px':
    print(f"stable solutions:{stable_solutions_px}")
    print(f"unstable solutions:{unstable_solutions_px}")
elif g == 'py':
    print(f"stable solutions:{stable_solutions_py}")
    print(f"unstable solutions:{unstable_solutions_py}")
elif g == 'vx':
    print(f"stable solutions:{stable_solutions_vx}")
    print(f"unstable solutions:{unstable_solutions_vx}")
elif g == 'vy':
    print(f"stable solutions:{stable_solutions_vy}")
    print(f"unstable solutions:{unstable_solutions_vy}")