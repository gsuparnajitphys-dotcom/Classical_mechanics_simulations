import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


#This class contains the initial position and velocity of the celestial body:

class body:
    def __init__(self,name,mass,pos,vel):
        self.name=name
        self.mass=mass
        self.pos=np.array(pos,dtype=float)
        self.vel=np.array(vel,dtype=float)
        
# Data from the NASA Horizons data :


b1 = body("Sun", 1988410e24, np.array([-1.602325979919614E+05,-7.005062866580029E+05 , -2.888835766883729E+05]), np.array([1.031902521729396E-02, 4.449029074274385E-03, 1.684369200411074E-03]))

b2 = body("Mercury", 3.302e23, np.array([5.005945338170964E+07, 9.740919593283530E+06, 8.434890081924666E+04]), np.array([-1.867497438964642E+01, 4.357107773137023E+01, 2.521169409511218E+01]))

b3 = body("Venus", 48.685e23, np.array([8.302432525358570E+07,6.425708656179664E+07 , 2.367777884417082E+07]), np.array([-2.247968325249581E+01,  2.391654264351647E+01,  1.218439937840489E+01]))

b4 = body("Earth", 5.97219e24, np.array([1.197286311109774E+08,  7.982463976381977E+07,  3.461601209038475E+07]), np.array([-1.805688133808997E+01,  2.195386395433621E+01,  9.517041959974549E+00]))

b5 = body("Mars", 6.4171e23, np.array([-3.932279784924307E+07,  2.124476993615854E+08,  9.853347780953923E+07]), np.array([-2.297156296757593E+01, -1.971991104355783E+00, -2.850371983145413E-01]))

b6 = body("Jupiter",18.9819e26 , np.array([ -5.469428826456959E+08,  5.251793803450833E+08,  2.384273303310489E+08]), np.array([-9.636786632066425E+00, -7.784841955694026E+00, -3.102163240189204E+00]))


bodies=[b1,b2,b3,b4,b5,b6]


N=len(bodies)

def acc(mass,pos):

    
    a=np.zeros((len(bodies),3))

    for i in range(len(bodies)):
        for j  in range(len(bodies)):

            
            if i!=j:
                G = 6.67430e-20
                
                r=pos[i]-pos[j]

                ax=-((G*mass[j]*r[0])/(np.linalg.norm(r)**3))
                ay=-((G*mass[j]*r[1])/(np.linalg.norm(r)**3))
                az=-((G*mass[j]*r[2])/(np.linalg.norm(r)**3))

                a[i]+=np.array([ax,ay,az])

    return a
                

            
def eq_o_mot(t,state):
    N=len(bodies)
    pos=state[:3*N].reshape(N,3)
    vel=state[3*N:].reshape(N,3)

    m=np.array([])
    for i in range(N):
        m=np.append(m,bodies[i].mass)

    ac=acc(m,pos)

    return np.concatenate((vel.flatten(),ac.flatten()))


initial_pos=np.array([])
initial_vel=np.array([])

for i in range(N):
    initial_pos=np.append(initial_pos,bodies[i].pos)
    initial_vel=np.append(initial_vel,bodies[i].vel)

state0=np.concatenate((initial_pos,initial_vel))

days=50

t_span = (0, days*24*3600)

t_eval = np.linspace(0, days*24*3600,50 )
    
sol=solve_ivp(eq_o_mot,t_span,state0,method='DOP853',t_eval=t_eval)

print(sol)

print("Success:", sol.success)
print("Position and velocity array shape:", sol.y.shape)
print("Number of output times:", len(sol.t))
print("Final time (seconds):", sol.t[-1])