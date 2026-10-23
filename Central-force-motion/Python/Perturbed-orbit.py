import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


# Define clean column names to override the messy header string
column_names = ['JDTDB', 'Calendar_Date', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'LT', 'RG', 'RR']

# Read the file line-by-line to isolate only the raw data numbers
data_lines = []
with open("/home/suparnajit2005/GitHub/Classical_mechanics_simulations/Central-force-motion/Jupyter/horizons_results.csv", "r") as f:
    in_data_zone = False
    for line in f:
        if "$$EOE" in line:
            break
        if in_data_zone:
            # Strip trailing commas and whitespace blocks
            cleaned_line = line.strip().rstrip(',')
            if cleaned_line:
                data_lines.append(cleaned_line.split(','))
        if "$$SOE" in line:
            in_data_zone = True

# Construct the clean DataFrame directly from parsed rows
df = pd.DataFrame(data_lines, columns=column_names)

# Strip out trailing spaces from text fields and convert columns to proper types
df['Calendar_Date'] = df['Calendar_Date'].str.strip()

numeric_cols = ['JDTDB', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'LT', 'RG', 'RR']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

# Preview your flawlessly formatted dataset
print(df.head(31))

Data_matrix=df.to_numpy()

# Time Initialisation
t0=np.ones_like(Data_matrix.T[0])*np.min(Data_matrix.T[0])
t_days=Data_matrix.T[0]-t0

# State Initialisation
state=np.array([Data_matrix.T[2].astype(float),Data_matrix.T[3].astype(float),Data_matrix.T[4].astype(float),Data_matrix.T[5].astype(float),Data_matrix.T[6].astype(float),Data_matrix.T[7].astype(float)])


# ==============================================================================
# MERCURY PHYSICAL CONSTANTS (Source: NASA JPL Horizons Ephemeris System)
# ==============================================================================

# --- Mercury Absolute Physical Constants ---
MERCURY_RADIUS_MEAN_KM = 2439.4       # Volumetric mean radius in km
MERCURY_RADIUS_EQ_KM   = 2440.53      # Equatorial radius (Re) in km
MERCURY_DENSITY_G_CM3  = 5.427        # Density in g/cm^3
MERCURY_VOLUME_KM3     = 6.085e10     # Volume in km^3 (derived from 6.085 x 10^10)
MERCURY_SURFACE_G_MS2  = 3.701        # Equatorial surface gravity in m/s^2
MERCURY_ESCAPE_VEL_KMS = 4.435        # Escape velocity in km/s

# --- Orbital Mechanics Constants ---
MERCURY_GM_KM3_S2      = 22031.86855  # Standard gravitational parameter (μ) in km^3/s^2
SUN_MERCURY_MASS_RATIO = 6023682.0    # Mass ratio of the Sun to Mercury (M_sun / M_mercury)
MERCURY_MEAN_VEL_KMS   = 47.362       # Mean orbital velocity in km/s
MERCURY_ORBIT_PERIOD_D = 87.969257    # Sidereal orbital period in solar days

# --- Rotational Constants ---
MERCURY_ROT_PERIOD_D   = 58.6463      # Sidereal rotation period in solar days
MERCURY_ROT_RATE_RAD_S = 1.24001e-6   # Sidereal rotation rate in rad/s

# ==============================================================================
# DYNAMIC MASS & G DERIVATIONS (Auto-converted to Standard SI / MKS Units)
# ==============================================================================

# 1. Absolute Masses (in kg)
MERCURY_MASS_KG = 3.302e23            # 3.302 x 10^23 kg
SUN_MASS_KG     = MERCURY_MASS_KG * SUN_MERCURY_MASS_RATIO

# 2. Standard Gravitational Parameter in SI Units (m^3 / s^2)
MERCURY_GM_M3_S2 = MERCURY_GM_KM3_S2 * 1e9

# 3. Universal Gravitational Constant 'G' Derived Dynamically from File State
# Formula: G = GM / M
G_SI = MERCURY_GM_M3_S2 / MERCURY_MASS_KG

# --- Print Validation Check ---
print(f"--- NASA Horizons Constants Imported Successfully ---")
print(f"Mercury Mass : {MERCURY_MASS_KG:.3e} kg")
print(f"Sun Mass     : {SUN_MASS_KG:.3e} kg")
print(f"Derived G    : {G_SI:.7e} m^3/(kg*s^2)")

G=G_SI
M=SUN_MASS_KG

m=MERCURY_MASS_KG


k=G*M*(1e-9)

def V(x,y,z):
    r=np.sqrt(x**2+y**2+z**2)

    return -k*m/r

def KE(vx,vy,vz):

    v=np.sqrt(vx**2+vy**2+vz**2)

    return 0.5*m*(v**2)

def U(sol):
    return np.array([KE(sol.T[3],sol.T[4],sol.T[5])+V(sol.T[0],sol.T[1],sol.T[2]),KE(sol.T[3],sol.T[4],sol.T[5]),V(sol.T[0],sol.T[1],sol.T[2])])


def acc(t,state):
    x,y,z,vx,vy,vz=state
    
    r=(x**2+y**2+z**2)**0.5
    
    a=-(k/r**3)-(1000/r**4)

    ax=a*x
    ay=a*y
    az=a*z
    return (vx,vy,vz,ax,ay,az)

        

    
days=31
t_span=(0,days*24*3600)
t_eval=np.linspace(0,max(t_span),31)

state0=(state[0][0],state[1][0],state[2][0],state[3][0],state[4][0],state[5][0])

#sol=odeint(acc,state0,t=t)
sol=solve_ivp(acc,t_span,state0,method='DOP853',t_eval=t_eval,rtol=1e-10,atol=1e-10)


from matplotlib import pyplot as plt

plt.style.use("dark_background")



#Plotting
fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(projection="3d")
ax.set_title("Mercury orbit analysis")
ax.set_xlabel("X - Axis")
ax.set_ylabel("Y - Axis")
ax.set_zlabel("Z - Axis")
# Set the background panel colors with a low alpha (e.g., 0.1 for 10% opacity)
# Format is (Red, Green, Blue, Alpha)
ax.xaxis.set_pane_color((0.2, 0.2, 0.2, 0.6))
ax.yaxis.set_pane_color((0.2, 0.2, 0.2, 0.6))
ax.zaxis.set_pane_color((0.2, 0.2, 0.2, 0.6))
ax.set_proj_type('ortho')


ax.plot(state[0],state[1],state[2],lw=0.51,color="#FF0054",label="Mercury-track ( JPL / NASA )")
ax.plot(sol.y[0],sol.y[1],sol.y[2],lw=0.51,color="#9E0059",label="Mercury-track (Simulation)")

ax.scatter(sol.y[0],sol.y[1],sol.y[2],s=10,color="#EFEFEF",label="Mercury")
ax.scatter(0,0,0,color="#FFBD00",s=100,label="Sun")
ax.scatter(0,0,0,color="#FF5400",s=500,alpha=0.6)
ax.legend(framealpha=0.3,shadow=False,loc=1)


plt.tight_layout()
plt.show()

