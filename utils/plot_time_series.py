import h5py
import matplotlib.pyplot as plt

with h5py.File('output/time_series.h5', 'r') as f:
    t = f['t'][:]
    energy = f['energy'][:]
    
plt.plot(t, energy)
plt.xlabel('Time')
plt.ylabel('Energy')
plt.savefig('energy_decay.png')