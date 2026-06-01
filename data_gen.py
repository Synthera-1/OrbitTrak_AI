import numpy as np
import pandas as pd

# --- Configuration Constants ---
G = 6.6743e-11  # Gravitational constant
M_PLANET = 5.972e24  # Mass of Earth (kg)
R_PLANET = 6371000  # Radius of Earth (meters)

def run_physics_simulation(angle, duration, power, mass):
    """
    Simulates a 2D rocket trajectory using Euler integration.
    Returns: (apoapsis_height, periapsis_height, stable_orbit_flag)
    """
    t = 0
    dt = 1.0  # 1-second time steps
    
    # Initial state: Position at North Pole (x=0, y=R), Velocity zero
    x, y = 0.0, float(R_PLANET)
    angle_rad = np.radians(angle)
    vx, vy = 0.0, 0.0
    
    max_altitude = 0.0
    min_altitude = float('inf')
    crashed = False
    
    # Simulate 2000 seconds of flight
    for _ in range(2000):
        r_vector = np.sqrt(x**2 + y**2)
        
        # Check for surface collision
        if r_vector <= R_PLANET and t > 0:
            crashed = True
            break
            
        # 1. Calculate Gravity (F = G*M*m / r^2)
        fg = G * M_PLANET * mass / (r_vector**2)
        ax_g = -fg * (x / r_vector) / mass
        ay_g = -fg * (y / r_vector) / mass
        
        # 2. Apply Thrust (only during burn duration)
        ax_t, ay_t = 0.0, 0.0
        if t < duration:
            # Force converted to acceleration (power is in kN -> N)
            ax_t = (power * 1000 * np.sin(angle_rad)) / mass
            ay_t = (power * 1000 * np.cos(angle_rad)) / mass
            
        # Update Velocity
        vx += (ax_g + ax_t) * dt
        vy += (ay_g + ay_t) * dt
        
        # Update Position
        x += vx * dt
        y += vy * dt
        
        # Calculate Altitude
        altitude = r_vector - R_PLANET
        
        # Track Peaks
        if altitude > max_altitude:
            max_altitude = altitude
        # Only track minimum altitude after engine cuts out
        if t > duration and altitude < min_altitude:
            min_altitude = altitude
            
        t += dt

    # CRITICAL FIX: If rocket crashed or flew directly out of orbit bounds instantly,
    # reset infinity to zero to protect the downstream TensorFlow optimization nodes
    if min_altitude == float('inf'):
        min_altitude = 0.0

    # To be stable, periapsis must be > 150km (150,000m) and not crashed
    stable_orbit = 1 if (min_altitude > 150000 and not crashed) else 0
    
    return max_altitude, max(0.0, min_altitude), stable_orbit

# --- Data Generation Loop ---
def generate_dataset(num_samples=10000):
    data = []
    
    print(f"Generating {num_samples} physics simulations...")
    for i in range(num_samples):
        # Randomize engineering parameters within defined bounds
        ang = np.random.uniform(10, 90)   # Launch angle 10-90 deg
        dur = np.random.uniform(20, 180) # Burn time 20-180s
        pwr = np.random.uniform(200, 1200)# Thrust 200-1200 kN
        ms = np.random.uniform(800, 4000)# Mass 800-4000 kg
        
        # Run simulation
        apo, peri, stable = run_physics_simulation(ang, dur, pwr, ms)
        data.append([ang, dur, pwr, ms, apo, peri, stable])
        
        if (i + 1) % 2000 == 0:
            print(f"  ... {i+1} completed")

    # Save to CSV
    df = pd.DataFrame(data, columns=['angle', 'duration', 'power', 'mass', 'apoapsis', 'periapsis', 'stable'])
    df.to_csv('orbit_training_data.csv', index=False)
    print("Dataset generation complete. Saved to 'orbit_training_data.csv'.")

if __name__ == "__main__":
    generate_dataset()
