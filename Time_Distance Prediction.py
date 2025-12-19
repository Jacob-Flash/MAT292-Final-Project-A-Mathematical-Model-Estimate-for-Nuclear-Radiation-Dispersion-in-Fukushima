"""
=============================================================================
                       ALPS GLOBAL DISPERSION MODEL V2.1
=============================================================================
***REQUIRED LIBRARIES:
You must have the following Python libraries installed (conduct the following command in terminal):
   pip install numpy matplotlib scipy cartopy

[1] USER GUIDE & INSTRUCTIONS
-----------------------------------------------------------------------------
This script simulates the long-term oceanic dispersion of treated water released
from Fukushima, tracking its hypothetical path across the globe over 30+ years.

HOW TO OPERATE:
1. Run the script. A menu will appear in the console.
2. Select Option [1] to generate a World Map at a specific year (e.g., "10.0").
3. Select Option [2] to calculate the arrival time for a specific city.
   - Example inputs: "London", "Dubai", "Shanghai", "Sydney".
   - You must enter the EXACT SAME city name to successfully operate.
4. Select Option [3] to check the list of available cities.
5. Select Option [Q] to quit the program.

Code Running Application:
- Python/Pyzo is HIGHLY RECOMMENDED to run this code. You can run in Pyzo/Python IDLE without limitation but remember to DOWNLOAD THE REQUIRED LIBRARY. If not, the code will not be operating.
- If you use Google Colab, you may not be able to continue to conduct next instruction after you received your first result. This limitation is caused by Google Colab itself. If you are using Google Colab, you must END the program manually after each image output to start the next one. REMEMBER to DOWNLOAD THE REQUIRED LIBRARY. If not, the code will not be operating.
- DO NOT USE VS code because it has some unspecified problem due to the application itself.
- Other applications are NOT TESTED. So, PLEASE use the Pyzo or Python IDLE to operate the program!!!!!!!!!!!

LIMITATIONS:
MAXIMUM YEAR: The model creates a track of ~40,000 km.
   - Inputting years > 45.0 will cause the plume to stop at the final waypoint (China).
   - The simulation does not "loop" continuously forever; it ends after one lap.

*** Gemini is used to generate the explanation and the guidelines for clarity。
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d
import datetime

# --- Part 1: PHYSICS ENGINE ---
def haversine_dist(lat1, lon1, lat2, lon2):
    """
    Calculates true sea distance between two points on Earth.
    This is necessary because earth is a sphere; simple euclidean
    distance is inaccurate over thousands of kilometers
    """
    R = 6371.0 # Earth's radius in km'
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def build_track_data(waypoints):
    """
    Interpolates the discrete waypoints into a continuous path.
    Creates mathematical functions f(distance) -> lat, lon, velocity).
    """
    lats, lons, vels = zip(*waypoints)
    dists = [0.0]

    # Calculate cumulative sea distance along the path.
    for i in range(1, len(waypoints)):
        d = haversine_dist(lats[i-1], lons[i-1], lats[i], lons[i])
        dists.append(dists[-1] + d)

    dists = np.array(dists)

    # Linear Interpolation allows to find position at any specific kilometers
    lat_f = interp1d(dists, lats, kind='linear')
    lon_f = interp1d(dists, lons, kind='linear')
    vel_f = interp1d(dists, vels, kind='linear')

    return dists, lat_f, lon_f, vel_f

class DispersionTrack:
    """
    Manages a single oceanic current pathway (Nothern Route VS Southern Route)
    """
    def __init__(self, name, color, waypoints, cities):
        self.name = name
        self.color = color
        self.cities = cities

        # Build Physics Interpolators
        self.dist_axis, self.lat_f, self.lon_f, self.vel_f = build_track_data(waypoints)
        self.max_dist = self.dist_axis[-1]

        # Snap Cities to Track
        # Cities are points on a map, but our track is a line.
        # Must find the exact point on the line closest to each city
        # to calculate its "Sea Distance" from Fukushima.
        self.city_locs = {}
        for city in self.cities:
            best_s, min_err = 0, 1e9
            # Scan the entire track with 2000 points to find the closest match.
            test_s = np.linspace(0, self.max_dist, 2000)
            for s in test_s:
                err = (self.lat_f(s) - city['lat'])**2 + (self.lon_f(s) - city['lon'])**2
                if err < min_err:
                    min_err = err
                    best_s = s
            self.city_locs[city['name']] = best_s

    def get_position_at_year(self, years):
        """
        Integrates velocity over time: Distance = Integral (Velocity * dt)
        """
        days = years * 365.25
        dt = 10.0 # Integration step size (days)
        t, s = 0, 0
        while t < days and s < self.max_dist:
            v = float(self.vel_f(s))
            s += v * dt
            t += dt
        return min(s, self.max_dist)

    def get_arrival_time(self, target_km):
        """
        Inverse calculation: How long does it take to travel X km?
        Time = Integral (1/Velocity * ds)
        """
        dt = 50.0 # Step size (km)
        s, days = 0, 0
        while s < target_km:
            v = float(self.vel_f(s))
            if v <= 0: v = 0.1 # Prevent division by zero
            days += dt / v
            s += dt
        return days / 365.25

# --- Part 2: Pathway Configuration ---
class GlobalSimulation:
    def __init__(self):
        """
        Note on Coordinates:
        Longitude > 180 are East of the Date Line (e.g., USA is ~240).
        Longitude > 360 represent a full circle (e.g., Dubai is ~415).
        """
        # TRACK A: NORTHERN ROUTE (The "Suez Return")

        pts_a = [
            # [Lat, Lon, Vel]
            # - Japan Coast: Fast Kuroshio current
            # - Open Pacific: Very slow diffusion
            # - Atlantic: Speeds up in Gulf Stream
            # - Suez/Malacca: Bottlenecks slow down transportation
            [37, 141, 15],  # Fukushima
            [42, 160, 10],  # Kuroshio Extension
            [48, 200, 6],   # Open Pacific (Slow Drift)
            [48, 230, 8],   # Reaching North America (Vancouver)
            [8,  280, 5],   # Panama Canal (Diffusion through locks is slow)
            [40, 320, 15],  # Atlantic / Gulf Stream
            [36, 360, 8],   # Gibraltar (Entering Mediterranean)
            [32, 392, 6],   # Suez Canal (Bottleneck)
            [12, 405, 10],  # Red Sea / Gulf of Aden
            [25, 415, 12],  # DUBAI / Persian Gulf
            [10, 435, 15],  # Indian Ocean Crossing
            [5,  460, 10],  # Malacca Strait
            [22, 474, 8],   # Hong Kong, China (Return to Asia)
            [31, 481, 8]    # Shanghai, China (End of Loop)
        ]

        cities_a = [
            {'name': 'Vancouver', 'lat': 49, 'lon': 236},
            {'name': 'San Francisco', 'lat': 37, 'lon': 237},
            {'name': 'Panama City', 'lat': 8, 'lon': 280},
            {'name': 'New York', 'lat': 40, 'lon': 286},
            {'name': 'London', 'lat': 51, 'lon': 359},
            {'name': 'Rome', 'lat': 41, 'lon': 372},
            {'name': 'Cairo', 'lat': 31, 'lon': 391},
            {'name': 'Dubai', 'lat': 25, 'lon': 415},
            {'name': 'Doha', 'lat': 25, 'lon': 411},
            {'name': 'Mumbai', 'lat': 19, 'lon': 432},
            {'name': 'Hong Kong', 'lat': 22, 'lon': 474},
            {'name': 'Shanghai', 'lat': 31, 'lon': 481}
        ]

        # TRACK B: SOUTHERN ROUTE (The "Good Hope Return")
        pts_b = [
            [37, 141, 12],  # Fukushima
            [-34, 151, 15], # Sydney (East Australian Current is fast)
            [-33, 288, 8],  # Chile (Long, slow Pacific crossing)
            [-34, 376, 10], # Cape Town
            [-20, 400, 15], # Madagascar Current
            [0,   430, 12], # Indian Ocean
            [1,   464, 10], # Singapore
            [22,  480, 8]   # Taipei, China (Return to Asia)
        ]

        cities_b = [
            {'name': 'Sydney', 'lat': -33, 'lon': 151},
            {'name': 'Santiago', 'lat': -33, 'lon': 289},
            {'name': 'Rio', 'lat': -22, 'lon': 316},
            {'name': 'Cape Town', 'lat': -33, 'lon': 378},
            {'name': 'Muscat', 'lat': 23, 'lon': 418},
            {'name': 'Colombo', 'lat': 6, 'lon': 439},
            {'name': 'Singapore', 'lat': 1, 'lon': 464},
            {'name': 'Taipei', 'lat': 25, 'lon': 481}
        ]

        self.tracks = [
            DispersionTrack("Northern (Suez Route)", 'red', pts_a, cities_a),
            DispersionTrack("Southern (Africa Route)", 'blue', pts_b, cities_b)
        ]
        self.start_date = datetime.date(2023, 8, 24)

    def run_dashboard(self):
        print("--- GLOBAL DISPERSION MODEL (EXTENDED) ---")
        while True:
            print("\n[1] Visualize Map at Year X")
            print("[2] Check Arrival Time for a City")
            print("[3] List Cities")
            print("[Q] Quit")
            choice = input("Select: ").upper()

            if choice == '1':
                try:
                    y = float(input("Enter Years (e.g., 20.0): "))
                    self.plot_map(y)
                except: print("Invalid number.")

            elif choice == '2':
                city_in = input("Enter City Name: ").strip()
                found = False
                for t in self.tracks:
                    if city_in in t.city_locs:
                        found = True
                        dist = t.city_locs[city_in]
                        years = t.get_arrival_time(dist)
                        date = self.start_date +        datetime.timedelta(days=years*365.25)
                        print(f"\nCity: {city_in.upper()}")
                        print(f"Est. Arrival: {date.strftime('%B %Y')} (T+{years:.1f} Yrs)")
                        if input("Map? (y/n): ") == 'y': self.plot_map(years, city_in)
                        break
                if not found: print("City not found.")

            elif choice == '3':
                print("Cities: " + ", ".join([c['name'] for t in self.tracks for c in t.cities]))

            elif choice == 'Q': break

    def plot_map(self, years, highlight_city=None):
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.set_facecolor('#e0f7fa') # Light blue ocean color
        ax.set_xlim(120, 500) # Extended view to show return to Asia
        ax.set_ylim(-60, 75)
        ax.set_title(f"Global Dispersion: T + {years:.1f} Years", fontsize=14, fontweight='bold')

        # Smart X-Axis Labels (Convert 360+ degrees back to E/W)
        ticks = np.arange(120, 501, 60)
        labels = [f"{t}E" if t<=180 else f"{360-t}W" if t<360 else f"{t-360}E" for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.grid(True, linestyle='--', alpha=0.5, color='white')

        # Draw Simplified Continents (Text labels for context)
        ax.text(140, 38, 'JAPAN', fontweight='bold')
        ax.text(260, 40, 'USA', color='gray')
        ax.text(360, 50, 'EUROPE', color='gray')
        ax.text(415, 28, 'MIDDLE\nEAST', ha='center', fontweight='bold')
        ax.text(475, 35, 'CHINA', ha='center', fontweight='bold')

        # Plot Data
        for t in self.tracks:
            # 1. Plot the Full Path (Faint dotted line)
            s_full = np.linspace(0, t.max_dist, 500)
            ax.plot(t.lon_f(s_full), t.lat_f(s_full), color=t.color, linestyle=':', alpha=0.3)

            # 2. Plot the Active Plume (Solid line up to current year)
            curr_s = t.get_position_at_year(years)
            if curr_s > 100:
                s_plume = np.linspace(0, curr_s, int(curr_s/20)+10)
                ax.plot(t.lon_f(s_plume), t.lat_f(s_plume), color=t.color, linewidth=2, alpha=0.7)

                # 3. Plot the "Head" of the plume
                head_lon, head_lat = t.lon_f(curr_s), t.lat_f(curr_s)
                ax.plot(head_lon, head_lat, marker='o', color=t.color)
                # 4. Dispersion Cloud (Cisual effect growing with time)
                ax.add_patch(mpatches.Circle((head_lon, head_lat), radius=5+years, color=t.color, alpha=0.2))

            # 5. Plot cities along the route
            for city, dist in t.city_locs.items():
                c_lon, c_lat = t.lon_f(dist), t.lat_f(dist)
                # If the plume has passed the city, mark it black, Else gray
                props = ('black', '.') if curr_s >= dist else ('gray', '.')
                ax.plot(c_lon, c_lat, props[1], color=props[0], markersize=5)

                # Highlight Logic for user selection
                if highlight_city and city == highlight_city:
                    ax.plot(c_lon, c_lat, 'r*', markersize=18)
                    t_arr = t.get_arrival_time(dist)
                    ax.text(c_lon+2, c_lat+2, f"{city.upper()}\n{t_arr:.1f} Yrs",
                            fontweight='bold', fontsize=9, bbox=dict(facecolor='white', edgecolor='red'))

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = GlobalSimulation()
    sim.run_dashboard()