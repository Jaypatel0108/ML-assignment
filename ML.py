import pandas as pd
import random
import datetime

# Adjust to generate ~1GB of data
duration_minutes = 720  # 12 hours of data
sampling_rate_hz = 10
num_samples = duration_minutes * 60 * sampling_rate_hz

timestamps = []
cpu_temperatures = []
cpu_usages = []
cpu_loads = []
memory_usages = []
battery_levels = []
cpu_powers = []

start_time = datetime.datetime.now()

for _ in range(num_samples):
    current_time = datetime.datetime.now()
    timestamps.append(current_time)

    # Simulate metrics
    cpu_temperatures.append(random.uniform(30, 80))
    cpu_usages.append(random.uniform(0, 100))
    cpu_loads.append(random.uniform(0, 10))
    memory_usages.append(random.uniform(10, 90))
    battery_levels.append(random.uniform(20, 100))
    cpu_powers.append(random.uniform(0, 50))

    # Introduce random anomalies
    if random.random() < 0.1:
        cpu_usages[-1] = random.uniform(90, 100)  # High CPU usage anomaly
    if random.random() < 0.1:
        cpu_temperatures[-1] = random.uniform(90, 110)  # High temp anomaly

# Save to a CSV
data = {
    'timestamp': timestamps,
    'cpu_temperature': cpu_temperatures,
    'cpu_usage': cpu_usages,
    'cpu_load': cpu_loads,
    'memory_usage': memory_usages,
    'battery_level': battery_levels,
    'cpu_power': cpu_powers
}
df = pd.DataFrame(data)
df.to_csv("hardware_monitor_data.csv", index=False)
from sklearn.ensemble import IsolationForest
import numpy as np

# Load data
df = pd.read_csv("hardware_monitor_data.csv")
df.ffill(inplace=True)

# Anomaly Detection
features = ['cpu_temperature', 'cpu_usage', 'cpu_load', 'memory_usage', 'cpu_power']
clf = IsolationForest(random_state=42, contamination=0.1)
df['anomaly'] = clf.fit_predict(df[features])

# Mark anomalies
df['is_anomaly'] = df['anomaly'] == -1

# Save results
df.to_csv("anomaly_detected.csv", index=False)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Example plot for CPU usage
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['cpu_usage'], label='CPU Usage')
plt.scatter(
    df['timestamp'][df['is_anomaly']],
    df['cpu_usage'][df['is_anomaly']],
    color='red', label='Anomalies'
)
plt.xlabel('Time')
plt.ylabel('CPU Usage (%)')
plt.title('CPU Usage with Anomalies')
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatic tick positioning
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Better formatting
plt.xticks(rotation=45)
plt.show()
