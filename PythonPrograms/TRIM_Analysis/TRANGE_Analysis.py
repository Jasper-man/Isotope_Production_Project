import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. PARSE EXYZ.txt SAFELY
# ==========================================
trange_file = "EXYZ.txt"
data_rows = []

try:
    with open(trange_file, "r", errors="ignore") as file:
        for line in file:
            parts = line.strip().split()
            # EXYZ has 5 columns: Ion_Num, Energy(eV), Depth_X(A), Y(A), Z(A)
            if len(parts) >= 5 and parts[0].isdigit():
                try:
                    ion_num = int(parts[0])
                    # parts[1] is Energy. parts[2] is Depth X.
                    depth_A = float(parts[2].replace(',', '.'))
                    y_A = float(parts[3].replace(',', '.'))
                    z_A = float(parts[4].replace(',', '.'))
                    data_rows.append((ion_num, depth_A, y_A, z_A))
                except ValueError:
                    continue

    df = pd.DataFrame(data_rows, columns=["Ion_Num", "Depth_A", "Y_A", "Z_A"])
    
    # Keep only the absolute last recorded coordinate for each unique proton
    df = df.drop_duplicates(subset=['Ion_Num'], keep='last')
    
    # Convert Angstroms to millimeters (1 mm = 1e7 Angstroms)
    df["Depth_mm"] = df["Depth_A"] / 1e7

except FileNotFoundError:
    print(f"Error: Could not find '{trange_file}' in the current directory.")
    exit()

if df.empty:
    print("No valid particle stopping data found in EXYZ.txt.")
    exit()

# ==========================================
# 2. BIN PARTICLES BY TARGET LAYER
# ==========================================
# Define cumulative layer boundaries in mm
be1_max = 4.0     # 0.0 to 4.0 mm (4.0 mm Be)
water_max = 4.7   # 4.0 to 4.7 mm (0.7 mm Water)
be2_max = 5.7     # 4.7 to 5.7 mm (1.0 mm Be)

stopped_be1 = df[df["Depth_mm"] <= be1_max]
stopped_water = df[(df["Depth_mm"] > be1_max) & (df["Depth_mm"] <= water_max)]
stopped_be2 = df[(df["Depth_mm"] > water_max) & (df["Depth_mm"] <= be2_max)]

counts = [len(stopped_be1), len(stopped_water), len(stopped_be2)]
total_stopped = len(df)
percentages = [(c / total_stopped) * 100 if total_stopped > 0 else 0 for c in counts]

layer_labels = [
    "Layer 1: Be\n(0 - 4.0 mm)",
    "Layer 2: Water\n(4.0 - 4.7 mm)",
    "Layer 3: Be\n(4.7 - 5.7 mm)"
]

print("--- PROTON STOPPING BREAKDOWN ---")
print(f"Total Stopped Protons : {total_stopped}")
for label, count, pct in zip(layer_labels, counts, percentages):
    clean_label = label.replace('\n', ' ')
    print(f"{clean_label:<28}: {count:>6} protons ({pct:.2f}%)")

# ==========================================
# 3. VISUALIZE THE BREAKDOWN
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("TRIM Proton Stopping Distribution (Be - Water - Be Target)", fontsize=14, fontweight="bold")

# --- Panel 1: Continuous Stopping Depth Histogram ---
bins = np.linspace(0, be2_max, 120)
ax1.hist(df["Depth_mm"], bins=bins, color="#2b5c8f", edgecolor="black", linewidth=0.4, zorder=3)

# Shade layer backgrounds
ax1.axvspan(0, be1_max, color="#d9d9d9", alpha=0.5, label="Layer 1: Be (4.0 mm)")
ax1.axvspan(be1_max, water_max, color="#a6bddb", alpha=0.5, label="Layer 2: Water (0.7 mm)")
ax1.axvspan(water_max, be2_max, color="#bdbdbd", alpha=0.5, label="Layer 3: Be (1.0 mm)")

# Draw boundary lines
ax1.axvline(be1_max, color="red", linestyle="--", linewidth=1.2, zorder=4)
ax1.axvline(water_max, color="red", linestyle="--", linewidth=1.2, zorder=4)

ax1.set_title("Depth Profile (Bragg Stopping Distribution)")
ax1.set_xlabel("Cumulative Target Depth (mm)")
ax1.set_ylabel("Number of Stopped Protons")
ax1.set_xlim(0, be2_max)
ax1.legend(loc="upper left")
ax1.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

# --- Panel 2: Total Stopped Per Layer Bar Chart ---
bar_colors = ["#737373", "#3690c0", "#525252"]
bars = ax2.bar(layer_labels, counts, color=bar_colors, edgecolor="black", width=0.55, zorder=3)

# Add count and percentage labels above each bar
max_count = max(counts) if counts else 1
ax2.set_ylim(0, max_count * 1.15)

for bar, count, pct in zip(bars, counts, percentages):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + (max_count * 0.02),
        f"{count:,}\n({pct:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

ax2.set_title("Total Protons Stopped by Layer")
ax2.set_ylabel("Proton Count")
ax2.grid(axis="y", linestyle=":", alpha=0.6, zorder=0)

plt.tight_layout()
plt.savefig("proton_stopping_breakdown.png", dpi=300)
plt.show()