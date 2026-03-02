import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Class names and percentages (MARIDA example)
classes = [
    'Marine Debris (Class 1)',
    'Marine Water (Class 7)',
    'Ship',
    'Sargassum',
    'Sea Ice',
    'Foam',
    'Clouds',
    'Shadow',
    'Other'
]
percentages = [1.2, 68.0, 5.0, 8.0, 3.0, 2.0, 7.0, 3.0, 2.8]

# Sort for nicer plotting
order = np.argsort(percentages)[::-1]
classes_sorted = [classes[i] for i in order]
perc_sorted = [percentages[i] for i in order]

# Colors: highlight Marine Water and Marine Debris
colors = []
for c in classes_sorted:
    if 'Marine Water' in c:
        colors.append('#2b83ba')  # blue
    elif 'Marine Debris' in c:
        colors.append('#d7191c')  # red
    else:
        colors.append('#b0b0b0')  # gray

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(classes_sorted, perc_sorted, color=colors)
ax.invert_yaxis()
ax.set_xlabel('Percentage of annotated pixels (%)')
ax.set_title('Class distribution of annotated pixels in the MARIDA dataset')

# Annotate percentages
for bar, pct in zip(bars, perc_sorted):
    w = bar.get_width()
    ax.text(w + 0.6, bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}%', va='center', fontsize=9)

# Compute and annotate imbalance ratio using Marine Water vs Marine Debris
try:
    p_water = perc_sorted[[i for i,c in enumerate(classes_sorted) if 'Marine Water' in c][0]]
    p_debris = perc_sorted[[i for i,c in enumerate(classes_sorted) if 'Marine Debris' in c][0]]
    ratio = p_water / p_debris if p_debris > 0 else float('inf')
    ax.text(0.99, 0.05, f'Imbalance ≈ 1:{ratio:.0f}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#555555'))
except Exception:
    pass

plt.tight_layout()
# Save as JPEG at high quality. Matplotlib's Agg JPEG writer doesn't accept
# a `quality` argument, so save as a temporary PNG then convert with Pillow.
out_path = 'figures/class_distribution.jpg'
tmp_png = out_path.replace('.jpg', '.png')
plt.savefig(tmp_png, dpi=300)
try:
    img = Image.open(tmp_png).convert('RGB')
    img.save(out_path, 'JPEG', quality=95)
    print(f'Saved {out_path}')
finally:
    try:
        os.remove(tmp_png)
    except OSError:
        pass
