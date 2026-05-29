from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Analysis_Outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT = OUT_DIR / "09_correct_typical_day_solar_vs_demand.png"


def font(size=18, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for item in candidates:
        if Path(item).exists():
            return ImageFont.truetype(item, size)
    return ImageFont.load_default()


W, H = 1500, 920
img = Image.new("RGB", (W, H), "#ffffff")
d = ImageDraw.Draw(img)

F_TITLE = font(34, True)
F_SUB = font(19)
F_AXIS = font(17)
F_SMALL = font(14)
F_NOTE = font(18, True)

left, top, right, bottom = 125, 165, 1390, 720

d.text((58, 38), "Exhibit 01", fill="#64748b", font=font(19, True))
d.text((58, 72), "A Correct Typical Day on India's Grid: Solar Output vs Demand", fill="#0f172a", font=F_TITLE)
d.text(
    (58, 120),
    "Illustrative normalized shape: solar peaks at midday, but demand often peaks after sunset.",
    fill="#475569",
    font=F_SUB,
)

# Normalized illustrative hourly profile.
hours = np.arange(0, 25)
solar = np.maximum(0, np.sin((hours - 6) / 12 * np.pi)) ** 1.55
solar[hours < 6] = 0
solar[hours > 18] = 0

morning = 0.20 * np.exp(-((hours - 9) / 4.2) ** 2)
evening = 0.40 * np.exp(-((hours - 20) / 3.2) ** 2)
base = 0.52 + 0.06 * np.sin((hours - 5) / 24 * 2 * np.pi)
demand = base + morning + evening
demand = demand / demand.max()

solar = solar * 0.82

def sx(hour):
    return left + (hour / 24) * (right - left)


def sy(value):
    return bottom - value * (bottom - top)


# Grid and axes.
for pct in np.linspace(0, 1, 6):
    y = sy(pct)
    d.line((left, y, right, y), fill="#e2e8f0", width=1)
    d.text((42, y - 9), f"{int(pct * 100)}%", fill="#64748b", font=F_SMALL)

for hour in [0, 6, 12, 15, 18, 20, 22, 24]:
    x = sx(hour)
    d.line((x, top, x, bottom), fill="#f1f5f9", width=1)
    label = "12 AM" if hour in [0, 24] else ("12 PM" if hour == 12 else f"{hour if hour <= 12 else hour - 12} {'AM' if hour < 12 else 'PM'}")
    d.text((x - 24, bottom + 16), label, fill="#475569", font=F_SMALL)

d.line((left, bottom, right, bottom), fill="#0f172a", width=2)
d.line((left, top, left, bottom), fill="#0f172a", width=2)
d.text((left, top - 32), "Normalized output / demand", fill="#334155", font=F_AXIS)
d.text((left + 535, bottom + 55), "Time of day", fill="#334155", font=F_AXIS)

# Evening gap shading: demand minus solar from 18 to 22.
gap_poly = []
for h in np.linspace(18, 22, 50):
    s = np.interp(h, hours, solar)
    dem = np.interp(h, hours, demand)
    gap_poly.append((sx(h), sy(dem)))
for h in np.linspace(22, 18, 50):
    s = np.interp(h, hours, solar)
    gap_poly.append((sx(h), sy(s)))
d.polygon(gap_poly, fill="#fee2e2")

# Midday surplus/good solar window shading.
solar_poly = [(sx(6), sy(0))]
for h in np.linspace(6, 18, 100):
    solar_poly.append((sx(h), sy(np.interp(h, hours, solar))))
solar_poly.append((sx(18), sy(0)))
d.polygon(solar_poly, fill="#dcfce7")

# Redraw grid lightly over fills.
for pct in np.linspace(0, 1, 6):
    y = sy(pct)
    d.line((left, y, right, y), fill="#e2e8f0", width=1)

# Curves.
solar_pts = [(sx(h), sy(v)) for h, v in zip(hours, solar)]
demand_pts = [(sx(h), sy(v)) for h, v in zip(hours, demand)]
d.line(solar_pts, fill="#16a34a", width=6, joint="curve")
d.line(demand_pts, fill="#2563eb", width=6, joint="curve")

for x, y in solar_pts[::3]:
    d.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#16a34a")
for x, y in demand_pts[::3]:
    d.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#2563eb")

# Legend.
legend_x, legend_y = 930, 555
d.rounded_rectangle((legend_x, legend_y, legend_x + 365, legend_y + 120), radius=12, fill="#ffffff", outline="#cbd5e1", width=2)
d.line((legend_x + 24, legend_y + 32, legend_x + 78, legend_y + 32), fill="#2563eb", width=6)
d.text((legend_x + 92, legend_y + 20), "Demand", fill="#0f172a", font=F_AXIS)
d.line((legend_x + 24, legend_y + 70, legend_x + 78, legend_y + 70), fill="#16a34a", width=6)
d.text((legend_x + 92, legend_y + 58), "Solar output", fill="#0f172a", font=F_AXIS)
d.rectangle((legend_x + 24, legend_y + 94, legend_x + 78, legend_y + 110), fill="#fee2e2")
d.text((legend_x + 92, legend_y + 89), "Evening peak gap", fill="#0f172a", font=F_AXIS)

# Callouts.
def callout(x, y, text, color, align="left"):
    box_w = 320
    lines = []
    for part in text.split("\n"):
        lines.extend([part] if len(part) < 34 else [part[:34], part[34:]])
    box_h = 30 + len(lines) * 22
    x0 = x if align == "left" else x - box_w
    d.rounded_rectangle((x0, y, x0 + box_w, y + box_h), radius=10, fill="#ffffff", outline=color, width=2)
    yy = y + 14
    for line in lines:
        d.text((x0 + 14, yy), line, fill=color, font=F_AXIS)
        yy += 22

callout(sx(6.4), sy(0.26), "6 AM\nSolar starts rising", "#16a34a")
callout(sx(10.1), sy(0.91), "12 PM\nSolar is near peak", "#16a34a", align="left")
callout(sx(14.7), sy(0.69), "3 PM\nSolar useful,\ndemand rising", "#ca8a04")
callout(sx(17.7), sy(0.44), "6 PM\nSolar drops quickly", "#ea580c")
callout(sx(23.2), sy(0.98), "8 PM\nDemand peaks,\nsolar is zero", "#dc2626", align="right")

d.text((84, 785), "Correct interpretation", fill="#0f172a", font=F_NOTE)
d.text(
    (84, 818),
    "Midday solar is abundant and cheap, but it does not directly meet the evening peak without storage, hydro, gas, coal, or demand response.",
    fill="#334155",
    font=F_SUB,
)
d.text(
    (84, 852),
    "So the right chart should compare two curves, not one unclear set of bars.",
    fill="#334155",
    font=F_SUB,
)

img.save(OUT)
print(OUT)
