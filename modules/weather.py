DRONE_MAX_RAINFALL_MM = 2.5
DRONE_MAX_WIND_KMPH = 25.0

def is_drone_weather_safe(rainfall_mm, wind_kmph):
    reasons = []
    if rainfall_mm > DRONE_MAX_RAINFALL_MM:
        reasons.append(f"rainfall {rainfall_mm:.1f} mm > {DRONE_MAX_RAINFALL_MM} mm limit")
    if wind_kmph > DRONE_MAX_WIND_KMPH:
        reasons.append(f"wind {wind_kmph:.1f} km/h > {DRONE_MAX_WIND_KMPH} km/h limit")
    if reasons:
        return False, " | ".join(reasons)
    return True, ""