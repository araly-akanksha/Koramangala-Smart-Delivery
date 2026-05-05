DRONE_MAX_RAINFALL_MM = 2.5      # Drones grounded above 2.5 mm/hr rainfall
DRONE_MAX_WIND_KMPH   = 25.0     # Drones grounded above 25 km/h wind speed

def is_drone_weather_safe(rainfall_mm: float, wind_kmph: float) -> tuple[bool, str]:
    """
    Returns (True, "") if drone can fly, or (False, reason) if grounded.
    Thresholds:
      - Rainfall > 2.5 mm/hr → grounded (motor/sensor damage risk)
      - Wind    > 25 km/h    → grounded (loss of stability risk)
    """
    reasons = []
    if rainfall_mm > DRONE_MAX_RAINFALL_MM:
        reasons.append(f"rainfall {rainfall_mm:.1f} mm > {DRONE_MAX_RAINFALL_MM} mm limit")
    if wind_kmph > DRONE_MAX_WIND_KMPH:
        reasons.append(f"wind {wind_kmph:.1f} km/h > {DRONE_MAX_WIND_KMPH} km/h limit")
    if reasons:
        return False, " | ".join(reasons)
    return True, ""