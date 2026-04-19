def run_ensemble(lat0, lon0, ocean_field, wind_field,
                 n_particles=DRIFT_ENSEMBLE_N,
                 dt=DRIFT_DT_SECONDS,
                 total_hours=DRIFT_HOURS,
                 wind_coeff=DRIFT_WIND_COEFF,
                 compute_ellipse_every=10):
    """
    Returns:
        {
          step: {
            "time_hours": float,
            "positions": (N,2),
            "mean_lat": float,
            "mean_lon": float,
            "ellipse": (...)  # optional
          }
        }
    """

    n_steps = int(total_hours * 3600 / dt)

    # Perturbed ocean fields for ensemble
    perturbed_fields = [_perturb_field(ocean_field) for _ in range(n_particles)]

    # Initial positions
    init_lats = lat0 + np.random.normal(0, 0.005, n_particles)
    init_lons = lon0 + np.random.normal(0, 0.005, n_particles)

    positions = np.column_stack([init_lats, init_lons])
    results = {}

    for step in range(1, n_steps + 1):
        new_pos = np.zeros_like(positions)

        for i in range(n_particles):
            new_pos[i] = _rk4_step(
                positions[i, 0], positions[i, 1],
                perturbed_fields[i], wind_field,
                wind_coeff, dt
            )

        positions = new_pos

        # Store EVERY epoch
        time_hours = step * dt / 3600.0

        entry = {
            "time_hours": time_hours,
            "positions": positions.copy(),
            "mean_lat": float(positions[:, 0].mean()),
            "mean_lon": float(positions[:, 1].mean()),
        }

        # Compute ellipse only every few steps (performance optimization)
        if step % compute_ellipse_every == 0:
            entry["ellipse"] = _confidence_ellipse(positions)

        results[step] = entry

    return results