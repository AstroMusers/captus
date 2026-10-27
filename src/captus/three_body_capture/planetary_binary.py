"""Equal-mass binary-planet formation through three-body capture."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy import constants as const
from astropy import units as u


def _positive_scalar_quantity(value: u.Quantity, unit: u.UnitBase, name: str) -> u.Quantity:
    try:
        quantity = u.Quantity(value).to(unit)
    except (TypeError, ValueError, u.UnitConversionError) as error:
        raise ValueError(f"{name} must be a quantity convertible to {unit}.") from error
    if not quantity.isscalar or not np.isfinite(quantity.value) or quantity.value <= 0:
        raise ValueError(f"{name} must be a finite, positive scalar.")
    return quantity


def compute_binary_planet_formation_rate(
    planet_mass: u.Quantity,
    planet_number_density: u.Quantity,
    velocity_dispersion: u.Quantity,
    maximum_semimajor_axis: u.Quantity | None = None,
    chi1: float = 10.0,
) -> dict[str, u.Quantity]:
    """Compute the hard-binary formation rate density for three equal-mass planets.

    This implements the equal-mass, point-particle fit in Eq. 23 of
    Atallah+2024 (arXiv:2402.12429). The three initially unbound planets share
    an isotropic Maxwellian velocity distribution with one-dimensional
    dispersion ``velocity_dispersion``. The rate counts all pairings and is
    cumulative for binary semimajor axes up to ``maximum_semimajor_axis``.
    Only the hard-binary regime, ``a <= a_fs``, and ``chi1 >= 10`` are supported.
    The number density is the density of each of the three identical populations.

    Returns the rate density, hard-binary semimajor-axis scale, and interaction
    radius as Astropy quantities.
    """

    mass = _positive_scalar_quantity(planet_mass, u.kg, "planet_mass")
    density = _positive_scalar_quantity(
        planet_number_density, 1 / u.m**3, "planet_number_density"
    )
    dispersion = _positive_scalar_quantity(
        velocity_dispersion, u.m / u.s, "velocity_dispersion"
    )
    if not np.isfinite(chi1) or chi1 < 10.0:
        raise ValueError("chi1 must be finite and at least 10.")

    # For identical Maxwellian populations, <v_rel^2> = 6 sigma^2.
    hard_semimajor_axis = (const.G * mass / (3.0 * dispersion**2)).to(u.m)  # [m]
    if maximum_semimajor_axis is None:
        maximum_axis = hard_semimajor_axis
    else:
        maximum_axis = _positive_scalar_quantity(
            maximum_semimajor_axis, u.m, "maximum_semimajor_axis"
        )
    if maximum_axis > hard_semimajor_axis:
        raise ValueError(
            "maximum_semimajor_axis must not exceed the hard-binary scale."
        )

    interaction_radius = (2.0 * chi1 * hard_semimajor_axis).to(u.m)  # [m]
    axis_fraction = (maximum_axis / hard_semimajor_axis).to_value(u.dimensionless_unscaled)
    formation_rate_density = (
        0.081
        * const.G**5
        * mass**5
        * density**3
        / dispersion**9
        * axis_fraction**3
        * np.sqrt(chi1)
    ).to(1 / (u.m**3 * u.s))  # [1 / (m^3 s)]

    return {
        "formation_rate_density": formation_rate_density,
        "hard_semimajor_axis": hard_semimajor_axis,
        "maximum_semimajor_axis": maximum_axis,
        "interaction_radius": interaction_radius,
    }


@dataclass(frozen=True)
class PlanetBody:
    """A gravitating body used in a three-body encounter."""

    name: str
    mass: u.Quantity
    radius: u.Quantity | None = None


@dataclass(frozen=True)
class ThreeBodyEncounter:
    """Initial state and integration controls for three unbound bodies."""

    bodies: tuple[PlanetBody, PlanetBody, PlanetBody]
    positions: u.Quantity
    velocities: u.Quantity
    integration_time: u.Quantity
    checkpoints: int = 200
    escape_separation_factor: float = 20.0
    interaction_radius: u.Quantity | None = None
    epoch_time: u.Quantity | None = None


@dataclass(frozen=True)
class BinaryCaptureResult:
    """Final classification and conservation diagnostics for an encounter."""

    status: str
    captured: bool
    binary_indices: tuple[int, int] | None
    single_index: int | None
    semi_major_axis: u.Quantity | None
    eccentricity: float | None
    integration_time: u.Quantity
    energy_error: float
    angular_momentum_error: float
    initial_energy: u.Quantity
    final_energy: u.Quantity

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record with physical quantities in explicit units."""

        return {
            "status": self.status,
            "captured": self.captured,
            "binary_indices": None if self.binary_indices is None else list(self.binary_indices),
            "single_index": self.single_index,
            "semi_major_axis_au": (
                None if self.semi_major_axis is None else float(self.semi_major_axis.to_value(u.au))
            ),
            "eccentricity": self.eccentricity,
            "integration_time_yr": float(self.integration_time.to_value(u.yr)),
            "energy_error": self.energy_error,
            "angular_momentum_error": self.angular_momentum_error,
            "initial_energy_j": float(self.initial_energy.to_value(u.J)),
            "final_energy_j": float(self.final_energy.to_value(u.J)),
        }


def _validate_three_body_encounter(encounter: ThreeBodyEncounter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(encounter.bodies) != 3:
        raise ValueError("A three-body encounter must contain exactly three bodies.")
    masses = np.asarray([body.mass.to_value(u.Msun) for body in encounter.bodies])
    positions = u.Quantity(encounter.positions).to_value(u.au)
    velocities = u.Quantity(encounter.velocities).to_value(u.au / u.yr)
    duration = _positive_scalar_quantity(encounter.integration_time, u.yr, "integration_time")
    if positions.shape != (3, 3) or velocities.shape != (3, 3):
        raise ValueError("positions and velocities must have shape (3, 3).")
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("All body masses must be finite and positive.")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError("Initial positions and velocities must be finite.")
    if isinstance(encounter.checkpoints, bool) or encounter.checkpoints < 1:
        raise ValueError("checkpoints must be a positive integer.")
    if not np.isfinite(encounter.escape_separation_factor) or encounter.escape_separation_factor <= 1:
        raise ValueError("escape_separation_factor must be finite and greater than one.")

    relative_pair_energies = []
    for first in range(3):
        for second in range(first + 1, 3):
            separation = np.linalg.norm(positions[second] - positions[first])
            relative_speed = np.linalg.norm(velocities[second] - velocities[first])
            if separation <= 0.0:
                raise ValueError("Initial positions must be distinct.")
            gravitational_parameter = 4.0 * np.pi**2 * (masses[first] + masses[second])
            relative_pair_energies.append(
                0.5 * relative_speed**2 - gravitational_parameter / separation
            )
    if min(relative_pair_energies) < 0.0:
        raise ValueError("All pairs must be initially unbound.")

    center_velocity = np.average(velocities, axis=0, weights=masses)
    kinetic_energy = 0.5 * np.sum(masses[:, None] * (velocities - center_velocity) ** 2)
    potential_energy = sum(
        -4.0 * np.pi**2 * masses[first] * masses[second]
        / np.linalg.norm(positions[second] - positions[first])
        for first in range(3)
        for second in range(first + 1, 3)
    )
    if kinetic_energy + potential_energy <= 0.0:
        raise ValueError("The initial three-body system must have positive total energy.")
    if not duration.isscalar:
        raise ValueError("integration_time must be a positive scalar.")
    return masses, positions, velocities


def sample_equal_mass_unbound_encounter(
    planet_mass: u.Quantity,
    velocity_dispersion: u.Quantity,
    *,
    chi1: float = 10.0,
    seed: int | None = 0,
    planet_radius: u.Quantity | None = None,
    checkpoints: int = 200,
    escape_separation_factor: float = 20.0,
    max_resamples: int = 10000,
) -> ThreeBodyEncounter:
    """Draw an initially unbound equal-mass encounter using Atallah+2024's geometry.

    Three positions are uniform in the interaction sphere, and all velocity
    components are drawn from one isotropic Maxwellian with the same dispersion.
    The positions are shifted backward by 15 interaction radii divided by the
    slowest body's speed. Bound initial pairs and nonpositive total energy are
    rejected. The setup isolates one encounter and omits the surrounding cluster.
    """

    mass = _positive_scalar_quantity(planet_mass, u.kg, "planet_mass")
    dispersion = _positive_scalar_quantity(
        velocity_dispersion, u.m / u.s, "velocity_dispersion"
    )
    if not np.isfinite(chi1) or chi1 < 10.0:
        raise ValueError("chi1 must be finite and at least 10.")
    if not np.isfinite(escape_separation_factor) or escape_separation_factor <= 1.0:
        raise ValueError("escape_separation_factor must be finite and greater than one.")
    if isinstance(max_resamples, bool) or max_resamples < 1:
        raise ValueError("max_resamples must be a positive integer.")
    radius = None if planet_radius is None else _positive_scalar_quantity(
        planet_radius, u.m, "planet_radius"
    )

    mass_kg = mass.to_value(u.kg)
    sigma = dispersion.to_value(u.m / u.s)
    hard_axis = (const.G * mass / (3.0 * dispersion**2)).to(u.m)
    interaction_radius = (2.0 * chi1 * hard_axis).to(u.m)
    rng = np.random.default_rng(seed)

    for _ in range(max_resamples):
        velocities = rng.normal(0.0, sigma, size=(3, 3))
        speeds = np.linalg.norm(velocities, axis=1)
        if np.min(speeds) <= 0.0:
            continue
        directions = rng.normal(size=(3, 3))
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms == 0.0):
            continue
        radii = interaction_radius.to_value(u.m) * rng.random(3) ** (1.0 / 3.0)
        positions_at_epoch = directions / norms[:, None] * radii[:, None]
        epoch_time = 15.0 * interaction_radius.to_value(u.m) / np.min(speeds)
        positions = positions_at_epoch - velocities * epoch_time
        encounter = ThreeBodyEncounter(
            bodies=tuple(
                PlanetBody(
                    name=f"planet_{index + 1}",
                    mass=mass,
                    radius=radius,
                )
                for index in range(3)
            ),
            positions=positions * u.m,
            velocities=velocities * u.m / u.s,
            integration_time=2.0 * epoch_time * u.s,
            checkpoints=checkpoints,
            escape_separation_factor=escape_separation_factor,
            interaction_radius=interaction_radius,
            epoch_time=epoch_time * u.s,
        )
        try:
            _validate_three_body_encounter(encounter)
        except ValueError:
            continue
        return encounter
    raise RuntimeError(
        f"Could not draw an initially unbound encounter in {max_resamples} attempts."
    )


def _find_escaped_binary(simulation: Any, masses: np.ndarray, separation_factor: float):
    particles = simulation.particles
    candidates = []
    for first in range(3):
        for second in range(first + 1, 3):
            single = next(index for index in range(3) if index not in (first, second))
            particle_first = particles[first]
            particle_second = particles[second]
            relative_position = np.array([
                particle_second.x - particle_first.x,
                particle_second.y - particle_first.y,
                particle_second.z - particle_first.z,
            ])
            relative_velocity = np.array([
                particle_second.vx - particle_first.vx,
                particle_second.vy - particle_first.vy,
                particle_second.vz - particle_first.vz,
            ])
            separation = np.linalg.norm(relative_position)
            gravitational_parameter = simulation.G * (masses[first] + masses[second])
            specific_energy = (
                0.5 * np.dot(relative_velocity, relative_velocity)
                - gravitational_parameter / separation
            )
            if specific_energy >= 0.0:
                continue
            semi_major_axis = -gravitational_parameter / (2.0 * specific_energy)
            angular_momentum = np.cross(relative_position, relative_velocity)
            eccentricity_vector = (
                np.cross(relative_velocity, angular_momentum) / gravitational_parameter
                - relative_position / separation
            )
            eccentricity = float(np.linalg.norm(eccentricity_vector))
            binary_mass = masses[first] + masses[second]
            binary_position = (
                masses[first] * np.array([particle_first.x, particle_first.y, particle_first.z])
                + masses[second] * np.array([particle_second.x, particle_second.y, particle_second.z])
            ) / binary_mass
            binary_velocity = (
                masses[first] * np.array([particle_first.vx, particle_first.vy, particle_first.vz])
                + masses[second] * np.array([particle_second.vx, particle_second.vy, particle_second.vz])
            ) / binary_mass
            particle_single = particles[single]
            outer_position = np.array([particle_single.x, particle_single.y, particle_single.z]) - binary_position
            outer_velocity = np.array([particle_single.vx, particle_single.vy, particle_single.vz]) - binary_velocity
            outer_separation = np.linalg.norm(outer_position)
            outer_energy = (
                0.5 * np.dot(outer_velocity, outer_velocity)
                - simulation.G * np.sum(masses) / outer_separation
            )
            moving_outward = np.dot(outer_position, outer_velocity) > 0.0
            if (
                outer_energy > 0.0
                and moving_outward
                and outer_separation >= separation_factor * semi_major_axis
                and eccentricity < 1.0
            ):
                candidates.append((specific_energy, (first, second), single, semi_major_axis, eccentricity))
    if not candidates:
        return None
    _, binary_indices, single_index, semi_major_axis, eccentricity = min(candidates)
    return binary_indices, single_index, semi_major_axis, eccentricity


def simulate_three_body_capture(encounter: ThreeBodyEncounter) -> BinaryCaptureResult:
    """Integrate one Newtonian three-planet encounter with REBOUND/IAS15."""

    import rebound

    masses, positions, velocities = _validate_three_body_encounter(encounter)
    simulation = rebound.Simulation()
    simulation.units = ("AU", "yr", "Msun")
    simulation.integrator = "ias15"
    simulation.collision = "direct"
    simulation.collision_resolve = "halt"
    for index, body in enumerate(encounter.bodies):
        body_radius = 0.0 if body.radius is None else body.radius.to_value(u.au)
        simulation.add(
            m=masses[index],
            r=body_radius,
            x=positions[index, 0],
            y=positions[index, 1],
            z=positions[index, 2],
            vx=velocities[index, 0],
            vy=velocities[index, 1],
            vz=velocities[index, 2],
        )
    simulation.move_to_com()
    initial_energy_code = float(simulation.energy())
    initial_angular_momentum = np.array([
        sum(particle.m * (particle.y * particle.vz - particle.z * particle.vy) for particle in simulation.particles),
        sum(particle.m * (particle.z * particle.vx - particle.x * particle.vz) for particle in simulation.particles),
        sum(particle.m * (particle.x * particle.vy - particle.y * particle.vx) for particle in simulation.particles),
    ])
    integration_time = encounter.integration_time.to(u.yr)
    capture = None
    status = "unresolved"
    time_grid = np.linspace(
        0.0,
        integration_time.to_value(u.yr),
        encounter.checkpoints + 1,
    )[1:]
    for endpoint in time_grid:
        try:
            simulation.integrate(float(endpoint), exact_finish_time=1)
        except rebound.Collision:
            status = "collision"
            break
        capture = _find_escaped_binary(
            simulation,
            masses,
            encounter.escape_separation_factor,
        )
        if capture is not None:
            status = "binary_capture"
            break

    final_energy_code = float(simulation.energy())
    final_angular_momentum = np.array([
        sum(particle.m * (particle.y * particle.vz - particle.z * particle.vy) for particle in simulation.particles),
        sum(particle.m * (particle.z * particle.vx - particle.x * particle.vz) for particle in simulation.particles),
        sum(particle.m * (particle.x * particle.vy - particle.y * particle.vx) for particle in simulation.particles),
    ])
    energy_error = abs(final_energy_code - initial_energy_code) / max(
        abs(initial_energy_code), np.finfo(float).tiny
    )
    angular_momentum_error = np.linalg.norm(
        final_angular_momentum - initial_angular_momentum
    ) / max(np.linalg.norm(initial_angular_momentum), np.finfo(float).tiny)
    energy_unit = u.Msun * (u.au / u.yr) ** 2

    binary_indices = None
    single_index = None
    semi_major_axis = None
    eccentricity = None
    if capture is not None:
        binary_indices, single_index, semi_major_axis_code, eccentricity = capture
        semi_major_axis = semi_major_axis_code * u.au

    return BinaryCaptureResult(
        status=status,
        captured=capture is not None,
        binary_indices=binary_indices,
        single_index=single_index,
        semi_major_axis=semi_major_axis,
        eccentricity=eccentricity,
        integration_time=simulation.t * u.yr,
        energy_error=float(energy_error),
        angular_momentum_error=float(angular_momentum_error),
        initial_energy=initial_energy_code * energy_unit.to(u.J) * u.J,
        final_energy=final_energy_code * energy_unit.to(u.J) * u.J,
    )


def _poisson_count_log_likelihood(values: np.ndarray, state: dict[str, float]) -> float:
    scale = float(values[0])
    if not np.isfinite(scale) or scale <= 0.0:
        return -np.inf
    expected_count = scale * state["base_expected_count"]
    observed_count = int(state["observed_count"])
    from pcat.likelihood import poisson_log_likelihood

    return float(poisson_log_likelihood(observed_count, expected_count, include_constant=True))


def infer_binary_planet_rate_normalization_pcat(
    formation_rate_density: u.Quantity,
    cluster_volume: u.Quantity,
    exposure_time: u.Quantity,
    observed_count: int,
    *,
    normalization_bounds: tuple[float, float] = (0.01, 5.0),
    chain_count: int = 4,
    sample_count: int = 1000,
    burn_count: int = 500,
    boolcheckconv: bool = False,
    seed: int | None = 0,
    pathbase: str | Path | None = None,
    typeverb: int = 0,
) -> dict[str, Any]:
    """Infer a rate-normalization factor with PCAT's fixed-dimensional sampler.

    The Poisson likelihood compares ``observed_count`` with the expected count
    from the supplied rate density, volume, exposure time, and fitted
    normalization. It omits survey selection, binary disruption, and
    contamination.
    """

    try:
        from pcat.fixed import sample_fixed_chains
    except ModuleNotFoundError as error:
        if error.name == "pcat" or (error.name or "").startswith("pcat."):
            raise ImportError(
                "PCAT inference requires the optional Captus extra. Install "
                "with `python -m pip install -e '.[inference]'`."
            ) from error
        raise

    rate_density = _positive_scalar_quantity(
        formation_rate_density, 1 / (u.m**3 * u.s), "formation_rate_density"
    )
    volume = _positive_scalar_quantity(cluster_volume, u.m**3, "cluster_volume")
    duration = _positive_scalar_quantity(exposure_time, u.s, "exposure_time")
    if isinstance(observed_count, bool) or not isinstance(observed_count, (int, np.integer)):
        raise ValueError("observed_count must be a nonnegative integer.")
    if observed_count < 0:
        raise ValueError("observed_count must be a nonnegative integer.")
    if chain_count < 1 or sample_count < 1 or burn_count < 0:
        raise ValueError("chain_count and sample_count must be positive; burn_count cannot be negative.")

    lower, upper = normalization_bounds
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= lower:
        raise ValueError("normalization_bounds must satisfy 0 < lower < upper.")
    base_expected_count = float((rate_density * volume * duration).to_value(u.dimensionless_unscaled))
    if not np.isfinite(base_expected_count) or base_expected_count <= 0.0:
        raise ValueError("The baseline expected formation count must be positive and finite.")

    sampler_output = sample_fixed_chains(
        {"observed_count": int(observed_count), "base_expected_count": base_expected_count},
        _poisson_count_log_likelihood,
        None,
        ["rate_normalization"],
        ["self"],
        np.array([lower]),
        np.array([upper]),
        np.zeros(1),
        np.ones(1),
        None,
        chain_count,
        sample_count,
        burn_count,
        pathbase=None if pathbase is None else str(pathbase),
        typeverb=typeverb,
        boolcheckconv=boolcheckconv,
        seed=seed,
    )
    normalization_samples = np.asarray(sampler_output[0], dtype=float)[..., 0]
    log_probability = np.asarray(sampler_output[1], dtype=float)
    return {
        "normalization_samples": normalization_samples,
        "formation_rate_density_samples": rate_density * normalization_samples,
        "expected_formation_count_samples": base_expected_count * normalization_samples,
        "log_probability": log_probability,
        "base_expected_count": base_expected_count,
        "sampler": "PCAT",
    }


def main(argv: list[str] | None = None) -> int:
    """Run one seeded equal-mass encounter and print its JSON-ready outcome."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass-mjup", type=float, default=1.0)
    parser.add_argument("--velocity-dispersion-kms", type=float, default=0.3)
    parser.add_argument("--chi1", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints", type=int, default=200)
    parser.add_argument("--escape-separation-factor", type=float, default=20.0)
    arguments = parser.parse_args(argv)
    encounter = sample_equal_mass_unbound_encounter(
        arguments.mass_mjup * u.M_jup,
        arguments.velocity_dispersion_kms * u.km / u.s,
        chi1=arguments.chi1,
        seed=arguments.seed,
        checkpoints=arguments.checkpoints,
        escape_separation_factor=arguments.escape_separation_factor,
    )
    result = simulate_three_body_capture(encounter)
    record = {
        "inputs": {
            "mass_mjup": arguments.mass_mjup,
            "velocity_dispersion_kms": arguments.velocity_dispersion_kms,
            "chi1": arguments.chi1,
            "seed": arguments.seed,
            "checkpoints": arguments.checkpoints,
        },
        "result": result.to_dict(),
    }
    print(json.dumps(record, indent=2))
    return 0