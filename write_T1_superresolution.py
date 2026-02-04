"""
Superresolution T1 Inversion Recovery with Radial Readout
based on Simone's Sequence
Hufnagel et al. "3D model-based super-resolution motion-corrected cardiac T1 mapping." PMB 2022 
     (Orientations 1, Translations 6, Slices per stack 6)
Hufnagel et al. "3D whole heart k-space-based super-resolution cardiac T1 mapping using rotated stacks", in submission
    (Orientations 4, Translations 3, Slices per Stack 5)
"""

from math import radians, sqrt, gcd, ceil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pypulseq as pp
from scipy.spatial.transform import Rotation
from datetime import datetime
import shutil
import sigpy.mri.rf
from itertools import zip_longest

################################# Settings #################################
filename: str | Path = "SRR"
system = pp.Opts(
    max_grad=50,  # could be increased if needed (50 or 80)
    grad_unit="mT/m",
    max_slew=160,  # limit 200
    slew_unit="T/m/s",
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)
use_trigger = False
split_stacks = True
superresolution_settings = dict(
    slice_thickness=6e-3,
    slice_gap=0,
    slices_per_stack=27,
    orientations=4,
    translations=1,
    rotation_axis=(0, 1, 0),
    stack_delay=6,
)
inversion_pulse_settings = dict(
    slice_thickness=superresolution_settings["slice_thickness"],
    adiabatic=True,
    slice_selective=True,
    duration=10.24e-3,
    IR_scale_factor=3.0,  ### ???
    spoiler_risetime=700e-6,
    spoiler_duration=9.6e-3,
    rf_tbp=6,
)
readout_settings = dict(
    slice_thickness=superresolution_settings["slice_thickness"],
    Nx=240,
    fov=240e-3,
    num_spokes=760,
    golden_angle=True,
    grad_spoiling=True,
    gx_pre_dur=1e-3,
    dwell_time=4e-6,
    flip_angle_degree=5,
    rf_duration=1e-3,
    rf_tbp=6,
    rf_spoiling_inc_degree=117.0,
    oversampling=2,
    TE=None,  # None: use_min TE/TR
    TR=None,
)
############################################################################

timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
filename = Path(filename)
filename = filename.with_name(f"{filename.name}_{timestamp}")


def write_seq(seq, filename, settings: dict | None = None, stack_number: int | None = None, report: bool = True):
    if settings is not None:
        for key, value in settings.items():
            seq.set_definition(key, value)
    filename_seq = Path(filename).with_suffix(".seq")
    if stack_number is not None:
        print(f"Saving stack {stack_number}")
        filename_seq = filename_seq.with_name(f"{filename_seq.stem}_stack{stack_number}").with_suffix(".seq")

    if report:
        print("\nCreating test report...")
        print(seq.test_report())
    else:
        ok, errors = seq.check_timing()
        if not ok:
            print("\nTiming check failed! Errors\n", errors)

    seq.write(filename_seq, create_signature=True)
    print(f"\nSaved sequence file as {filename_seq}")


def rotate(
    *args: SimpleNamespace,
    rotation: Rotation,
) -> list[SimpleNamespace]:
    """
    Rotate events using a scipy.spatial.transform.Rotation.

    Allows for arbitrary Rotations
    """
    EPS = 1e-6

    def grad_abs_mag(grad: SimpleNamespace) -> float:
        if grad.type == "trap":
            return abs(grad.amplitude)
        return float(np.max(np.abs(grad.waveform)))

    bypass = []
    max_mag = 0.0
    rotated = ([], [], [])
    for event in args:
        if event.type not in ["grad", "trap"]:
            # only need to rotate these
            bypass.append(event)
            continue
        event_axis = (event.channel == "x", event.channel == "y", event.channel == "z")
        if abs(np.dot(rotation.as_rotvec(), ~np.array(event_axis))) < EPS:
            # parallel to rotation axis or zero angle
            bypass.append(event)
            continue
        max_mag = max(max_mag, grad_abs_mag(event))
        rotated_axis = rotation.apply(event_axis)
        for axis_name, scale, rotatated_dest in zip("xyz", rotated_axis, rotated):
            if abs(scale) < EPS:
                continue
            new = pp.scale_grad(grad=event, scale=scale)
            new.channel = axis_name
            rotatated_dest.append(new)

    # Add gradients on the corresponding axis together
    new_gradients = [pp.add_gradients(grads=g) for g in rotated if g]
    # filter zero amplitude gradients out
    new_gradients = [g for g in new_gradients if grad_abs_mag(g) > EPS * max_mag]
    return [*bypass, *new_gradients]


def inversion_pulse(
    system,
    seq,
    duration: float,
    IR_scale_factor: float,
    slice_thickness: float,
    spoiler_duration: float,
    spoiler_risetime: float,
    adiabatic: bool,
    slice_selective: bool,
    slice_shift: float,
    slice_rotation: Rotation,
    rf_tbp,
) -> dict:
    """
    Slice selective inversion pulse
    """
    # create slice selective inversion pulse and gradient
    if adiabatic and slice_selective:
        rf_inv, gz_inv, gz_inv_rew = pp.make_adiabatic_pulse(
            pulse_type="hypsec",
            adiabaticity=6,
            beta=800,
            mu=4.9,
            duration=duration,
            system=system,
            return_gz=True,
            slice_thickness=IR_scale_factor * slice_thickness,
        )
        rf_inv.freq_offset = sqrt(gz_inv.amplitude**2) * slice_shift
        gz_inv = rotate(gz_inv, rotation=slice_rotation)
        seq.add_block(rf_inv, *gz_inv)

    elif adiabatic and not slice_selective:
        rf_inv = pp.make_adiabatic_pulse(
            pulse_type="hypsec",
            adiabaticity=6,
            beta=800,
            mu=4.9,
            duration=duration,
            system=system,
            return_gz=False,
            slice_thickness=0.0,
        )
        seq.add_block(rf_inv)

    elif not adiabatic and slice_selective:
        rf_inv, gz_inv, gz_inv_rew = pp.make_sinc_pulse(
            flip_angle=np.pi,
            duration=duration,
            apodization=0.5,
            time_bw_product=rf_tbp,
            system=system,
            return_gz=True,
            slice_thickness=IR_scale_factor * slice_thickness,
        )

        rf_inv.freq_offset = sqrt(gz_inv.amplitude**2) * slice_shift
        gz_inv = rotate(gz_inv, rotation=slice_rotation)
        seq.add_block(rf_inv, *gz_inv)

    elif not adiabatic and not slice_selective:
        rf_inv = pp.make_sinc_pulse(
            flip_angle=np.pi,
            duration=duration,
            apodization=0.5,
            time_bw_product=rf_tbp,
            system=system,
            return_gz=False,
            slice_thickness=0.0,
        )
        seq.add_block(rf_inv)

    gz_inv_spoil = pp.make_trapezoid(channel="z", amplitude=0.4 * system.max_grad, duration=spoiler_duration, rise_time=spoiler_risetime)
    seq.add_block(*rotate(gz_inv_spoil, rotation=slice_rotation))
    return {}


def radial_readout(
    seq,
    system,
    flip_angle_degree: float,
    rf_duration: float,
    rf_tbp: float,
    slice_thickness: float,
    slice_shift: float,
    num_spokes,
    golden_angle: bool,
    slice_rotation,
    fov: float,
    Nx: int,
    oversampling: int,
    dwell_time: float,
    gx_pre_dur: float,
    rf_spoiling_inc_degree: float,
    grad_spoiling: bool,
    TE: float | None = None,
    TR: float | None = None,
    calculate_slice_profile: bool = False,
):
    """
    Radial readout.
    Returns dict of parameters that can differ from the inputs
    """
    # adc and read out
    delta_k = 1 / fov
    # dwell time must be on adc raster and flat time on grad raster
    g = gcd(Nx * oversampling, round(system.grad_raster_time / system.adc_raster_time))
    dwell_time = round(dwell_time / system.grad_raster_time * g) * system.grad_raster_time / g
    gx_flat_time = dwell_time * Nx * oversampling

    gx = pp.make_trapezoid(
        channel="x",
        flat_area=Nx * delta_k,
        flat_time=gx_flat_time,
        system=system,
    )
    adc = pp.make_adc(
        num_samples=Nx * oversampling,
        dwell=dwell_time,
        delay=gx.rise_time,
        system=system,
    )
    gx_pre = pp.make_trapezoid(channel="x", area=-gx.area / 2, system=system, duration=gx_pre_dur)
    # slice selection pulse and gradient
    rf, gz, gzr = pp.make_sinc_pulse(
        flip_angle=radians(flip_angle_degree),
        duration=rf_duration,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=rf_tbp,
        system=system,
        return_gz=True,
    )

    # gradient spoiling
    A_gz_spoil = 4 / slice_thickness - gz.area / 2
    gz_spoil = pp.make_trapezoid(channel="z", area=A_gz_spoil, system=system)
    gx_spoil = pp.make_trapezoid(channel="x", area=2 * Nx * delta_k, system=system)
    if grad_spoiling:
        duration_spoil = max(pp.calc_duration(gx_spoil), pp.calc_duration(gz_spoil))
    else:
        duration_spoil = max(pp.calc_duration(gx_pre), pp.calc_duration(gz_spoil))

    # TE and TR
    min_TE = (
        gz.fall_time
        + gz.flat_time / 2  # half rf pulse
        + pp.calc_duration(gzr)  # slice selection re-phasing gradient
        + pp.calc_duration(gx_pre)  # readout pre-winder gradient
        + pp.calc_duration(gx) / 2  # half readout gradient
    )
    min_TE = np.ceil(min_TE / system.grad_raster_time) * system.grad_raster_time  # put on raster
    min_TE = np.ceil(min_TE * 1e8) / 1e8  # round to 2 decimal values in ms
    min_TR = (
        pp.calc_duration(gz)  # rf pulse
        + pp.calc_duration(gzr)  # slice selection re-phasing gradient
        + pp.calc_duration(gx_pre)  # readout pre-winder gradient
        + pp.calc_duration(gx)  # readout gradient
        + duration_spoil  # gradient spoiler or readout-re-winder
    )
    min_TR = np.ceil(min_TR / system.grad_raster_time) * system.grad_raster_time  # put on raster
    min_TR = np.ceil(min_TR * 1e8) / 1e8  # round to 2 decimal values in ms
    if TR is None:
        TR = min_TR
    if TE is None:
        TE = min_TE
    delay_TE = np.ceil((TE - min_TE) / system.grad_raster_time) * system.grad_raster_time
    current_TR = np.ceil((min_TR + delay_TE) / system.grad_raster_time) * system.grad_raster_time
    delay_TR = np.ceil((TR - current_TR) / system.grad_raster_time) * system.grad_raster_time
    if not delay_TE >= 0:
        raise ValueError(f"TE must be larger than {min_TE * 1000:.2f} ms. Current value is {TE * 1000:.2f} ms.")
    if not delay_TR >= 0:
        raise ValueError(f"TR must be larger than {current_TR.max() * 1000:.2f} ms. Current value is {TR * 1000:.2f} ms.")

    if calculate_slice_profile:
        g_amp = pp.convert.convert(from_value=gz.amplitude, from_unit="Hz/m", to_unit="mT/m") * 1e-3  # T/m
        signal = pp.convert.convert(from_value=rf.signal, from_unit="Hz/m", to_unit="mT/m") * 1e-3  # T/m
        gamma_dt = np.mean(np.diff(rf.t)) * 267.52218744e6
        z = np.linspace(-2 * slice_thickness, 2 * slice_thickness, 200)  # m
        a, b = sigpy.mri.rf.optcont.blochsim(signal, z, g_amp * gamma_dt * np.ones_like(rf.signal))
        slice_profile = np.abs(2 * np.conj(a) * b)
        slice_profile /= slice_profile.max()
        slice_profile = np.round(slice_profile, 8)
        slice_info = dict(slice_profile_m=slice_profile, slice_profile_z=z)

    # slice rotation
    gz = rotate(gz, rotation=slice_rotation)
    gzr = rotate(gzr, rotation=slice_rotation)

    # slice shift
    rf.freq_offset = slice_shift * sqrt(sum([g.amplitude**2 for g in gz]))

    # spoke angle
    if golden_angle:
        angle_spoke_delta = 2 * np.pi / (1 + sqrt(5))  # angular increment # full spokes # angle = 111.25°
    else:
        angle_spoke_delta = np.pi / num_spokes  # angular increment

    rf_phase = rf_inc = 0.0
    for idx_spoke in range(num_spokes):
        # rf_spoiling
        if rf_spoiling_inc_degree > 0:
            rf.phase_offset = radians(rf_phase)
            adc.phase_offset = radians(rf_phase)
            rf_inc = (rf_inc + rf_spoiling_inc_degree) % 360
            rf_phase = (rf_phase + rf_inc) % 360
        # slice selective excitation pulse
        seq.add_block(rf, *gz)
        # slice selection re-phasing gradient
        seq.add_block(*gzr)
        # TE
        if delay_TE > 0:
            seq.add_block(pp.make_delay(delay_TE))
        # spoke rotation would be a rotation around z axis in non-rotated slice
        spoke_rotation = slice_rotation * Rotation.from_rotvec((0, 0, angle_spoke_delta * idx_spoke))
        # readout pre-winder gradient.
        seq.add_block(*rotate(gx_pre, rotation=spoke_rotation))
        # readout gradient and ADC
        label = pp.make_label(label="LIN", type="SET", value=idx_spoke)
        seq.add_block(*rotate(gx, adc, rotation=spoke_rotation), label)
        if grad_spoiling:
            seq.add_block(*rotate(gx_spoil, gz_spoil, rotation=spoke_rotation))
        else:
            # readout re-winder gradient
            seq.add_block(*rotate(gx_pre, gz_spoil, rotation=spoke_rotation))
        # TR
        if delay_TR > 0:
            seq.add_block(pp.make_delay(delay_TR))

    # increase counter for next acquisition
    seq.add_block(pp.make_label(label="SLC", type="INC", value=1))

    ret = {"TE": TE, "TR": TR, "angle_spoke_delta": angle_spoke_delta, "dwell_time": dwell_time}
    if calculate_slice_profile:
        ret |= slice_info
    return ret


def interleaved(N):
    if N > 16:
        n = ceil(N**0.5)
        lists = [list(range(i, N, n)) for i in range(n)]
        lists = [lists[i] for i in interleaved(n)]
        return [item for sublist in lists for item in sublist]
    elif N > 6:
        n = 4
    else:
        n = 2
    step = ceil(N / n)
    lists = [(range(i * step, min((i + 1) * step, N))) for i in range(n)]
    if N % n != n - 1:
        lists = reversed(lists)
    return [item for sublist in zip_longest(*lists) for item in sublist if item is not None]


def create_stacks(slice_thickness, slice_gap, slices_per_stack, orientations, translations, rotation_axis=(0, 1, 0), stack_delay: float = 0.0):
    slice_order = interleaved(slices_per_stack)
    stacks = []
    for idx_orientation in range(orientations):
        for idx_translation in range(translations):
            slice_rotations = []
            slice_shifts = []
            for idx_slice in slice_order:
                angle = np.pi * idx_orientation / orientations
                rot_vec = angle * np.array(rotation_axis) / np.linalg.norm(rotation_axis)
                slice_rotations.append(Rotation.from_rotvec(rot_vec))
                shift = (idx_slice - slices_per_stack // 2 + idx_translation / translations) * (slice_gap + slice_thickness)
                slice_shifts.append(shift)
            delays = [0.0] * (len(slice_order) - 1) + [stack_delay]
            stacks.append(dict(slice_rotations=slice_rotations, slice_shifts=slice_shifts, delays=delays))
    return stacks


def print_stacks(stacks):
    print("Stack Information:")
    for i, stack in enumerate(stacks):
        unique_rot = np.unique([r.as_rotvec(degrees=True) for r in stack["slice_rotations"]], axis=0)
        unique_shifts, order = np.unique(np.round(stack["slice_shifts"], 8), return_inverse=True)

        print(f" Stack {i+1}/{len(stacks)}:")
        print("   Unique Rotations:", unique_rot.tolist())
        print("   Unique Shifts:", unique_shifts.tolist())
        if len(unique_rot) > 1:
            print("   Rotations:", [r.as_rotvec(degrees=True).tolist() for r in stack["slice_rotations"]])
            print("   Shifts:", np.round(stack["slice_shifts"], 8))
        else:
            print("   Shift order:", order.tolist())
            print("")


def stack_information(stacks, flatten=True):
    quaternions = np.round(np.array([[rotation.as_quat() for rotation in s["slice_rotations"]] for s in stacks]), 12)
    positions = np.round(np.array([[rot.apply((0, 0, z), inverse=True) for z, rot in zip(s["slice_shifts"], s["slice_rotations"])] for s in stacks]), 12)
    shifts = np.round([s["slice_shifts"] for s in stacks], 12)
    if flatten:
        shifts = shifts.ravel()
        quaternions = quaternions.reshape(-1, 4)
        positions = positions.reshape(-1, 3)
        return {
            "slice_quaternions_x": quaternions[:, 0],
            "slice_quaternions_y": quaternions[:, 1],
            "slice_quaternions_z": quaternions[:, 2],
            "slice_quaternions_w": quaternions[:, 3],
            "slice_positions_x": positions[:, 0],
            "slice_positions_y": positions[:, 1],
            "slice_positions_z": positions[:, 2],
            "slice_shifts": shifts,
        }
    return {"slice_quaternions": quaternions, "slice_positions": positions, "slice_shifts": shifts}


stacks = create_stacks(**superresolution_settings)
print_stacks(stacks)
settings = (
    dict(use_trigger=use_trigger, split_stacks=split_stacks)
    | readout_settings
    | inversion_pulse_settings
    | superresolution_settings
    | stack_information(stacks)
)
seq = pp.Sequence()
for idx_stack, stack in enumerate(stacks):
    for slice_shift, slice_rotation, delay in zip(stack["slice_shifts"], stack["slice_rotations"], stack["delays"]):
        if use_trigger:
            seq.add_block(pp.make_trigger(channel="physio1", duration=250e-3))
        settings |= inversion_pulse(
            seq=seq,
            system=system,
            slice_rotation=slice_rotation,
            slice_shift=slice_shift,
            **inversion_pulse_settings,
        )

        settings |= radial_readout(
            seq=seq,
            system=system,
            slice_rotation=slice_rotation,
            slice_shift=slice_shift,
            **readout_settings,
            calculate_slice_profile="slice_profile_m" not in settings,
        )
    if delay:
        seq.add_block(pp.make_delay(delay))
    if split_stacks:
        settings |= stack_information([stack])
        write_seq(seq, settings=settings, filename=filename, stack_number=idx_stack)
        seq = pp.Sequence()

if not split_stacks:
    settings |= stack_information(stacks)
    write_seq(seq, settings=settings, filename=filename)

print(f"Time per slice {settings['TR']*settings['num_spokes']} s")
filename_py = Path(filename).with_suffix(".py")
shutil.copy(__file__, filename_py)
print(f"Saved copy of script as {filename_py}")
