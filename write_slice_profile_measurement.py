import math
from pathlib import Path
from datetime import datetime
import pypulseq as pp

### SETTINGS ###

timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
filename = Path(f"profile_{timestamp}.seq")

system = pp.Opts(
    max_grad=50,
    grad_unit="mT/m",
    max_slew=150,
    slew_unit="T/m/s",
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)
excitation_settings = dict(
    slice_thickness=6e-3,
    flip_angle_degree=5,
    rf_duration=1e-3,
    rf_tbp=6,
)
profile_settings = dict(
    Nz=512,
    dwell_time=1e-5,
    repeats=10,
    slice_scale=2,
    fov_scale=5,
    TR=10,
)

################


def slice_selective_exication(
    system, flip_angle_degree, rf_tbp, rf_duration, slice_thickness, **kwargs
):
    return pp.make_sinc_pulse(
        flip_angle=math.radians(flip_angle_degree),
        duration=rf_duration,
        slice_thickness=slice_thickness,
        apodization=0.5,
        time_bw_product=rf_tbp,
        system=system,
        return_gz=True,
    )


seq = pp.Sequence()
rf_phase = rf_inc = 0.0
for i in range(profile_settings["repeats"]):
    # The slice selective pulse
    rf, gz, gzr = slice_selective_exication(system, **excitation_settings)

    # scale gradient to make slice profile thicker to get better SNR

    if profile_settings["slice_scale"] != 1:
        max_scale = system.max_grad / gz.amplitude
        if profile_settings["slice_scale"] > max_scale:
            print()
            print(
                f"WARNING: slice scale would result in too high gradient. changing to {max_scale} "
            )
            print()
            profile_settings["slice_scale"] = max_scale
        gz = pp.make_trapezoid(
            channel="z",
            flat_time=gz.flat_time,
            amplitude=gz.amplitude * profile_settings["slice_scale"],
            system=system,
        )
    fov = (
        excitation_settings["slice_thickness"]
        * profile_settings["slice_scale"]
        * profile_settings["fov_scale"]
    )
    delta_k = 1 / fov

    gz_readout = pp.make_trapezoid(
        channel="z",
        flat_area=profile_settings["Nz"] * delta_k,
        flat_time=profile_settings["Nz"] * profile_settings["dwell_time"],
        system=system,
    )
    adc = pp.make_adc(
        num_samples=profile_settings["Nz"],
        duration=gz_readout.flat_time,
        delay=gz_readout.rise_time,
        system=system,
    )
    gz_readout_pre = pp.make_trapezoid(
        channel="z", amplitude=system.max_grad, area=-gz_readout.area / 2, system=system
    )

    ## RF Spoiling
    rf.phase_offset = math.radians(rf_phase)
    adc.phase_offset = math.radians(rf_phase)
    rf_inc = (rf_inc + 117) % 360
    rf_phase = (rf_phase + rf_inc) % 360

    seq.add_block(rf, gz)
    seq.add_block(gzr)
    seq.add_block(gz_readout_pre)
    label = pp.make_label(label="REP", type="SET", value=i)
    seq.add_block(adc, gz_readout, label)

    # Gradient spoiling
    spoiler_xy = pp.make_trapezoid(
        channel="xy"[i % 2], system=system, area=2 * gz.area, duration=4 * gz.flat_time
    )
    seq.add_block(spoiler_xy)

    if (delay := profile_settings["TR"]) > 0:
        seq.add_block(pp.make_delay(delay))


print("Test report....")
print(seq.test_report())


for key, value in (profile_settings | excitation_settings).items():
    seq.set_definition(key, value)
seq.set_definition(
    "FOV",
    [
        1,
        1,
        excitation_settings["slice_thickness"]
        * profile_settings["slice_scale"]
        * profile_settings["fov_scale"],
    ],
)
seq.write(filename)
print(f"written to {filename}")
