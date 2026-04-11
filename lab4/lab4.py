import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
import scipy.signal as signal

t_min, t_max = 0, 10
n_points = 1000
t = np.linspace(t_min, t_max, n_points)
fs = n_points / (t_max - t_min)

init_amplitude = 1.0
init_frequency = 0.5
init_phase = 0.0
init_noise_mean = 0.0
init_noise_covariance = 0.1
init_cutoff = 2.0
init_show_noise = True

base_noise = np.random.randn(n_points)

def harmonic_with_noise(t, amplitude, frequency, phase, noise_mean, noise_covariance, show_noise, base_noise):
    omega = 2 * np.pi * frequency
    harmonic = amplitude * np.sin(omega * t + phase)
    
    std_dev = np.sqrt(max(0, noise_covariance))
    current_noise = noise_mean + std_dev * base_noise
    
    if show_noise:
        return harmonic, harmonic + current_noise
    else:
        return harmonic, harmonic

def apply_filter(data, cutoff, fs):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    if normal_cutoff <= 0 or normal_cutoff >= 1:
        return data
        
    b, a = signal.iirfilter(N=4, Wn=normal_cutoff, btype='low', ftype='butter')
    
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.45)
ax.set_title("Harmonic Signal Analysis")
ax.set_xlabel("Time (t)")
ax.set_ylabel("Amplitude y(t)")
ax.grid(True, linestyle='--', alpha=0.6)

pure_harmonic, noisy_signal = harmonic_with_noise(
    t, init_amplitude, init_frequency, init_phase, 
    init_noise_mean, init_noise_covariance, init_show_noise, base_noise
)
filtered_signal = apply_filter(noisy_signal, init_cutoff, fs)

line_noisy, = ax.plot(t, noisy_signal, color='orange', label='Noisy Signal', alpha=0.8)
line_filtered, = ax.plot(t, filtered_signal, color='blue', linestyle='--', linewidth=2, label='Filtered Signal')
line_pure, = ax.plot(t, pure_harmonic, color='red', linestyle='-', linewidth=2, alpha=1.0, label='Pure Harmonic')
ax.legend(loc='upper right')

axcolor = 'lightgoldenrodyellow'
ax_amp    = plt.axes([0.15, 0.35, 0.65, 0.03], facecolor=axcolor)
ax_freq   = plt.axes([0.15, 0.30, 0.65, 0.03], facecolor=axcolor)
ax_phase  = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_nmean  = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_ncov   = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_cutoff = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor='lightblue')

s_amp    = Slider(ax_amp, 'Amplitude', 0.1, 5.0, valinit=init_amplitude)
s_freq   = Slider(ax_freq, 'Frequency', 0.1, 5.0, valinit=init_frequency)
s_phase  = Slider(ax_phase, 'Phase', 0.0, 2 * np.pi, valinit=init_phase)
s_nmean  = Slider(ax_nmean, 'Noise Mean', -2.0, 2.0, valinit=init_noise_mean)
s_ncov   = Slider(ax_ncov, 'Noise Covariance', 0.0, 2.0, valinit=init_noise_covariance)
s_cutoff = Slider(ax_cutoff, 'Filter Cutoff', 0.1, 20.0, valinit=init_cutoff)

ax_check = plt.axes([0.85, 0.35, 0.12, 0.05])
check = CheckButtons(ax_check, ['Show Noise'], [init_show_noise])

ax_reset = plt.axes([0.85, 0.10, 0.1, 0.04])
btn_reset = Button(ax_reset, 'Reset', hovercolor='0.975')

def update(val=None):
    amp = s_amp.val
    freq = s_freq.val
    phase = s_phase.val
    n_mean = s_nmean.val
    n_cov = s_ncov.val
    cutoff = s_cutoff.val
    show_noise = check.get_status()[0]

    pure_h, sig = harmonic_with_noise(t, amp, freq, phase, n_mean, n_cov, show_noise, base_noise)
    filt_sig = apply_filter(sig, cutoff, fs)

    line_noisy.set_ydata(sig)
    line_filtered.set_ydata(filt_sig)
    line_pure.set_ydata(pure_h)
    
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)
    
    fig.canvas.draw_idle()

def reset(event):
    s_amp.reset()
    s_freq.reset()
    s_phase.reset()
    s_nmean.reset()
    s_ncov.reset()
    s_cutoff.reset()
    if not check.get_status()[0]:
        check.set_active(0)
    update()

s_amp.on_changed(update)
s_freq.on_changed(update)
s_phase.on_changed(update)
s_nmean.on_changed(update)
s_ncov.on_changed(update)
s_cutoff.on_changed(update)
check.on_clicked(update)
btn_reset.on_clicked(reset)

plt.show()