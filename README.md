
import numpy as np
import adi
import matplotlib.pyplot as plt

sample_rate = 1e6 # Hz
center_freq = 100e6 # Hz
num_samps = 10000 # number of samples returned per call to rx()

sdr = adi.Pluto('ip:192.168.2.1')
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 70.0 # dB
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter width, just set it to the same as sample rate for now
sdr.rx_buffer_size = num_samps

samples = sdr.rx() # receive samples off Pluto
print(samples[0:10])

# ---- FFT y guardado de figura ----
fft_samples = np.fft.fftshift(np.fft.fft(samples))
fft_magnitude = 20*np.log10(np.abs(fft_samples))
freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sample_rate))

plt.figure()
plt.plot(freqs, fft_magnitude)
plt.title("FFT de la señal recibida")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)






# Escala fija en eje Y
plt.ylim(10, 120)
plt.yticks(np.arange(10, 121, 20))

plt.savefig("fft_signal.png", dpi=300)
plt.close()






import adi
sdr = adi.Pluto('ip:192.168.2.1')
sdr.gain_control_mode_chan0 = "slow_attack"
print(sdr._get_iio_attr('voltage0','hardwaregain', False))







import numpy as np
import adi
import matplotlib.pyplot as plt  # <-- solo esto se añade para la gráfica

sample_rate = 1e6 # Hz
center_freq = 915e6 # Hz







sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = -50 # Increase to increase tx power, valid range is -90 to 0 dB

N = 10000 # number of samples to transmit at once
t = np.arange(N)/sample_rate
samples = 0.5*np.exp(2.0j*np.pi*100e3*t) # Simulate a sinusoid of 100 kHz, so it should show up at 915.1 MHz at the receiver
samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

# Transmit our batch of samples 100 times, so it should be 1 second worth of samples total, if USB can keep up
for i in range(100):
    sdr.tx(samples) # transmit the batch of samples once

# ---- FFT de la señal TRANSMITIDA ----
fft_samples = np.fft.fftshift(np.fft.fft(samples))
fft_magnitude = 20*np.log10(np.abs(fft_samples))
freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sample_rate))

plt.figure()
plt.plot(freqs, fft_magnitude)
plt.title("FFT de la señal transmitida")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.ylim(10, 120)
plt.yticks(np.arange(10, 121, 20))
plt.savefig("fft_transmitida.png", dpi=300)
plt.close()




 import numpy as np
import adi
import matplotlib.pyplot as plt

sample_rate = 1e6 # Hz
center_freq = 915e6 # Hz
num_samps = 100000 # number of samples per call to rx()

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# Config Tx
sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = -50 # Increase to increase tx power, valid range is -90 to 0 dB

# Config Rx
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 0.0 # dB, increase to increase the receive gain, but be careful not to saturate the ADC

# Create transmit waveform (QPSK, 16 samples per symbol)
num_symbols = 1000
x_int = np.random.randint(0, 4, num_symbols) # 0 to 3
x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 degrees
x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # this produces our QPSK complex symbols
samples = np.repeat(x_symbols, 16) # 16 samples per symbol (rectangular pulses)



# Start the transmitter
sdr.tx_cyclic_buffer = True # Enable cyclic buffers
sdr.tx(samples) # start transmitting

# Clear buffer just to be safe
for i in range (0, 10):
    raw_data = sdr.rx()

# Receive samples
rx_samples = sdr.rx()
print(rx_samples)

# Stop transmitting
sdr.tx_destroy_buffer()

# --- Procesamiento para gráficas ---
# FFT/PSD transmitida
psd_tx = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2
psd_tx_dB = 10*np.log10(psd_tx)
f_tx = np.linspace(sample_rate/-2, sample_rate/2, len(psd_tx))

# FFT/PSD recibida
psd_rx = np.abs(np.fft.fftshift(np.fft.fft(rx_samples)))**2
psd_rx_dB = 10*np.log10(psd_rx)
f_rx = np.linspace(sample_rate/-2, sample_rate/2, len(psd_rx))

# --- Figura con 4 subgráficas ---
plt.figure(figsize=(12, 8))

# 1. Temporal transmitida (solo 100 muestras)
plt.subplot(2, 2, 1)
plt.plot(np.real(samples[:100]), label="Real")
plt.plot(np.imag(samples[:100]), label="Imag")
plt.title("Señal transmitida (dominio temporal)")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

# 2. FFT transmitida
plt.subplot(2, 2, 2)
plt.plot(f_tx/1e6, psd_tx_dB)
plt.title("Señal transmitida (FFT / PSD)")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("PSD [dB]")
plt.grid(True)
plt.ylim(-10, 180)
plt.yticks(np.arange(-10, 181, 20))

# 3. Temporal recibida (solo 100 muestras)
plt.subplot(2, 2, 3)
plt.plot(np.real(rx_samples[:100]), label="Real")
plt.plot(np.imag(rx_samples[:100]), label="Imag")
plt.title("Señal recibida (dominio temporal)")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)

# 4. FFT recibida
plt.subplot(2, 2, 4)
plt.plot(f_rx/1e6, psd_rx_dB)
plt.title("Señal recibida (FFT / PSD)")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("PSD [dB]")
plt.grid(True)
plt.ylim(-10, 180)
plt.yticks(np.arange(-10, 181, 20))

plt.tight_layout()
plt.savefig("tx_rx_completo.png", dpi=300)
plt.close()

