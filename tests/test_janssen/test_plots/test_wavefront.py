
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from janssen.utils import make_optical_wavefront
from janssen.plots.wavefront import (
    plot_complex_wavefront,
    plot_amplitude,
    plot_intensity,
    plot_phase,
)

def test_plotting():
    # Create a dummy scalar wavefront
    N = 128
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Gaussian amplitude, spiral phase
    amplitude = np.exp(-R**2 / 0.5)
    phase = np.arctan2(Y, X)
    field = amplitude * np.exp(1j * phase)
    
    wavefront = make_optical_wavefront(
        field=jnp.array(field),
        wavelength=500e-9,
        dx=1e-6,
        z_position=0.0
    )
    
    print("Testing scalar wavefront plotting...")
    try:
        fig, ax = plot_complex_wavefront(wavefront, title="Complex Wavefront")
        plt.close(fig)
        print("plot_complex_wavefront: OK")
        
        fig, ax = plot_amplitude(wavefront, title="Amplitude")
        plt.close(fig)
        print("plot_amplitude: OK")
        
        fig, ax = plot_intensity(wavefront, title="Intensity")
        plt.close(fig)
        print("plot_intensity: OK")
        
        fig, ax = plot_phase(wavefront, title="Phase")
        plt.close(fig)
        print("plot_phase: OK")
    except Exception as e:
        print(f"FAILED: {e}")
        raise

    # Create a dummy polarized wavefront
    print("\nTesting polarized wavefront plotting...")
    field_pol = np.stack([field, field * 0.5], axis=-1)
    
    wavefront_pol = make_optical_wavefront(
        field=jnp.array(field_pol),
        wavelength=500e-9,
        dx=1e-6,
        z_position=0.0
    )
    
    try:
        fig, axes = plot_complex_wavefront(wavefront_pol, title="Polarized Complex")
        plt.close(fig)
        print("plot_complex_wavefront (polarized): OK")
        
        fig, axes = plot_amplitude(wavefront_pol, title="Polarized Amplitude")
        plt.close(fig)
        print("plot_amplitude (polarized): OK")
        
        fig, axes = plot_intensity(wavefront_pol, title="Polarized Intensity")
        plt.close(fig)
        print("plot_intensity (polarized): OK")
        
        fig, axes = plot_phase(wavefront_pol, title="Polarized Phase")
        plt.close(fig)
        print("plot_phase (polarized): OK")
    except Exception as e:
        print(f"FAILED: {e}")
        raise

if __name__ == "__main__":
    test_plotting()
