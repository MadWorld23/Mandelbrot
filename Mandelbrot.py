import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit

# --- 1. Numba-optimierte Kernfunktion ---
@jit(nopython=True, cache=True)
def mandelbrot_optimized(h, w, max_iter, x_center, y_center, zoom):
    # Bereich berechnen
    scale = 3.0 / zoom
    x_min = x_center - scale
    x_max = x_center + scale
    y_min = y_center - scale
    y_max = y_center + scale
    
    # Gitter erstellen (NumPy ist hier schnell genug für die Initialisierung)
    x = np.linspace(x_min, x_max, w)
    y = np.linspace(y_min, y_max, h)
    
    
    # Initialisierung
    c_real = np.empty(h * w, dtype=np.float64)
    c_imag = np.empty(h * w, dtype=np.float64)
    
    idx = 0
    for j in range(h):
        for i in range(w):
            c_real[idx] = x[i]
            c_imag[idx] = y[j]
            idx += 1
            
    z_real = np.zeros(h * w, dtype=np.float64)
    z_imag = np.zeros(h * w, dtype=np.float64)
    iterations = np.full(h * w, max_iter, dtype=np.int64)
    active = np.ones(h * w, dtype=np.bool_)
    
    # Hauptberechnungsschleife
    for n in range(max_iter):
        
        count_active = 0
        for k in range(h * w):
            if active[k]:
                # z = z^2 + c
                # (a+bi)^2 = a^2 - b^2 + 2abi
                zr = z_real[k]
                zi = z_imag[k]
                
                z_real_new = zr * zr - zi * zi + c_real[k]
                z_imag_new = 2 * zr * zi + c_imag[k]
                
                z_real[k] = z_real_new
                z_imag[k] = z_imag_new
                
                # Divergenzprüfung: |z|^2 > 4
                if z_real_new * z_real_new + z_imag_new * z_imag_new > 4.0:
                    iterations[k] = n
                    active[k] = False
                else:
                    count_active += 1
        
        # Wenn keine aktiven Pixel mehr übrig sind, abbrechen
        if count_active == 0:
            break
            
    # Ergebnis zurückgeben als 2D-Array
    return iterations.reshape((h, w))


def main():
    # ── Einstellungen ──────────────────────────────────────────
    width, height = 550, 550  
    num_frames = 300          
    zoom_factor = 1.15     
    
    # Startpunkt (Seahorse Valley)
    x_center = -0.7436438870371587
    y_center = 0.1318259042053119

    # ── Figur vorbereiten ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor('black')

    # Erstes Bild berechnen (Cache wird hier aufgebaut)
    initial_zoom = 1.0
    initial_max_iter = 100
    
    print("Erstes Bild wird berechnet (Numba JIT-Compilation läuft im Hintergrund)...")
    data = mandelbrot_optimized(height, width, initial_max_iter, x_center, y_center, initial_zoom)
    print("Berechnung fertig. Animation startet...")

    img = ax.imshow(data, cmap='magma', origin='lower', vmin=0, vmax=initial_max_iter)
    title = ax.set_title("", color="white", fontsize=14, pad=15)

    # ── Animationsfunktion ─────────────────────────────────────
    current_zoom = [initial_zoom]

    def update(frame):
        current_zoom[0] *= zoom_factor
        
        # Dynamische Iterationen für Detailtiefe
        max_iter = int(100 + 60 * np.log2(current_zoom[0]))
        
        # Berechnung (sehr schnell dank Numba)
        data = mandelbrot_optimized(height, width, max_iter, x_center, y_center, current_zoom[0])
        
        img.set_data(data)
        img.set_clim(vmin=0, vmax=max_iter)
        title.set_text(f"Zoom: {current_zoom[0]:.1f}x  |  Iterationen: {max_iter}")
        
        return [img, title]

    # Animation starten
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=50, blit=True, repeat=False  # Interval auf 50ms gesenkt für flüssigeres Video
    )

    plt.show()

if __name__ == "__main__":
    main()