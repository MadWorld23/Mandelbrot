import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import atexit
import shutil
import os
import tempfile

# ── Cache-Verwaltung ──────────────────────────────────────────
_cache_dir = None
_script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()

def _setup_temp_cache():
    """Erstellt ein temporäres Cache-Verzeichnis für Numba."""
    global _cache_dir
    _cache_dir = tempfile.mkdtemp(prefix="numba_mandelbrot_")
    os.environ['NUMBA_CACHE_DIR'] = _cache_dir

def _cleanup_cache():
    """Löscht ALLE Cache-Verzeichnisse beim Programmende."""
    global _cache_dir
    
    # 1) Numba-Temp-Ordner löschen
    if _cache_dir and os.path.exists(_cache_dir):
        try:
            shutil.rmtree(_cache_dir)
        except Exception:
            pass
    
    # 2) __pycache__ im Skript-Verzeichnis löschen
    pycache_path = os.path.join(_script_dir, "__pycache__")
    if os.path.exists(pycache_path):
        try:
            shutil.rmtree(pycache_path)
        except Exception:
            pass

# Setup & Cleanup registrieren
_setup_temp_cache()
atexit.register(_cleanup_cache)

# ── Numba-optimierte Mandelbrot-Berechnung ────────────────────
@jit(nopython=True, cache=True)
def mandelbrot_compute(h, w, max_iter, x_min, x_max, y_min, y_max):
    c_real = np.empty(h * w, dtype=np.float64)
    c_imag = np.empty(h * w, dtype=np.float64)
    
    dx = (x_max - x_min) / w
    dy = (y_max - y_min) / h
    
    idx = 0
    for j in range(h):
        for i in range(w):
            c_real[idx] = x_min + i * dx
            c_imag[idx] = y_min + j * dy
            idx += 1
    
    z_real = np.zeros(h * w, dtype=np.float64)
    z_imag = np.zeros(h * w, dtype=np.float64)
    iterations = np.full(h * w, max_iter, dtype=np.int64)
    active = np.ones(h * w, dtype=np.bool_)
    
    for n in range(max_iter):
        count_active = 0
        for k in range(h * w):
            if active[k]:
                zr = z_real[k]
                zi = z_imag[k]
                cr = c_real[k]
                ci = c_imag[k]
                
                zr_new = zr * zr - zi * zi + cr
                zi_new = 2.0 * zr * zi + ci
                
                z_real[k] = zr_new
                z_imag[k] = zi_new
                
                if zr_new * zr_new + zi_new * zi_new > 4.0:
                    iterations[k] = n
                    active[k] = False
                else:
                    count_active += 1
        
        if count_active == 0:
            break
    
    return iterations.reshape((h, w))


# ── Interaktiver Viewer ───────────────────────────────────────
class MandelbrotViewer:
    def __init__(self, width=800, height=800):
        self.width = width
        self.height = height
        
        self.x_min = -2.5
        self.x_max = 1.0
        self.y_min = -1.25
        self.y_max = 1.25
        
        self.zoom_step = 3.0
        self.max_iter = 100
        
        self.cmap = 'magma'
        self.colormaps = ['magma', 'plasma', 'inferno', 'viridis', 
                          'twilight_shifted', 'hot', 'cool', 'ocean']
        self.cmap_idx = 0
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.fig.patch.set_facecolor('black')
        
        self.img = self.ax.imshow(
            self._compute(), cmap=self.cmap, origin='lower',
            extent=[self.x_min, self.x_max, self.y_min, self.y_max],
            vmin=0, vmax=self.max_iter
        )
        
        self.info_text = self.ax.text(
            0.01, 0.01, '', transform=self.ax.transAxes,
            color='white', fontsize=10, family='monospace',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
        )
        self._update_info()
        
        self.ax.text(
            0.99, 0.99, 
            'Links=Zoom rein | Rechts=Zoom raus | Rad=Feinzoom\n'
            'r=Reset | c=Farbe wechseln | q=Beenden',
            transform=self.ax.transAxes,
            color='white', fontsize=9, family='monospace',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
        )
        
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        print("\n╔══════════════════════════════════════╗")
        print("║   Interaktiver Mandelbrot-Viewer     ║")
        print("║   (Numba + Auto-Cleanup alles)       ║")
        print("╠══════════════════════════════════════╣")
        print("║ Linksklick    → Zoom rein           ║")
        print("║ Rechtsklick   → Zoom raus           ║")
        print("║ Mausrad       → Feinzoom rein/raus  ║")
        print("║ r             → Reset               ║")
        print("║ c             → Farbe wechseln      ║")
        print("║ +/-           → Iterationen ändern  ║")
        print("║ q             → Beenden             ║")
        print("╚══════════════════════════════════════╝\n")
        
        plt.show()
    
    def _compute(self):
        return mandelbrot_compute(
            self.height, self.width, self.max_iter,
            self.x_min, self.x_max, self.y_min, self.y_max
        )
    
    def _refresh(self):
        self.img.set_data(self._compute())
        self.img.set_extent([self.x_min, self.x_max, self.y_min, self.y_max])
        self.img.set_clim(vmin=0, vmax=self.max_iter)
        self._update_info()
        self.fig.canvas.draw_idle()
    
    def _update_info(self):
        x_range = self.x_max - self.x_min
        zoom = 3.5 / x_range
        cx = (self.x_min + self.x_max) / 2
        cy = (self.y_min + self.y_max) / 2
        self.info_text.set_text(
            f'Zoom: {zoom:>10.1f}x | '
            f'Mitte: ({cx:+.10f}, {cy:+.10f}i) | '
            f'Iter: {self.max_iter}'
        )
    
    def _zoom_at(self, x, y, factor):
        x_range = (self.x_max - self.x_min) / factor
        y_range = (self.y_max - self.y_min) / factor
        
        self.x_min = x - x_range / 2
        self.x_max = x + x_range / 2
        self.y_min = y - y_range / 2
        self.y_max = y + y_range / 2
        
        zoom = 3.5 / x_range
        self.max_iter = max(100, int(100 + 60 * np.log2(max(1, zoom))))
        
        self._refresh()
    
    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:
            self._zoom_at(event.xdata, event.ydata, self.zoom_step)
        elif event.button == 3:
            self._zoom_at(event.xdata, event.ydata, 1.0 / self.zoom_step)
    
    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        
        if event.button == 'up':
            factor = 1.3
        elif event.button == 'down':
            factor = 1.0 / 1.3
        else:
            return
        
        self._zoom_at(event.xdata, event.ydata, factor)
    
    def _on_key(self, event):
        if event.key == 'r':
            self.x_min = -2.5
            self.x_max = 1.0
            self.y_min = -1.25
            self.y_max = 1.25
            self.max_iter = 100
            self._refresh()
        elif event.key == 'c':
            self.cmap_idx = (self.cmap_idx + 1) % len(self.colormaps)
            self.cmap = self.colormaps[self.cmap_idx]
            self.img.set_cmap(self.cmap)
            self.fig.canvas.draw_idle()
        elif event.key in ('+', '='):
            self.max_iter = int(self.max_iter * 1.5)
            self._refresh()
        elif event.key == '-':
            self.max_iter = max(50, int(self.max_iter / 1.5))
            self._refresh()
        elif event.key == 'q':
            plt.close(self.fig)


if __name__ == "__main__":
    viewer = MandelbrotViewer(width=800, height=800)