import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def mandelbrot(h, w, max_iter, x_center, y_center, zoom):
    scale = 3.0 / zoom
    x_min = x_center - scale
    x_max = x_center + scale
    y_min = y_center - scale
    y_max = y_center + scale

    x = np.linspace(x_min, x_max, w)
    y = np.linspace(y_min, y_max, h)
    c = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    z = np.zeros_like(c)
    iterations = np.full(z.shape, max_iter, dtype=int)
    mask = np.ones(z.shape, dtype=bool)

    for i in range(max_iter):
        z[mask] = z[mask]**2 + c[mask]
        diverged = np.abs(z) > 2
        iterations[mask & diverged] = i
        mask &= ~diverged
        if not np.any(mask):
            break

    return iterations


def main():
    # ── Einstellungen ──────────────────────────────────────────
    width, height = 600, 600
    num_frames = 300          # Anzahl der Zoom-Schritte
    zoom_factor = 1.2        # pro Frame wird der Zoom um diesen Faktor erhöht

    # Startpunkt (interessante Stelle am Rand der Mandelbrot-Menge)
    x_center = -0.7436438870371587
    y_center = 0.1318259042053119

    # ── Figur vorbereiten ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Erstes Bild berechnen
    initial_zoom = 1.0
    initial_max_iter = 100
    data = mandelbrot(height, width, initial_max_iter, x_center, y_center, initial_zoom)
    img = ax.imshow(data, cmap='magma', origin='lower', vmin=0, vmax=initial_max_iter)

    title = ax.set_title("", color="white", fontsize=12, pad=10)

    # Hintergrund schwarz für den Titel
    fig.patch.set_facecolor('black')

    # ── Animationsfunktion ─────────────────────────────────────
    current_zoom = [initial_zoom]

    def update(frame):
        current_zoom[0] *= zoom_factor

        # Iterationen mit dem Zoom erhöhen für mehr Detailtiefe
        max_iter = int(100 + 50 * np.log2(current_zoom[0]))

        data = mandelbrot(height, width, max_iter, x_center, y_center, current_zoom[0])
        img.set_data(data)
        img.set_clim(vmin=0, vmax=max_iter)
        title.set_text(f"Zoom: {current_zoom[0]:.1f}x  |  Iterationen: {max_iter}")

        return [img, title]

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=80, blit=True, repeat=False
    )

    plt.show()


if __name__ == "__main__":
    main()