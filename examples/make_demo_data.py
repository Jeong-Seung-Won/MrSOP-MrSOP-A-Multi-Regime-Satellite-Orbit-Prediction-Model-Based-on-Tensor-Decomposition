from pathlib import Path
import argparse
import csv
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic circular orbits for a local pipeline smoke run.')
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mu = 398600.4418
    n_steps = 4800
    time = np.arange(n_steps, dtype=np.float64) * 60
    for split, phase in (("train", 0.0), ("val", 0.3), ("test", 0.6)):
        rows = []
        for regime, radius, inclination in (("LEO", 7000, .8), ("MEO", 15000, .6),
                                           ("NSO", 26560, .96), ("GEO", 42164, .01)):
            angle = phase + time * np.sqrt(mu / radius**3)
            speed = np.sqrt(mu / radius)
            c, s = np.cos(angle), np.sin(angle)
            ci, si = np.cos(inclination), np.sin(inclination)
            states = np.column_stack((radius*c, radius*s*ci, radius*s*si,
                                      -speed*s, speed*c*ci, speed*c*si)).astype(np.float32)
            filename = f"{split}_{regime}.npy"
            np.save(args.output/filename, states)
            rows.append(dict(path=filename, seg_start=0, seg_end=n_steps, regime=regime))
            if split == "test" and regime in ("LEO", "NSO", "GEO"):
                if regime == "LEO":
                    np.save(args.output/'input.npy', states[:192*15:15])
                source = {"LEO": "starlink", "NSO": "gnss", "GEO": "beidou"}[regime]
                step = 1 if source == "starlink" else 15

                with (args.output/f'{source}_demo.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['epoch_ms', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
                    for timestamp, row in zip(time[::step], states[::step]):
                        writer.writerow([int(timestamp*1000), *row])
        with (args.output/f'{split}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['path', 'seg_start', 'seg_end', 'regime'])
            writer.writeheader()
            writer.writerows(rows)
    print(f"Synthetic demo data written to {args.output}")


if __name__ == '__main__':
    main()
