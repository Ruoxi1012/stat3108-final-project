"""Audit EmotionalCanines and create reproducible paired noise experiments."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.io import wavfile
from scipy.signal import resample_poly, stft, istft
from math import gcd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_audio(path):
    rate, raw = wavfile.read(path)
    if raw.size == 0:
        raise ValueError('empty audio')
    channels = raw.shape[1] if raw.ndim == 2 else 1
    if raw.dtype == np.uint8:
        audio = (raw.astype(np.float64) - 128) / 128
    elif np.issubdtype(raw.dtype, np.signedinteger):
        audio = raw.astype(np.float64) / (2 ** (np.iinfo(raw.dtype).bits - 1))
    elif np.issubdtype(raw.dtype, np.floating):
        audio = raw.astype(np.float64)
    else:
        raise ValueError('unsupported WAV encoding')
    if not np.isfinite(audio).all():
        raise ValueError('nonfinite samples')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return rate, audio, channels


def audit(root, output):
    root, output = Path(root), Path(output)
    rows, problems, label_hashes = [], [], {}
    seen_ids, seen_audio = {}, defaultdict(list)
    for breed in ('husky', 'shiba'):
        for split in ('train', 'test'):
            label_file = root / f'{breed}_{split}_labels.csv'
            if not label_file.exists():
                problems.append({'path': str(label_file), 'reason': 'missing label file'})
                continue
            label_hashes[label_file.name] = sha256(label_file)
            with label_file.open(newline='', encoding='utf-8-sig') as handle:
                reader = csv.DictReader(handle)
                if not {'audio_id', 'arousal', 'valence'} <= set(reader.fieldnames or []):
                    problems.append({'path': str(label_file), 'reason': 'invalid CSV columns'})
                    continue
                for item in reader:
                    audio_id = item['audio_id']
                    if Path(audio_id).name != audio_id or not audio_id.startswith(f'{breed}_{split}_'):
                        problems.append({'audio_id': audio_id, 'reason': 'invalid source ID'})
                        continue
                    path = root / breed / split / f'{audio_id}.wav'
                    row = dict(item, breed=breed, split=split, path=str(path.resolve()))
                    try:
                        if audio_id in seen_ids:
                            raise ValueError('duplicate label ID')
                        seen_ids[audio_id] = split
                        if item['arousal'] not in ('Low', 'Medium', 'High') or item['valence'] not in ('Negative', 'Neutral', 'Positive'):
                            raise ValueError('invalid affect label')
                        if path.exists() and path.read_bytes()[:42].startswith(b'version https://git-lfs.github.com/spec/v1'):
                            raise ValueError('Git LFS pointer: actual audio not downloaded')
                        rate, audio, channels = read_audio(path)
                        digest = hashlib.sha256(np.asarray(audio, dtype='<f8').tobytes() + str(rate).encode()).hexdigest()
                        seen_audio[digest].append({'audio_id': audio_id, 'split': split, 'breed': breed})
                        row.update(sample_rate=rate, channels=channels, duration_seconds=len(audio)/rate,
                                   rms=float(np.sqrt(np.mean(audio**2))), peak=float(np.max(np.abs(audio))),
                                   clipped_fraction=float(np.mean(np.abs(audio) >= 1)), sha256=sha256(path), status='ok')
                    except (ValueError, OSError) as error:
                        row.update(status='excluded', reason=str(error))
                        problems.append({'audio_id': audio_id, 'reason': str(error)})
                    rows.append(row)
    duplicates = [group for group in seen_audio.values() if len(group) > 1]
    for group in duplicates:
        if len({item['split'] for item in group}) > 1:
            problems.append({'reason': 'identical decoded audio crosses official splits', 'clips': group})
    summary = {'label_sha256': label_hashes, 'clips': len(rows), 'readable': sum(r['status']=='ok' for r in rows),
               'class_counts': {}, 'duplicates': duplicates, 'issues': problems,
               'limitations': 'No public dog IDs; validation cannot be certified dog-disjoint. Exact decoded duplicates only; near-duplicate review remains manual.'}
    for breed in ('husky', 'shiba'):
        for split in ('train', 'test'):
            selected = [r for r in rows if r['breed']==breed and r['split']==split]
            summary['class_counts'][f'{breed}/{split}'] = {target: dict(Counter(r[target] for r in selected)) for target in ('arousal','valence')}
            if len(selected) != (600 if split == 'train' else 100):
                problems.append({'reason': 'unexpected released clip count', 'group': f'{breed}/{split}', 'count': len(selected)})
    output.mkdir(parents=True, exist_ok=True)
    (output/'manifest.json').write_text(json.dumps(rows, indent=2)+'\n')
    (output/'audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def mix_noise(audio, noise, snr_db, seed):
    if not np.isfinite(snr_db) or not len(noise) or not len(audio):
        raise ValueError('finite SNR and nonempty signals required')
    rng = np.random.default_rng(seed)
    noise = np.tile(noise, int(np.ceil((len(audio)+len(noise))/len(noise))))
    start = int(rng.integers(0, len(noise)-len(audio)+1))
    noise = noise[start:start+len(audio)].copy()
    signal_rms, noise_rms = np.sqrt(np.mean(audio**2)), np.sqrt(np.mean(noise**2))
    if min(signal_rms, noise_rms) <= 1e-12:
        raise ValueError('cannot define SNR for silent signal or noise')
    noise *= signal_rms / noise_rms / (10 ** (snr_db/20))
    # Common gain preserves SNR and keeps both reference and mixture in range.
    gain = min(1.0, .99/max(np.max(np.abs(audio+noise)), np.max(np.abs(audio))))
    return audio*gain, (audio+noise)*gain, noise*gain, gain


def enhance(audio, noise, rate):
    """Oracle noise-profile spectral subtraction baseline, NOT an AI denoiser."""
    if len(audio) < 32:
        raise ValueError('audio too short for spectral subtraction')
    n = min(512, len(audio))
    _, _, spectrum = stft(audio, fs=rate, nperseg=n, noverlap=n//2)
    _, _, noise_spectrum = stft(noise, fs=rate, nperseg=n, noverlap=n//2)
    profile = np.mean(np.abs(noise_spectrum)**2, axis=1, keepdims=True)
    power = np.abs(spectrum)**2
    mask = np.sqrt(np.maximum(power-profile, .02*power)/(power+1e-12))
    _, restored = istft(spectrum*mask, fs=rate, nperseg=n, noverlap=n//2)
    return restored[:len(audio)]


def pairs(manifest, noise_path, output, snr_db, seed):
    rows = json.loads(Path(manifest).read_text())
    if any(r['status'] != 'ok' for r in rows):
        raise ValueError('resolve audit exclusions before generating pairs')
    if not rows or len({r['audio_id'] for r in rows}) != len(rows):
        raise ValueError('manifest must contain unique source IDs')
    noise_rate, background, _ = read_audio(Path(noise_path))
    output = Path(output)
    if output.exists():
        raise ValueError('output already exists; choose a fresh directory')
    records = []
    for row in rows:
        if row['split'] not in ('train','test') or row['breed'] not in ('husky','shiba') or Path(row['audio_id']).name != row['audio_id']:
            raise ValueError('invalid manifest identity or split')
        source = Path(row['path'])
        if sha256(source) != row['sha256']:
            raise ValueError(f'source changed since audit: {source}')
        rate, audio, _ = read_audio(source)
        divisor = gcd(rate, noise_rate)
        noise = resample_poly(background, rate//divisor, noise_rate//divisor)
        clip_seed = int.from_bytes(hashlib.sha256(f"{seed}:{row['audio_id']}".encode()).digest()[:8], 'big')
        reference, noisy, added, gain = mix_noise(audio, noise, snr_db, clip_seed)
        conditions = {'original': reference, 'noisy': noisy, 'enhanced': enhance(noisy, added, rate),
                      'original_enhanced': enhance(reference, added, rate)}
        for condition, signal in conditions.items():
            destination = output/row['breed']/row['split']/row['audio_id']/f'{condition}.wav'
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Float WAV avoids clipping and preserves enhancer overshoot for audit.
            wavfile.write(destination, rate, signal.astype(np.float32))
            records.append(dict(row, condition=condition, path=str(destination.resolve()), source_sha256=row['sha256'],
                                sha256=sha256(destination), requested_snr_db=snr_db, seed=clip_seed, common_gain=gain,
                                noise_sha256=sha256(Path(noise_path)), peak=float(np.max(np.abs(signal)))))
    (output/'pairs.json').write_text(json.dumps(records, indent=2)+'\n')
    (output/'config.json').write_text(json.dumps({'seed':seed,'snr_db':snr_db,'enhancer':'oracle spectral subtraction, power floor=0.02, FFT<=512, 50% overlap',
        'numpy':np.__version__, 'scipy':scipy.__version__, 'manifest_sha256':sha256(Path(manifest)), 'noise_path':str(Path(noise_path).resolve()), 'note':'Known added-noise profile is an optimistic control; no real-world or AI enhancement claim.'}, indent=2)+'\n')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('audit')
    command.add_argument('--root', required=True)
    command.add_argument('--output', required=True)
    command = commands.add_parser('pairs')
    command.add_argument('--manifest', required=True)
    command.add_argument('--noise', required=True)
    command.add_argument('--output', required=True)
    command.add_argument('--snr-db', type=float, required=True)
    command.add_argument('--seed', type=int, default=3108)
    args = parser.parse_args()
    if args.command == 'audit':
        summary = audit(args.root, args.output)
        print(f"Audited {summary['clips']} clips; {summary['readable']} readable; {len(summary['issues'])} issues")
        return int(bool(summary['issues']))
    records = pairs(args.manifest, args.noise, args.output, args.snr_db, args.seed)
    print(f'Wrote {len(records)} paired condition records')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
