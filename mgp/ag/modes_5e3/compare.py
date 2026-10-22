import re
import sys

def parse_modes(filepath):
    """Parse vibrational mode frequencies from a file.
    Expects lines like:   3: 60.22 cm⁻¹  or   3: 60.22 cm^-1
    """
    modes = {}
    pattern = re.compile(r'^\s*(\d+):\s*([-\d.]+)\s*cm')
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                idx = int(m.group(1))
                freq = float(m.group(2))
                modes[idx] = freq
    return modes

def compare_modes(file_a, file_b):
    modes_a = parse_modes(file_a)
    modes_b = parse_modes(file_b)

    all_indices = sorted(set(modes_a) | set(modes_b))

    print(f"{'Mode':>6}  {'File A (cm⁻¹)':>16}  {'File B (cm⁻¹)':>16}  {'Diff (B-A) cm⁻¹':>18}")
    print("-" * 62)

    for idx in all_indices:
        a = modes_a.get(idx)
        b = modes_b.get(idx)

        a_str = f"{a:>16.2f}" if a is not None else f"{'N/A':>16}"
        b_str = f"{b:>16.2f}" if b is not None else f"{'N/A':>16}"

        if a is not None and b is not None:
            diff = b - a
            diff_str = f"{diff:>+18.2f}"
        else:
            diff_str = f"{'N/A':>18}"

        print(f"{idx:>6}  {a_str}  {b_str}  {diff_str}")

if __name__ == "__main__":
    file_a = sys.argv[1] if len(sys.argv) > 1 else "0.md"
    file_b = sys.argv[2] if len(sys.argv) > 2 else "mgp.md"
    compare_modes(file_a, file_b)
