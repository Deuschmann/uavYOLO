import os
from pathlib import Path

# Scan all label files (train + val) to find max class index and detailed info
for split in ['train', 'val']:
    label_dir = Path(f'clean_baseline_dataset/labels/{split}')
    max_class = -1
    files_with_issues = []
    total_lines = 0
    bad_lines = []

    if label_dir.exists():
        for label_file in sorted(label_dir.glob('*.txt')):
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    total_lines += 1
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            class_idx = int(parts[0])
                            if class_idx > max_class:
                                max_class = class_idx
                            if class_idx >= 3:
                                files_with_issues.append((label_file.name, line_num, class_idx, line.strip()))
                        except (ValueError, IndexError) as e:
                            bad_lines.append((label_file.name, line_num, str(e), line.strip()))

        print(f"\n{split.upper()} SET:")
        print(f"  Max class index: {max_class}")
        print(f"  Num classes config: 3")
        print(f"  Total lines: {total_lines}")
        print(f"  Problematic class indices: {len(files_with_issues)}")
        print(f"  Bad format lines: {len(bad_lines)}")
        if files_with_issues:
            print("  Sample class index issues (first 5):")
            for fname, line_num, class_idx, content in files_with_issues[:5]:
                print(f"    {fname}:{line_num}: class_idx={class_idx}, line={content}")
        if bad_lines:
            print("  Sample format issues (first 5):")
            for fname, line_num, err, content in bad_lines[:5]:
                print(f"    {fname}:{line_num}: {err}, line={content}")
    else:
        print(f"\n{split.upper()} SET: Directory not found")
