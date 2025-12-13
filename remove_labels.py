import argparse
import sys
from pathlib import Path


def remove_labels_from_dataset(labels_dir, image_dir, labels_to_remove=None, keep_label=None, dry_run=True):
    """Remove specified label ids (or keep a single label) from YOLO .txt files.

    If a file becomes empty it will be deleted along with its corresponding image file.
    """
    labels_path = Path(labels_dir)
    images_path = Path(image_dir)

    if not labels_path.exists():
        print(f"Labels directory not found: {labels_dir}")
        return
    if not images_path.exists():
        print(f"Images directory not found: {image_dir}")
        return

    labels_to_remove = set(labels_to_remove or [])
    removed_count = 0
    kept_count = 0
    deleted_images = 0
    files_changed = 0

    label_files = list(labels_path.glob("*.txt"))
    print(f"Found {len(label_files)} label files in {labels_path}")

    for label_file in label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = [l for l in f.readlines() if l.strip()]

            file_labels = [int(line.split()[0]) for line in lines]

            if keep_label is not None and keep_label in file_labels:
                new_lines = [line for line in lines if int(line.split()[0]) == keep_label]
                removed_count += len(lines) - len(new_lines)
                kept_count += len(new_lines)
            else:
                new_lines = [line for line in lines if int(line.split()[0]) not in labels_to_remove]
                removed_count += len(lines) - len(new_lines)

            if new_lines:
                if new_lines != lines:
                    files_changed += 1
                    print(f"Updating: {label_file.name} -> {len(new_lines)} labels")
                    if not dry_run:
                        with open(label_file, 'w', encoding='utf-8') as f:
                            f.writelines(new_lines)
            else:
                print(f"Deleting: {label_file.name} (no labels remain)")
                if not dry_run:
                    label_file.unlink()
                    # delete corresponding image
                    image_name = label_file.stem
                    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        image_file = images_path / f"{image_name}{ext}"
                        if image_file.exists():
                            image_file.unlink()
                            deleted_images += 1
                            break

        except Exception as e:
            print(f"Error processing {label_file.name}: {e}")

    print(f"\n=== Summary ({'dry-run' if dry_run else 'applied'}) ===")
    print(f"Labels removed: {removed_count}")
    print(f"Labels kept (explicit): {kept_count}")
    print(f"Files changed: {files_changed}")
    print(f"Images deleted: {deleted_images}")


def remap_labels(labels_dir, mapping, dry_run=True):
    """Remap label ids according to mapping dict {old_id: new_id} in all .txt files."""
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        print(f"Labels directory not found: {labels_dir}")
        return

    files_changed = 0
    remapped_count = 0

    for label_file in labels_path.glob('*.txt'):
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = [l for l in f.readlines() if l.strip()]

            new_lines = []
            changed = False
            changes_in_file = 0
            for line in lines:
                parts = line.split()
                cls = int(parts[0])
                if cls in mapping:
                    parts[0] = str(mapping[cls])
                    changed = True
                    remapped_count += 1
                    changes_in_file += 1
                new_lines.append(' '.join(parts) + '\n')

            if changed:
                files_changed += 1
                print(f"Remapping in: {label_file.name} (changes: {changes_in_file})")
                if not dry_run:
                    with open(label_file, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)

        except Exception as e:
            print(f"Error processing {label_file.name}: {e}")

    print(f"\n=== Remap Summary ({'dry-run' if dry_run else 'applied'}) ===")
    print(f"Files changed: {files_changed}")
    print(f"Labels remapped: {remapped_count}")


def _parse_remap_arg(arg_str):
    mapping = {}
    for pair in arg_str.split(','):
        if not pair.strip():
            continue
        old, new = pair.split(':')
        mapping[int(old)] = int(new)
    return mapping


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove or remap YOLO label ids (dry-run by default)')
    parser.add_argument('--remove', action='store_true', help='Run the remove_labels_from_dataset logic')
    parser.add_argument('--labels-to-remove', type=str, default='', help='Comma-separated label ids to remove, e.g. "0,1,3"')
    parser.add_argument('--keep-label', type=int, default=None, help='If present in a file, keep only this label')
    parser.add_argument('--remap', type=str, default='', help='Remap labels, format: "old:new,old2:new2" e.g. "2:0"')
    parser.add_argument('--apply', action='store_true', help='Actually write changes; default is dry-run')
    parser.add_argument('--dataset', type=str, default=r"D:\\HAR\\mine2_divided_dataset", help='Root dataset folder')

    args = parser.parse_args()

    dry_run = not args.apply
    root = Path(args.dataset)

    # Define paths used by functions
    train_labels = root /'labels'/ 'train'
    train_images = root /'images'/ 'train'
    valid_labels = root /'labels'/ 'val'
    valid_images = root /'images'/'val'
    test_labels = root /'labels'/ 'test'
    test_images = root /'images'/ 'test'

    if args.remove:
        labels_to_remove = [int(x) for x in args.labels_to_remove.split(',')] if args.labels_to_remove else []
        print('Processing TRAIN set...')
        remove_labels_from_dataset(train_labels, train_images, labels_to_remove=labels_to_remove, keep_label=args.keep_label, dry_run=dry_run)
        print('\n' + '='*50)
        print('\nProcessing VALID set...')
        remove_labels_from_dataset(valid_labels, valid_images, labels_to_remove=labels_to_remove, keep_label=args.keep_label, dry_run=dry_run)
        print('\n' + '='*50)
        print('\nProcessing TEST set...')
        remove_labels_from_dataset(test_labels, test_images, labels_to_remove=labels_to_remove, keep_label=args.keep_label, dry_run=dry_run)

    if args.remap:
        mapping = _parse_remap_arg(args.remap)
        print(f"Remapping labels with mapping: {mapping} (root: {root})")
        print('TRAIN labels:')
        remap_labels(train_labels, mapping, dry_run=dry_run)
        print('\n' + '='*50)
        print('VALID labels:')
        remap_labels(valid_labels, mapping, dry_run=dry_run)
        print('\n' + '='*50)
        print('TEST labels:')
        remap_labels(test_labels, mapping, dry_run=dry_run)

    if not args.remove and not args.remap:
        parser.print_help()
