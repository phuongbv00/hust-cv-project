import os
from pathlib import Path

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent / 'data'
    tmpl_path = base_dir / 'template.jpg'
    scene_mini_dir = base_dir / 'scene' / 'positives'
    scene_others_dir = base_dir / 'scene' / 'negatives'

    if not tmpl_path.is_file() or not scene_mini_dir.is_dir() or not scene_others_dir.is_dir():
        print(
            'Please organize images under p2/data/template.jpg and p2/data/scene/{positives,negatives} (template.jpg and both directories should exist).')
    else:
        scenes = []
        exts = ('.png', '.jpg', '.jpeg', '.bmp')

        # Collect scenes from both positives and negatives
        for dir_path in (scene_mini_dir, scene_others_dir):
            for (_, _, files) in os.walk(dir_path):
                for file in files:
                    if file.lower().endswith(exts):
                        scenes.append(str(dir_path / file))
                        break

        if len(scenes) == 0:
            print('No scene image files found in p2/data/scene/{positives,negatives}')
        else:
            scenes_args = ' '.join(scenes)
            cmd = f"python -m p2.main {tmpl_path} {scenes_args}"
            os.system(cmd)
