import os

if __name__ == '__main__':
    base_dir = './input'
    tmpl_dir = os.path.join(base_dir, 'template')
    scene_dir = os.path.join(base_dir, 'scene')

    if not os.path.isdir(tmpl_dir) or not os.path.isdir(scene_dir):
        print('Please organize images under ./input/template and ./input/scene (both directories should exist).')
    else:
        templates = []
        scenes = []

        for (_, _, files) in os.walk(tmpl_dir):
            for file in files:
                if file.lower().endswith('.png') or file.lower().endswith('.jpg'):
                    templates.append(file)
            break

        for (_, _, files) in os.walk(scene_dir):
            for file in files:
                if file.lower().endswith('.png') or file.lower().endswith('.jpg'):
                    scenes.append(file)
            break

        if len(templates) == 0:
            print('No template .png files found in ./input/template')
        elif len(scenes) == 0:
            print('No scene .png files found in ./input/scene')
        else:
            scenes_args = ' '.join(os.path.join(scene_dir, s) for s in scenes)
            for template in templates:
                tmpl_path = os.path.join(tmpl_dir, template)
                cmd = f"python -m p2.main {tmpl_path} {scenes_args}"
                os.system(cmd)
